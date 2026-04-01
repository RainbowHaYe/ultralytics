"""
YOLOv8n 小零件检测改进模块集合

基于 20 组消融实验结果，精选表现最优的互补模块组合：
1. SPDConv  — SPD无损下采样 (v2_spd +0.10%, FPS仅降1%)
2. GSConv   — 分组混洗卷积 (v3_gsc_neck +0.44%, 精度最高)
3. C3_GSC   — GSConv优化的C3 Neck结构
4. ECA      — 高效通道注意力 (误检最少 FP=28, FN=11)

学术参考：
- SPD-Conv: Sunkara & Luo, "No More Strided Convolutions or Pooling",
  https://arxiv.org/abs/2208.03641
- GSConv:   Li et al., "Slim-neck by GSConv", https://arxiv.org/abs/2206.02424
- ECA-Net:  Wang et al., "ECA-Net: Efficient Channel Attention",
  https://arxiv.org/abs/1910.03151
"""

import math

import torch
import torch.nn as nn

__all__ = [
    "SPDConv", "SPDLiteConv", "GSConv", "GSCBottleneck", "C3_GSC", "ECA",
    "C2f_CBAM", "StarBottleneck", "StarC2f", "StarC2f_CBAM",
    "CoordAtt", "SimBottleneck", "SimC2f", "WAFF",
    "BiFPNFuse", "C3Ghost_CBAM",
]


# ============================================================================
# 1. SPDConv — Space-to-Depth Convolution（无损下采样）
# ============================================================================
class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution.

    通过 PixelUnshuffle 将空间像素重排到通道维度实现下采样，
    相比 stride-2 Conv 不丢失任何像素信息，对小目标极为友好。

    H×W → (H/2)×(W/2)，通道扩展 4 倍后用 Conv 融合回目标通道数。

    References:
        Sunkara & Luo, "No More Strided Convolutions or Pooling:
        A New Building Block for CNNs", https://arxiv.org/abs/2208.03641
    """

    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv, autopad

        if s != 2:
            raise ValueError("SPDConv only supports stride=2")
        self.spd = nn.PixelUnshuffle(2)  # [B,C,H,W] → [B,4C,H/2,W/2]
        self.conv = Conv(c1 * 4, c2, k, 1, autopad(k, None))

    def forward(self, x):
        return self.conv(self.spd(x))


# ============================================================================
# 2. GSConv — Group Shuffle Convolution（分组混洗卷积）
# ============================================================================
class GSConv(nn.Module):
    """
    Group Shuffle Convolution.

    Depthwise Conv 提取空间特征 → 分组 Pointwise Conv 降低参数 →
    Channel Shuffle 确保跨组信息交互。

    适合在 Neck 中替代标准 Conv 进行特征融合，降低计算量同时保持
    跨通道信息流动（Channel Shuffle 仅是内存重排，开销极小）。

    References:
        Li et al., "Slim-neck by GSConv: A better design paradigm of
        detector architectures for autonomous vehicles",
        https://arxiv.org/abs/2206.02424
    """

    def __init__(self, c1, c2, k=1, s=1, g=4):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv, autopad

        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, None), groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, groups=g, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.g = g

    def forward(self, x):
        x = self.act(self.bn1(self.dwconv(x)))
        x = self.bn2(self.pwconv(x))
        # Channel Shuffle
        b, c, h, w = x.size()
        x = x.view(b, self.g, c // self.g, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, c, h, w)
        return self.act(x)


# ============================================================================
# 3. GSCBottleneck — GSConv 瓶颈层
# ============================================================================
class GSCBottleneck(nn.Module):
    """Bottleneck using GSConv, designed for lightweight Neck fusion."""

    def __init__(self, c1, c2, shortcut=True, g=4, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        c_ = max(8, int(c2 * e))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = GSConv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ============================================================================
# 4. C3_GSC — GSConv 优化的 C3 结构（Neck 专用）
# ============================================================================
class C3_GSC(nn.Module):
    """
    C3 module with GSConv-based bottlenecks.

    C3 结构（双分支 + 拼接）保证梯度流稳定性，
    内部 GSCBottleneck 实现特征融合极致减脂。

    在 20 组消融实验中，v3_gsc_neck 取得最高 mAP50-95 (+0.44%)，
    同时参数量减少 16%。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=4, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        c_ = max(8, int(c2 * e))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[GSCBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv2(x)), self.cv1(x)], 1))


# ============================================================================
# 5. ECA — Efficient Channel Attention（高效通道注意力）
# ============================================================================
class ECA(nn.Module):
    """
    Efficient Channel Attention.

    仅用 1 个 1D 卷积实现通道注意力，参数量仅约 5 个（取决于核大小），
    几乎不增加计算开销。核大小根据通道数自适应计算。

    在错误分析中 ECA 模型取得最少误检 (FP=28) 和最少遗漏 (FN=11)。

    References:
        Wang et al., "ECA-Net: Efficient Channel Attention for Deep
        Convolutional Neural Networks", https://arxiv.org/abs/1910.03151
    """

    def __init__(self, c1):
        super().__init__()
        # 自适应核大小：k = |log2(C)/2 + 0.5|_odd
        t = int(abs(math.log2(c1) + 1) / 2)
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # (B, C, 1, 1)
        return x * y.expand_as(x)


# ============================================================================
# 6. C2f_CBAM — C2f + CBAM Attention (weight-transfer-friendly)
# ============================================================================
class C2f_CBAM(nn.Module):
    """C2f module with CBAM attention applied to the output.

    Internally mirrors the C2f parameter structure (cv1, cv2, m) so that
    pretrained YOLOv8 weights transfer seamlessly — only the extra ``cbam``
    sub-module is randomly initialised.

    Uses the built-in CBAM from ``ultralytics.nn.modules.conv`` which
    implements Channel Attention (global avg pool → 1×1 conv → sigmoid)
    followed by Spatial Attention (concat avg+max → 7×7 conv → sigmoid).

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of Bottleneck repeats.
        shortcut (bool): Whether to use residual connections inside Bottleneck.
        g (int): Group convolution groups.
        e (float): Expansion ratio for hidden channels.

    References:
        Woo et al., "CBAM: Convolutional Block Attention Module",
        ECCV 2018, https://arxiv.org/abs/1807.06521
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.block import Bottleneck
        from ultralytics.nn.modules.conv import CBAM, Conv

        self.c = int(c2 * e)  # hidden channels (kept for compatibility)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
        self.cbam = CBAM(c2)

    def forward(self, x):
        """C2f forward pass followed by CBAM attention."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cbam(self.cv2(torch.cat(y, 1)))


# ============================================================================
# 7. StarBottleneck — Star Operation Bottleneck (CVPR 2024)
# ============================================================================
class StarBottleneck(nn.Module):
    """Star operation bottleneck from StarNet.

    Replaces the standard dual-3×3-conv Bottleneck with a star-operation
    paradigm: DWConv for spatial mixing → two 1×1 branches → element-wise
    multiply (implicit high-dimensional feature mapping) → 1×1 project back
    → DWConv for spatial refinement.

    The element-wise multiplication of two d-dimensional vectors is
    mathematically equivalent to a polynomial feature mapping into d²
    dimensions, dramatically expanding model capacity WITHOUT actually
    computing in that high-dimensional space.

    Compared to standard Bottleneck (two 3×3 convs ≈ 18c² params, 18c²HW FLOPs):
        StarBottleneck ≈ 3c²+98c params, ~6c²HW+98cHW FLOPs → **~60% fewer FLOPs**

    Args:
        c (int): Number of input/output channels.
        shortcut (bool): Whether to use residual connection.
        mlp_ratio (int): Expansion ratio for the star branches.

    References:
        Ma et al., "Rewrite the Stars", CVPR 2024,
        https://arxiv.org/abs/2403.19967
    """

    def __init__(self, c, shortcut=True, mlp_ratio=3):
        super().__init__()
        mid = int(c * mlp_ratio)
        # Spatial mixing
        self.dwconv = nn.Sequential(
            nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False),
            nn.BatchNorm2d(c),
        )
        # Star operation branches
        self.f1 = nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.ReLU6(inplace=True))
        self.f2 = nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.ReLU6(inplace=True))
        # Project back
        self.g = nn.Conv2d(mid, c, 1, bias=False)
        # Second spatial mixing + residual fusion
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False),
            nn.BatchNorm2d(c),
        )
        self.add = shortcut

    def forward(self, x):
        """Forward pass with star operation."""
        res = x
        x = self.dwconv(x)
        x = self.g(self.f1(x) * self.f2(x))  # star operation
        x = self.dwconv2(x + res if self.add else x)
        return x


# ============================================================================
# 8. StarC2f — C2f with Star Operation Bottleneck
# ============================================================================
class StarC2f(nn.Module):
    """C2f module with StarBottleneck replacing standard Bottleneck.

    Keeps C2f's split-concat-conv structure (cv1, cv2, m) for partial
    pretrained weight transfer. Only the internal bottleneck changes
    from standard 3×3 convs to star-operation paradigm.

    Benefits over standard C2f:
      - ~60% fewer FLOPs per bottleneck (DWConv + 1×1 vs 3×3)
      - Implicit high-dimensional feature interaction via element-wise multiply
      - Full TensorRT/ONNX compatibility (all standard ops)

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of StarBottleneck repeats.
        shortcut (bool): Whether to use residual connections.
        g (int): Unused (kept for API compatibility with C2f).
        e (float): Expansion ratio for hidden channels.

    References:
        Ma et al., "Rewrite the Stars", CVPR 2024,
        https://arxiv.org/abs/2403.19967
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            StarBottleneck(self.c, shortcut) for _ in range(n)
        )

    def forward(self, x):
        """StarC2f forward: split → star bottlenecks → concat → conv."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ============================================================================
# 9. StarC2f_CBAM — StarC2f + CBAM Attention
# ============================================================================
class StarC2f_CBAM(nn.Module):
    """StarC2f with CBAM attention on the output.

    Combines two complementary improvements:
    1. Star operation: Implicit high-dim feature expansion via element-wise
       multiply, reducing FLOPs while maintaining capacity.
    2. CBAM attention: Channel + spatial attention to refine features,
       improving recall and localization accuracy.

    Designed for edge deployment (Jetson Orin Nano): fewer FLOPs than
    baseline C2f while achieving higher accuracy.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of StarBottleneck repeats.
        shortcut (bool): Whether to use residual connections.
        g (int): Unused (API compatibility).
        e (float): Expansion ratio for hidden channels.

    References:
        Ma et al., "Rewrite the Stars", CVPR 2024
        Woo et al., "CBAM", ECCV 2018
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import CBAM, Conv

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            StarBottleneck(self.c, shortcut) for _ in range(n)
        )
        self.cbam = CBAM(c2)

    def forward(self, x):
        """StarC2f forward + CBAM attention refinement."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cbam(self.cv2(torch.cat(y, 1)))


# ============================================================================
# 10. SPDLiteConv — Lightweight SPD Convolution (1×1 融合)
# ============================================================================
class SPDLiteConv(nn.Module):
    """Lightweight Space-to-Depth Convolution.

    Similar to SPDConv but uses a 1×1 convolution for channel fusion instead
    of 3×3, significantly reducing parameters and FLOPs while preserving the
    lossless spatial-to-channel rearrangement via PixelUnshuffle.

    H×W → (H/2)×(W/2), channels expand 4× then fused back via 1×1 Conv.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        k (int): Kernel size for the fusion conv (default 1).
        s (int): Stride factor (must be 2).

    References:
        Adapted from SPD-Conv (Sunkara & Luo, arXiv:2208.03641).
    """

    def __init__(self, c1, c2, k=1, s=2):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        if s != 2:
            raise ValueError("SPDLiteConv only supports stride=2")
        self.spd = nn.PixelUnshuffle(2)  # [B, C, H, W] → [B, 4C, H/2, W/2]
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x):
        """Forward: PixelUnshuffle → 1×1 Conv fusion."""
        return self.conv(self.spd(x))


# ============================================================================
# 11. CoordAtt — Coordinate Attention
# ============================================================================
class CoordAtt(nn.Module):
    """Coordinate Attention module.

    Encodes channel relationships and long-range dependencies with precise
    positional information along both H and W directions. Two 1D global
    pooling operations (along H and W) capture long-range spatial interactions
    with precise positional encoding.

    The module first pools along H and W separately, concatenates them,
    passes through a shared 1×1 bottleneck, then splits back and generates
    per-direction attention maps via sigmoid.

    Args:
        c1 (int): Input/output channels.
        reduction (int): Channel reduction ratio for the bottleneck.

    References:
        Hou et al., "Coordinate Attention for Efficient Mobile Network
        Design", CVPR 2021, https://arxiv.org/abs/2103.02907
    """

    def __init__(self, c1, reduction=32):
        super().__init__()
        mid = max(8, c1 // reduction)
        # 不使用 nn.AdaptiveAvgPool2d，改用 torch.mean 以保证 CUDA 确定性
        self.conv1 = nn.Conv2d(c1, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, c1, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, c1, 1, bias=False)

    def forward(self, x):
        """Forward: H-pool + W-pool → shared bottleneck → split → attention."""
        _, _, H, W = x.shape
        x_h = x.mean(dim=3, keepdim=True)  # (B, C, H, 1) — 沿 W 取均值
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B, C, 1, W) → (B, C, W, 1)
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))  # (B, mid, H+W, 1)
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        x_w = self.conv_w(x_w).permute(0, 1, 3, 2).sigmoid()  # (B, C, 1, W)
        return x * x_h * x_w


# ============================================================================
# 12. SimBottleneck — Simplified Bottleneck (1×1 + 3×3)
# ============================================================================
class SimBottleneck(nn.Module):
    """Simplified Bottleneck with 1×1 + 3×3 convolutions.

    Replaces the standard Bottleneck's 3×3 + 3×3 structure with 1×1 + 3×3,
    reducing parameters by ~44% per bottleneck while maintaining receptive
    field through the 3×3 convolution.

    Args:
        c (int): Input/output channels.
        shortcut (bool): Whether to add residual connection.
    """

    def __init__(self, c, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        self.cv1 = Conv(c, c, 1, 1)  # 1×1 channel mixing
        self.cv2 = Conv(c, c, 3, 1)  # 3×3 spatial conv
        self.add = shortcut

    def forward(self, x):
        """Forward: 1×1 → 3×3 with optional residual."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ============================================================================
# 13. SimC2f — C2f with Simplified Bottleneck
# ============================================================================
class SimC2f(nn.Module):
    """C2f module with SimBottleneck (1×1+3×3) replacing standard Bottleneck.

    Keeps C2f's split-concat-conv structure for pretrained weight partial
    transfer (cv1, cv2 are compatible). Internal bottleneck changes from
    3×3+3×3 to 1×1+3×3, cutting bottleneck params by ~44%.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of SimBottleneck repeats.
        shortcut (bool): Whether to use residual connections.
        g (int): Unused (API compatibility with C2f).
        e (float): Expansion ratio for hidden channels.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(SimBottleneck(self.c, shortcut) for _ in range(n))

    def forward(self, x):
        """SimC2f forward: split → sim-bottlenecks → concat → conv."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ============================================================================
# 14. WAFF — Weighted Adaptive Feature Fusion
# ============================================================================
class WAFF(nn.Module):
    """Weighted Adaptive Feature Fusion.

    Combines BiFPN-style learnable per-branch weights with ECA channel
    attention for adaptive multi-scale feature fusion. Designed to replace
    standard Concat in the top-down pathway of the Neck.

    Each input branch receives a softmax-normalized learnable weight,
    enabling the network to automatically balance contributions from
    different scales. After concatenation, ECA channel attention further
    refines the fused features with negligible overhead.

    Args:
        c1s (list[int]): List of input channel sizes from each branch.

    References:
        BiFPN: Tan et al., "EfficientDet", CVPR 2020
        ECA-Net: Wang et al., arXiv:1910.03151
    """

    def __init__(self, c1s):
        super().__init__()
        n = len(c1s)
        self.weights = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.eps = 1e-4
        c_out = sum(c1s)
        # ECA on concatenated output
        t = int(abs(math.log2(c_out) + 1) / 2)
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward: weighted branches → concat → ECA attention."""
        w = torch.softmax(self.weights, dim=0)
        x = torch.cat([w[i] * xi for i, xi in enumerate(x)], dim=1)
        # ECA channel attention
        y = x.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.eca_conv(y)  # (B, 1, C)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # (B, C, 1, 1)
        return x * y.expand_as(x)


# ============================================================================
# 15. BiFPNFuse — BiFPN-style Weighted Feature Fusion (pure)
# ============================================================================
class BiFPNFuse(nn.Module):
    """BiFPN-style fast normalized weighted feature fusion.

    Implements the fast normalized fusion from EfficientDet / BiFPN.
    Each input branch receives a learnable weight (ReLU-activated) with
    fast normalization: w_i / (sum(w_j) + eps). Unlike WAFF, this module
    does NOT include ECA channel attention — it is a pure weighted fusion
    baseline to verify BiFPN's standalone contribution.

    Args:
        c1s (list[int]): List of input channel sizes from each branch.

    References:
        Tan et al., "EfficientDet: Scalable and Efficient Object Detection",
        CVPR 2020, https://arxiv.org/abs/1911.09070
    """

    def __init__(self, c1s):
        super().__init__()
        n = len(c1s)
        # Learnable per-branch weights, initialized to 1.0
        self.weights = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, x):
        """Forward: fast normalized weighted fusion → concat."""
        # ReLU ensures weights >= 0 (fast normalized fusion)
        w = torch.relu(self.weights)
        w = w / (w.sum() + self.eps)
        return torch.cat([w[i] * xi for i, xi in enumerate(x)], dim=1)


# ============================================================================
# 16. C3Ghost_CBAM — C3Ghost with CBAM Attention
# ============================================================================
class C3Ghost_CBAM(nn.Module):
    """C3 module with GhostBottleneck and CBAM attention.

    Combines GhostNet's cheap-operation feature generation with CBAM's
    channel + spatial attention for lightweight yet attentive feature
    extraction. This represents the "GhostNet backbone + CBAM" design
    from the original proposal.

    Architecture:
        Input → conv1(1×1) → conv2(1×1) → GhostBottleneck×n → concat → conv3 → CBAM → output

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of GhostBottleneck repeats.
        shortcut (bool): Unused (API compatibility with C3).
        g (int): Unused (API compatibility with C3).
        e (float): Expansion ratio for hidden channels.

    References:
        Han et al., "GhostNet", CVPR 2020, https://arxiv.org/abs/1911.11907
        Woo et al., "CBAM", ECCV 2018, https://arxiv.org/abs/1807.06521
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.block import GhostBottleneck
        from ultralytics.nn.modules.conv import CBAM, Conv

        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))
        self.cbam = CBAM(c2)

    def forward(self, x):
        """Forward: split → GhostBottlenecks → concat → conv → CBAM."""
        return self.cbam(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)))
