"""
消融实验结果汇总分析 — selfpart499_augmented 数据集
=================================================
生成:
  1. 消融实验汇总表格 (控制台 + LaTeX)
  2. 训练过程曲线可视化 (mAP / Loss / LR)
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ─── 配置 ───────────────────────────────────────────────────────────
BASE = Path("results/runs/detect/results/runs/self499_aug")
OUT_DIR = Path("results/ablation_analysis_self499aug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 实验定义: (文件夹名, 显示名, 组件描述, Layers, Params, GFLOPs)
EXPERIMENTS = [
    ("baseline",        "Baseline (YOLOv8n)",      "—",                                      129, 3157200, 8.8575),
    ("ablation_SimC2f", "+SimC2f",                  "SimC2f",                                  129, 2592373, 7.1520),
    ("ablation_S",      "+SimC2f+S",                "SimC2f + SPDLiteConv",                    133, 2374773, 6.6277),
    ("ablation_CA",     "+SimC2f+CA",               "SimC2f + CoordAtt",                       134, 2598533, 7.1586),
    ("ablation_W",      "+SimC2f+W",                "SimC2f + WAFF",                           131, 2592387, 7.1543),
    ("ablation_SW",     "+SimC2f+S+W",              "SimC2f + SPDLiteConv + WAFF",             135, 2374787, 6.6300),
    ("ablation_SCA",    "+SimC2f+S+CA",             "SimC2f + SPDLiteConv + CoordAtt",         138, 2380933, 6.6343),
    ("ablation_CAW",    "+SimC2f+CA+W",             "SimC2f + CoordAtt + WAFF",                136, 2598547, 7.1609),
    ("SCAW",            "SCAW (全部)",              "SimC2f + SPDLiteConv + CoordAtt + WAFF",  140, 2380947, 6.6366),
]

COLORS = [
    "#888888",  # baseline - grey
    "#1f77b4",  # SimC2f
    "#ff7f0e",  # +S
    "#2ca02c",  # +CA
    "#d62728",  # +W
    "#9467bd",  # +SW
    "#8c564b",  # +SCA
    "#e377c2",  # +CAW
    "#000000",  # SCAW - black
]


# ─── 数据读取 ───────────────────────────────────────────────────────
def load_csv(folder_name):
    fpath = BASE / folder_name / "results.csv"
    with open(fpath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # 清理 key 中可能的空格
    cleaned = []
    for row in rows:
        cleaned.append({k.strip(): v.strip() for k, v in row.items()})
    return cleaned


def get_col(rows, col):
    return np.array([float(r[col]) for r in rows])


# ─── 1. 汇总表格 ────────────────────────────────────────────────────
print("\n" + "=" * 150)
print("消融实验汇总表格  —  selfpart499_augmented 数据集 (所有实验: epochs=300, batch=64, AdamW, NWD loss)")
print("=" * 150)

header = f"{'实验配置':<25} {'组件':<35} {'Params(M)':>9} {'GFLOPs':>7} {'Epochs':>6} {'Best mAP50':>10} {'Best mAP50-95':>14} {'Precision':>10} {'Recall':>10} {'Δ mAP50-95':>11}"
print(header)
print("-" * 150)

table_data = []
baseline_best_map5095 = None
baseline_params = None
baseline_gflops = None

for folder, display, desc, layers, params, gflops in EXPERIMENTS:
    rows = load_csv(folder)
    best_row = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    best_map50_row = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))

    best_map5095 = float(best_row["metrics/mAP50-95(B)"])
    best_map50 = float(best_map50_row["metrics/mAP50(B)"])
    precision = float(best_row["metrics/precision(B)"])
    recall = float(best_row["metrics/recall(B)"])
    total_ep = int(rows[-1]["epoch"])
    params_m = params / 1e6

    if baseline_best_map5095 is None:
        baseline_best_map5095 = best_map5095
        baseline_params = params
        baseline_gflops = gflops
        delta = "—"
    else:
        d = best_map5095 - baseline_best_map5095
        delta = f"{d:+.3f}" if d != 0 else "0.000"

    print(f"{display:<25} {desc:<35} {params_m:>9.2f} {gflops:>7.2f} {total_ep:>6} {best_map50:>10.5f} {best_map5095:>14.5f} {precision:>10.5f} {recall:>10.5f} {delta:>11}")

    table_data.append({
        "folder": folder,
        "display": display,
        "desc": desc,
        "epochs": total_ep,
        "layers": layers,
        "params": params,
        "params_m": params_m,
        "gflops": gflops,
        "best_map50": best_map50,
        "best_map5095": best_map5095,
        "precision": precision,
        "recall": recall,
    })

print("=" * 150)
print("注: Best mAP50-95 取各实验训练过程中的历史最佳值; Δ mAP50-95 为相对于 Baseline 的提升量")
print(f"     Baseline Params = {baseline_params/1e6:.2f}M, GFLOPs = {baseline_gflops:.2f}")
print()

# ─── 组件贡献分析 ────────────────────────────────────────────────────
simc2f_map = next(d for d in table_data if d["folder"] == "ablation_SimC2f")["best_map5095"]
baseline_map = table_data[0]["best_map5095"]
s_map = next(d for d in table_data if d["folder"] == "ablation_S")["best_map5095"]
ca_map = next(d for d in table_data if d["folder"] == "ablation_CA")["best_map5095"]
w_map = next(d for d in table_data if d["folder"] == "ablation_W")["best_map5095"]
scaw_map = next(d for d in table_data if d["folder"] == "SCAW")["best_map5095"]

print("组件独立贡献分析 (相对于 SimC2f 基底):")
print(f"  SimC2f 基底 vs Baseline:   Δ = {simc2f_map - baseline_map:+.5f}")
print(f"  +SPDLiteConv (S):          Δ = {s_map - simc2f_map:+.5f}")
print(f"  +CoordAtt (CA):            Δ = {ca_map - simc2f_map:+.5f}")
print(f"  +WAFF (W):                 Δ = {w_map - simc2f_map:+.5f}")
print(f"  SCAW (全部) vs Baseline:   Δ = {scaw_map - baseline_map:+.5f}")
print()


# ─── 2. 训练曲线可视化 ──────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

all_data = {}
for folder, display, *_ in EXPERIMENTS:
    rows = load_csv(folder)
    all_data[folder] = {
        "display": display,
        "epoch": get_col(rows, "epoch"),
        "mAP50": get_col(rows, "metrics/mAP50(B)"),
        "mAP50-95": get_col(rows, "metrics/mAP50-95(B)"),
        "precision": get_col(rows, "metrics/precision(B)"),
        "recall": get_col(rows, "metrics/recall(B)"),
        "train_box": get_col(rows, "train/box_loss"),
        "train_cls": get_col(rows, "train/cls_loss"),
        "train_dfl": get_col(rows, "train/dfl_loss"),
        "val_box": get_col(rows, "val/box_loss"),
        "val_cls": get_col(rows, "val/cls_loss"),
        "val_dfl": get_col(rows, "val/dfl_loss"),
        "lr": get_col(rows, "lr/pg0"),
    }


def plot_metric(ax, key, ylabel, title):
    for i, (folder, display, *_) in enumerate(EXPERIMENTS):
        d = all_data[folder]
        ax.plot(d["epoch"], d[key], color=COLORS[i], label=display, linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# ── 图1: mAP 曲线 ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
plot_metric(axes[0], "mAP50", "mAP@0.5", "mAP@0.5 训练曲线")
plot_metric(axes[1], "mAP50-95", "mAP@0.5:0.95", "mAP@0.5:0.95 训练曲线")
axes[1].legend(loc="lower right", ncol=1, framealpha=0.9)
fig.suptitle("消融实验 — mAP 指标对比 (selfpart499_augmented)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_mAP_curves.png", bbox_inches="tight")
print(f"已保存: {OUT_DIR / 'ablation_mAP_curves.png'}")

# ── 图2: Precision & Recall ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
plot_metric(axes[0], "precision", "Precision", "Precision 训练曲线")
plot_metric(axes[1], "recall", "Recall", "Recall 训练曲线")
axes[1].legend(loc="lower right", ncol=1, framealpha=0.9)
fig.suptitle("消融实验 — Precision & Recall 对比", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_PR_curves.png", bbox_inches="tight")
print(f"已保存: {OUT_DIR / 'ablation_PR_curves.png'}")

# ── 图3: 验证损失 ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (key, title) in zip(axes, [
    ("val_box", "Val Box Loss"),
    ("val_cls", "Val Cls Loss"),
    ("val_dfl", "Val DFL Loss"),
]):
    plot_metric(ax, key, "Loss", title)
axes[2].legend(loc="upper right", ncol=1, framealpha=0.9, fontsize=7)
fig.suptitle("消融实验 — 验证损失对比", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_val_loss_curves.png", bbox_inches="tight")
print(f"已保存: {OUT_DIR / 'ablation_val_loss_curves.png'}")

# ── 图4: 训练损失 ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (key, title) in zip(axes, [
    ("train_box", "Train Box Loss"),
    ("train_cls", "Train Cls Loss"),
    ("train_dfl", "Train DFL Loss"),
]):
    plot_metric(ax, key, "Loss", title)
axes[2].legend(loc="upper right", ncol=1, framealpha=0.9, fontsize=7)
fig.suptitle("消融实验 — 训练损失对比", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_train_loss_curves.png", bbox_inches="tight")
print(f"已保存: {OUT_DIR / 'ablation_train_loss_curves.png'}")

# ── 图5: 学习率 ──
fig, ax = plt.subplots(figsize=(10, 4))
plot_metric(ax, "lr", "Learning Rate", "学习率调度曲线")
ax.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_lr_curves.png", bbox_inches="tight")
print(f"已保存: {OUT_DIR / 'ablation_lr_curves.png'}")

# ── 图6: 柱状图对比 Best mAP50-95 ──
fig, ax = plt.subplots(figsize=(12, 5))
names = [d["display"] for d in table_data]
vals = [d["best_map5095"] for d in table_data]
bars = ax.bar(range(len(names)), vals, color=COLORS, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Best mAP@0.5:0.95")
ax.set_title("消融实验 — Best mAP@0.5:0.95 对比")
ax.set_ylim(min(vals) - 0.005, max(vals) + 0.005)
ax.grid(axis="y", alpha=0.3)
# 添加数值标签
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_best_mAP5095_bar.png", bbox_inches="tight")
print(f"已保存: {OUT_DIR / 'ablation_best_mAP5095_bar.png'}")

# ── 图6b: Params (M) + GFLOPs 柱状图 ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
for ax_i, (metric, ylabel, unit) in enumerate([
    ("params_m", "Params (M)", "M"),
    ("gflops", "GFLOPs", ""),
]):
    ax = axes[ax_i]
    vals_m = [d[metric] for d in table_data]
    bars_m = ax.bar(range(len(names)), vals_m, color=COLORS, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"消融实验 — {ylabel} 对比")
    ax.set_ylim(0, max(vals_m) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars_m, vals_m):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals_m) * 0.01,
                f"{val:.2f}{unit}", ha="center", va="bottom", fontsize=8)
fig.suptitle("消融实验 — 模型复杂度对比", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_params_gflops_bar.png", bbox_inches="tight")
print(f"已保存: {OUT_DIR / 'ablation_params_gflops_bar.png'}")

# ── 图6c: Params vs mAP50-95 散点图 (效率-精度权衡) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax_i, (x_key, x_label) in enumerate([
    ("params_m", "Params (M)"),
    ("gflops", "GFLOPs"),
]):
    ax = axes[ax_i]
    for i, d in enumerate(table_data):
        ax.scatter(d[x_key], d["best_map5095"], color=COLORS[i], s=100, zorder=5,
                   edgecolors="black", linewidth=0.5)
        ax.annotate(d["display"], (d[x_key], d["best_map5095"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Best mAP@0.5:0.95")
    ax.set_title(f"{x_label} vs mAP@0.5:0.95")
    ax.grid(True, alpha=0.3)
fig.suptitle("消融实验 — 效率-精度权衡", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_efficiency_accuracy_tradeoff.png", bbox_inches="tight")
print(f"已保存: {OUT_DIR / 'ablation_efficiency_accuracy_tradeoff.png'}")

# ── 图7: 组件消融热力图 (勾选矩阵) ──
components = ["SimC2f", "SPDLiteConv\n(S)", "CoordAtt\n(CA)", "WAFF\n(W)"]
config_matrix = np.array([
    [0, 0, 0, 0],  # baseline
    [1, 0, 0, 0],  # SimC2f
    [1, 1, 0, 0],  # +S
    [1, 0, 1, 0],  # +CA
    [1, 0, 0, 1],  # +W
    [1, 1, 0, 1],  # +SW
    [1, 1, 1, 0],  # +SCA
    [1, 0, 1, 1],  # +CAW
    [1, 1, 1, 1],  # SCAW
], dtype=float)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(config_matrix, cmap="Greens", aspect="auto", vmin=-0.2, vmax=1.2)

# 标注勾选
for i in range(config_matrix.shape[0]):
    for j in range(config_matrix.shape[1]):
        mark = "✓" if config_matrix[i, j] else "✗"
        color = "white" if config_matrix[i, j] else "#999999"
        ax.text(j, i, mark, ha="center", va="center", fontsize=14, fontweight="bold", color=color)

# 在右侧添加 mAP50-95 / Params / GFLOPs
for i, d in enumerate(table_data):
    ax.text(config_matrix.shape[1] + 0.3, i,
            f"mAP50-95: {d['best_map5095']:.4f}  |  {d['params_m']:.2f}M  |  {d['gflops']:.2f}G",
            ha="left", va="center", fontsize=8)

ax.set_xticks(range(len(components)))
ax.set_xticklabels(components, fontsize=10)
ax.set_yticks(range(len(table_data)))
ax.set_yticklabels([d["display"] for d in table_data], fontsize=9)
ax.set_title("消融实验组件矩阵 + mAP@0.5:0.95 / Params / GFLOPs", fontsize=13)
ax.set_xlim(-0.5, config_matrix.shape[1] + 4.5)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_component_matrix.png", bbox_inches="tight")
print(f"已保存: {OUT_DIR / 'ablation_component_matrix.png'}")

# ── 图8: mAP50-95 后期稳定区 (epoch 200+) 放大对比 ──
fig, ax = plt.subplots(figsize=(12, 5))
for i, (folder, display, *_) in enumerate(EXPERIMENTS):
    d = all_data[folder]
    mask = d["epoch"] >= 200
    if mask.any():
        ax.plot(d["epoch"][mask], d["mAP50-95"][mask], color=COLORS[i],
                label=display, linewidth=1.5, alpha=0.85)
ax.set_xlabel("Epoch")
ax.set_ylabel("mAP@0.5:0.95")
ax.set_title("mAP@0.5:0.95 后期收敛对比 (Epoch ≥ 200)")
ax.legend(loc="lower left", ncol=2, framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_mAP5095_late_zoom.png", bbox_inches="tight")
print(f"已保存: {OUT_DIR / 'ablation_mAP5095_late_zoom.png'}")

print(f"\n所有图表已保存到: {OUT_DIR.resolve()}")
print("完成!")
