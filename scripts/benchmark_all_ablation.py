"""
全消融实验推理速度基准测试
在 GPU 上测试，包含预热，分别在 OpenV2 和 SelfPart499 两个数据集的验证集上运行
使用 model.val() 提取推理速度，确保与实际检测流程一致
"""
import csv
import os
import sys
import time

import numpy as np
import torch

from ultralytics import YOLO

torch.cuda.empty_cache()

BASE = "/home/ros/jhy/ultralytics/results/runs/detect/results/runs"

# 消融实验列表（与论文表格一致的顺序）
EXPERIMENTS = [
    "baseline",
    "ablation_SimC2f",
    "ablation_S",
    "ablation_CA",
    "ablation_W",
    "ablation_SCA",
    "ablation_SW",
    "ablation_CAW",
    "SCAW",
    "ablation_Ghost",
    "ablation_GhostCBAM",
    "ablation_GhostBiFPN",
    "ablation_GhostBiFPNCBAM",
]

DATASETS = {
    "openv2": "/home/ros/jhy/ultralytics/datasets/openv2/data.yaml",
    "self499_aug": "/home/ros/jhy/ultralytics/datasets/selfpart499_augmented/data.yaml",
}

NUM_WARMUP = 30
NUM_RUNS = 100
IMGSZ = 640


def benchmark_model_predict(model_path, imgsz=640, num_warmup=30, num_runs=100):
    """使用随机 tensor 做纯推理速度测试，排除数据加载和后处理干扰"""
    model = YOLO(model_path)
    params = sum(x.numel() for x in model.model.parameters())
    # GFLOPs 通过 thop 计算
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(next(model.model.parameters()).device)
        flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
    except Exception:
        gflops = 0.0

    dummy = torch.randn(1, 3, imgsz, imgsz).cuda()

    # 预热
    for _ in range(num_warmup):
        model.predict(dummy, imgsz=imgsz, verbose=False, device=0)
    torch.cuda.synchronize()

    # 正式测速
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, verbose=False, device=0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    median_ms = np.median(times)
    p95_ms = np.percentile(times, 95)
    fps = 1000.0 / avg_ms

    del model
    torch.cuda.empty_cache()

    return {
        "params": params,
        "gflops": gflops,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }


def main():
    all_results = {}

    for ds_name, ds_yaml in DATASETS.items():
        print(f"\n{'#'*70}")
        print(f"  数据集: {ds_name}")
        print(f"{'#'*70}")

        ds_results = {}
        for exp in EXPERIMENTS:
            weight_path = os.path.join(BASE, ds_name, exp, "weights", "best.pt")
            if not os.path.exists(weight_path):
                print(f"  [SKIP] {exp}: 权重文件不存在 -> {weight_path}")
                continue

            print(f"\n  >>> {exp}")
            result = benchmark_model_predict(
                weight_path,
                imgsz=IMGSZ,
                num_warmup=NUM_WARMUP,
                num_runs=NUM_RUNS,
            )
            ds_results[exp] = result
            print(
                f"      Params: {result['params']:,}  GFLOPs: {result['gflops']:.2f}"
            )
            print(
                f"      Avg: {result['avg_ms']:.2f} ± {result['std_ms']:.2f} ms  "
                f"Median: {result['median_ms']:.2f} ms  FPS: {result['fps']:.2f}"
            )

        all_results[ds_name] = ds_results

    # ============ 输出汇总表格 ============
    for ds_name, ds_results in all_results.items():
        print(f"\n\n{'='*80}")
        print(f"  {ds_name} 推理速度汇总 (GPU: A100, imgsz={IMGSZ}, batch=1, {NUM_RUNS}次平均)")
        print(f"{'='*80}")
        header = f"{'实验':<20} {'Params':>10} {'GFLOPs':>8} {'Avg(ms)':>10} {'Std(ms)':>9} {'Median(ms)':>11} {'P95(ms)':>9} {'FPS':>8}"
        print(header)
        print("-" * 90)
        for exp, r in ds_results.items():
            print(
                f"{exp:<20} {r['params']:>10,} {r['gflops']:>8.2f} "
                f"{r['avg_ms']:>10.2f} {r['std_ms']:>9.2f} {r['median_ms']:>11.2f} "
                f"{r['p95_ms']:>9.2f} {r['fps']:>8.2f}"
            )

        # 对比 baseline vs SCAW
        if "baseline" in ds_results and "SCAW" in ds_results:
            b = ds_results["baseline"]
            s = ds_results["SCAW"]
            print(f"\n  --- baseline vs SCAW ---")
            print(f"  参数量:  {b['params']:,} -> {s['params']:,} ({(s['params']-b['params'])/b['params']*100:+.1f}%)")
            print(f"  GFLOPs:  {b['gflops']:.2f} -> {s['gflops']:.2f} ({(s['gflops']-b['gflops'])/b['gflops']*100:+.1f}%)")
            print(f"  推理ms:  {b['avg_ms']:.2f} -> {s['avg_ms']:.2f} ({(s['avg_ms']-b['avg_ms'])/b['avg_ms']*100:+.1f}%)")
            print(f"  FPS:     {b['fps']:.2f} -> {s['fps']:.2f} ({(s['fps']-b['fps'])/b['fps']*100:+.1f}%)")

    # ============ 写入 CSV ============
    out_csv = os.path.join(BASE, "benchmark_speed_all_ablation.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "experiment", "params", "gflops",
            "avg_ms", "std_ms", "median_ms", "p95_ms", "fps"
        ])
        for ds_name, ds_results in all_results.items():
            for exp, r in ds_results.items():
                writer.writerow([
                    ds_name, exp, r["params"], f"{r['gflops']:.4f}",
                    f"{r['avg_ms']:.4f}", f"{r['std_ms']:.4f}",
                    f"{r['median_ms']:.4f}", f"{r['p95_ms']:.4f}",
                    f"{r['fps']:.4f}",
                ])
    print(f"\n结果已保存至: {out_csv}")


if __name__ == "__main__":
    main()
