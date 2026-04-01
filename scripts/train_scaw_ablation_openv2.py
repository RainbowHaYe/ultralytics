"""
==============================================================================
 YOLO-SCAW 消融实验训练脚本 —— OpenV2 公开数据集
==============================================================================

实验设计：
  基于 SCW 三项创新 (S=SPDLiteConv, C=CoordAtt, W=WAFF)，
  以 SimC2f 轻量化 Bottleneck 为公共基底，进行全组合消融实验。

  ┌──────────────────────────────────────────────────────────────────────┐
  │ 编号  │ 实验名称          │  S  │  C  │  W  │  说明                │
  ├──────────────────────────────────────────────────────────────────────┤
  │  0    │ baseline          │  ✗  │  ✗  │  ✗  │  原生 YOLOv8n        │
  │  1    │ ablation_SimC2f   │  ✗  │  ✗  │  ✗  │  SimC2f 轻量化基底   │
  │  2    │ ablation_S        │  ✓  │  ✗  │  ✗  │  +SPDLiteConv        │
  │  3    │ ablation_CA        │  ✗  │  ✓  │  ✗  │  +CoordAtt           │
  │  4    │ ablation_W        │  ✗  │  ✗  │  ✓  │  +WAFF               │
  │  5    │ ablation_SCA       │  ✓  │  ✓  │  ✗  │  +SPDLiteConv+Coord  │
  │  6    │ ablation_SW       │  ✓  │  ✗  │  ✓  │  +SPDLiteConv+WAFF   │
  │  7    │ ablation_CAW       │  ✗  │  ✓  │  ✓  │  +CoordAtt+WAFF      │
  │  8    │ SCW               │  ✓  │  ✓  │  ✓  │  完整 YOLO-SCAW       │
  └──────────────────────────────────────────────────────────────────────┘

数据集  ：OpenV2 公开机械零件数据集 (6 类, 5914 张)
硬件环境：NVIDIA A100-PCIE-40GB, CUDA 12.6

用法：
  # 运行全部 9 组实验
  python scripts/train_scw_ablation_openv2.py

  # 仅运行指定实验
  python scripts/train_scw_ablation_openv2.py --experiments 0,8

  # 从第 5 组开始继续
  python scripts/train_scw_ablation_openv2.py --start-from 5

结果路径：results/runs/openv2/<name>/
==============================================================================
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

# ============================================================================
# 实验配置
# ============================================================================
EXPERIMENTS = [
    # (编号, 名称, 模型来源, 说明)
    (0, "baseline",          "yolov8n.yaml",                                                   "原生 YOLOv8n (C2f + Conv stride-2)"),
    (1, "ablation_SimC2f",   "ultralytics/cfg/models/v8/yolov8n_ablation_SimC2f.yaml",        "SimC2f 轻量化基底"),
    (2, "ablation_S",        "ultralytics/cfg/models/v8/yolov8n_ablation_S.yaml",             "+SPDLiteConv 无损下采样"),
    (3, "ablation_CA",        "ultralytics/cfg/models/v8/yolov8n_ablation_CA.yaml",             "+CoordAtt 坐标注意力"),
    (4, "ablation_W",        "ultralytics/cfg/models/v8/yolov8n_ablation_W.yaml",             "+WAFF 自适应融合"),
    (5, "ablation_SCA",       "ultralytics/cfg/models/v8/yolov8n_ablation_SCA.yaml",            "+SPDLiteConv + CoordAtt"),
    (6, "ablation_SW",       "ultralytics/cfg/models/v8/yolov8n_ablation_SW.yaml",            "+SPDLiteConv + WAFF"),
    (7, "ablation_CAW",       "ultralytics/cfg/models/v8/yolov8n_ablation_CAW.yaml",            "+CoordAtt + WAFF"),
    (8, "SCAW",               "ultralytics/cfg/models/v8/yolov8n_SCAW.yaml",                    "完整 YOLO-SCAW (S+C+W)"),
]

DATA_YAML = "datasets/openv2/data.yaml"

# ============================================================================
# 公共训练超参数
# ============================================================================
TRAIN_ARGS = dict(
    # ── 数据与任务 ──
    data=DATA_YAML,
    task="detect",

    # ── 训练规模 ──
    epochs=300,
    patience=50,
    imgsz=640,
    batch=64,

    # ── 优化器与学习率 ──
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,
    cos_lr=True,
    weight_decay=5e-4,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # ── 损失函数权重 (保持默认) ──
    box=7.5,
    cls=0.5,
    dfl=1.5,

    # ── 数据增强 (小目标友好) ──
    mosaic=0.5,
    close_mosaic=20,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    scale=0.3,
    translate=0.1,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    erasing=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    bgr=0.0,

    # ── 训练控制 ──
    device=0,
    workers=16,
    cache=False,
    amp=True,

    # ── 可复现性 ──
    seed=0,
    deterministic=True,

    # ── 验证与保存 ──
    val=True,
    plots=True,
    save=True,
    exist_ok=False,

    # ── 其他 ──
    rect=False,
    resume=False,
    pretrained=True,
    single_cls=False,
    multi_scale=False,
    nbs=64,
    verbose=True,
)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="YOLO-SCAW 消融实验 (OpenV2)")
    parser.add_argument(
        "--experiments", type=str, default=None,
        help="要运行的实验，逗号分隔。支持编号(0-8)或名称。默认运行全部。"
    )
    parser.add_argument(
        "--start-from", type=int, default=0,
        help="从第几组实验开始 (0-8)"
    )
    parser.add_argument(
        "--project", type=str, default="results/runs/openv2",
        help="实验结果保存的项目目录"
    )
    return parser.parse_args()


def select_experiments(args):
    """根据命令行参数选择要运行的实验列表。"""
    if args.experiments is not None:
        selected = []
        for token in args.experiments.split(","):
            token = token.strip()
            if token.isdigit():
                idx = int(token)
                match = [e for e in EXPERIMENTS if e[0] == idx]
            else:
                match = [e for e in EXPERIMENTS if e[1].lower() == token.lower()]
            if not match:
                print(f"  [警告] 未找到实验: {token}，跳过")
            else:
                selected.extend(match)
        return selected
    else:
        return [e for e in EXPERIMENTS if e[0] >= args.start_from]


def run_experiment(exp_id, exp_name, model_src, description, project):
    """运行单组消融实验。"""
    run_name = exp_name

    print("\n" + "=" * 72)
    print(f"  实验 [{exp_id}/8] — {exp_name}")
    print(f"  {description}")
    print(f"  模型来源: {model_src}")
    print(f"  保存路径: {project}/{run_name}/")
    print("=" * 72)

    model = YOLO(model_src)
    info = model.info()
    print(f"  [模型信息] Layers={info[0]}, Params={info[1]/1e6:.2f}M, "
          f"GFLOPs={info[3]:.1f}")

    t0 = time.time()
    results = model.train(
        **TRAIN_ARGS,
        name=run_name,
        project=project,
    )
    elapsed = time.time() - t0

    save_dir = model.trainer.save_dir
    best_epoch = model.trainer.stopper.best_epoch
    total_epochs = model.trainer.epoch + 1

    result_dict = {
        "name": exp_name,
        "description": description,
        "params_m": info[1] / 1e6,
        "gflops": info[3],
        "map50": results.box.map50,
        "map50_95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "elapsed_min": elapsed / 60,
        "save_dir": str(save_dir),
    }

    print(f"\n  ── 实验 {exp_name} 完成 ──")
    print(f"  训练时长     : {elapsed/60:.1f} min")
    print(f"  最佳 Epoch   : {best_epoch} / {total_epochs}")
    print(f"  mAP50        : {results.box.map50:.4f}")
    print(f"  mAP50-95     : {results.box.map:.4f}")
    print(f"  Precision    : {results.box.mp:.4f}")
    print(f"  Recall       : {results.box.mr:.4f}")
    print(f"  保存目录     : {save_dir}")

    # 清理 GPU/CPU 内存
    del model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  [GPU 清理] allocated={allocated:.0f}MB, reserved={reserved:.0f}MB")
    print("-" * 72)

    return result_dict


def print_summary_table(all_results):
    """打印所有消融实验的汇总对比表。"""
    if not all_results:
        return

    print("\n\n" + "=" * 100)
    print("  YOLO-SCAW 消融实验汇总 (OpenV2 数据集)")
    print("=" * 100)
    header = (
        f"{'实验':<22} {'Params(M)':>10} {'GFLOPs':>8} "
        f"{'mAP50':>8} {'mAP50-95':>10} {'Prec':>8} {'Recall':>8} "
        f"{'Best Ep':>8} {'时长(min)':>10}"
    )
    print(header)
    print("-" * 100)

    baseline_map = None
    for r in all_results:
        if r["name"] == "baseline":
            baseline_map = r["map50_95"]
            break

    for r in all_results:
        delta = ""
        if baseline_map is not None and r["name"] != "baseline":
            diff = r["map50_95"] - baseline_map
            delta = f" ({diff:+.2%})"

        line = (
            f"{r['name']:<22} {r['params_m']:>10.2f} {r['gflops']:>8.1f} "
            f"{r['map50']:>8.4f} {r['map50_95']:>10.4f}{delta:>8} "
            f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
            f"{r['best_epoch']:>8} {r['elapsed_min']:>10.1f}"
        )
        print(line)

    print("=" * 100)
    print("  数据集: OpenV2 (6 类机械零件, 5914 张)")
    print("  超参数: AdamW, lr0=1e-3, cos_lr, batch=64, imgsz=640, epochs≤300")
    print("  pretrained=True (COCO 预训练权重)")
    print("=" * 100)


def main():
    args = parse_args()
    experiments = select_experiments(args)

    if not experiments:
        print("[错误] 未选择任何实验。使用 --help 查看用法。")
        sys.exit(1)

    print("=" * 72)
    print("  YOLO-SCAW 消融实验 (OpenV2 数据集)")
    print(f"  共 {len(experiments)} 组实验待运行:")
    for exp_id, exp_name, _, desc in experiments:
        print(f"    [{exp_id}] {exp_name:22s} — {desc}")
    print(f"  数据集: {DATA_YAML}")
    print(f"  结果目录: {args.project}/")
    print("=" * 72)

    all_results = []
    for exp_id, exp_name, model_src, desc in experiments:
        try:
            result = run_experiment(exp_id, exp_name, model_src, desc,
                                    project=args.project)
            all_results.append(result)
        except Exception as e:
            print(f"\n  [错误] 实验 {exp_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"  跳过，继续下一组...\n")
            continue

    # 打印汇总表
    print_summary_table(all_results)

    # 保存结果到 CSV
    if all_results:
        import csv
        csv_path = Path(args.project) / "ablation_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  汇总 CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
