"""
漏检率 & 过检率 评估脚本 —— SelfPart499_aug 全部消融实验
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

BASE_DIR = ROOT / "results" / "runs" / "detect" / "results" / "runs" / "self499_aug"
DATA_YAML = str(ROOT / "datasets" / "selfpart499_augmented" / "data.yaml")

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
]


def evaluate_model(model_path, data_yaml):
    model = YOLO(str(model_path))
    metrics = model.val(data=data_yaml, plots=False, verbose=False,
                        imgsz=640, batch=64, device=0,
                        conf=0.25, iou=0.45,
                        save_json=False, save_txt=False)

    cm = metrics.confusion_matrix
    matrix = cm.matrix

    tp_per_class = matrix.diagonal()[:-1]
    fp_per_class = matrix.sum(axis=1)[:-1] - tp_per_class
    fn_per_class = matrix.sum(axis=0)[:-1] - tp_per_class

    total_tp = tp_per_class.sum()
    total_fp = fp_per_class.sum()
    total_fn = fn_per_class.sum()

    miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    false_alarm_rate = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    precision = metrics.box.mp
    recall = metrics.box.mr

    class_names = metrics.names if hasattr(metrics, 'names') else \
                  {i: str(i) for i in range(len(tp_per_class))}

    per_class = {}
    for i in range(len(tp_per_class)):
        cls_name = class_names.get(i, str(i))
        tp_i, fp_i, fn_i = tp_per_class[i], fp_per_class[i], fn_per_class[i]
        mr_i = fn_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
        far_i = fp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
        per_class[cls_name] = {
            "tp": int(tp_i), "fp": int(fp_i), "fn": int(fn_i),
            "miss_rate": float(mr_i), "false_alarm_rate": float(far_i),
        }

    return {
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "miss_rate": float(miss_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "precision": float(precision),
        "recall": float(recall),
        "per_class": per_class,
    }


def main():
    print("=" * 100)
    print("  漏检率 & 过检率评估 —— SelfPart499_aug 消融实验")
    print("=" * 100)

    all_results = []

    for exp_name in EXPERIMENTS:
        model_path = BASE_DIR / exp_name / "weights" / "best.pt"
        if not model_path.exists():
            print(f"  [跳过] {exp_name}: best.pt 不存在")
            continue

        print(f"\n  正在评估: {exp_name} ...")
        result = evaluate_model(model_path, DATA_YAML)
        result["name"] = exp_name
        all_results.append(result)
        print(f"    TP={result['total_tp']:>5d}  FP={result['total_fp']:>5d}  "
              f"FN={result['total_fn']:>5d}  |  "
              f"漏检率={result['miss_rate']:.4f}  "
              f"过检率={result['false_alarm_rate']:.4f}")

    if not all_results:
        print("\n  无有效结果")
        return

    class_names = list(all_results[0]["per_class"].keys())

    # === 整体汇总 ===
    print("\n\n" + "=" * 100)
    print("  整体汇总")
    print("=" * 100)
    print(f"{'实验':<22} {'TP':>6} {'FP':>6} {'FN':>6} {'漏检率':>10} {'虚警率':>10} {'P':>8} {'R':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['name']:<22} {r['total_tp']:>6d} {r['total_fp']:>6d} {r['total_fn']:>6d} "
              f"{r['miss_rate']:>10.6f} {r['false_alarm_rate']:>10.6f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f}")

    # === 分类别 ===
    print("\n\n" + "=" * 100)
    print("  分类别详细")
    print("=" * 100)
    for cls_name in class_names:
        print(f"\n  ── {cls_name} ──")
        print(f"  {'实验':<22} {'TP':>6} {'FP':>6} {'FN':>6} {'漏检率':>10} {'虚警率':>10}")
        print("  " + "-" * 70)
        for r in all_results:
            pc = r["per_class"].get(cls_name)
            if pc:
                print(f"  {r['name']:<22} {pc['tp']:>6d} {pc['fp']:>6d} {pc['fn']:>6d} "
                      f"{pc['miss_rate']:>10.6f} {pc['false_alarm_rate']:>10.6f}")

    # === 写 CSV ===
    out_csv = BASE_DIR / "miss_false_alarm_results.csv"
    fieldnames = ["experiment", "total_TP", "total_FP", "total_FN",
                  "miss_rate", "false_alarm_rate", "precision", "recall"]
    for cls in class_names:
        fieldnames.extend([f"{cls}_TP", f"{cls}_FP", f"{cls}_FN",
                           f"{cls}_miss_rate", f"{cls}_false_alarm_rate"])

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {
                "experiment": r["name"],
                "total_TP": r["total_tp"],
                "total_FP": r["total_fp"],
                "total_FN": r["total_fn"],
                "miss_rate": f"{r['miss_rate']:.6f}",
                "false_alarm_rate": f"{r['false_alarm_rate']:.6f}",
                "precision": f"{r['precision']:.6f}",
                "recall": f"{r['recall']:.6f}",
            }
            for cls in class_names:
                pc = r["per_class"].get(cls, {})
                row[f"{cls}_TP"] = pc.get("tp", 0)
                row[f"{cls}_FP"] = pc.get("fp", 0)
                row[f"{cls}_FN"] = pc.get("fn", 0)
                row[f"{cls}_miss_rate"] = f"{pc.get('miss_rate', 0):.6f}"
                row[f"{cls}_false_alarm_rate"] = f"{pc.get('false_alarm_rate', 0):.6f}"
            writer.writerow(row)

    print(f"\n结果已保存: {out_csv}")


if __name__ == "__main__":
    main()
