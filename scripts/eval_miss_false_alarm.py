"""
==============================================================================
 漏检率 & 过检率 评估脚本 —— OpenV2 全部 13 组实验
==============================================================================

计算指标：
  - Miss Rate（漏检率）= FN / (TP + FN) = 1 - Recall
      含义：真实目标未被检出的比例
  - False Alarm Rate（过检率 / 虚警率）= FP / (TP + FP) = 1 - Precision
      含义：检测结果中误报的比例

方法：
  对每个实验的 best.pt 模型，在 OpenV2 验证集上运行 model.val()，
  从混淆矩阵中提取 TP/FP/FN，计算整体及每类的漏检率与过检率。

用法：
  python scripts/eval_miss_false_alarm.py
==============================================================================
"""

import csv
import sys
from pathlib import Path

import numpy as np

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

# ============================================================================
# 实验配置
# ============================================================================
BASE_DIR = ROOT / "results" / "runs" / "detect" / "results" / "runs" / "openv2"
DATA_YAML = str(ROOT / "datasets" / "openv2" / "data.yaml")

# 所有实验（按逻辑分组排序）
EXPERIMENTS = [
    "baseline",
    # SCAW 消融系列
    "ablation_SimC2f",  # S
    "ablation_CA",       # CA (CoordAtt)
    "ablation_W",       # W (WAFF)
    "ablation_S",       # S (SPDLiteConv)
    "ablation_SCA",
    "ablation_SW",
    "ablation_CAW",
    "SCAW",
    # 开题方案系列
    "ablation_Ghost",
    "ablation_GhostCBAM",
    "ablation_GhostBiFPN",
    "ablation_GhostBiFPNCBAM",
]


def evaluate_model(model_path, data_yaml):
    """
    对单个模型运行验证，提取 TP/FP/FN 及漏检率/过检率。

    Returns:
        dict: 包含整体和每类的 TP, FP, FN, miss_rate, false_alarm_rate
    """
    model = YOLO(str(model_path))
    # plots=True 以确保混淆矩阵被计算（confusion_matrix 仅在 plots=True 时填充）
    metrics = model.val(data=data_yaml, plots=True, verbose=False,
                        imgsz=640, batch=64, device=0,
                        conf=0.25, iou=0.45,
                        save_json=False, save_txt=False)

    # 从混淆矩阵提取
    cm = metrics.confusion_matrix
    matrix = cm.matrix  # shape: (nc+1, nc+1)

    # 每类 TP / FP / FN
    tp_per_class = matrix.diagonal()[:-1]             # 对角线（去背景）
    fp_per_class = matrix.sum(axis=1)[:-1] - tp_per_class  # 行和 - TP
    fn_per_class = matrix.sum(axis=0)[:-1] - tp_per_class  # 列和 - TP

    # 整体汇总
    total_tp = tp_per_class.sum()
    total_fp = fp_per_class.sum()
    total_fn = fn_per_class.sum()

    # 漏检率 & 过检率
    miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    false_alarm_rate = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    # 也从 metrics 对象获取 precision/recall 用于交叉验证
    precision = metrics.box.mp   # mean precision
    recall = metrics.box.mr      # mean recall

    # 类别名称
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
    print("  漏检率 & 过检率评估 —— OpenV2 全部 13 组实验")
    print("=" * 100)
    print(f"  数据集     : {DATA_YAML}")
    print(f"  置信度阈值 : 0.25")
    print(f"  IoU 阈值   : 0.45")
    print(f"  实验数     : {len(EXPERIMENTS)}")
    print("=" * 100)

    all_results = []

    for exp_name in EXPERIMENTS:
        model_path = BASE_DIR / exp_name / "weights" / "best.pt"
        if not model_path.exists():
            print(f"  [跳过] {exp_name}: best.pt 不存在")
            continue

        print(f"\n  正在评估: {exp_name} ...")
        try:
            result = evaluate_model(model_path, DATA_YAML)
            result["name"] = exp_name
            all_results.append(result)
            print(f"    TP={result['total_tp']:>5d}  FP={result['total_fp']:>5d}  "
                  f"FN={result['total_fn']:>5d}  |  "
                  f"漏检率={result['miss_rate']:.4f}  "
                  f"过检率={result['false_alarm_rate']:.4f}")
        except Exception as e:
            print(f"    [错误] {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    # ==================== 汇总表 ====================
    if not all_results:
        print("\n  [错误] 无有效结果")
        return

    # 获取类别名称列表（以第一个模型为准）
    class_names = list(all_results[0]["per_class"].keys())

    print("\n\n" + "=" * 110)
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║         漏检率 & 过检率 汇总表（整体）                          ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print("=" * 110)

    header = (
        f"{'实验':<26} {'TP':>6} {'FP':>6} {'FN':>6}"
        f" │ {'漏检率':>8} {'过检率':>8}"
        f" │ {'Precision':>10} {'Recall':>8}"
        f" │ {'vs baseline':>12}"
    )
    print(header)
    print("─" * 110)

    baseline_mr = None
    baseline_far = None

    for r in all_results:
        if r["name"] == "baseline":
            baseline_mr = r["miss_rate"]
            baseline_far = r["false_alarm_rate"]

        delta_str = ""
        if baseline_mr is not None and r["name"] != "baseline":
            delta_mr = r["miss_rate"] - baseline_mr
            delta_far = r["false_alarm_rate"] - baseline_far
            # 漏检率下降为好(负数好)，过检率下降也好(负数好)
            mr_arrow = "↓" if delta_mr < 0 else "↑" if delta_mr > 0 else "="
            far_arrow = "↓" if delta_far < 0 else "↑" if delta_far > 0 else "="
            delta_str = f"MR{mr_arrow}{abs(delta_mr):.3f} FA{far_arrow}{abs(delta_far):.3f}"

        line = (
            f"{r['name']:<26} {r['total_tp']:>6d} {r['total_fp']:>6d} {r['total_fn']:>6d}"
            f" │ {r['miss_rate']:>8.4f} {r['false_alarm_rate']:>8.4f}"
            f" │ {r['precision']:>10.4f} {r['recall']:>8.4f}"
            f" │ {delta_str:>12}"
        )
        print(line)

    print("─" * 110)

    # ==================== 每类详细表 ====================
    print("\n\n" + "=" * 110)
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║         漏检率 & 过检率 —— 分类别详细                           ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print("=" * 110)

    for cls_name in class_names:
        print(f"\n  ── 类别: {cls_name} ──")
        print(f"  {'实验':<26} {'TP':>6} {'FP':>6} {'FN':>6}"
              f" │ {'漏检率':>8} {'过检率':>8}")
        print("  " + "─" * 75)

        for r in all_results:
            pc = r["per_class"].get(cls_name)
            if pc is None:
                continue
            print(f"  {r['name']:<26} {pc['tp']:>6d} {pc['fp']:>6d} {pc['fn']:>6d}"
                  f" │ {pc['miss_rate']:>8.4f} {pc['false_alarm_rate']:>8.4f}")

    # ==================== 保存 CSV ====================
    csv_path = BASE_DIR / "miss_false_alarm_results.csv"

    rows = []
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
        for cls_name in class_names:
            pc = r["per_class"].get(cls_name, {})
            row[f"{cls_name}_TP"] = pc.get("tp", "")
            row[f"{cls_name}_FP"] = pc.get("fp", "")
            row[f"{cls_name}_FN"] = pc.get("fn", "")
            row[f"{cls_name}_miss_rate"] = f"{pc['miss_rate']:.6f}" if pc else ""
            row[f"{cls_name}_false_alarm_rate"] = f"{pc['false_alarm_rate']:.6f}" if pc else ""
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\n  结果已保存: {csv_path}")

    # ==================== 结论 ====================
    print("\n\n" + "=" * 110)
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║                      分析结论                                   ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print("=" * 110)

    # 找 baseline 和 SCW
    bl = next((r for r in all_results if r["name"] == "baseline"), None)
    scw = next((r for r in all_results if r["name"] == "SCAW"), None)

    if bl and scw:
        print(f"\n  baseline  → 漏检率: {bl['miss_rate']:.4f}  过检率: {bl['false_alarm_rate']:.4f}")
        print(f"  SCAW      → 漏检率: {scw['miss_rate']:.4f}  过检率: {scw['false_alarm_rate']:.4f}")
        mr_diff = scw['miss_rate'] - bl['miss_rate']
        far_diff = scw['false_alarm_rate'] - bl['false_alarm_rate']
        print(f"  SCAW vs BL → 漏检率变化: {mr_diff:+.4f} ({'改善' if mr_diff < 0 else '恶化'})")
        print(f"            → 过检率变化: {far_diff:+.4f} ({'改善' if far_diff < 0 else '恶化'})")

    # 找漏检率/过检率综合最优
    # 综合得分 = miss_rate + false_alarm_rate (越低越好)
    print(f"\n  综合评分 (漏检率 + 过检率, 越低越好):")
    scored = [(r["name"], r["miss_rate"] + r["false_alarm_rate"],
               r["miss_rate"], r["false_alarm_rate"]) for r in all_results]
    scored.sort(key=lambda x: x[1])
    for rank, (name, score, mr, far) in enumerate(scored, 1):
        marker = " ★" if name == "SCAW" else ""
        print(f"    {rank:>2}. {name:<26} 综合={score:.4f}  "
              f"(漏检={mr:.4f}, 过检={far:.4f}){marker}")

    print("\n" + "=" * 110)


if __name__ == "__main__":
    main()
