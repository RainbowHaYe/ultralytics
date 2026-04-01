"""
==============================================================================
 YOLO-SCAW 消融实验结果分析脚本 (通用)
==============================================================================

用法：
  # 分析 selfpart499
  python scripts/analyze_ablation.py results/runs/detect/results/runs/self499_aug

  # 分析 openv2
  python scripts/analyze_ablation.py results/runs/detect/results/runs/openv2

  # 分析任意目录
  python scripts/analyze_ablation.py <结果目录>

目录结构要求：
  <结果目录>/
    baseline/results.csv
    ablation_SimC2f/results.csv
    ablation_S/results.csv
    ...
    SCW/results.csv
==============================================================================
"""

import argparse
import os
import sys


# ============================================================================
# 实验定义
# ============================================================================
EXPERIMENT_ORDER = [
    "baseline", "ablation_SimC2f",
    "ablation_S", "ablation_CA", "ablation_W",
    "ablation_SCA", "ablation_SW", "ablation_CAW",
    "SCAW",
]

FLAGS = {
    "baseline":        ("", "", ""),
    "ablation_SimC2f": ("", "", ""),
    "ablation_S":      ("✓", "", ""),
    "ablation_CA":      ("", "✓", ""),
    "ablation_W":      ("", "", "✓"),
    "ablation_SCA":     ("✓", "✓", ""),
    "ablation_SW":     ("✓", "", "✓"),
    "ablation_CAW":     ("", "✓", "✓"),
    "SCAW":             ("✓", "✓", "✓"),
}

LABELS = {
    "baseline":        "原生 YOLOv8n",
    "ablation_SimC2f": "SimC2f 轻量化基底",
    "ablation_S":      "+SPDLiteConv",
    "ablation_CA":      "+CoordAtt",
    "ablation_W":      "+WAFF",
    "ablation_SCA":     "+SPDLiteConv+CoordAtt",
    "ablation_SW":     "+SPDLiteConv+WAFF",
    "ablation_CAW":     "+CoordAtt+WAFF",
    "SCAW":             "完整 YOLO-SCAW",
}

# 已知模型结构信息
MODEL_INFO = {
    "baseline":        {"params_m": 3.01, "gflops": 8.2},
    "ablation_SimC2f": {"params_m": 2.59, "gflops": 7.2},
    "ablation_S":      {"params_m": 2.61, "gflops": 7.4},
    "ablation_CA":      {"params_m": 2.59, "gflops": 7.2},
    "ablation_W":      {"params_m": 2.59, "gflops": 7.2},
    "ablation_SCA":     {"params_m": 2.62, "gflops": 7.4},
    "ablation_SW":     {"params_m": 2.61, "gflops": 7.4},
    "ablation_CAW":     {"params_m": 2.60, "gflops": 7.2},
    "SCAW":             {"params_m": 2.38, "gflops": 6.6},
}


# ============================================================================
# 数据读取
# ============================================================================
def parse_results_csv(csv_path):
    """从 results.csv 中提取最佳 epoch 的指标。"""
    with open(csv_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if len(lines) < 2:
        return None

    header = [h.strip() for h in lines[0].split(",")]

    best_map = -1
    best_row = None
    best_epoch = 0

    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        row = dict(zip(header, vals))
        try:
            m = float(row.get("metrics/mAP50-95(B)", 0))
            if m > best_map:
                best_map = m
                best_row = row
                best_epoch = int(float(row.get("epoch", 0)))
        except (ValueError, TypeError):
            continue

    if best_row is None:
        return None

    return {
        "map50": float(best_row.get("metrics/mAP50(B)", 0)),
        "map50_95": float(best_row.get("metrics/mAP50-95(B)", 0)),
        "prec": float(best_row.get("metrics/precision(B)", 0)),
        "recall": float(best_row.get("metrics/recall(B)", 0)),
        "best_epoch": best_epoch,
        "total_epochs": len(lines) - 1,
    }


def load_all_results(base_dir):
    """加载目录下所有实验的结果。"""
    results = {}
    found = []
    missing = []

    for name in EXPERIMENT_ORDER:
        csv_path = os.path.join(base_dir, name, "results.csv")
        if os.path.exists(csv_path):
            data = parse_results_csv(csv_path)
            if data:
                results[name] = data
                found.append(name)
            else:
                missing.append(name)
        else:
            missing.append(name)

    # 也扫描目录中存在但不在预定义列表中的实验
    if os.path.isdir(base_dir):
        for d in sorted(os.listdir(base_dir)):
            if d not in EXPERIMENT_ORDER and os.path.isdir(os.path.join(base_dir, d)):
                csv_path = os.path.join(base_dir, d, "results.csv")
                if os.path.exists(csv_path):
                    data = parse_results_csv(csv_path)
                    if data:
                        results[d] = data
                        found.append(d)

    return results, found, missing


# ============================================================================
# 输出格式化
# ============================================================================
def print_main_table(results):
    """打印主消融实验对比表。"""
    bl = results.get("baseline", {})
    bl_map50 = bl.get("map50", 0)
    bl_map95 = bl.get("map50_95", 0)

    print()
    print(f"{'实验':<20} {'S':>2} {'C':>2} {'W':>2} {'Params':>7} {'GFLOPs':>7} "
          f"{'mAP50':>8} {'mAP50-95':>9} {'Δ50(%)':>8} {'Δ95(%)':>8} "
          f"{'Prec':>7} {'Recall':>7} {'BestEp':>7}")
    print("=" * 105)

    # 先打印预定义顺序的实验
    printed = set()
    for name in EXPERIMENT_ORDER:
        if name not in results:
            continue
        _print_row(name, results[name], bl_map50, bl_map95)
        printed.add(name)

    # 再打印额外发现的实验
    for name in sorted(results.keys()):
        if name not in printed:
            _print_row(name, results[name], bl_map50, bl_map95)

    print("=" * 105)


def _print_row(name, r, bl_map50, bl_map95):
    """打印表格中的一行。"""
    mi = MODEL_INFO.get(name, {"params_m": 0, "gflops": 0})
    s, c, w = FLAGS.get(name, ("?", "?", "?"))

    d50 = (r["map50"] - bl_map50) * 100
    d95 = (r["map50_95"] - bl_map95) * 100
    d50_s = f"{d50:+.2f}" if name != "baseline" else "  -"
    d95_s = f"{d95:+.2f}" if name != "baseline" else "  -"

    params_s = f"{mi['params_m']:.2f}M" if mi["params_m"] > 0 else "  N/A"
    gflops_s = f"{mi['gflops']:.1f}" if mi["gflops"] > 0 else " N/A"

    print(f"{name:<20} {s:>2} {c:>2} {w:>2} {params_s:>7} {gflops_s:>7} "
          f"{r['map50']:>8.4f} {r['map50_95']:>9.4f} {d50_s:>8} {d95_s:>8} "
          f"{r['prec']:>7.4f} {r['recall']:>7.4f} {r['best_epoch']:>7}")


def print_ranking(results):
    """按 mAP50-95 排名。"""
    bl_map95 = results.get("baseline", {}).get("map50_95", 0)

    print()
    print("=== mAP50-95 排名 ===")
    sorted_exps = sorted(results.items(), key=lambda x: x[1]["map50_95"], reverse=True)
    for i, (name, r) in enumerate(sorted_exps, 1):
        delta = (r["map50_95"] - bl_map95) * 100
        marker = " ★" if delta > 0 and name != "baseline" else ""
        print(f"  {i}. {name:<20} {r['map50_95']:.4f}  ({delta:+.2f}%){marker}")


def print_module_contribution(results):
    """单模块贡献分析（相对 SimC2f 基底）。"""
    sim = results.get("ablation_SimC2f")
    if sim is None:
        return

    sim_map = sim["map50_95"]
    print()
    print("=== 单模块贡献分析 (相对 SimC2f 基底) ===")
    for name, label in [("ablation_S", "S (SPDLiteConv)"),
                         ("ablation_CA", "CA (CoordAtt)"),
                         ("ablation_W", "W (WAFF)")]:
        r = results.get(name)
        if r:
            delta = (r["map50_95"] - sim_map) * 100
            sign = "+" if delta > 0 else ""
            print(f"  {label:<22} vs SimC2f: {sign}{delta:.2f}%")


def print_combination_analysis(results):
    """组合效应分析。"""
    sim = results.get("ablation_SimC2f")
    if sim is None:
        return

    sim_map = sim["map50_95"]

    combos = [
        ("ablation_SCA", "S+CA", "ablation_S", "ablation_CA"),
        ("ablation_SW", "S+W", "ablation_S", "ablation_W"),
        ("ablation_CAW", "CA+W", "ablation_CA", "ablation_W"),
    ]

    print()
    print("=== 组合效应分析 (是否存在协同/冲突) ===")
    print(f"  {'组合':<10} {'实际Δ':>8} {'预期Δ(加和)':>12} {'协同效应':>10}")
    print(f"  {'─'*44}")

    for combo_name, label, a_name, b_name in combos:
        combo = results.get(combo_name)
        a = results.get(a_name)
        b = results.get(b_name)
        if combo and a and b:
            actual_delta = (combo["map50_95"] - sim_map) * 100
            expected_delta = ((a["map50_95"] - sim_map) + (b["map50_95"] - sim_map)) * 100
            synergy = actual_delta - expected_delta
            print(f"  {label:<10} {actual_delta:>+7.2f}% {expected_delta:>+11.2f}% {synergy:>+9.2f}%")


def print_lightweight_summary(results):
    """轻量化效果汇总。"""
    bl = results.get("baseline")
    scw = results.get("SCAW")
    if not bl or not scw:
        return

    bl_mi = MODEL_INFO.get("baseline", {})
    scw_mi = MODEL_INFO.get("SCAW", {})

    if bl_mi.get("params_m", 0) == 0:
        return

    print()
    print("=== 轻量化效果 (SCW vs Baseline) ===")
    print(f"  参数量:   {bl_mi['params_m']:.2f}M → {scw_mi['params_m']:.2f}M "
          f"({(scw_mi['params_m'] - bl_mi['params_m']) / bl_mi['params_m'] * 100:+.1f}%)")
    print(f"  计算量:   {bl_mi['gflops']:.1f}G → {scw_mi['gflops']:.1f}G "
          f"({(scw_mi['gflops'] - bl_mi['gflops']) / bl_mi['gflops'] * 100:+.1f}%)")
    print(f"  mAP50:    {bl['map50']:.4f} → {scw['map50']:.4f} "
          f"({(scw['map50'] - bl['map50']) * 100:+.2f}%)")
    print(f"  mAP50-95: {bl['map50_95']:.4f} → {scw['map50_95']:.4f} "
          f"({(scw['map50_95'] - bl['map50_95']) * 100:+.2f}%)")


def print_best_configs(results):
    """推荐最优配置。"""
    bl_map95 = results.get("baseline", {}).get("map50_95", 0)
    if bl_map95 == 0:
        return

    # 超过baseline的实验
    better = [(n, r) for n, r in results.items()
              if n != "baseline" and r["map50_95"] > bl_map95]

    print()
    if better:
        print("=== 超过 Baseline 的配置 ===")
        for name, r in sorted(better, key=lambda x: x[1]["map50_95"], reverse=True):
            mi = MODEL_INFO.get(name, {})
            delta = (r["map50_95"] - bl_map95) * 100
            params_info = f", Params {mi['params_m']:.2f}M" if mi.get("params_m") else ""
            print(f"  ✅ {name:<20} mAP50-95={r['map50_95']:.4f} ({delta:+.2f}%){params_info}")
    else:
        print("=== 无配置超过 Baseline ===")
        print("  轻量化方案以精度-效率权衡为目标，非精度最大化")

    # 精度-效率最优（帕累托前沿）
    print()
    print("=== 精度-效率权衡推荐 ===")
    for name in ["SCAW", "ablation_CAW", "ablation_W"]:
        r = results.get(name)
        mi = MODEL_INFO.get(name, {})
        if r and mi.get("params_m"):
            delta = (r["map50_95"] - bl_map95) * 100
            params_delta = (mi["params_m"] - MODEL_INFO["baseline"]["params_m"]) / MODEL_INFO["baseline"]["params_m"] * 100
            print(f"  {name:<20} mAP50-95 {delta:+.2f}%, Params {params_delta:+.1f}%, "
                  f"GFLOPs {(mi['gflops'] - MODEL_INFO['baseline']['gflops']) / MODEL_INFO['baseline']['gflops'] * 100:+.1f}%")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="YOLO-SCAW 消融实验结果分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/analyze_ablation.py results/runs/detect/results/runs/self499_aug
  python scripts/analyze_ablation.py results/runs/detect/results/runs/openv2
        """,
    )
    parser.add_argument("result_dir", help="消融实验结果目录")
    args = parser.parse_args()

    base_dir = args.result_dir

    if not os.path.isdir(base_dir):
        print(f"[错误] 目录不存在: {base_dir}")
        sys.exit(1)

    # 加载数据
    results, found, missing = load_all_results(base_dir)

    if not results:
        print(f"[错误] 未在 {base_dir} 中找到任何实验结果")
        sys.exit(1)

    # 打印标题
    print()
    print("=" * 105)
    print(f"  YOLO-SCAW 消融实验分析")
    print(f"  结果目录: {base_dir}")
    print(f"  已加载: {len(found)} 组实验  |  缺失: {len(missing)} 组")
    if missing:
        print(f"  缺失实验: {', '.join(missing)}")
    print("=" * 105)

    # 输出各项分析
    print_main_table(results)
    print_ranking(results)
    print_module_contribution(results)
    print_combination_analysis(results)
    print_lightweight_summary(results)
    print_best_configs(results)

    print()


if __name__ == "__main__":
    main()
