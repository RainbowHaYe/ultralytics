#!/bin/bash
# ==============================================================================
#  YOLO-SCAW 消融实验串联执行脚本
#  先运行 selfpart499 全部 9 组，再运行 openv2 全部 9 组
# ==============================================================================
set -e

cd /home/ros/jhy/ultralytics
source .venv/bin/activate

mkdir -p logs

echo "============================================================"
echo "  YOLO-SCAW 消融实验串联执行"
echo "  第一阶段: selfpart499 (9 组)"
echo "  第二阶段: openv2 (9 组)"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---- 第一阶段: selfpart499 ----
echo ""
echo "[阶段 1/2] selfpart499 消融实验开始: $(date '+%Y-%m-%d %H:%M:%S')"
python scripts/train_scw_ablation.py 2>&1 | tee logs/ablation_self499_$(date +%Y%m%d_%H%M%S).log
echo "[阶段 1/2] selfpart499 消融实验完成: $(date '+%Y-%m-%d %H:%M:%S')"

# ---- 第二阶段: openv2 ----
echo ""
echo "[阶段 2/2] openv2 消融实验开始: $(date '+%Y-%m-%d %H:%M:%S')"
python scripts/train_scw_ablation_openv2.py 2>&1 | tee logs/ablation_openv2_$(date +%Y%m%d_%H%M%S).log
echo "[阶段 2/2] openv2 消融实验完成: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "============================================================"
echo "  全部消融实验完成!"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  selfpart499 结果: results/runs/self499_aug/"
echo "  openv2 结果:      results/runs/openv2/"
echo "============================================================"
