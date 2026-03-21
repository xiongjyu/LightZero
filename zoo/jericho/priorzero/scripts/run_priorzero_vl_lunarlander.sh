#!/bin/bash
# PriorZero VL Training on LunarLander-v2 (Image Input)
#
# Usage:
#   bash run_priorzero_vl_lunarlander.sh [NUM_GPUS] [VL_MODEL] [SEED] [EXTRA_ARGS...]
#
# Examples:
#   bash run_priorzero_vl_lunarlander.sh 4 Qwen2.5-VL-7b 0
#   bash run_priorzero_vl_lunarlander.sh 2 Qwen3-VL-2b 42
#   bash run_priorzero_vl_lunarlander.sh 1 Qwen3-VL-2b 0 --quick_test
#   bash run_priorzero_vl_lunarlander.sh 1 Qwen3-VL-2b 0 --quick_test --no_cot --no_vl_fixed --mcts_mode wm_logits

set -euo pipefail

# ===================== Configurable Parameters =====================
NUM_GPUS=${1:-4}
VL_MODEL=${2:-"Qwen3-VL-2b"}
SEED=${3:-0}
EXTRA_ARGS="${@:4}"
# ===================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ID="LunarLander-v2"
TIMESTAMP="$(date +%y%m%d_%H%M%S)"

# Build structured log directory: logs/<env>/<model>/
LOG_DIR="${SCRIPT_DIR}/logs/LunarLander/${VL_MODEL}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/seed${SEED}_gpu${NUM_GPUS}_${TIMESTAMP}.log"

echo "========================================"
echo "PriorZero VL - LunarLander-v2 (Image)"
echo "========================================"
echo "GPUs:        ${NUM_GPUS}"
echo "VL Model:    ${VL_MODEL}"
echo "Seed:        ${SEED}"
echo "Extra Args:  ${EXTRA_ARGS}"
echo "Log File:    ${LOG_FILE}"
echo "========================================"

cd "${SCRIPT_DIR}"

torchrun \
    --nproc_per_node "${NUM_GPUS}" \
    --master_port 29501 \
    priorzero_entry_unified.py \
    --input_type image \
    --env_id "${ENV_ID}" \
    --vl_model "${VL_MODEL}" \
    --seed "${SEED}" \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"
