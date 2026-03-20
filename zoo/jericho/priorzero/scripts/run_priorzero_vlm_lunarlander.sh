#!/bin/bash
# PriorZero VLM Training on LunarLander-v2 (Image Input)
#
# Usage:
#   bash run_priorzero_vlm_lunarlander.sh [NUM_GPUS] [VLM_MODEL] [SEED]
#
# Examples:
#   bash run_priorzero_vlm_lunarlander.sh 4 Qwen2.5-VL-7b 0
#   bash run_priorzero_vlm_lunarlander.sh 2 Qwen2.5-VL-2b 42
#   bash run_priorzero_vlm_lunarlander.sh 1 Qwen2.5-VL-2b 0 --quick_test

set -euo pipefail

NUM_GPUS=${1:-4}
VLM_MODEL=${2:-"Qwen2.5-VL-7b"}
SEED=${3:-0}
EXTRA_ARGS="${@:4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ID="LunarLander-v2"
EXP_NAME="data_priorzero_complete/image_LunarLander_${VLM_MODEL}_seed${SEED}"

echo "========================================"
echo "PriorZero VLM - LunarLander-v2 (Image)"
echo "========================================"
echo "GPUs: ${NUM_GPUS}"
echo "VLM Model: ${VLM_MODEL}"
echo "Seed: ${SEED}"
echo "Exp Name: ${EXP_NAME}"
echo "Extra Args: ${EXTRA_ARGS}"
echo "========================================"

cd "${SCRIPT_DIR}"

torchrun \
    --nproc_per_node "${NUM_GPUS}" \
    --master_port 29501 \
    priorzero_entry_unified.py \
    --input_type image \
    --env_id "${ENV_ID}" \
    --vlm_model "${VLM_MODEL}" \
    --seed "${SEED}" \
    ${EXTRA_ARGS}
