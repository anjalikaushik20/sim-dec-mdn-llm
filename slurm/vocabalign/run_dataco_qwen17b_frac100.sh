#!/bin/bash
# One-shot: train the missing DataCo Qwen3-1.7B frac1.00 adapter on GPU 0.
# Exp 2 is running on GPU 1; this runs independently on GPU 0.
#
# Usage: bash run_nohup.sh run_dataco_qwen17b_frac100.sh

set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_DIR="output/decision_maker/DataCo/ckpts/frac1.00/qwen3-1.7B"
mkdir -p "${CKPT_DIR}"

echo "Training DataCo Qwen3-1.7B frac1.00 → ${CKPT_DIR}"

CUDA_VISIBLE_DEVICES=0 python3 main/cb_main_llm.py \
    --use_gpu 1 --device_id 0 \
    --dataset DataCo \
    --train_mode 2 \
    --wandb 1 \
    --hf_model_name Qwen/Qwen3-1.7B \
    --save 1 \
    --ckpt_dir "${CKPT_DIR}" \
    --dm_epochs 200 \
    --train_frac 1.00 \
    --otr_reward_coeff 2 \
    --ckpt output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth

echo ""
echo "Done. Adapter saved to:"
ls "${CKPT_DIR}/"
echo ""
echo "Re-run Exp 1 with:"
echo "  bash run_nohup.sh run_exp1_observational_matching.sh"
