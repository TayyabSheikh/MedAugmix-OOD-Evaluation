#!/bin/bash

# Script to evaluate HypO OOD detection performance on Camelyon17 (WILDS)
# using a pre-trained HypO model.

# --- Configuration ---
# !! IMPORTANT: Update this path if you use a different HypO checkpoint !!
HYPO_CKPT_PATH="/home/tsheikh/Thesis/hypo_impl/scripts/checkpoints/camelyon17/16_04_21:45_hypo_densenet121_lr_0.0005_cosine_False_bsz_256_head_mlp_wd_2.0_200_128_trial_medmnistc_0_temp_0.1_camelyon17_pm_0.95_medmnistc/checkpoint_max.pth.tar"

DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
MODEL="densenet121" # Must match the architecture of the checkpoint
HEAD="mlp"       # Must match the head of the checkpoint
GPU_ID=0
BATCH_SIZE=128   # Evaluation batch size

# --- Activate Conda Environment (Optional - uncomment if needed) ---
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate fresh

# --- Construct and Run Command ---
# Assumes the script is run from the hypo_impl/scripts directory
# Uses ../eval_hypo_camelyon.py to point to the script in the parent directory

echo "Running HypO evaluation script..."
echo "Checkpoint: ${HYPO_CKPT_PATH}"
echo "Command:"

# Construct the python command
# Note: --in-dataset, --model, --head are passed to ensure model loading is correct
#       --wilds_root_dir is needed for the dataloader

# Print the command before executing
echo "python ../eval_hypo_camelyon.py \\"
echo "  --ckpt_path ${HYPO_CKPT_PATH} \\"
echo "  --in-dataset ${DATASET} \\"
echo "  --wilds_root_dir ${WILDS_ROOT_DIR} \\"
echo "  --model ${MODEL} \\"
echo "  --head ${HEAD} \\"
echo "  --gpu ${GPU_ID} \\"
echo "  --batch_size ${BATCH_SIZE}"

# Execute the command directly
python ../eval_hypo_camelyon.py \
  --ckpt_path ${HYPO_CKPT_PATH} \
  --in-dataset ${DATASET} \
  --wilds_root_dir ${WILDS_ROOT_DIR} \
  --model ${MODEL} \
  --head ${HEAD} \
  --gpu ${GPU_ID} \
  --batch_size ${BATCH_SIZE}

echo "HypO evaluation script finished."
echo "Results saved in logs/eval/camelyon17/..." # Check specific log directory name in output
