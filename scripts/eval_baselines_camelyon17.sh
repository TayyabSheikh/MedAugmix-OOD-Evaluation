#!/bin/bash

# Script to evaluate baseline OOD methods (MSP, Energy) on Camelyon17 (WILDS)
# using a pre-trained ERM model.

# --- Configuration ---
# !! IMPORTANT: Update this path to the specific ERM checkpoint you want to evaluate !!
ERM_CKPT_PATH="/home/tsheikh/Thesis/hypo_impl/scripts/checkpoints/camelyon17/16_04_10:28_erm_densenet121_lr_0.0005_cosine_False_bsz_256_head_mlp_wd_2_200_128_trial_erm_1_temp_0.1_camelyon17_pm_0.95/checkpoint_max.pth.tar"

DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
MODEL="densenet121" # Must match the architecture of the checkpoint
HEAD="mlp"       # Must match the head of the checkpoint
GPU_ID=0
BATCH_SIZE=128
BASELINE_METHODS="msp energy" # Space-separated list
OUTPUT_DIR="../baseline_results/erm_${MODEL}" # Relative path from scripts/

# --- Activate Conda Environment (Optional - uncomment if needed) ---
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate fresh

# --- Construct and Run Command ---
# Assumes the script is run from the hypo_impl/scripts directory
# Uses ../evaluate_baselines.py to point to the script in the parent directory

echo "Running baseline evaluation script..."
echo "Checkpoint: ${ERM_CKPT_PATH}"
echo "Methods: ${BASELINE_METHODS}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Command:"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Construct the python command
CMD="python ../evaluate_baselines.py \
  --ckpt_path ${ERM_CKPT_PATH} \
  --in-dataset ${DATASET} \
  --wilds_root_dir ${WILDS_ROOT_DIR} \
  --model ${MODEL} \
  --head ${HEAD} \
  --gpu ${GPU_ID} \
  --batch_size ${BATCH_SIZE} \
  --baseline_methods ${BASELINE_METHODS} \
  --output_dir ${OUTPUT_DIR}"

# Print the command
echo ${CMD}

# Execute the command
${CMD}

echo "Baseline evaluation script finished."
echo "Results saved in: ${OUTPUT_DIR}"
