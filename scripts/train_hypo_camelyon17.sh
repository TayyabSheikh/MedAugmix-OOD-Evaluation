#!/bin/bash

# Script to train HypO on Camelyon17 (WILDS)

# Activate conda environment (adjust if your conda init is different)
# source $(conda info --base)/etc/profile.d/conda.sh # Common way, might vary
# conda activate fresh # Assuming 'fresh' is the environment name

# Define arguments
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
MODEL="densenet121" # Or resnet50, etc.
GPU_ID=0
EPOCHS=200 # Start with a small number for testing
BATCH_SIZE=256
LEARNING_RATE=5e-4
WANDB_MODE="online" # Set to "disabled" to turn off wandb logging
FEAT_DIM=128
PROTO_M=0.95
LOSS_SCALE_W=2
TRIAL="0" # Increment for multiple runs

# Construct the command
# Note: Assumes the script is run from the hypo_impl directory root
# If running from scripts/ directory, adjust paths accordingly (e.g., python ../train_hypo.py)
# Running from root (hypo_impl) is generally easier for relative paths.

echo "Running training script from hypo_impl directory..."
echo "Command:"
echo "python train_hypo.py \\"
echo "  --in-dataset ${DATASET} \\"
echo "  --wilds_root_dir ${WILDS_ROOT_DIR} \\"
echo "  --model ${MODEL} \\"
echo "  --gpu ${GPU_ID} \\"
echo "  --epochs ${EPOCHS} \\"
echo "  --batch_size ${BATCH_SIZE} \\"
echo "  --learning_rate ${LEARNING_RATE} \\"
echo "  --feat_dim ${FEAT_DIM} \\"
echo "  --proto_m ${PROTO_M} \\"
echo "  --w ${LOSS_SCALE_W} \\"
echo "  --trial ${TRIAL} \\"
echo "  --mode ${WANDB_MODE}"

# Execute the command (relative to the script's parent directory)
# Use ../ to point to train_hypo.py in the hypo_impl directory
python ../train_hypo.py \
  --in-dataset ${DATASET} \
  --wilds_root_dir ${WILDS_ROOT_DIR} \
  --model ${MODEL} \
  --gpu ${GPU_ID} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --feat_dim ${FEAT_DIM} \
  --proto_m ${PROTO_M} \
  --w ${LOSS_SCALE_W} \
  --trial ${TRIAL} \
  --mode ${WANDB_MODE}

echo "Training script finished."
