#!/bin/bash

# Script to train ERM (standard classifier) on Camelyon17 (WILDS)

# Activate conda environment (adjust if your conda init is different)
# source $(conda info --base)/etc/profile.d/conda.sh # Common way, might vary
# conda activate fresh # Assuming 'fresh' is the environment name

# Define arguments
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
MODEL="densenet121" # Or resnet50, etc.
GPU_ID=0
EPOCHS=200 # Start with a small number for testing (adjust as needed)
BATCH_SIZE=256
LEARNING_RATE=5e-4 # May need tuning for ERM
WANDB_MODE="online" # Set to "disabled" to turn off wandb logging
FEAT_DIM=128 # Feature dim might be less relevant for ERM head, but keep consistent if model structure depends on it
HEAD="mlp" # Standard classifier head usually linear, but follow model setup. Check set_model if needed.
LOSS_TYPE="erm" # Changed loss type
TRIAL="erm_1" # Differentiate trial name

# Construct the command
echo "Running ERM training script from hypo_impl directory..."
echo "Command:"
echo "python ../train_hypo.py \\" # Still uses train_hypo.py, but with --loss erm
echo "  --in-dataset ${DATASET} \\"
echo "  --wilds_root_dir ${WILDS_ROOT_DIR} \\"
echo "  --model ${MODEL} \\"
echo "  --gpu ${GPU_ID} \\"
echo "  --epochs ${EPOCHS} \\"
echo "  --batch_size ${BATCH_SIZE} \\"
echo "  --learning_rate ${LEARNING_RATE} \\"
echo "  --feat_dim ${FEAT_DIM} \\"
echo "  --head ${HEAD} \\"
echo "  --loss ${LOSS_TYPE} \\" # Use the ERM loss type
echo "  --trial ${TRIAL} \\"
echo "  --mode ${WANDB_MODE}"

# Execute the command (relative to the script's parent directory)
python ../train_hypo.py \
  --in-dataset ${DATASET} \
  --wilds_root_dir ${WILDS_ROOT_DIR} \
  --model ${MODEL} \
  --gpu ${GPU_ID} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --feat_dim ${FEAT_DIM} \
  --head ${HEAD} \
  --loss ${LOSS_TYPE} \
  --trial ${TRIAL} \
  --mode ${WANDB_MODE}

echo "ERM Training script finished."
