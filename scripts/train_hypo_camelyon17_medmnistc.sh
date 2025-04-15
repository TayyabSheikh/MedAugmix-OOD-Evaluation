#!/bin/bash

# Script to train HypO with MedMNIST-C augmentations on Camelyon17 (WILDS)

# Activate conda environment (adjust if your conda init is different)
# source $(conda info --base)/etc/profile.d/conda.sh # Common way, might vary
# conda activate fresh # Assuming 'fresh' is the environment name

# Define arguments
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
MODEL="resnet50" # Or resnet50, etc.
GPU_ID=0
EPOCHS=200 # Start with a small number for testing
BATCH_SIZE=256
LEARNING_RATE=5e-4
WANDB_MODE="online" # Set to "disabled" to turn off wandb logging
FEAT_DIM=128
PROTO_M=0.95
LOSS_SCALE_W=2
TRIAL="medmnistc_0" # Updated trial ID for augmented run
TEMP=0.1 # Added temp based on previous script
LR_DECAY_EPOCHS="100,150,180" # Added based on previous script
LR_DECAY_RATE=0.1 # Added based on previous script
WEIGHT_DECAY=1e-4 # Added based on previous script
MOMENTUM=0.9 # Added based on previous script
PREFETCH=4 # Added based on previous script
HEAD="mlp" # Added based on previous script
LOSS_TYPE="hypo" # Added based on previous script

# Construct the command
# Assumes the script is run from the hypo_impl/scripts directory

echo "Running training script with MedMNIST-C augmentations from scripts directory..."
echo "Command:"
# Updated python script name
echo "python ../train_hypo_medmnistc.py \\"
echo "  --in-dataset ${DATASET} \\"
echo "  --wilds_root_dir ${WILDS_ROOT_DIR} \\"
echo "  --model ${MODEL} \\"
echo "  --head ${HEAD} \\"
echo "  --loss ${LOSS_TYPE} \\"
echo "  --gpu ${GPU_ID} \\"
echo "  --epochs ${EPOCHS} \\"
echo "  --batch_size ${BATCH_SIZE} \\"
echo "  --learning_rate ${LEARNING_RATE} \\"
echo "  --lr_decay_epochs ${LR_DECAY_EPOCHS} \\"
echo "  --lr_decay_rate ${LR_DECAY_RATE} \\"
echo "  --weight-decay ${WEIGHT_DECAY} \\"
echo "  --momentum ${MOMENTUM} \\"
echo "  --prefetch ${PREFETCH} \\"
echo "  --feat_dim ${FEAT_DIM} \\"
echo "  --proto_m ${PROTO_M} \\"
echo "  --w ${LOSS_SCALE_W} \\"
echo "  --temp ${TEMP} \\"
echo "  --trial ${TRIAL} \\"
echo "  --mode ${WANDB_MODE}"

# Execute the command (relative to the script's parent directory)
# Use ../ to point to train_hypo_medmnistc.py in the hypo_impl directory
python ../train_hypo_medmnistc.py \
  --in-dataset ${DATASET} \
  --wilds_root_dir ${WILDS_ROOT_DIR} \
  --model ${MODEL} \
  --head ${HEAD} \
  --loss ${LOSS_TYPE} \
  --gpu ${GPU_ID} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --lr_decay_epochs ${LR_DECAY_EPOCHS} \
  --lr_decay_rate ${LR_DECAY_RATE} \
  --weight-decay ${WEIGHT_DECAY} \
  --momentum ${MOMENTUM} \
  --prefetch ${PREFETCH} \
  --feat_dim ${FEAT_DIM} \
  --proto_m ${PROTO_M} \
  --w ${LOSS_SCALE_W} \
  --temp ${TEMP} \
  --trial ${TRIAL} \
  --mode ${WANDB_MODE}

echo "Training script finished."
