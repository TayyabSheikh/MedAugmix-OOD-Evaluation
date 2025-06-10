#!/bin/bash

# Script to train ERM with MedMNIST-C augmentations AND AugMix on Camelyon17 (WILDS)

# Activate conda environment (adjust if your conda init is different)
# source $(conda info --base)/etc/profile.d/conda.sh # Common way, might vary
# conda activate fresh # Assuming 'fresh' is the environment name

# Define arguments
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
MODEL="densenet121" # Or resnet50, etc.
GPU_ID=0
EPOCHS=200
BATCH_SIZE=256
LEARNING_RATE=5e-4
WANDB_MODE="online" # Set to "disabled" to turn off wandb logging
FEAT_DIM=128
HEAD="mlp"
LOSS_TYPE="erm" # Ensure loss is ERM
TRIAL="medmnistc_in_augmix_erm_0" # Differentiate trial name to reflect MedMNIST-C ops in AugMix
TEMP=0.1 # Temperature (might not be used by ERM loss but needed by script)
LOSS_SCALE_W=2 # Loss scale (might not be used by ERM loss but needed by script)
PROTO_M=0.95 # Proto m (might not be used by ERM loss but needed by script)
LR_DECAY_EPOCHS="100,150,180"
LR_DECAY_RATE=0.1
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
PREFETCH=4

# Construct the command
echo "Running ERM training script with MedMNIST-C + AugMix augmentations from scripts directory..."
echo "Command:"
# Point to the correct python script
echo "python ../train_erm_medmnistc.py \\"
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
echo "  --proto_m ${PROTO_M} \\" # Include args expected by script
echo "  --w ${LOSS_SCALE_W} \\" # Include args expected by script
echo "  --temp ${TEMP} \\" # Include args expected by script
echo "  --trial ${TRIAL} \\"
echo "  --use_med_augmix \\" # Changed to --use_med_augmix
echo "  --augmix_severity 3 \\" # Default severity
echo "  --augmix_mixture_width 3 \\" # Default mixture width
echo "  --mode ${WANDB_MODE}"

# Execute the command (relative to the script's parent directory)
python ../train_erm_medmnistc.py \
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
  --use_med_augmix \
  --augmix_severity 3 \
  --augmix_mixture_width 3 \
  --mode ${WANDB_MODE}

echo "ERM Training script finished."
