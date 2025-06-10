#!/bin/bash

# Script to train HypO with MedMNIST-C operations inside AugMix on Camelyon17 (WILDS)
# Parameters based on run_multiple_seeds_densenet121_subset.sh

# Activate conda environment (adjust if your conda init is different)
# source $(conda info --base)/etc/profile.d/conda.sh # Common way, might vary
# conda activate fresh # Assuming 'fresh' is the environment name

# --- Parameters ---
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
MODEL="densenet121"
GPU_ID=0 # Adjust GPU ID as needed
EPOCHS=1
BATCH_SIZE=384
LEARNING_RATE=5e-4
WANDB_MODE="online" # Set to "disabled" to turn off wandb logging
FEAT_DIM=128
HEAD="mlp"
TEMP=0.1
LOSS_SCALE_W=2.0
PROTO_M=0.95
LR_DECAY_EPOCHS="30,40" # Adjusted for 50 epochs
LR_DECAY_RATE=0.1
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
PREFETCH=16
SEED=0 # Specific seed for this run
LOSS_TYPE="hypo" # Training HypO
PYTHON_SCRIPT="train_hypo_medmnistc.py" # Script that handles MedAugMix dataloader
TRIAL="hypo_medaugmix_seed_${SEED}" # Unique trial name
AUGMIX_SEVERITY=3 # Default severity
AUGMIX_MIXTURE_WIDTH=3 # Default mixture width

echo "Starting HypO training with MedAugMix on Camelyon17..."
echo "Seed: ${SEED}"
echo "Trial Name: ${TRIAL}"

# Construct the command
CMD="python ../${PYTHON_SCRIPT} \\
  --in-dataset ${DATASET} \\
  --wilds_root_dir ${WILDS_ROOT_DIR} \\
  --model ${MODEL} \\
  --head ${HEAD} \\
  --loss ${LOSS_TYPE} \\
  --gpu ${GPU_ID} \\
  --epochs ${EPOCHS} \\
  --batch_size ${BATCH_SIZE} \\
  --learning_rate ${LEARNING_RATE} \\
  --lr_decay_epochs ${LR_DECAY_EPOCHS} \\
  --lr_decay_rate ${LR_DECAY_RATE} \\
  --weight-decay ${WEIGHT_DECAY} \\
  --momentum ${MOMENTUM} \\
  --prefetch ${PREFETCH} \\
  --feat_dim ${FEAT_DIM} \\
  --temp ${TEMP} \\
  --w ${LOSS_SCALE_W} \\
  --proto_m ${PROTO_M} \\
  --trial ${TRIAL} \\
  --seed ${SEED} \\
  --use_med_augmix \\
  --augmix_severity ${AUGMIX_SEVERITY} \\
  --augmix_mixture_width ${AUGMIX_MIXTURE_WIDTH} \\
  --mode ${WANDB_MODE}"

# Print the command
echo "Command:"
echo -e "$CMD\n" # Use -e to interpret escape sequences like newline

# Execute the command
eval "$CMD"

echo "Training script finished."
