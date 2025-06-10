#!/bin/bash

# Script to run HypO and ERM with MedMNIST-C in AugMix using the
# best identified hyperparameters (Severity 5, Width 1)
# for 50 epochs with multiple random seeds.

# --- Common Arguments ---
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
MODEL="resnet50"
GPU_ID=0 # Adjust GPU ID as needed
EPOCHS=50 # As per the analyzed logs
BATCH_SIZE=384
LEARNING_RATE=5e-4
WANDB_MODE="online" # Set to "disabled" to turn off wandb logging
FEAT_DIM=128
HEAD="mlp"
TEMP=0.1
LOSS_SCALE_W=2.0 # 'w' from logs
PROTO_M=0.95
LR_DECAY_EPOCHS="30,40" # Default from template, adjust if needed for 10 epochs (likely not critical for 10 epochs)
LR_DECAY_RATE=0.1
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
PREFETCH=16

# --- Best AugMix Hyperparameters ---
AUGMIX_SEVERITY=5
AUGMIX_MIXTURE_WIDTH=1

# --- Seeds to run ---
SEEDS=(0 1 2) # Run with seeds 0, 1, and 2 for 3 runs each

# --- Training Configurations ---
# Format: LOSS_TYPE PYTHON_SCRIPT_NAME TRIAL_BASE_PREFIX
CONFIGS=(
    "hypo train_hypo_medmnistc.py hypo_medaugmix_best"
    "erm train_erm_medmnistc.py erm_medaugmix_best"
)

echo "Starting multiple seed training runs with best MedMNIST-C AugMix hyperparameters..."

# Loop through each configuration
for config in "${CONFIGS[@]}"; do
    read -r LOSS_TYPE PYTHON_SCRIPT TRIAL_BASE_PREFIX <<< "$config"

    # Loop through each seed
    for SEED in "${SEEDS[@]}"; do
        TRIAL="${TRIAL_BASE_PREFIX}_sev${AUGMIX_SEVERITY}_w${AUGMIX_MIXTURE_WIDTH}_seed_${SEED}" # Unique trial name

        echo "Running ${LOSS_TYPE} (${TRIAL_BASE_PREFIX}) with AugMix Severity ${AUGMIX_SEVERITY}, Width ${AUGMIX_MIXTURE_WIDTH}, Seed ${SEED}..."
        echo "Trial Name: ${TRIAL}"

   
        # --augmix_severity, and --augmix_mixture_width flags.
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

        echo "${LOSS_TYPE} (${TRIAL_BASE_PREFIX}) with seed ${SEED} finished."
        echo "----------------------------------------"
    done
done

echo "All specified training runs with best MedMNIST-C AugMix hyperparameters completed."
