#!/bin/bash

# Script to run expanded ResNet50 experiments on Camelyon17,
# training from scratch (no pretrained weights).
# - Baseline ERM/HypO for seeds 1, 2.
# - MedAugMix (sev3_w5 and sev5_w1) ERM/HypO for seeds 0, 1, 2.

# --- Common Arguments ---
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data"
GPU_ID=0
EPOCHS=50
BATCH_SIZE=384
LEARNING_RATE=5e-4
WANDB_MODE="online"
FEAT_DIM=128
HEAD="mlp"
TEMP=0.1
LOSS_SCALE_W=2.0
PROTO_M=0.95
LR_DECAY_EPOCHS="30,40"
LR_DECAY_RATE=0.1
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
PREFETCH=16
MODEL_NAME="resnet50"

# --- Seeds ---
SEEDS_FOR_BASELINE=(1 2) # Seed 0 assumed to be done by run_cam17_baseline_scratch.sh
SEEDS_FOR_MEDAUGMIX=(0 1 2)

# --- MedAugMix Configurations ---
MEDAUGMIX_CONFIGS=(
    "3 5"  # Severity 3, Width 5
    "5 1"  # Severity 5, Width 1
)

echo "Starting ResNet50 Camelyon17 scratch experiments (expanded)..."

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASELINE_PYTHON_SCRIPT_PATH="${SCRIPT_DIR}/../train_hypo.py"
MEDAUGMIX_ERM_PYTHON_SCRIPT_PATH="${SCRIPT_DIR}/../train_erm_medmnistc.py"
MEDAUGMIX_HYPO_PYTHON_SCRIPT_PATH="${SCRIPT_DIR}/../train_hypo_medmnistc.py"

# --- Baseline Runs (ERM and HypO, Seeds 1, 2) ---
echo ""
echo "# --- Starting Baseline ERM/HypO Runs (ResNet50 Scratch, Seeds 1, 2) ---"
for LOSS_TYPE in "erm" "hypo"; do
    for SEED in "${SEEDS_FOR_BASELINE[@]}"; do
        TRIAL_NAME="${MODEL_NAME}_${LOSS_TYPE}_baseline_scratch_seed_${SEED}"

        echo ""
        echo "# Running: Model=${MODEL_NAME}, Loss=${LOSS_TYPE}, Aug=Baseline, Seed=${SEED}, Pretrained=False"
        echo "# Trial Name: ${TRIAL_NAME}"

        CMD="python ${BASELINE_PYTHON_SCRIPT_PATH} \\
          --in-dataset ${DATASET} \\
          --wilds_root_dir ${WILDS_ROOT_DIR} \\
          --model ${MODEL_NAME} \\
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
          --trial ${TRIAL_NAME} \\
          --seed ${SEED} \\
          --model_pretrained False \\
          --mode ${WANDB_MODE}"
        
        echo "Command:"
        echo -e "${CMD//\\$'\n'/}"
        echo ""
        eval "$CMD"
        echo "Finished run for: ${TRIAL_NAME}"
        echo "----------------------------------------"
    done
done

# --- MedAugMix Runs (ERM and HypO, Seeds 0, 1, 2 for each MedAugMix config) ---
echo ""
echo "# --- Starting MedAugMix ERM/HypO Runs (ResNet50 Scratch, Seeds 0, 1, 2) ---"
for LOSS_TYPE in "erm" "hypo"; do
    for MEDAUG_CONFIG in "${MEDAUGMIX_CONFIGS[@]}"; do
        IFS=' ' read -r AUGMIX_SEVERITY AUGMIX_MIXTURE_WIDTH <<< "$MEDAUG_CONFIG"
        
        PYTHON_SCRIPT_PATH=""
        if [ "$LOSS_TYPE" == "erm" ]; then
            PYTHON_SCRIPT_PATH=$MEDAUGMIX_ERM_PYTHON_SCRIPT_PATH
        else
            PYTHON_SCRIPT_PATH=$MEDAUGMIX_HYPO_PYTHON_SCRIPT_PATH
        fi

        for SEED in "${SEEDS_FOR_MEDAUGMIX[@]}"; do
            TRIAL_NAME="${MODEL_NAME}_${LOSS_TYPE}_medaugmix_sev${AUGMIX_SEVERITY}_w${AUGMIX_MIXTURE_WIDTH}_scratch_seed_${SEED}"

            echo ""
            echo "# Running: Model=${MODEL_NAME}, Loss=${LOSS_TYPE}, Aug=MedAugMix (Sev=${AUGMIX_SEVERITY} Width=${AUGMIX_MIXTURE_WIDTH}), Seed=${SEED}, Pretrained=False"
            echo "# Trial Name: ${TRIAL_NAME}"

            CMD="python ${PYTHON_SCRIPT_PATH} \\
              --in-dataset ${DATASET} \\
              --wilds_root_dir ${WILDS_ROOT_DIR} \\
              --model ${MODEL_NAME} \\
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
              --trial ${TRIAL_NAME} \\
              --seed ${SEED} \\
              --model_pretrained False \\
              --use_med_augmix \\
              --augmix_severity ${AUGMIX_SEVERITY} \\
              --augmix_mixture_width ${AUGMIX_MIXTURE_WIDTH} \\
              --mode ${WANDB_MODE}"
            
            echo "Command:"
            echo -e "${CMD//\\$'\n'/}"
            echo ""
            eval "$CMD"
            echo "Finished run for: ${TRIAL_NAME}"
            echo "----------------------------------------"
        done
    done
done

echo ""
echo "All ResNet50 scratch expanded experiments initiated."
