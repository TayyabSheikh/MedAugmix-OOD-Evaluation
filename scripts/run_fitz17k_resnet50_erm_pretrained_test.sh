#!/bin/bash

# Script to run an initial test for Fitzpatrick17k dataset with MedAugMix.
# Model: ResNet50 (pretrained)
# Loss: ERM
# Augmentation: MedAugMix (dermamnist, sev3, w3)
# Epochs: 1
# Seed: 0
# Primary Task: 3-class high-level skin condition classification
# OOD Setup: Train on FST Groups 1 & 2, Validate/Test on FST Group 3

# --- Arguments ---
# Ensure WILDS_ROOT_DIR points to the actual 'finalfitz17k' directory
# e.g., if your data is in /home/tsheikh/Thesis/data/finalfitz17k, set WILDS_ROOT_DIR accordingly.
# The default in train_hypo_fitzpatrick.py is "./data/finalfitz17k", which means it expects
# 'data/finalfitz17k' relative to where the script is run (usually project root).
# For clarity, we specify it here.
WILDS_ROOT_DIR="../../data/finalfitz17k" # Relative to project root (Thesis/)
GPU_ID=0
EPOCHS=10
BATCH_SIZE=128
LEARNING_RATE=5e-4
WANDB_MODE="online" 
MODEL_NAME="densenet121"
LOSS_TYPE="erm"
SEED=0
LABEL_PARTITION=3 
TARGET_DOMAIN_OOD="56" # FST Group 3 (Scales 5 & 6) for OOD test
MODEL_PRETRAINED=True 
USE_MED_AUGMIX=True
AUGMIX_CORRUPTION_DATASET="dermamnist"
AUGMIX_SEVERITY=3
AUGMIX_MIXTURE_WIDTH=3
# AUGMENT_TRAIN flag in train_hypo_fitzpatrick.py is still True by default, 
# but use_med_augmix in get_fitzpatrick17k_dataloaders will take precedence.

# --- Other common args from the training script (can be adjusted if needed) ---
FEAT_DIM=128
HEAD="mlp"
TEMP=0.1 
LOSS_SCALE_W=2.0 
PROTO_M=0.95    
LR_DECAY_EPOCHS="30,40" 
LR_DECAY_RATE=0.1
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
PREFETCH=4


echo "Starting Fitzpatrick17k initial test run..."
echo "Model: ${MODEL_NAME}, Loss: ${LOSS_TYPE}, Pretrained: ${MODEL_PRETRAINED}, Epochs: ${EPOCHS}, Seed: ${SEED}"
echo "Label Partition: ${LABEL_PARTITION}, OOD Test Group (target_domain): ${TARGET_DOMAIN_OOD}"
echo "Dataset Root: ${WILDS_ROOT_DIR}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SCRIPT_PATH="${SCRIPT_DIR}/../train_hypo_fitzpatrick.py" 

TRIAL_NAME="${MODEL_NAME}_${LOSS_TYPE}_lp${LABEL_PARTITION}_ood${TARGET_DOMAIN_OOD}"
if [ "$MODEL_PRETRAINED" = true ]; then
    TRIAL_NAME+="_pretrained"
else
    TRIAL_NAME+="_scratch"
fi
TRIAL_NAME+="_testrun_ep${EPOCHS}_seed_${SEED}"

echo "Trial Name: ${TRIAL_NAME}"

# Construct the command
CMD_BASE="python ${PYTHON_SCRIPT_PATH} \\
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
  --model_pretrained ${MODEL_PRETRAINED} \\
  --label_partition ${LABEL_PARTITION} \\
  --target_domain ${TARGET_DOMAIN_OOD}"

CMD_AUGMIX_ARGS=""
if [ "$USE_MED_AUGMIX" = true ] ; then
  CMD_AUGMIX_ARGS=" \\
  --use_med_augmix \\
  --augmix_corruption_dataset ${AUGMIX_CORRUPTION_DATASET} \\
  --augmix_severity ${AUGMIX_SEVERITY} \\
  --augmix_mixture_width ${AUGMIX_MIXTURE_WIDTH}"
fi

CMD="${CMD_BASE}${CMD_AUGMIX_ARGS} \\
  --mode ${WANDB_MODE}"

# The --in-dataset argument defaults to "fitzpatrick17k" in train_hypo_fitzpatrick.py, so not explicitly passed here.

echo "Command:"
echo -e "${CMD//\\$'\n'/}" 
echo ""

eval "$CMD"

echo "Finished initial test run for: ${TRIAL_NAME}"
echo "----------------------------------------"

echo "Fitzpatrick17k initial test script finished."
