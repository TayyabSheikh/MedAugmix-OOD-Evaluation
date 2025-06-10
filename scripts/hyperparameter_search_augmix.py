import subprocess
import os
import logging
import pprint
from datetime import datetime # Import datetime

# --- Configuration ---
TRAIN_SCRIPT_ERM = "/home/tsheikh/Thesis/hypo_impl/train_erm_medmnistc.py" # Corrected path
TRAIN_SCRIPT_HYPO = "/home/tsheikh/Thesis/hypo_impl/train_hypo_medmnistc.py" # Corrected path
DATA_ROOT_DIR = "/home/tsheikh/Thesis/data" # Use absolute path
PYTHON_EXECUTABLE = "python" # Or specify path to your python env if needed
SHORT_EPOCHS = 10 # Number of epochs for short runs
SINGLE_SEED = 0 # Seed to use for short runs

# --- Parameters to align with run_multiple_seeds_densenet121_subset.sh ---
# DATA_FLAG removed as it's hardcoded in training scripts now
MODEL_NAME = "densenet121"
CONFIG_BATCH_SIZE = 384
CONFIG_LEARNING_RATE = 5e-4 # Matches default in run_multiple_seeds
CONFIG_FEAT_DIM = 128
CONFIG_HEAD = "mlp"
CONFIG_TEMP = 0.1 # For Hypo
CONFIG_LOSS_SCALE_W = 2.0 # For Hypo (passed as --w)
CONFIG_PROTO_M = 0.95 # For Hypo
CONFIG_LR_DECAY_EPOCHS = "30,40" # Note: for SHORT_EPOCHS, may not be very effective but included for consistency
CONFIG_LR_DECAY_RATE = 0.1
CONFIG_WEIGHT_DECAY = 1e-4
CONFIG_MOMENTUM = 0.9
CONFIG_PREFETCH = 16
# --- End Alignment Parameters ---

# Define parameter combinations to test (Severity, Mixture Width)
param_grid = [
    (1, 1), (1, 3), (1, 5),
    (3, 1), (3, 3), (3, 5),
    (5, 1), (5, 3), (5, 5),
]
# --- End Configuration ---

# --- Setup Logging ---
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Console Handler
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
streamHandler.setFormatter(formatter)
log.addHandler(streamHandler)

# File Handler for results summary
results_log_dir = "hyperparameter_search_logs"
os.makedirs(results_log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_log_path = os.path.join(results_log_dir, f"augmix_hp_search_results_{timestamp}.log")
fileHandler = logging.FileHandler(results_log_path, mode='w')
fileHandler.setFormatter(formatter)
log.addHandler(fileHandler)
# --- End Logging Setup ---

log.info(f"Starting AugMix Hyperparameter Search for ERM and Hypo.")
log.info(f"Results will be logged to: {results_log_path}")
log.info(f"Running each combination for {SHORT_EPOCHS} epochs with seed {SINGLE_SEED}.")
log.info(f"Parameter Grid (Severity, Mixture Width): {param_grid}")

results = {}

for severity, width in param_grid:
    log.info(f"\n--- Testing Severity={severity}, Width={width} ---")

    for algo_script, algo_name in [(TRAIN_SCRIPT_ERM, "ERM"), (TRAIN_SCRIPT_HYPO, "Hypo")]:
        log.info(f"Running {algo_name}...")

        # Construct the command
        command = [
            PYTHON_EXECUTABLE,
            algo_script,
            # "--data_flag" removed
            "--wilds_root_dir", DATA_ROOT_DIR,
            "--model", MODEL_NAME,
            "--epochs", str(SHORT_EPOCHS),
            "--seed", str(SINGLE_SEED),
            "--batch_size", str(CONFIG_BATCH_SIZE),
            "--learning_rate", str(CONFIG_LEARNING_RATE),
            "--lr_decay_epochs", CONFIG_LR_DECAY_EPOCHS,
            "--lr_decay_rate", str(CONFIG_LR_DECAY_RATE),
            "--weight-decay", str(CONFIG_WEIGHT_DECAY),
            "--momentum", str(CONFIG_MOMENTUM),
            "--prefetch", str(CONFIG_PREFETCH),
            "--feat_dim", str(CONFIG_FEAT_DIM),
            "--head", CONFIG_HEAD,
            "--loss", algo_name.lower(), # Set loss explicitly (erm or hypo)
            "--gpu", "0", # Assuming GPU 0
            "--mode", "disabled", # Disable wandb
            "--use_med_augmix", # Enable AugMix
            "--augmix_severity", str(severity),
            "--augmix_mixture_width", str(width),
        ]

        # Add Hypo-specific parameters if running Hypo
        if algo_name == "Hypo":
            command.extend([
                "--temp", str(CONFIG_TEMP),
                "--w", str(CONFIG_LOSS_SCALE_W),
                "--proto_m", str(CONFIG_PROTO_M),
            ])

        log.debug(f"Command: {' '.join(command)}")

        try:
            # Execute the command and capture output
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            log.debug("Script executed successfully.")
            log.info(f"--- Output from {algo_name} (Severity={severity}, Width={width}) ---")
            if result.stdout:
                for line in result.stdout.splitlines():
                    log.info(line)
            if result.stderr:
                log.warning(f"--- Errors from {algo_name} (Severity={severity}, Width={width}) ---")
                for line in result.stderr.splitlines():
                    log.warning(line)
            log.info(f"--- End Output from {algo_name} (Severity={severity}, Width={width}) ---")

            # --- Extract Validation Accuracy from Output ---
            # We need to parse the output to find the validation accuracy.
            # Looking at the training script output:
            # INFO:root:Epoch: X, ID Val Acc: Y.YYYYYY, OOD Test Acc: Z.ZZZZZZ
            # We want the final ID Val Acc from the last epoch.
            val_acc = None
            for line in result.stdout.splitlines():
                if f"Epoch: {SHORT_EPOCHS-1}," in line and "ID Val Acc:" in line:
                    try:
                        # Find the ID Val Acc part and extract the number
                        parts = line.split("ID Val Acc:")
                        if len(parts) > 1:
                            acc_str = parts[1].split(',')[0].strip()
                            val_acc = float(acc_str)
                            log.info(f"Extracted ID Val Acc: {val_acc:.6f}")
                            break # Found the last epoch's accuracy
                    except Exception as e:
                        log.warning(f"Could not parse accuracy from line: {line} - {e}")

            if val_acc is not None:
                if algo_name not in results:
                    results[algo_name] = {}
                results[algo_name][(severity, width)] = val_acc
            else:
                log.warning(f"Could not extract validation accuracy for {algo_name} (Severity={severity}, Width={width}).")

        except subprocess.CalledProcessError as e:
            log.error(f"Error running {algo_name} script for Severity={severity}, Width={width}:")
            log.error(f"Command: {' '.join(e.cmd)}")
            log.error(f"Return Code: {e.returncode}")
            log.error(f"Stderr:\n{e.stderr}")
            # log.error(f"Stdout:\n{e.stdout}") # Avoid logging potentially large stdout on error
        except FileNotFoundError:
            log.error(f"Error: Could not find Python executable '{PYTHON_EXECUTABLE}' or training script '{algo_script}'. Check paths.")
            break # Stop if python or script not found
        except Exception as e:
            log.error(f"An unexpected error occurred for {algo_name} (Severity={severity}, Width={width}): {e}")

# --- Summarize Results ---
log.info("\n--- Hyperparameter Search Results Summary ---")
log.info(f"Results logged to: {results_log_path}")

for algo_name, algo_results in results.items():
    log.info(f"\nResults for {algo_name}:")
    # Sort results by validation accuracy (descending)
    sorted_results = sorted(algo_results.items(), key=lambda item: item[1], reverse=True)
    for (severity, width), acc in sorted_results:
        log.info(f"  Severity={severity}, Width={width}: ID Val Acc = {acc:.6f}")

    if sorted_results:
        best_params, best_acc = sorted_results[0]
        log.info(f"\nBest parameters for {algo_name}: Severity={best_params[0]}, Width={best_params[1]} (ID Val Acc = {best_acc:.6f})")
    else:
        log.info(f"No successful runs recorded for {algo_name}.")

log.info("\n--- Hyperparameter Search Finished ---")
