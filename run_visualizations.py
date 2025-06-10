import subprocess
import os

# --- Configuration ---
VISUALIZATION_SCRIPT = "/home/tsheikh/Thesis/hypo_impl/visualize_augmix_medmnistc.py"
DATA_ROOT_DIR = "/home/tsheikh/Thesis/data" # Use absolute path
OUTPUT_DIR = "/home/tsheikh/Thesis/visualizations/big_time" # Use absolute path
PYTHON_EXECUTABLE = "python" # Or specify path to your python env if needed

# Define parameter combinations to test
# Each dictionary represents one run of the visualization script
param_sets = [
  {"severity": 1, "width": 1, "corruption": "bloodmnist"},
  {"severity": 1, "width": 2, "corruption": "bloodmnist"},
  {"severity": 1, "width": 3, "corruption": "bloodmnist"},
  {"severity": 1, "width": 4, "corruption": "bloodmnist"},
  {"severity": 1, "width": 5, "corruption": "bloodmnist"},

  {"severity": 2, "width": 1, "corruption": "bloodmnist"},
  {"severity": 2, "width": 2, "corruption": "bloodmnist"},
  {"severity": 2, "width": 3, "corruption": "bloodmnist"},
  {"severity": 2, "width": 4, "corruption": "bloodmnist"},
  {"severity": 2, "width": 5, "corruption": "bloodmnist"},

  {"severity": 3, "width": 1, "corruption": "bloodmnist"},
  {"severity": 3, "width": 2, "corruption": "bloodmnist"},
  {"severity": 3, "width": 3, "corruption": "bloodmnist"},
  {"severity": 3, "width": 4, "corruption": "bloodmnist"},
  {"severity": 3, "width": 5, "corruption": "bloodmnist"},

  {"severity": 4, "width": 1, "corruption": "bloodmnist"},
  {"severity": 4, "width": 2, "corruption": "bloodmnist"},
  {"severity": 4, "width": 3, "corruption": "bloodmnist"},
  {"severity": 4, "width": 4, "corruption": "bloodmnist"},
  {"severity": 4, "width": 5, "corruption": "bloodmnist"},

  {"severity": 5, "width": 1, "corruption": "bloodmnist"},
  {"severity": 5, "width": 2, "corruption": "bloodmnist"},
  {"severity": 5, "width": 3, "corruption": "bloodmnist"},
  {"severity": 5, "width": 4, "corruption": "bloodmnist"},
  {"severity": 5, "width": 5, "corruption": "bloodmnist"}
]

# --- End Configuration ---

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Running visualizations for {len(param_sets)} parameter sets...")

for i, params in enumerate(param_sets):
    severity = params["severity"]
    width = params["width"]
    corruption = params["corruption"]

    # Construct unique output filename
    output_filename = f"augmix_viz_corr_{corruption}_sev_{severity}_width_{width}.png"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\n--- Running Set {i+1}/{len(param_sets)} ---")
    print(f"Params: Severity={severity}, Width={width}, Corruption={corruption}")
    print(f"Output File: {output_filepath}")

    # Construct the command
    command = [
        PYTHON_EXECUTABLE,
        VISUALIZATION_SCRIPT,
        "--wilds_root_dir", DATA_ROOT_DIR,
        "--output_dir", OUTPUT_DIR,
        "--output_filename", output_filename,
        "--augmix_severity", str(severity),
        "--augmix_mixture_width", str(width),
        "--corruption_dataset", corruption,
        # Add other arguments like --num_samples if needed
        # "--num_samples", "8",
    ]

    # Execute the command
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Script executed successfully.")
        # print("Output:\n", result.stdout) # Uncomment to see script output
    except subprocess.CalledProcessError as e:
        print(f"Error running script for set {i+1}:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"Stderr:\n{e.stderr}")
        print(f"Stdout:\n{e.stdout}")
    except FileNotFoundError:
        print(f"Error: Could not find Python executable '{PYTHON_EXECUTABLE}' or script '{VISUALIZATION_SCRIPT}'. Check paths.")
        break # Stop if python or script not found
    except Exception as e:
        print(f"An unexpected error occurred for set {i+1}: {e}")

print("\n--- Visualization runs finished ---")
