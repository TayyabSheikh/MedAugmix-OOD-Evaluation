import os
import re
import shutil # For copying files
import numpy as np
# --- Reused parse_filename function from parse_experiment_logs.py ---
def parse_filename(filename):
    """
    Parses the log filename to extract hyperparameters.
    Returns a dictionary of parameters or None if pattern doesn't match or is filtered out.
    """
    params = {
        'filename': filename, 'model': None, 'loss': None, 'seed': None,
        'augmix_type': 'none', 'augmix_severity': None, 'augmix_mixture_width': None,
        'base_trial': None, 'epochs_from_filename': None
    }

    # Determine model from filename
    if 'densenet121' in filename:
        params['model'] = 'densenet121'
    elif 'resnet50' in filename:
        params['model'] = 'resnet50'
    else:
        return None # Skip if model is not recognized or not in filename

    # Try to match MedMNISTCAugMix filenames first (more specific)
    match_medaugmix = re.search(r'_trial_(hypo|erm)_medaugmix_best_sev(\d+)_w(\d+)_seed_(\d+)', filename)
    if match_medaugmix:
        params['loss'] = match_medaugmix.group(1)
        params['augmix_type'] = 'custom_medmnistc_augmix'
        params['augmix_severity'] = int(match_medaugmix.group(2))
        params['augmix_mixture_width'] = int(match_medaugmix.group(3))
        params['seed'] = int(match_medaugmix.group(4))
        params['base_trial'] = f"{params['model']}_{params['loss']}_medaugmix_sev{params['augmix_severity']}_w{params['augmix_mixture_width']}"
    else:
        # Try to match basic MedMNIST-C runs
        match_medc = re.search(r'_trial_(hypo|erm)_medmnistc_seed_(\d+)', filename)
        if match_medc:
            params['loss'] = match_medc.group(1)
            params['augmix_type'] = 'basic_medmnistc'
            params['seed'] = int(match_medc.group(2))
            params['base_trial'] = f"{params['model']}_{params['loss']}_medmnistc"
        else:
            # Try to match standard ERM/HypO runs (no MedC, no custom AugMix)
            match_standard = re.search(r'_trial_(hypo|erm)_seed_(\d+)', filename)
            if match_standard:
                params['loss'] = match_standard.group(1)
                params['augmix_type'] = 'standard_erm_hypo'
                params['seed'] = int(match_standard.group(2))
                params['base_trial'] = f"{params['model']}_{params['loss']}"
            else:
                return None # Filename doesn't match any of the targeted patterns

    epoch_match_in_filename = re.search(r'_wd_[\d\.]+_(\d+)_\d+_trial_', filename)
    if epoch_match_in_filename:
        params['epochs_from_filename'] = int(epoch_match_in_filename.group(1))

    return params
# --- End of reused parse_filename function ---

def main_copy():
    source_log_dir = 'hypo_impl/scripts/epoch_summary_logs/camelyon17/'
    # The destination directory will be relative to where the script is run from.
    # Assuming the script is in hypo_impl/scripts/, then 'checkpoint_copies'
    # will be hypo_impl/scripts/checkpoint_copies/
    destination_log_dir = 'checkpoint_copies' 

    if not os.path.isdir(source_log_dir):
        print(f"Error: Source log directory not found at {source_log_dir}")
        return

    if not os.path.exists(destination_log_dir):
        os.makedirs(destination_log_dir)
        print(f"Created destination directory: {destination_log_dir}")
    else:
        print(f"Destination directory already exists: {destination_log_dir}")

    copied_files_count = 0
    skipped_files_count = 0

    print(f"\nScanning source directory: {os.path.abspath(source_log_dir)}")
    for filename in sorted(os.listdir(source_log_dir)):
        if filename.endswith('.log'):
            source_filepath = os.path.join(source_log_dir, filename)
            file_params = parse_filename(filename) # Use the filename parsing logic
            
            if file_params: # If the filename matches basic criteria (model, trial type)
                # Now, check for completeness (50 epochs)
                log_metrics_for_completeness = parse_log_content_for_completeness_check(source_filepath)
                
                if log_metrics_for_completeness.get('epochs_from_args') == 50 and \
                   log_metrics_for_completeness.get('is_complete_50_epochs'):
                    
                    destination_filepath = os.path.join(destination_log_dir, filename)
                    try:
                        shutil.copy2(source_filepath, destination_filepath) # copy2 preserves metadata
                        print(f"Copied (complete 50 epochs): {filename} -> {destination_log_dir}/")
                        copied_files_count += 1
                    except Exception as e:
                        print(f"Error copying {filename}: {e}")
                else:
                    # print(f"Skipped (incomplete or not 50 epochs): {filename}")
                    skipped_files_count += 1
            else:
                # This file did not match the initial filename parsing criteria
                # print(f"Skipped (filename pattern mismatch): {filename}")
                skipped_files_count += 1

    print(f"\n--- Summary ---")
    print(f"Total COMPLETED 50-epoch log files copied: {copied_files_count}")
    print(f"Total files skipped (filename pattern mismatch, incomplete, or not 50 epochs): {skipped_files_count}")
    print(f"Copied logs are in: {os.path.abspath(destination_log_dir)}")

# Minimal version of parse_log_content just for completeness check
def parse_log_content_for_completeness_check(filepath):
    metrics = {
        'epochs_from_args': 0,
        'is_complete_50_epochs': False,
        'final_epoch_number': np.nan,
        'num_epoch_lines': 0
    }
    try:
        with open(filepath, 'r') as f:
            content = f.read()

            epochs_arg_match = re.search(r"'epochs': (\d+)", content)
            if epochs_arg_match:
                metrics['epochs_from_args'] = int(epochs_arg_match.group(1))
            
            match_final_id = re.search(r'Final Epoch \((\d+)\) ID Val Acc: (\d+\.\d+)', content)
            if match_final_id:
                metrics['final_epoch_number'] = int(match_final_id.group(1))

            epoch_accuracies = re.findall(r'Epoch: \d+,\s*ID Val Acc: (\d+\.\d+),\s*OOD Test Acc: (\d+\.\d+)', content)
            metrics['num_epoch_lines'] = len(epoch_accuracies)

            if metrics['epochs_from_args'] == 50:
                if metrics['final_epoch_number'] == 49: # 0-indexed
                    metrics['is_complete_50_epochs'] = True
                elif metrics['num_epoch_lines'] == 50:
                     metrics['is_complete_50_epochs'] = True
            
    except Exception:
        pass # Ignore errors for this simple check, will result in not being copied
    return metrics

if __name__ == '__main__':
    main_copy()
