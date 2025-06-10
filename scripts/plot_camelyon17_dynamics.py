import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from collections import defaultdict

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(module)s - %(message)s')

# --- Filename Parsing (adapted from parse_experiment_logs.py for Camelyon17) ---
def parse_camelyon17_filename(filename):
    """
    Parses the Camelyon17 log filename to extract hyperparameters.
    """
    params = {
        'filename': filename, 'model': None, 'loss': None, 'seed': None,
        'augmentation_strategy': 'none', 
        'augmix_severity': None, 'augmix_mixture_width': None, 
        'base_trial': None, 
        'epochs_in_filename': None, 
        'pretrained_status': 'pretrained' 
    }

    if 'densenet121' in filename:
        params['model'] = 'densenet121'
    elif 'resnet50' in filename:
        params['model'] = 'resnet50'
    else:
        logging.debug(f"Model not densenet121 or resnet50 in {filename}")
        return None

    match_medaugmix_scratch = re.search(
        r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_medaugmix_sev(?P<sev>\d+)_w(?P<width>\d+)_scratch_seed_(?P<seed>\d+)',
        filename)
    if match_medaugmix_scratch:
        params['loss'] = match_medaugmix_scratch.group('loss')
        params['augmentation_strategy'] = 'medmnist_c_augmix_scratch'
        params['augmix_severity'] = int(match_medaugmix_scratch.group('sev'))
        params['augmix_mixture_width'] = int(match_medaugmix_scratch.group('width'))
        params['seed'] = int(match_medaugmix_scratch.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_medaugmix_sev{params['augmix_severity']}_w{params['augmix_mixture_width']}_scratch"
        params['pretrained_status'] = 'scratch'
        return params

    match_medaugmix = re.search(
        r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_medaugmix(?:_best)?_sev(?P<sev>\d+)_w(?P<width>\d+)_seed_(?P<seed>\d+)',
        filename)
    if match_medaugmix:
        params['loss'] = match_medaugmix.group('loss')
        params['augmentation_strategy'] = 'medmnist_c_augmix'
        params['augmix_severity'] = int(match_medaugmix.group('sev'))
        params['augmix_mixture_width'] = int(match_medaugmix.group('width'))
        params['seed'] = int(match_medaugmix.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_medaugmix_sev{params['augmix_severity']}_w{params['augmix_mixture_width']}"
        return params

    match_baseline_scratch = re.search(
        r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_baseline_scratch_seed_(?P<seed>\d+)',
        filename)
    if match_baseline_scratch:
        params['loss'] = match_baseline_scratch.group('loss')
        params['augmentation_strategy'] = 'baseline_scratch'
        params['seed'] = int(match_baseline_scratch.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_baseline_scratch"
        params['pretrained_status'] = 'scratch'
        return params

    match_plain_augmix = re.search(r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_plain_augmix_seed_(?P<seed>\d+)', filename)
    if match_plain_augmix:
        params['loss'] = match_plain_augmix.group('loss')
        params['augmentation_strategy'] = 'plain_augmix'
        params['seed'] = int(match_plain_augmix.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_plain_augmix"
        return params

    match_rsmedc = re.search(
        r'trial_(?P<model_in_trial>resnet50|densenet121)_(?P<loss>hypo|erm)_rsmedc_(?P<source_ds>\w+)_seed_(?P<seed>\d+)',
        filename)
    if match_rsmedc:
        params['loss'] = match_rsmedc.group('loss')
        if params['model'] != match_rsmedc.group('model_in_trial'):
             params['model'] = match_rsmedc.group('model_in_trial')
        params['augmentation_strategy'] = f"random_single_medmnist_c_{match_rsmedc.group('source_ds')}"
        params['seed'] = int(match_rsmedc.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_random_single_medmnist_c_{match_rsmedc.group('source_ds')}"
        return params
        
    match_medmnistc_proper = re.search(
        r'_trial_(?:(?P<trial_model_inner>densenet121|resnet50)_)?(?P<loss>hypo|erm)_medmnistc_seed_(?P<seed>\d+)',
        filename)
    if match_medmnistc_proper:
        params['loss'] = match_medmnistc_proper.group('loss')
        model_in_trial = match_medmnistc_proper.group('trial_model_inner')
        if model_in_trial and params['model'] != model_in_trial:
            params['model'] = model_in_trial
        params['augmentation_strategy'] = 'medmnistc' 
        params['seed'] = int(match_medmnistc_proper.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_medmnistc"
        return params

    match_standard = re.search(r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_seed_(?P<seed>\d+)', filename)
    if match_standard:
        params['loss'] = match_standard.group('loss')
        params['augmentation_strategy'] = 'baseline'
        params['seed'] = int(match_standard.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_baseline"
        return params
    
    loss_in_fn_prefix_match = re.search(r'_(hypo|erm)_', filename.split('_trial_')[0] if '_trial_' in filename else filename)
    if loss_in_fn_prefix_match:
        params['loss'] = loss_in_fn_prefix_match.group(1)
        old_medaugmix_match = re.search(r'_trial_medmnistc_augmix_(\d+)', filename)
        if old_medaugmix_match:
            params['augmentation_strategy'] = 'medmnist_c_augmix_old'
            params['seed'] = int(old_medaugmix_match.group(1))
            params['base_trial'] = f"{params['model']}_{params['loss']}_medmnist_c_augmix_old"
            return params
        elif (old_baseline_match := re.search(r'_trial_(\d+)', filename)) and old_baseline_match.group(1).isdigit():
            params['augmentation_strategy'] = 'baseline_old'
            params['seed'] = int(old_baseline_match.group(1))
            params['base_trial'] = f"{params['model']}_{params['loss']}_baseline_old"
            return params

    logging.warning(f"All patterns failed for Camelyon17 filename: {filename}")
    return None

def extract_epoch_data(filepath, metrics_to_extract, target_epochs=50):
    epoch_data_list = []
    max_epoch_found = -1
    
    metric_patterns = {
        'ID Val Acc': r"ID Val Acc:\s*(\d+\.\d+)",
        'OOD Test Acc': r"OOD Test Acc(?: \(FSG \d+\))?:\s*(\d+\.\d+)",
        'ID Val Bal Acc': r"ID Val Acc:\s*\d+\.\d+\s*\(Bal:\s*(\d+\.\d+)\)", 
        'OOD Test Bal Acc': r"OOD Test Acc(?: \(FSG \d+\))?:\s*\d+\.\d+\s*\(Bal:\s*(\d+\.\d+)\)",
    }
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                epoch_line_match = re.search(r"Epoch:\s*(\d+),", line) 
                if epoch_line_match:
                    current_epoch = int(epoch_line_match.group(1))
                    max_epoch_found = max(max_epoch_found, current_epoch)

                    if current_epoch >= target_epochs: # Only consider epochs < target_epochs (0-49 for 50 epochs)
                        continue

                    epoch_metrics = {'epoch': current_epoch}
                    found_any_metric_for_epoch = False
                    for metric_key in metrics_to_extract:
                        if metric_key in metric_patterns:
                            value_match = re.search(metric_patterns[metric_key], line)
                            if value_match:
                                epoch_metrics[metric_key] = float(value_match.group(1))
                                found_any_metric_for_epoch = True
                        else:
                            logging.warning(f"No regex pattern defined for metric: {metric_key}")
                    
                    if found_any_metric_for_epoch:
                        for mk in metrics_to_extract:
                            if mk not in epoch_metrics:
                                epoch_metrics[mk] = np.nan
                        epoch_data_list.append(epoch_metrics)
    except FileNotFoundError:
        logging.error(f"Log file not found: {filepath}")
        return [], -1 # Return empty list and -1 for max_epoch if file not found
    except Exception as e:
        logging.error(f"Error parsing content of {filepath}: {e}")
        return [], max_epoch_found # Return what was parsed so far and max_epoch
    
    # Check if the run completed exactly target_epochs (e.g., 0-49)
    # max_epoch_found will be target_epochs - 1 if it completed fully.
    # Number of unique epochs should be target_epochs.
    if not epoch_data_list: # No data extracted
        return [], max_epoch_found

    df_temp = pd.DataFrame(epoch_data_list)
    if df_temp.empty or df_temp['epoch'].nunique() != target_epochs or max_epoch_found != (target_epochs -1) :
        logging.info(f"File {filepath} did not complete exactly {target_epochs} epochs (0 to {target_epochs-1}). Max epoch found: {max_epoch_found}, Unique epochs: {df_temp['epoch'].nunique() if not df_temp.empty else 0}. Excluding this run.")
        return [], max_epoch_found # Exclude if not exactly target_epochs epochs of data (0 to target_epochs-1)
        
    return epoch_data_list, max_epoch_found

def main(args):
    log_dir = args.log_dir
    output_dir = args.output_dir
    metrics_to_plot_keys = args.metrics_to_plot
    target_epochs_for_plot = args.epochs_to_plot # Use new argument

    if not os.path.isdir(log_dir):
        logging.error(f"Log directory not found: {log_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    all_epoch_data = []
    for filename in sorted(os.listdir(log_dir)):
        if filename.endswith('_epoch_summary.log') or filename.endswith('.log'):
            filepath = os.path.join(log_dir, filename)
            file_params = parse_camelyon17_filename(filename) 
            if file_params:
                # Pass target_epochs_for_plot to extract_epoch_data
                epoch_data, max_epoch_in_file = extract_epoch_data(filepath, metrics_to_plot_keys, target_epochs=target_epochs_for_plot)
                
                # Check if the file was kept (i.e., it had exactly target_epochs_for_plot epochs)
                if epoch_data: # extract_epoch_data returns empty list if not exactly target_epochs
                    for record in epoch_data:
                        combined_record = {**file_params, **record}
                        all_epoch_data.append(combined_record)
                else:
                    logging.info(f"Skipping file {filename} as it did not contain exactly {target_epochs_for_plot} epochs of data (0 to {target_epochs_for_plot-1}).")
            else:
                logging.info(f"Skipping file (did not match Camelyon17 filename pattern): {filename}")

    if not all_epoch_data:
        logging.warning("No data extracted from Camelyon17 logs that met the epoch criteria. Exiting.")
        return

    df = pd.DataFrame(all_epoch_data)
    logging.info(f"Collected data for {len(df)} Camelyon17 epoch records from {df['filename'].nunique()} files that met criteria.")
    if df.empty:
        logging.warning("DataFrame is empty after processing all Camelyon17 logs. No plots will be generated.")
        return

    df['exp_group_id'] = df['model'].fillna('unk_model') + '_' + \
                         df['loss'].fillna('unk_loss') + '_' + \
                         df['augmentation_strategy'].fillna('unk_aug')
    
    if args.distinguish_pretrained and 'pretrained_status' in df.columns:
         df['exp_group_id'] += '_' + df['pretrained_status'].fillna('unk_pt')


    for metric_key in metrics_to_plot_keys:
        if metric_key not in df.columns or df[metric_key].isna().all():
            logging.warning(f"Metric '{metric_key}' not found in Camelyon17 data or all values are NaN. Skipping plot.")
            continue

        plt.figure(figsize=(12, 8))
        
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df[metric_key] = pd.to_numeric(df[metric_key], errors='coerce')
        df_cleaned = df.dropna(subset=['epoch', metric_key])

        agg_metrics = df_cleaned.groupby(['exp_group_id', 'epoch'])[metric_key].agg(['mean', 'std']).reset_index()

        for group_name, group_data in agg_metrics.groupby('exp_group_id'):
            # Ensure data is sorted by epoch for correct line plotting
            group_data = group_data.sort_values(by='epoch')
            plt.plot(group_data['epoch'], group_data['mean'], label=group_name, marker='o', linestyle='-')
            if 'std' in group_data.columns and args.plot_std_dev:
                plt.fill_between(group_data['epoch'], 
                                 group_data['mean'] - group_data['std'], 
                                 group_data['mean'] + group_data['std'], 
                                 alpha=0.2)
        
        plt.title(f'Training Dynamics: {metric_key} vs. Epoch (Camelyon17 - {target_epochs_for_plot} Epochs)')
        plt.xlabel('Epoch')
        plt.ylabel(metric_key.replace('_', ' '))
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(-1, target_epochs_for_plot) # Set x-axis limit
        
        plot_filename = f"camelyon17_dynamics_ep{target_epochs_for_plot}_{metric_key.lower().replace(' ', '_')}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_filepath)
            logging.info(f"Saved plot: {plot_filepath}")
        except Exception as e:
            logging.error(f"Error saving plot {plot_filepath}: {e}")
        plt.close()

    logging.info("Camelyon17 plotting complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot training dynamics from Camelyon17 experiment logs.")
    parser.add_argument('--log_dir', type=str, 
                        default='hypo_impl/scripts/epoch_summary_logs/camelyon17/',
                        help='Directory containing the Camelyon17 _epoch_summary.log files.')
    parser.add_argument('--output_dir', type=str, 
                        default='visualizations/training_dynamics/camelyon17/',
                        help='Directory to save the generated plots for Camelyon17.')
    parser.add_argument('--metrics_to_plot', nargs='+', 
                        default=['ID Val Acc', 'OOD Test Acc'], 
                        help="List of metric keys to plot (e.g., 'ID Val Acc' 'OOD Test Acc').")
    parser.add_argument('--epochs_to_plot', type=int, default=50,
                        help="Filter logs to include only those that ran for exactly this many epochs (e.g., 50 for epochs 0-49).")
    parser.add_argument('--plot_std_dev', action='store_true', help="Plot shaded region for standard deviation across seeds.")
    parser.add_argument('--distinguish_pretrained', action='store_true', help="Distinguish pretrained vs scratch in plot lines/legend.")

    cli_args = parser.parse_args()
    main(cli_args)
