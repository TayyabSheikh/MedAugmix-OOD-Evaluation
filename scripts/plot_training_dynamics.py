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

# --- Filename Parsing (adapted from parse_fitzpatrick_logs.py) ---
def parse_experiment_filename(filename):
    """
    Parses log filename to extract hyperparameters for grouping.
    Adjusted for FitzPatrick17k naming conventions.
    """
    params = {
        'filename': filename, 'model': None, 'loss': None, 
        'augmentation_strategy': 'unknown', 'seed': None,
        # Add other relevant params if they are in the filename and needed for grouping
        'label_partition': None, 'ood_target': None, 'epochs_in_filename': None,
        'pretrained_status': None
    }

    # Regex to capture the core trial name structure from run_all_fitzpatrick17k_experiments.sh
    # Example: resnet50_erm_plain_tv_augmix_lp3_ood56_ep50_seed0_pt
    core_pattern_match = re.search(
        r'(?P<model>resnet50|densenet121)_'
        r'(?P<loss>erm|hypo)_'
        r'(?P<aug_strategy_base>(?:baseline_noaug)|(?:medmnistc_dermamnist_s\d+w\d+)|(?:medaugmix_dermamnist_s\d+w\d+)|(?:plain_tv_augmix)|(?:plain_medmnistc_random))_'
        r'lp(?P<lp>\d+)_'
        r'ood(?P<ood>\d+)_'
        r'ep(?P<ep>\d+)_'
        r'seed(?P<seed>\d+)_'
        r'(?P<pt_status>pt|scratch)',
        filename 
    )

    if core_pattern_match:
        params['model'] = core_pattern_match.group('model')
        params['loss'] = core_pattern_match.group('loss')
        params['augmentation_strategy'] = core_pattern_match.group('aug_strategy_base')
        params['label_partition'] = int(core_pattern_match.group('lp'))
        params['ood_target'] = core_pattern_match.group('ood')
        params['epochs_in_filename'] = int(core_pattern_match.group('ep'))
        params['seed'] = int(core_pattern_match.group('seed'))
        params['pretrained_status'] = core_pattern_match.group('pt_status')
        logging.debug(f"Parsed filename {filename} -> {params}")
        return params
    else:
        # Fallback for potentially more detailed names from train_hypo_fitzpatrick.py's args.trial
        # These might appear if logs were named directly using args.trial
        detailed_tv_augmix_match = re.search(
            r'(?P<model>resnet50|densenet121)_'
            r'(?P<loss>erm|hypo)_'
            r'tvaugmix_sev\d+_mw\d+_alpha\d+\.\d+_' # Simplified to just detect tvaugmix
            r'lp(?P<lp>\d+)_.*_seed(?P<seed>\d+)_(?P<pt_status>pt|scratch)', filename)
        if detailed_tv_augmix_match:
            params['model'] = detailed_tv_augmix_match.group('model')
            params['loss'] = detailed_tv_augmix_match.group('loss')
            params['augmentation_strategy'] = 'plain_tv_augmix' # Standardize
            params['seed'] = int(detailed_tv_augmix_match.group('seed'))
            params['label_partition'] = int(detailed_tv_augmix_match.group('lp'))
            params['pretrained_status'] = detailed_tv_augmix_match.group('pt_status')
            logging.debug(f"Parsed filename (detailed tv_augmix) {filename} -> {params}")
            return params

        detailed_plain_medc_match = re.search(
            r'(?P<model>resnet50|densenet121)_'
            r'(?P<loss>erm|hypo)_'
            r'plainmedc_col_\w+_rand_sev_'
            r'lp(?P<lp>\d+)_.*_seed(?P<seed>\d+)_(?P<pt_status>pt|scratch)', filename)
        if detailed_plain_medc_match:
            params['model'] = detailed_plain_medc_match.group('model')
            params['loss'] = detailed_plain_medc_match.group('loss')
            params['augmentation_strategy'] = 'plain_medmnistc_random' # Standardize
            params['seed'] = int(detailed_plain_medc_match.group('seed'))
            params['label_partition'] = int(detailed_plain_medc_match.group('lp'))
            params['pretrained_status'] = detailed_plain_medc_match.group('pt_status')
            logging.debug(f"Parsed filename (detailed plain_medc) {filename} -> {params}")
            return params
            
        logging.warning(f"Could not parse filename with primary or detailed patterns: {filename}")
        return None

# --- Log Content Parsing (Per-Epoch Data) ---
def extract_epoch_data(filepath, metrics_to_extract):
    """
    Extracts per-epoch data for specified metrics from a log file.
    Args:
        filepath (str): Path to the log file.
        metrics_to_extract (list): List of metric names (keys) to extract, 
                                   e.g., ['ID Val Bal Acc', 'OOD Test Bal Acc'].
    Returns:
        list: A list of dictionaries, each representing an epoch's data.
              e.g., [{'epoch': 0, 'ID Val Bal Acc': 0.33, 'OOD Test Bal Acc': 0.33}, ...]
    """
    epoch_data_list = []
    # Regex to capture epoch and the relevant metrics with their balanced counterparts
    # Example line: INFO:__main__:Epoch: 0, ID Val Acc: 0.7176 (Bal: 0.3333), OOD Test Acc (FSG 56): 0.8104 (Bal: 0.3333)
    # This regex is flexible for different metrics that might be logged.
    # We will map desired metrics like 'ID Val Bal Acc' to specific groups in the regex.
    
    # Define patterns for each metric. This makes it more extensible.
    metric_patterns = {
        'ID Val Acc': r"ID Val Acc:\s*(\d+\.\d+)",
        'ID Val Bal Acc': r"ID Val Acc:\s*\d+\.\d+\s*\(Bal:\s*(\d+\.\d+)\)",
        'OOD Test Acc': r"OOD Test Acc(?: \(FSG \d+\))?:\s*(\d+\.\d+)", # Optional FSG part
        'OOD Test Bal Acc': r"OOD Test Acc(?: \(FSG \d+\))?:\s*\d+\.\d+\s*\(Bal:\s*(\d+\.\d+)\)",
        # Add more patterns here if other metrics like loss are needed
        # 'Train CE Loss': r"Train CE Loss:\s*(\d+\.\d+)" # Example if train loss is logged per epoch summary
    }

    try:
        with open(filepath, 'r') as f:
            for line in f:
                epoch_match = re.search(r"Epoch:\s*(\d+),", line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    epoch_metrics = {'epoch': current_epoch}
                    found_any_metric_for_epoch = False
                    for metric_key in metrics_to_extract:
                        if metric_key in metric_patterns:
                            value_match = re.search(metric_patterns[metric_key], line)
                            if value_match:
                                epoch_metrics[metric_key] = float(value_match.group(1))
                                found_any_metric_for_epoch = True
                            else:
                                epoch_metrics[metric_key] = np.nan # Metric not found in this line for this epoch
                        else:
                            logging.warning(f"No regex pattern defined for metric: {metric_key}")
                            epoch_metrics[metric_key] = np.nan
                    
                    if found_any_metric_for_epoch: # Only add if at least one requested metric was found
                        epoch_data_list.append(epoch_metrics)
    except FileNotFoundError:
        logging.error(f"Log file not found: {filepath}")
    except Exception as e:
        logging.error(f"Error parsing content of {filepath}: {e}")
    
    return epoch_data_list


# --- Main Function ---
def main(args):
    log_dir = args.log_dir
    output_dir = args.output_dir
    metrics_to_plot_keys = args.metrics_to_plot

    if not os.path.isdir(log_dir):
        logging.error(f"Log directory not found: {log_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    all_epoch_data = []
    for filename in sorted(os.listdir(log_dir)):
        if filename.endswith('_epoch_summary.log'):
            filepath = os.path.join(log_dir, filename)
            file_params = parse_experiment_filename(filename)
            if file_params:
                epoch_data = extract_epoch_data(filepath, metrics_to_plot_keys)
                for record in epoch_data:
                    # Combine file params with epoch data
                    combined_record = {**file_params, **record}
                    all_epoch_data.append(combined_record)
            else:
                logging.info(f"Skipping file (did not match filename pattern): {filename}")

    if not all_epoch_data:
        logging.warning("No data extracted from logs. Exiting.")
        return

    df = pd.DataFrame(all_epoch_data)
    logging.info(f"Collected data for {len(df)} epoch records across all files.")
    if df.empty:
        logging.warning("DataFrame is empty after processing all logs. No plots will be generated.")
        return

    # Define how to group experiments for plotting lines (e.g., by model, loss, aug_strategy)
    # This creates a unique identifier for each experimental line on the plot
    df['exp_group_id'] = df['model'] + '_' + df['loss'] + '_' + df['augmentation_strategy']
    if args.distinguish_pretrained and 'pretrained_status' in df.columns:
         df['exp_group_id'] += '_' + df['pretrained_status']


    for metric_key in metrics_to_plot_keys:
        if metric_key not in df.columns or df[metric_key].isna().all():
            logging.warning(f"Metric '{metric_key}' not found in data or all values are NaN. Skipping plot.")
            continue

        plt.figure(figsize=(12, 8))
        
        # Group by experiment configuration and epoch, then average over seeds
        # Ensure 'epoch' and metric_key are numeric for mean calculation
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df[metric_key] = pd.to_numeric(df[metric_key], errors='coerce')

        # Drop rows where epoch or metric_key could not be coerced to numeric (became NaT/NaN)
        df_cleaned = df.dropna(subset=['epoch', metric_key])

        # Average across seeds for each exp_group_id and epoch
        # Grouping by 'exp_group_id' and 'epoch'
        # Calculating mean and std for the metric_key
        agg_metrics = df_cleaned.groupby(['exp_group_id', 'epoch'])[metric_key].agg(['mean', 'std']).reset_index()

        for group_name, group_data in agg_metrics.groupby('exp_group_id'):
            plt.plot(group_data['epoch'], group_data['mean'], label=group_name, marker='o', linestyle='-')
            if 'std' in group_data.columns and args.plot_std_dev:
                plt.fill_between(group_data['epoch'], 
                                 group_data['mean'] - group_data['std'], 
                                 group_data['mean'] + group_data['std'], 
                                 alpha=0.2)
        
        plt.title(f'Training Dynamics: {metric_key} vs. Epoch (FitzPatrick17k)')
        plt.xlabel('Epoch')
        plt.ylabel(metric_key.replace('_', ' ')) # Make label more readable
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plot_filename = f"fitzpatrick17k_dynamics_{metric_key.lower().replace(' ', '_')}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_filepath)
            logging.info(f"Saved plot: {plot_filepath}")
        except Exception as e:
            logging.error(f"Error saving plot {plot_filepath}: {e}")
        plt.close()

    logging.info("Plotting complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot training dynamics from experiment logs.")
    parser.add_argument('--log_dir', type=str, 
                        default='hypo_impl/scripts/epoch_summary_logs/fitzpatrick17k/',
                        help='Directory containing the _epoch_summary.log files.')
    parser.add_argument('--output_dir', type=str, 
                        default='visualizations/training_dynamics/fitzpatrick17k/',
                        help='Directory to save the generated plots.')
    parser.add_argument('--metrics_to_plot', nargs='+', 
                        default=['ID Val Bal Acc', 'OOD Test Bal Acc'],
                        help="List of metric keys to plot from log files (e.g., 'ID Val Bal Acc' 'OOD Test Bal Acc').")
    parser.add_argument('--plot_std_dev', action='store_true', help="Plot shaded region for standard deviation across seeds.")
    parser.add_argument('--distinguish_pretrained', action='store_true', help="Distinguish pretrained vs scratch in plot lines/legend.")


    cli_args = parser.parse_args()
    main(cli_args)
