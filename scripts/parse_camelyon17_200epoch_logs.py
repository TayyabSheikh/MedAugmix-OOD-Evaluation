import os
import re
import pandas as pd
import numpy as np
import ast # For safely evaluating string representations of dicts
import logging
from pathlib import Path

# Configure basic logging for the parser itself
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(filename)s - %(message)s')

def parse_folder_name_and_args(folder_path):
    """
    Parses the experiment folder name and train_args.txt to extract hyperparameters.
    Returns a dictionary of parameters or None if essential parts are missing or don't match.
    """
    folder_name = os.path.basename(folder_path)
    params = {
        'folder_name': folder_name,
        'model': None, 'loss': None, 'lr': None, 'cosine': None,
        'bsz': None, 'head': None, 'loss_scale_w': None, 
        'epochs_from_folder': None, 'feat_dim': None,
        'trial_base': None, 'temp': None, 'dataset': 'camelyon17', 'proto_m': None,
        'seed': None, 'declared_epochs_from_args': None, 'augment_from_args': None,
        'augmentation_strategy': 'unknown'
    }

    # Regex for folder name: e.g., 10_04_10:31_erm_resnet50_lr_0.0005_cosine_False_bsz_256_head_mlp_wd_2_200_128_trial_erm_1_temp_0.1_camelyon17_pm_0.95
    # Made more flexible for trial part and wd (loss_scale_w)
    pattern = re.compile(
        r"(\d{2}_\d{2}_\d{2}:\d{2})_"  # Timestamp
        r"(?P<loss>erm|hypo)_"
        r"(?P<model>resnet50|densenet121|resnet18)_" # Added resnet18
        r"lr_(?P<lr>\d+\.\d+)_"
        r"cosine_(?P<cosine>True|False)_"
        r"bsz_(?P<bsz>\d+)_"
        r"head_(?P<head>mlp|linear)_" 
        r"wd_(?P<w>\d+(?:\.\d+)?)_" # Allow float for w (loss_scale_w)
        r"(?P<epochs>\d+)_" 
        r"(?P<feat_dim>\d+)_"
        r"trial_(?P<trial_full>[a-zA-Z0-9_.-]+?)_" # Capture more general trial string, including potential dots or hyphens
        r"temp_(?P<temp>\d+\.\d+)_"
        r"camelyon17_" 
        r"pm_(?P<pm>\d+\.\d+)"
        r"(?P<suffix>_medmnistc|_plain_augmix|_medaugmix(?:_best_sev\d_w\d)?)?" # Optional suffix for aug types
    )
    
    match = pattern.match(folder_name)
    if not match:
        logging.warning(f"Folder name {folder_name} did not match expected pattern.")
        return None
    
    data = match.groupdict()
    params['loss'] = data['loss']
    params['model'] = data['model']
    params['lr'] = float(data['lr'])
    params['cosine'] = data['cosine'] == 'True'
    params['bsz'] = int(data['bsz'])
    params['head'] = data['head']
    params['loss_scale_w'] = float(data['w']) # Changed to float
    params['epochs_from_folder'] = int(data['epochs'])
    params['feat_dim'] = int(data['feat_dim'])
    params['temp'] = float(data['temp'])
    params['proto_m'] = float(data['pm'])
    
    # Post-process trial_full to extract seed and base trial name
    trial_full = data['trial_full']
    seed_match = re.search(r"_seed_(\d+)$", trial_full)
    if seed_match:
        params['seed_from_folder'] = int(seed_match.group(1))
        params['trial_base'] = trial_full[:seed_match.start()]
    else:
        # If no _seed_ at the end, check if trial_full itself is just a number (old style trial_0)
        if trial_full.isdigit():
            params['seed_from_folder'] = int(trial_full) # Assuming this number is the seed
            params['trial_base'] = params['loss'] # Or some other default base
        else:
            params['trial_base'] = trial_full # No explicit seed found in trial_full string
            params['seed_from_folder'] = None # Or a default like 0 or -1

    # Handle augmentation strategy based on folder name parts if not clear from args
    # This is a placeholder, train_args.txt 'augment' is more reliable for basic augs.
    # Suffix or trial_full might indicate specific augs like medaugmix.
    folder_suffix = data.get('suffix', '')
    if folder_suffix == '_medmnistc' and 'medmnistc' not in trial_full and 'medaugmix' not in trial_full:
         # This might indicate a specific type of medmnistc run not covered by trial_full alone
         params['augmentation_strategy'] = 'medmnistc_variant_from_suffix'
    elif 'medaugmix' in trial_full or folder_suffix == '_medaugmix':
        params['augmentation_strategy'] = 'medaugmix_from_folder'
    elif 'plain_augmix' in trial_full or folder_suffix == '_plain_augmix':
        params['augmentation_strategy'] = 'plain_augmix_from_folder'
    elif 'medmnistc' in trial_full and 'augmix' not in trial_full : # e.g. trial_hypo_medmnistc_seed_0
        params['augmentation_strategy'] = 'medmnistc_from_folder'
    # This will be refined/overwritten by train_args.txt parsing for 'augment' flag

    # Now parse train_args.txt
    args_file_path = os.path.join(folder_path, 'train_args.txt')
    if not os.path.exists(args_file_path):
        logging.warning(f"train_args.txt not found in {folder_path}")
        return None # Or handle differently, e.g., rely only on folder name

    try:
        with open(args_file_path, 'r') as f:
            args_content_str = f.read()
            # The content is a string representation of a dict
            train_args = ast.literal_eval(args_content_str)
        
        params['declared_epochs_from_args'] = train_args.get('epochs')
        params['seed'] = train_args.get('seed')
        params['augment_from_args'] = train_args.get('augment')

        if params['augment_from_args'] is True:
            # For Camelyon17, 'augment: True' in train_hypo.py means ToTensor + Normalize
            # as per dataloader.camelyon17_wilds.py
            params['augmentation_strategy'] = 'baseline_minimal (ToTensor+Normalize)'
        elif params['augment_from_args'] is False:
            params['augmentation_strategy'] = 'none'
        else: # if augment arg is missing, treat as unknown or default based on script
            # Assuming train_hypo.py defaults augment to True for Camelyon17 if not specified otherwise
            # However, the train_args.txt should reflect the actual value used.
             params['augmentation_strategy'] = 'unknown_from_args_file'


    except Exception as e:
        logging.error(f"Error parsing train_args.txt in {folder_path}: {e}")
        return None

    # Filter: Only process if declared_epochs_from_args is 200
    if params['declared_epochs_from_args'] != 200:
        logging.info(f"Skipping {folder_name}: Declared epochs ({params['declared_epochs_from_args']}) is not 200.")
        return None
    
    # Also check if epochs from folder name matches, for consistency
    if params['epochs_from_folder'] != 200:
        logging.warning(f"Epochs from folder name ({params['epochs_from_folder']}) "
                        f"does not match target 200 for {folder_name}. Proceeding based on args file.")
        # Decide if this is a critical error or just a warning

    return params

def parse_train_info_log(log_file_path):
    """
    Parses the train_info.log file to extract metrics from the summary block.
    """
    metrics = {
        'best_id_val_acc': np.nan, 'best_id_val_epoch': np.nan,
        'ood_at_best_id_val': np.nan,
        'best_ood_test_acc': np.nan, 'best_ood_test_epoch': np.nan, # Optional
        'id_val_at_best_ood': np.nan, # Optional
        'final_epoch_id_val_acc': np.nan, 'final_epoch_ood_test_acc': np.nan,
        'final_epoch_number_from_log': np.nan,
        'is_complete_200_epochs': False
    }
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

            # Check for completion by looking for the final epoch log line
            # Example: 2025-04-10 20:12:19,326 : INFO : Epoch: 199, ID Val Acc: 0.911750, OOD Test Acc: 0.817632
            # Make regex for Epoch 199 line more general (ignore log level prefix and allow variable spacing)
            final_epoch_line_match = re.search(r"Epoch:\s*199,\s*ID Val Acc:\s*(\d+\.\d+),\s*OOD Test Acc:\s*(\d+\.\d+)", content)
            if final_epoch_line_match:
                metrics['is_complete_200_epochs'] = True
                metrics['final_epoch_number_from_log'] = 199
                # These might be overwritten by the summary block, which is fine
                metrics['final_epoch_id_val_acc'] = float(final_epoch_line_match.group(1))
                metrics['final_epoch_ood_test_acc'] = float(final_epoch_line_match.group(2))

            # Try to parse the summary block first
            summary_block_match = re.search(r"INFO : --- Training Summary ---.*"
                                            r"INFO : Best ID Val Acc: (?P<best_id_val_acc>\d+\.\d+) \(Epoch (?P<best_id_val_epoch>\d+)\).*"
                                            r"INFO : OOD Test Acc at Best ID Val Epoch: (?P<ood_at_best_id_val>\d+\.\d+).*"
                                            r"(?:INFO : Best OOD Test Acc: (?P<best_ood_test_acc>\d+\.\d+) \(Epoch (?P<best_ood_test_epoch>\d+)\).*)?" # Optional
                                            r"(?:INFO : ID Val Acc at Best OOD Test Epoch: (?P<id_val_at_best_ood>\d+\.\d+).*)?" # Optional
                                            r"INFO : Final Epoch \((?P<final_epoch_num_summary>\d+)\) ID Val Acc: (?P<final_epoch_id_val_acc>\d+\.\d+).*"
                                            r"INFO : Final Epoch \((?P<final_epoch_num_summary_ood>\d+)\) OOD Test Acc: (?P<final_epoch_ood_test_acc>\d+\.\d+)",
                                            content, re.DOTALL)
            
            if summary_block_match:
                data = summary_block_match.groupdict()
                metrics['best_id_val_acc'] = float(data['best_id_val_acc'])
                metrics['best_id_val_epoch'] = int(data['best_id_val_epoch'])
                metrics['ood_at_best_id_val'] = float(data['ood_at_best_id_val'])
                
                if data.get('best_ood_test_acc') and data.get('best_ood_test_epoch'):
                    metrics['best_ood_test_acc'] = float(data['best_ood_test_acc'])
                    metrics['best_ood_test_epoch'] = int(data['best_ood_test_epoch'])
                if data.get('id_val_at_best_ood'):
                    metrics['id_val_at_best_ood'] = float(data['id_val_at_best_ood'])

                metrics['final_epoch_id_val_acc'] = float(data['final_epoch_id_val_acc'])
                metrics['final_epoch_ood_test_acc'] = float(data['final_epoch_ood_test_acc'])
                
                # Confirm final epoch number from summary matches 199
                if int(data['final_epoch_num_summary']) == 199 and int(data['final_epoch_num_summary_ood']) == 199 :
                    metrics['final_epoch_number_from_log'] = 199 # Redundant if already set, but confirms
                    metrics['is_complete_200_epochs'] = True # Stronger confirmation
                else:
                    logging.warning(f"Final epoch in summary block for {log_file_path} is not 199. "
                                    f"ID: {data['final_epoch_num_summary']}, OOD: {data['final_epoch_num_summary_ood']}")
                    metrics['is_complete_200_epochs'] = False # If summary doesn't confirm 199
            
            elif not metrics['is_complete_200_epochs']: # If summary block not found and not confirmed by initial epoch line
                logging.warning(f"Summary block not found or Epoch 199 not confirmed by summary in {log_file_path}. "
                                f"Attempting line-by-line parsing for completion and metrics.")
                
                all_epoch_metrics = []
                # Regex for individual epoch lines, could be INFO or DEBUG
                # Example: 2025-04-10 10:34:32,454 : INFO : Epoch: 0, ID Val Acc: 0.899096, OOD Test Acc: 0.878211
                epoch_line_pattern = re.compile(r"Epoch:\s*(\d+),\s*ID Val Acc:\s*(\d+\.\d+),\s*OOD Test Acc:\s*(\d+\.\d+)")
                
                max_epoch_found = -1
                for line in content.splitlines():
                    match = epoch_line_pattern.search(line)
                    if match:
                        epoch = int(match.group(1))
                        id_val = float(match.group(2))
                        ood_test = float(match.group(3))
                        all_epoch_metrics.append({'epoch': epoch, 'id_val_acc': id_val, 'ood_test_acc': ood_test})
                        if epoch > max_epoch_found:
                            max_epoch_found = epoch
                
                if max_epoch_found >= 199: # Check if at least 200 epochs (0-199) were logged
                    metrics['is_complete_200_epochs'] = True
                    metrics['final_epoch_number_from_log'] = max_epoch_found # Could be > 199 if overran

                    # Find best ID Val Acc from all parsed epochs
                    if all_epoch_metrics:
                        best_id_val_run = max(all_epoch_metrics, key=lambda x: x['id_val_acc'])
                        metrics['best_id_val_acc'] = best_id_val_run['id_val_acc']
                        metrics['best_id_val_epoch'] = best_id_val_run['epoch']
                        # Find OOD acc for that specific epoch
                        for m in all_epoch_metrics:
                            if m['epoch'] == best_id_val_run['epoch']:
                                metrics['ood_at_best_id_val'] = m['ood_test_acc']
                                break
                        
                        # Find metrics for epoch 199 specifically for final values
                        epoch_199_data = next((m for m in all_epoch_metrics if m['epoch'] == 199), None)
                        if epoch_199_data:
                            metrics['final_epoch_id_val_acc'] = epoch_199_data['id_val_acc']
                            metrics['final_epoch_ood_test_acc'] = epoch_199_data['ood_test_acc']
                            # final_epoch_number_from_log is already set to max_epoch_found,
                            # but if we specifically want to note it's based on 199:
                            # metrics['final_epoch_number_from_log'] = 199 
                        else: # Should not happen if max_epoch_found >= 199, but as a safeguard
                            logging.warning(f"Max epoch was {max_epoch_found} but data for epoch 199 not found in parsed list for {log_file_path}.")
                            metrics['is_complete_200_epochs'] = False # Revert if epoch 199 data is missing
                
                else: # Fallback parsing did not confirm 200 epochs
                    logging.warning(f"Line-by-line parsing of {log_file_path} did not confirm 200 epochs (max_epoch_found: {max_epoch_found}).")
                    return None # Skip if not confirmed complete by fallback

            # If neither summary block nor initial "Epoch: 199" line (nor fallback) confirmed completion, it will be caught by the final check.

    except FileNotFoundError:
        logging.error(f"Log file not found: {log_file_path}")
        return None
    except Exception as e:
        logging.error(f"Error parsing content of {log_file_path}: {e}")
        return None
        
    if not metrics['is_complete_200_epochs']:
        logging.info(f"Skipping {log_file_path} as it's not confirmed complete for 200 epochs.")
        return None
        
    return metrics

def main():
    base_log_dir = Path('hypo_impl/scripts/logs/camelyon17/')
    all_results = []

    logging.info(f"Scanning directory: {base_log_dir.resolve()}")
    if not base_log_dir.is_dir():
        logging.error(f"Error: Log directory not found at {base_log_dir}")
        return

    for experiment_folder in sorted(base_log_dir.iterdir()):
        if not experiment_folder.is_dir():
            continue
        
        logging.info(f"Processing folder: {experiment_folder.name}")
        
        folder_params = parse_folder_name_and_args(str(experiment_folder))
        if not folder_params:
            continue # Skip if folder name parsing failed or not 200 epochs

        train_info_log_path = experiment_folder / 'train_info.log'
        if not train_info_log_path.exists():
            logging.warning(f"train_info.log not found in {experiment_folder.name}")
            continue
            
        log_metrics = parse_train_info_log(str(train_info_log_path))
        if not log_metrics:
            continue # Skip if log parsing failed or not confirmed complete

        combined_data = {**folder_params, **log_metrics}
        all_results.append(combined_data)
    
    if not all_results:
        logging.warning("No completed 200-epoch log files found matching criteria.")
        return

    df = pd.DataFrame(all_results)
    
    # Define column order for the detailed CSV
    detailed_columns = [
        'folder_name', 'model', 'loss', 'lr', 'bsz', 'seed', 
        'declared_epochs_from_args', 'epochs_from_folder', 'augmentation_strategy',
        'best_id_val_acc', 'best_id_val_epoch', 'ood_at_best_id_val',
        'final_epoch_id_val_acc', 'final_epoch_ood_test_acc', 'final_epoch_number_from_log',
        'is_complete_200_epochs',
        # Add other parsed params if needed: 'cosine', 'head', 'loss_scale_w', 'feat_dim', 'trial_base', 'temp', 'proto_m'
    ]
    # Ensure only existing columns are selected
    df_detailed_ordered = df[[col for col in detailed_columns if col in df.columns]]

    detailed_csv_path = Path('hypo_impl/scripts/') / 'parsed_camelyon17_200epoch_details.csv'
    try:
        df_detailed_ordered.to_csv(detailed_csv_path, index=False)
        logging.info(f"Detailed summary of 200-epoch runs saved to: {detailed_csv_path}")
    except Exception as e:
        logging.error(f"Error saving detailed CSV to {detailed_csv_path}: {e}")

    # --- Generate Aggregated Report ---
    report_path = Path('hypo_impl/scripts/') / 'camelyon17_200epoch_summary_report.txt'
    generate_aggregated_report(df, report_path)


def generate_aggregated_report(df, report_path):
    if df.empty:
        logging.info("DataFrame for aggregation is empty, skipping report generation.")
        return

    report_content = []
    report_content.append("Camelyon17 200-Epoch Experiments - Aggregated Summary Report\n")
    report_content.append("=" * 60 + "\n")
    report_content.append(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Define metrics for aggregation
    metrics_to_agg = {
        'best_id_val_acc': ['mean', 'std', 'count'],
        'ood_at_best_id_val': ['mean', 'std'],
        'final_epoch_id_val_acc': ['mean', 'std'],
        'final_epoch_ood_test_acc': ['mean', 'std'],
    }
    
    # Group by relevant parameters
    # Ensure 'seed' is present for meaningful aggregation if runs differ only by seed
    grouping_keys = ['model', 'loss', 'lr', 'bsz', 'augmentation_strategy', 'loss_scale_w', 'head'] 
    # Filter out any grouping keys not present in the DataFrame
    valid_grouping_keys = [key for key in grouping_keys if key in df.columns]

    if not valid_grouping_keys:
        logging.warning("No valid grouping keys found in DataFrame columns. Cannot generate aggregated report.")
        report_content.append("Could not generate aggregated summary: No valid grouping keys found in the data.\n")
    else:
        grouped = df.groupby(valid_grouping_keys, dropna=False)
        summary_df = grouped.agg(metrics_to_agg)
        # Flatten MultiIndex columns
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        summary_df = summary_df.rename(columns={'best_id_val_acc_count': 'num_runs'}) # Assuming count is on a primary metric
        summary_df = summary_df.reset_index()

        report_content.append("--- Aggregated Results (Mean ± Std over runs/seeds) ---\n\n")
        
        # Select and order columns for the report table
        # Start with grouping keys
        report_table_columns = valid_grouping_keys + ['num_runs']
        # Add aggregated metrics
        for metric_base in metrics_to_agg.keys():
            mean_col = f"{metric_base}_mean"
            std_col = f"{metric_base}_std"
            if mean_col in summary_df.columns:
                report_table_columns.append(mean_col)
            if std_col in summary_df.columns:
                report_table_columns.append(std_col)
        
        report_table_df = summary_df[[col for col in report_table_columns if col in summary_df.columns]].copy()
        
        # Formatting for display (similar to other report)
        for col in report_table_df.columns:
            if '_mean' in col:
                report_table_df[col] = report_table_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")
            elif '_std' in col:
                report_table_df[col] = report_table_df[col].apply(lambda x: f"±{x:.4f}" if pd.notnull(x) and x != 0 else "") # Hide std if 0 or NaN
        
        # Sort by a relevant metric if available
        if 'ood_at_best_id_val_mean' in report_table_df.columns:
            report_table_df = report_table_df.sort_values(by='ood_at_best_id_val_mean', ascending=False)
        elif 'best_id_val_acc_mean' in report_table_df.columns:
            report_table_df = report_table_df.sort_values(by='best_id_val_acc_mean', ascending=False)

        report_content.append(report_table_df.to_markdown(index=False))
        report_content.append("\n\n")

    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(report_content))
        logging.info(f"Aggregated summary report saved to: {report_path}")
    except Exception as e:
        logging.error(f"Error writing aggregated report to {report_path}: {e}")

if __name__ == '__main__':
    main()
