import os
import re
import pandas as pd
import numpy as np # For calculating mean if lists are empty

def parse_filename(filename):
    """
    Parses log filenames specifically for 'legacy_medmnistc_config' runs.
    Returns a dictionary of parameters or None if pattern doesn't match.
    """
    # print(f"\n[DEBUG-LEGACY] Parsing filename: {filename}") # DEBUG
    params = {
        'filename': filename, 'model': None, 'loss': None, 'seed': None,
        'augmentation_strategy': 'none', 
        'augmix_severity': None, 'augmix_mixture_width': None, # Should be N/A for these
        'base_trial': None, 'epochs_from_filename': None
    }

    # Determine model from filename
    if 'densenet121' in filename:
        params['model'] = 'densenet121'
    elif 'resnet50' in filename:
        params['model'] = 'resnet50'
    else:
        # print(f"[DEBUG-LEGACY] Model not densenet121 or resnet50 in {filename}. Skipping.")
        return None

    # Specific pattern for legacy/mislabeled "medmnist_c" runs
    match_legacy_medc = re.search(
        r'_trial_(?:(?P<trial_model_inner>densenet121|resnet50)_)?(?P<loss>hypo|erm)_medmnistc_seed_(?P<seed>\d+)', 
        filename
    )

    if match_legacy_medc:
        # print(f"[DEBUG-LEGACY] Matched legacy medmnist_c pattern: {filename}")
        params['loss'] = match_legacy_medc.group('loss')
        
        # Ensure model consistency if model_in_trial is present and differs
        # This handles cases where the filename prefix might be generic but trial name is specific
        model_in_trial = match_legacy_medc.group('trial_model_inner')
        if model_in_trial and params['model'] != model_in_trial:
            # print(f"[DEBUG-LEGACY] Model in trial part '{model_in_trial}' differs from initial model parse '{params['model']}'. Using model from trial part.")
            params['model'] = model_in_trial
        
        params['augmentation_strategy'] = 'legacy_medmnistc_config' # Special label for this report
        params['seed'] = int(match_legacy_medc.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_legacy_medmnistc_config"
        
        # Attempt to get epochs from filename (copied from original parse_experiment_logs.py)
        epoch_match_in_filename = re.search(r'_wd_[\d\.]+_(\d+)_\d+_trial_', filename)
        if epoch_match_in_filename:
            params['epochs_from_filename'] = int(epoch_match_in_filename.group(1))
        else: # Fallback if wd_epochs_featdim_trial pattern is not there (older names)
            epoch_match_simple = re.search(r'_(\d+)_trial_', filename) # e.g. _50_trial_
            if epoch_match_simple:
                 params['epochs_from_filename'] = int(epoch_match_simple.group(1))


        return params
    else:
        # If it doesn't match this specific pattern, ignore it for this script
        # print(f"[DEBUG-LEGACY] Filename did not match legacy medmnist_c pattern: {filename}")
        return None

def parse_log_content(filepath):
    """
    Parses the content of a single log file.
    Returns a dictionary of metrics. (Copied from parse_experiment_logs.py)
    """
    metrics = {
        'best_id_val_acc': np.nan, 'best_id_val_epoch': np.nan,
        'ood_at_best_id_val': np.nan,
        'best_ood_test_acc': np.nan, 'best_ood_test_epoch': np.nan,
        'id_val_at_best_ood': np.nan,
        'final_epoch_id_val_acc': np.nan, 'final_epoch_ood_test_acc': np.nan,
        'final_epoch_number': np.nan,
        'avg_id_val_acc': np.nan, 'avg_ood_test_acc': np.nan,
        'avg_top5_id_val_acc': np.nan, 'avg_top5_ood_test_acc': np.nan, 
        'all_id_val_acc': [], 'all_ood_test_acc': [],
        'is_complete_50_epochs': False # Default to False
    }
    try:
        with open(filepath, 'r') as f:
            content = f.read()

            epochs_arg_match = re.search(r"'epochs': (\d+)", content)
            declared_epochs = 0
            if epochs_arg_match:
                declared_epochs = int(epochs_arg_match.group(1))
                metrics['epochs_from_args'] = declared_epochs
            
            # Check for completion based on declared epochs (e.g., 50)
            # More robust check for completion: look for summary line or last epoch log
            final_epoch_summary_line = f'Final Epoch ({declared_epochs - 1})'
            if declared_epochs > 0:
                if final_epoch_summary_line in content or f"Epoch: {declared_epochs -1}," in content:
                    metrics['is_complete_50_epochs'] = True


            match = re.search(r'Best ID Val Acc: (\d+\.\d+)\s*\(Epoch (\d+)\)', content)
            if match:
                metrics['best_id_val_acc'] = float(match.group(1))
                metrics['best_id_val_epoch'] = int(match.group(2))

            match = re.search(r'OOD Test Acc at Best ID Val Epoch: (\d+\.\d+)', content)
            if match:
                metrics['ood_at_best_id_val'] = float(match.group(1))

            match = re.search(r'Best OOD Test Acc: (\d+\.\d+)\s*\(Epoch (\d+)\)', content)
            if match:
                metrics['best_ood_test_acc'] = float(match.group(1))
                metrics['best_ood_test_epoch'] = int(match.group(2))
            
            match = re.search(r'ID Val Acc at Best OOD Test Epoch: (\d+\.\d+)', content)
            if match:
                metrics['id_val_at_best_ood'] = float(match.group(1))

            match_final_id = re.search(r'Final Epoch \((\d+)\) ID Val Acc: (\d+\.\d+)', content)
            if match_final_id:
                metrics['final_epoch_number'] = int(match_final_id.group(1))
                metrics['final_epoch_id_val_acc'] = float(match_final_id.group(2))
            
            match_final_ood = re.search(r'Final Epoch \((\d+)\) OOD Test Acc: (\d+\.\d+)', content)
            if match_final_ood:
                # Ensure final_epoch_number is consistent if already set
                if pd.isna(metrics['final_epoch_number']) or metrics['final_epoch_number'] == int(match_final_ood.group(1)):
                    metrics['final_epoch_number'] = int(match_final_ood.group(1))
                metrics['final_epoch_ood_test_acc'] = float(match_final_ood.group(2))

            epoch_accuracies = re.findall(r'Epoch: \d+,\s*ID Val Acc: (\d+\.\d+),\s*OOD Test Acc: (\d+\.\d+)', content)
            if epoch_accuracies:
                metrics['all_id_val_acc'] = sorted([float(acc_pair[0]) for acc_pair in epoch_accuracies], reverse=True)
                metrics['all_ood_test_acc'] = sorted([float(acc_pair[1]) for acc_pair in epoch_accuracies], reverse=True)
                
                if metrics['all_id_val_acc']:
                    metrics['avg_id_val_acc'] = np.mean(metrics['all_id_val_acc']) if metrics['all_id_val_acc'] else np.nan
                    metrics['avg_top5_id_val_acc'] = np.mean(metrics['all_id_val_acc'][:5]) if len(metrics['all_id_val_acc']) >=5 else (np.mean(metrics['all_id_val_acc']) if metrics['all_id_val_acc'] else np.nan)
                if metrics['all_ood_test_acc']:
                    metrics['avg_ood_test_acc'] = np.mean(metrics['all_ood_test_acc']) if metrics['all_ood_test_acc'] else np.nan
                    metrics['avg_top5_ood_test_acc'] = np.mean(metrics['all_ood_test_acc'][:5]) if len(metrics['all_ood_test_acc']) >=5 else (np.mean(metrics['all_ood_test_acc']) if metrics['all_ood_test_acc'] else np.nan)
            
            # Refined completion check based on declared_epochs
            if declared_epochs > 0: # Only if epochs were declared
                # Check if the logged final epoch number matches declared_epochs - 1
                # Or if the number of logged epoch entries matches declared_epochs
                if (not pd.isna(metrics['final_epoch_number']) and metrics['final_epoch_number'] == (declared_epochs - 1)) or \
                   (len(epoch_accuracies) == declared_epochs):
                    metrics['is_complete_50_epochs'] = (declared_epochs == 50) # Specifically for 50 epoch runs
                else:
                    metrics['is_complete_50_epochs'] = False


    except Exception as e:
        print(f"Error parsing file content {filepath}: {e}") # Keep error prints for runtime issues
    return metrics

def main():
    log_dir = 'hypo_impl/scripts/epoch_summary_logs/camelyon17/' # Source directory
    all_results = []
    completed_run_results = []

    # print(f"Scanning directory for LEGACY MedMNIST-C runs: {os.path.abspath(log_dir)}")
    if not os.path.isdir(log_dir):
        print(f"Error: Log directory not found at {log_dir}") # Keep error prints
        return

    for filename in sorted(os.listdir(log_dir)):
        if filename.endswith('.log'):
            filepath = os.path.join(log_dir, filename)
            file_params = parse_filename(filename) 
            
            if file_params: 
                log_metrics = parse_log_content(filepath)
                
                actual_epochs_val = file_params.get('epochs_from_filename')
                if pd.isna(actual_epochs_val) and 'epochs_from_args' in log_metrics:
                    actual_epochs_val = log_metrics['epochs_from_args']
                
                combined_data = {**file_params, **log_metrics, 'actual_epochs_run': actual_epochs_val}
                all_results.append(combined_data)

                if log_metrics.get('epochs_from_args') == 50 and log_metrics.get('is_complete_50_epochs'):
                    completed_run_results.append(combined_data)
    
    if not completed_run_results:
        # print("No completed 50-epoch LEGACY MedMNIST-C log files found matching criteria.")
        if all_results:
            df_all_legacy = pd.DataFrame(all_results)
            legacy_debug_csv_path = os.path.join('hypo_impl/scripts/', 'debug_legacy_medmnistc_parsed_logs.csv')
            try:
                df_all_legacy.to_csv(legacy_debug_csv_path, index=False)
                # print(f"\nDebug CSV with all parsed legacy files saved to: {legacy_debug_csv_path}")
            except Exception as e:
                print(f"\nError saving legacy debug CSV: {e}") # Keep error prints
        return

    df = pd.DataFrame(completed_run_results)
    
    columns_ordered = [
        'filename', 'model', 'loss', 'seed', 'augmentation_strategy', 
        'base_trial', 'actual_epochs_run',
        'best_id_val_acc', 'best_id_val_epoch', 'ood_at_best_id_val',
        'best_ood_test_acc', 'best_ood_test_epoch', 'id_val_at_best_ood',
        'final_epoch_id_val_acc', 'final_epoch_ood_test_acc', 'final_epoch_number',
        'avg_top5_id_val_acc', 'avg_top5_ood_test_acc'
    ]
    df_ordered = df[[col for col in columns_ordered if col in df.columns]]

    # print("\n--- Parsed LEGACY MedMNIST-C Log Summary (DataFrame of COMPLETED 50-epoch runs) ---")
    # print(df_ordered.to_string())
    
    csv_filename = 'legacy_medmnistc_runs_summary.csv' 
    output_csv_path = os.path.join('hypo_impl/scripts/', csv_filename)
    try:
        df_ordered.to_csv(output_csv_path, index=False)
        # print(f"\nSummary of LEGACY MedMNIST-C runs saved to: {output_csv_path}")
    except Exception as e:
        print(f"\nError saving LEGACY CSV to {output_csv_path}: {e}") # Keep error prints

    report_path = os.path.join('hypo_impl/scripts/', 'legacy_medmnistc_runs_report.txt') 
    generate_analysis_report(df_ordered, report_path)

def generate_analysis_report(df, report_path):
    if df.empty:
        # print("DataFrame for LEGACY MedMNIST-C report is empty, skipping report generation.")
        return

    report_content = []
    report_content.append("Legacy MedMNIST-C Configuration Runs Analysis Report (Completed 50 Epoch Runs)\n")
    report_content.append("================================================================================\n\n")
    report_content.append(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    unique_models = df['model'].unique()
    report_content.append(f"Models included: {', '.join(map(str, unique_models))}\n")
    if 'filename' in df.columns and df['filename'].notna().any():
         report_content.append(f"Data source: Logs matching '_trial_..._medmnist_c_seed_X' pattern from {os.path.basename(df['filename'].dropna().iloc[0])[:12]}... etc.\n")
    else:
        report_content.append("Data source: Parsed log files matching legacy pattern.\n")

    if 'filename' in df.columns and df['filename'].notna().any():
        report_content.append("\n--- Parsed Log Filenames Included in this Report ---\n")
        for fname in df['filename'].unique():
            report_content.append(f"- {fname}\n")
        report_content.append("\n")
    
    report_content.append("This report summarizes model performance for runs identified as 'legacy_medmnistc_config'.\n")
    report_content.append("These runs were named with '_medmnist_c_seed_X' in their trial string and are believed\n")
    report_content.append("to have used a dataloader configuration that resulted in baseline augmentations only\n")
    report_content.append("(ToTensor + Normalize), despite their naming.\n\n")


    report_content.append("--- Explanation of Metrics ---\n")
    report_content.append("The table below summarizes performance metrics averaged over multiple seeds for each configuration.\n")
    report_content.append("- **model**: Neural network architecture (e.g., 'densenet121', 'resnet50').\n")
    report_content.append("- **base_trial**: Identifier for the core experimental setup.\n")
    report_content.append("- **loss**: Loss function used ('erm' or 'hypo').\n")
    report_content.append("- **augmentation_strategy**: Should be 'legacy_medmnistc_config' for this report.\n")
    report_content.append("- **num_seeds**: Number of runs (with different random seeds) averaged.\n")
    report_content.append("- **best_id_val_acc_mean/std**: Mean/std of the highest In-Distribution (ID) validation accuracy from each run.\n")
    report_content.append("- **best_id_val_epoch_mean**: Mean epoch where `best_id_val_acc` occurred.\n")
    report_content.append("- **ood_at_best_id_val_mean/std**: Mean/std of Out-of-Distribution (OOD) test accuracy at the epoch of `best_id_val_acc`.\n")
    report_content.append("- **best_ood_test_acc_mean/std**: Mean/std of the highest OOD test accuracy from each run.\n")
    report_content.append("- **best_ood_test_epoch_mean**: Mean epoch where `best_ood_test_acc` occurred.\n")
    report_content.append("- **id_val_at_best_ood_mean/std**: Mean/std of ID validation accuracy at the epoch of `best_ood_test_acc`.\n")
    report_content.append("- **final_epoch_id_val_acc_mean/std**: Mean/std of ID validation accuracy at the final training epoch.\n")
    report_content.append("- **final_epoch_ood_test_acc_mean/std**: Mean/std of OOD test accuracy at the final training epoch.\n")
    report_content.append("- **avg_top5_id_val_acc_mean/std**: Mean/std of the average of the top 5 ID validation accuracies from each run.\n")
    report_content.append("- **avg_top5_ood_test_acc_mean/std**: Mean/std of the average of the top 5 OOD test accuracies from each run.\n")
    report_content.append("- **gap_at_best_id_epoch_mean**: (`best_id_val_acc_mean` - `ood_at_best_id_val_mean`).\n")
    report_content.append("- **gap_at_best_ood_epoch_mean**: (`id_val_at_best_ood_mean` - `best_ood_test_acc_mean`).\n")
    report_content.append("- **gap_at_final_epoch_mean**: (`final_epoch_id_val_acc_mean` - `final_epoch_ood_test_acc_mean`).\n\n")


    metrics_to_agg = {
        'best_id_val_acc': ['mean', 'std'], 'best_id_val_epoch': ['mean', 'std'],
        'ood_at_best_id_val': ['mean', 'std'],
        'best_ood_test_acc': ['mean', 'std'], 'best_ood_test_epoch': ['mean', 'std'],
        'id_val_at_best_ood': ['mean', 'std'],
        'final_epoch_id_val_acc': ['mean', 'std'], 'final_epoch_ood_test_acc': ['mean', 'std'],
        'avg_top5_id_val_acc': ['mean', 'std'], 'avg_top5_ood_test_acc': ['mean', 'std'],
        'actual_epochs_run': ['count'] 
    }
    
    grouping_keys = ['model', 'base_trial', 'loss', 'augmentation_strategy'] 
    valid_grouping_keys = [key for key in grouping_keys if key in df.columns]
    grouped = df.groupby(valid_grouping_keys, dropna=False)
    
    summary_df = grouped.agg(metrics_to_agg).reset_index()
    summary_df.columns = ['_'.join(map(str, col)).strip('_') if isinstance(col, tuple) else col for col in summary_df.columns]
    summary_df = summary_df.rename(columns={'actual_epochs_run_count': 'num_seeds'})

    if 'best_id_val_acc_mean' in summary_df.columns and 'ood_at_best_id_val_mean' in summary_df.columns:
        summary_df['gap_at_best_id_epoch_mean'] = summary_df['best_id_val_acc_mean'] - summary_df['ood_at_best_id_val_mean']
    if 'id_val_at_best_ood_mean' in summary_df.columns and 'best_ood_test_acc_mean' in summary_df.columns:
        summary_df['gap_at_best_ood_epoch_mean'] = summary_df['id_val_at_best_ood_mean'] - summary_df['best_ood_test_acc_mean']
    if 'final_epoch_id_val_acc_mean' in summary_df.columns and 'final_epoch_ood_test_acc_mean' in summary_df.columns:
        summary_df['gap_at_final_epoch_mean'] = summary_df['final_epoch_id_val_acc_mean'] - summary_df['final_epoch_ood_test_acc_mean']


    report_content.append("--- Summary of Results (Mean ± Std over Seeds) ---\n\n")
    if 'ood_at_best_id_val_mean' in summary_df.columns:
        summary_df_sorted = summary_df.sort_values(by='ood_at_best_id_val_mean', ascending=False)
    else:
        summary_df_sorted = summary_df 
    
    report_columns = [
        'model', 'base_trial', 'loss', 'augmentation_strategy', 'num_seeds', 
        'best_id_val_acc_mean', 'best_id_val_acc_std', 'best_id_val_epoch_mean',
        'ood_at_best_id_val_mean', 'ood_at_best_id_val_std',
        'best_ood_test_acc_mean', 'best_ood_test_acc_std', 'best_ood_test_epoch_mean',
        'id_val_at_best_ood_mean', 'id_val_at_best_ood_std',
        'final_epoch_id_val_acc_mean', 'final_epoch_id_val_acc_std',
        'final_epoch_ood_test_acc_mean', 'final_epoch_ood_test_acc_std',
        'avg_top5_id_val_acc_mean', 'avg_top5_id_val_acc_std',
        'avg_top5_ood_test_acc_mean', 'avg_top5_ood_test_acc_std',
        'gap_at_best_id_epoch_mean', 'gap_at_best_ood_epoch_mean', 'gap_at_final_epoch_mean'
    ]
    report_table_df = summary_df_sorted[[col for col in report_columns if col in summary_df_sorted.columns]].copy()

    float_cols_format = {col: '{:.4f}' for col in report_table_df.columns if '_mean' in col or '_std' in col}
    for col, fmt in float_cols_format.items():
        if '_std' in col: 
            report_table_df[col] = report_table_df[col].apply(lambda x: f"± {x:.4f}" if pd.notnull(x) else "")
        else: 
            report_table_df[col] = report_table_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")
            
    for col_fill_na in ['best_id_val_epoch_mean', 'best_ood_test_epoch_mean']:
        if col_fill_na in report_table_df.columns:
            def format_epoch_mean(val):
                if isinstance(val, str) and val not in ['NaN', 'N/A']: 
                    try: return f"{round(float(val))}"
                    except ValueError: return val 
                elif pd.notnull(val) and not isinstance(val, str): 
                    return f"{round(val)}"
                return "N/A" 
            report_table_df[col_fill_na] = report_table_df[col_fill_na].apply(format_epoch_mean)

    report_content.append(report_table_df.to_markdown(index=False))
    report_content.append("\n\n")

    report_content.append("--- End of Legacy MedMNIST-C Report ---\n")

    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(report_content))
        # print(f"Legacy MedMNIST-C analysis report saved to: {report_path}")
    except Exception as e:
        print(f"Error writing legacy report to {report_path}: {e}") # Keep error prints

if __name__ == '__main__':
    main()
