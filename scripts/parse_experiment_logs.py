import os
import re
import pandas as pd
import numpy as np # For calculating mean if lists are empty

def parse_filename(filename):
    """
    Parses the log filename to extract hyperparameters.
    Returns a dictionary of parameters or None if pattern doesn't match or is filtered out.
    """
    print(f"\n[DEBUG] Parsing filename: {filename}") # DEBUG
    params = {
        'filename': filename, 'model': None, 'loss': None, 'seed': None,
        'augmentation_strategy': 'none', 
        'augmix_severity': None, 'augmix_mixture_width': None,
        'base_trial': None, 'epochs_from_filename': None
    }

    # Determine model from filename (this part is fine and should remain first)
    if 'densenet121' in filename:
        params['model'] = 'densenet121'
    elif 'resnet50' in filename:
        params['model'] = 'resnet50'
    else:
        print(f"[DEBUG] Model not densenet121 or resnet50 in {filename}")
        return None

    # --- New Ordered Regex Matching on full filename ---

    # 1. MedMNIST-C AugMix from SCRATCH (most specific for scratch augmix)
    match_medaugmix_scratch = re.search(
        r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_medaugmix_sev(?P<sev>\d+)_w(?P<width>\d+)_scratch_seed_(?P<seed>\d+)',
        filename
    )
    if match_medaugmix_scratch:
        print(f"[DEBUG] Matched medmnist_c_augmix_scratch for {filename}")
        params['loss'] = match_medaugmix_scratch.group('loss')
        params['augmentation_strategy'] = 'medmnist_c_augmix_scratch'
        params['augmix_severity'] = int(match_medaugmix_scratch.group('sev'))
        params['augmix_mixture_width'] = int(match_medaugmix_scratch.group('width'))
        params['seed'] = int(match_medaugmix_scratch.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_medaugmix_sev{params['augmix_severity']}_w{params['augmix_mixture_width']}_scratch"

    # 2. MedMNIST-C AugMix (standard, potentially pretrained)
    # Pattern includes optional model prefix and optional _best_
    elif (match_medaugmix := re.search(
        r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_medaugmix(?:_best)?_sev(?P<sev>\d+)_w(?P<width>\d+)_seed_(?P<seed>\d+)',
        filename
    )):
        print(f"[DEBUG] Matched medmnist_c_augmix for {filename}")
        params['loss'] = match_medaugmix.group('loss')
        params['augmentation_strategy'] = 'medmnist_c_augmix'
        params['augmix_severity'] = int(match_medaugmix.group('sev'))
        params['augmix_mixture_width'] = int(match_medaugmix.group('width'))
        params['seed'] = int(match_medaugmix.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_medaugmix_sev{params['augmix_severity']}_w{params['augmix_mixture_width']}"

    # 3. Baseline SCRATCH
    elif (match_baseline_scratch := re.search(
        r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_baseline_scratch_seed_(?P<seed>\d+)',
        filename
    )):
        print(f"[DEBUG] Matched baseline_scratch for {filename}")
        params['loss'] = match_baseline_scratch.group('loss')
        params['augmentation_strategy'] = 'baseline_scratch'
        params['seed'] = int(match_baseline_scratch.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_baseline_scratch"

    # 4. Plain AugMix
    elif (match_plain_augmix := re.search(r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_plain_augmix_seed_(?P<seed>\d+)', filename)):
        print(f"[DEBUG] Matched plain_augmix for {filename}")
        params['loss'] = match_plain_augmix.group('loss')
        params['augmentation_strategy'] = 'plain_augmix'
        params['seed'] = int(match_plain_augmix.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_plain_augmix"

    # NEW RULE for Random Single MedMNIST-C
    # Example filename part from trial: ..._trial_resnet50_hypo_rsmedc_bloodmnist_seed_0_...
    # The args.name in train_hypo.py will build the full log name around this trial string.
    elif (match_rsmedc := re.search(
        r'trial_(?P<model_in_trial>resnet50|densenet121)_(?P<loss>hypo|erm)_rsmedc_(?P<source_ds>\w+)_seed_(?P<seed>\d+)',
        filename 
    )):
        print(f"[DEBUG] Matched random_single_medmnist_c ({match_rsmedc.group('source_ds')}) for {filename}")
        params['loss'] = match_rsmedc.group('loss')
        # params['model'] is already set from the filename prefix if this script is called by train_hypo.py's naming convention
        # If the model in trial differs from prefix, prioritize trial for this specific strategy
        if params['model'] != match_rsmedc.group('model_in_trial'):
            print(f"[DEBUG] Model in trial part '{match_rsmedc.group('model_in_trial')}' differs from initial model parse '{params['model']}'. Using model from trial part for rsmedc.")
            params['model'] = match_rsmedc.group('model_in_trial')

        params['augmentation_strategy'] = f"random_single_medmnist_c_{match_rsmedc.group('source_ds')}"
        params['seed'] = int(match_rsmedc.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_random_single_medmnist_c_{match_rsmedc.group('source_ds')}"

    # NEW: MedMNIST-C (specific naming, formerly "legacy")
    elif (match_medmnistc_proper := re.search(
        r'_trial_(?:(?P<trial_model_inner>densenet121|resnet50)_)?(?P<loss>hypo|erm)_medmnistc_seed_(?P<seed>\d+)', # Note: medmnistc without underscore before c
        filename
    )):
        print(f"[DEBUG] Matched medmnistc for {filename}")
        params['loss'] = match_medmnistc_proper.group('loss')
        
        model_in_trial = match_medmnistc_proper.group('trial_model_inner')
        if model_in_trial and params['model'] != model_in_trial:
            print(f"[DEBUG] Model in trial part '{model_in_trial}' differs from initial model parse '{params['model']}'. Using model from trial part for medmnistc.")
            params['model'] = model_in_trial
            
        params['augmentation_strategy'] = 'medmnistc' 
        params['seed'] = int(match_medmnistc_proper.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_medmnistc"

    # 5. Basic MedMNIST-C (Excluding these runs as they are confusingly named baselines)
    elif (match_medc := re.search(r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_medmnist_c_seed_(?P<seed>\d+)', filename)):
        print(f"[DEBUG] Matched filename pattern for 'medmnist_c' (e.g., ..._trial_..._medmnist_c_seed_X). EXCLUDING this run: {filename}")
        return None # Exclude these confusingly named baseline runs

    # 6. Baseline (standard ERM/HypO, implies pretrained if not caught by _scratch_ variants)
    elif (match_standard := re.search(r'_trial_(?:(?P<trial_model>densenet121|resnet50)_)?(?P<loss>hypo|erm)_seed_(?P<seed>\d+)', filename)):
        print(f"[DEBUG] Matched baseline for {filename}")
        params['loss'] = match_standard.group('loss')
        params['augmentation_strategy'] = 'baseline' # This implies pretrained or not explicitly scratch
        params['seed'] = int(match_standard.group('seed'))
        params['base_trial'] = f"{params['model']}_{params['loss']}_baseline"

    # 7. Fallback for very old trial names (e.g., _trial_medmnistc_augmix_0 or _trial_0)
    else:
        print(f"[DEBUG] Trying fallback patterns for {filename}")
        loss_in_fn_prefix_match = re.search(r'_(hypo|erm)_', filename.split('_trial_')[0])
        if loss_in_fn_prefix_match:
            params['loss'] = loss_in_fn_prefix_match.group(1)
            print(f"[DEBUG] Fallback: Inferred loss '{params['loss']}' from filename prefix for {filename}")

            old_medaugmix_match = re.search(r'_trial_medmnistc_augmix_(\d+)', filename)
            if old_medaugmix_match:
                print(f"[DEBUG] Fallback: Matched old medmnistc_augmix format for {filename}")
                params['augmentation_strategy'] = 'medmnist_c_augmix'
                params['seed'] = int(old_medaugmix_match.group(1))
                params['base_trial'] = f"{params['model']}_{params['loss']}_medmnist_c_augmix_unknown_sev_wid"
            elif (old_baseline_match := re.search(r'_trial_(\d+)', filename)): 
                trial_content = old_baseline_match.group(1)
                if trial_content.isdigit(): # Ensure it's purely a digit for seed
                    print(f"[DEBUG] Fallback: Matched old baseline format (e.g. _trial_0) for {filename}")
                    params['augmentation_strategy'] = 'baseline'
                    params['seed'] = int(trial_content)
                    params['base_trial'] = f"{params['model']}_{params['loss']}_baseline"
                else:
                    print(f"[DEBUG] Fallback: _trial_(\\d+) pattern matched non-digit part '{trial_content}'. No match. {filename}")
                    return None
            else:
                print(f"[DEBUG] Fallback: No old pattern matched for {filename} despite inferred loss.")
                return None
        else:
            print(f"[DEBUG] All patterns failed for {filename}, including fallback loss inference.")
            return None
            
    epoch_match_in_filename = re.search(r'_wd_[\d\.]+_(\d+)_\d+_trial_', filename)
    if epoch_match_in_filename:
        params['epochs_from_filename'] = int(epoch_match_in_filename.group(1))

    return params

def parse_log_content(filepath):
    """
    Parses the content of a single log file.
    Returns a dictionary of metrics.
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
        'is_complete_50_epochs': False
    }
    try:
        with open(filepath, 'r') as f:
            content = f.read()

            epochs_arg_match = re.search(r"'epochs': (\d+)", content)
            declared_epochs = 0
            if epochs_arg_match:
                declared_epochs = int(epochs_arg_match.group(1))
                metrics['epochs_from_args'] = declared_epochs

            final_epoch_summary_line = f'Final Epoch ({declared_epochs - 1})'
            if declared_epochs == 50 and final_epoch_summary_line in content:
                 metrics['is_complete_50_epochs'] = True
            elif declared_epochs > 0 and f"Epoch: {declared_epochs -1}," in content: 
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
                if pd.isna(metrics['final_epoch_number']) or metrics['final_epoch_number'] == int(match_final_ood.group(1)):
                    metrics['final_epoch_number'] = int(match_final_ood.group(1))
                    metrics['final_epoch_ood_test_acc'] = float(match_final_ood.group(2))

            epoch_accuracies = re.findall(r'Epoch: \d+,\s*ID Val Acc: (\d+\.\d+),\s*OOD Test Acc: (\d+\.\d+)', content)
            if epoch_accuracies:
                metrics['all_id_val_acc'] = sorted([float(acc_pair[0]) for acc_pair in epoch_accuracies], reverse=True)
                metrics['all_ood_test_acc'] = sorted([float(acc_pair[1]) for acc_pair in epoch_accuracies], reverse=True)
                
                if metrics['all_id_val_acc']:
                    metrics['avg_id_val_acc'] = np.mean(metrics['all_id_val_acc'])
                    metrics['avg_top5_id_val_acc'] = np.mean(metrics['all_id_val_acc'][:5]) if len(metrics['all_id_val_acc']) >=5 else np.mean(metrics['all_id_val_acc'])
                if metrics['all_ood_test_acc']:
                    metrics['avg_ood_test_acc'] = np.mean(metrics['all_ood_test_acc'])
                    metrics['avg_top5_ood_test_acc'] = np.mean(metrics['all_ood_test_acc'][:5]) if len(metrics['all_ood_test_acc']) >=5 else np.mean(metrics['all_ood_test_acc'])
            
            if declared_epochs == 50:
                if metrics['final_epoch_number'] == 49 or \
                   (len(metrics['all_id_val_acc']) == 50 and f"Epoch: {declared_epochs -1}," in content) or \
                   (final_epoch_summary_line in content) :
                    metrics['is_complete_50_epochs'] = True
                else:
                    metrics['is_complete_50_epochs'] = False
            elif declared_epochs > 0 : 
                 if metrics['final_epoch_number'] == (declared_epochs -1) or \
                    (len(metrics['all_id_val_acc']) == declared_epochs and f"Epoch: {declared_epochs -1}," in content):
                     metrics['is_complete_50_epochs'] = True
                 else:
                     metrics['is_complete_50_epochs'] = False
    except Exception as e:
        print(f"Error parsing file content {filepath}: {e}")
    return metrics

def main():
    log_dir = 'hypo_impl/scripts/epoch_summary_logs/camelyon17/'
    all_results = []
    completed_run_results = []

    print(f"Scanning directory: {os.path.abspath(log_dir)}")
    if not os.path.isdir(log_dir):
        print(f"Error: Log directory not found at {log_dir}")
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
        print("No completed 50-epoch log files found matching criteria for the report.")
        if all_results:
            df_all = pd.DataFrame(all_results)
            all_csv_path = os.path.join('hypo_impl/scripts/', 'debug_all_parsed_logs.csv')
            try:
                df_all.to_csv(all_csv_path, index=False)
                print(f"\nDebug CSV with all parsed files (including incomplete) saved to: {all_csv_path}")
            except Exception as e:
                print(f"\nError saving debug CSV: {e}")
        return

    df = pd.DataFrame(completed_run_results)
    
    columns_ordered = [
        'filename', 'model', 'loss', 'seed', 'augmentation_strategy', 
        'augmix_severity', 'augmix_mixture_width', 'base_trial', 'actual_epochs_run',
        'best_id_val_acc', 'best_id_val_epoch', 'ood_at_best_id_val',
        'best_ood_test_acc', 'best_ood_test_epoch', 'id_val_at_best_ood',
        'final_epoch_id_val_acc', 'final_epoch_ood_test_acc', 'final_epoch_number',
        'avg_id_val_acc', 'avg_ood_test_acc',
        'avg_top5_id_val_acc', 'avg_top5_ood_test_acc'
    ]
    df_ordered = df[[col for col in columns_ordered if col in df.columns]]

    print("\n--- Parsed Log Summary (DataFrame of COMPLETED 50-epoch runs) ---")
    print(df_ordered.to_string())
    
    csv_filename = 'all_completed_50epoch_logs_summary.csv' 
    output_csv_path = os.path.join('hypo_impl/scripts/', csv_filename)
    try:
        df_ordered.to_csv(output_csv_path, index=False)
        print(f"\nSummary of completed 50-epoch runs saved to: {output_csv_path}")
    except Exception as e:
        print(f"\nError saving CSV to {output_csv_path}: {e}")

    report_path = os.path.join('hypo_impl/scripts/', 'experiment_summary_report_extended.txt') 
    generate_analysis_report(df_ordered, report_path)

def generate_analysis_report(df, report_path):
    if df.empty:
        print("DataFrame is empty, skipping report generation.")
        return

    report_content = []
    report_content.append("Extended Experiment Analysis Report (All Models and Configurations - Completed 50 Epoch Runs)\n")
    report_content.append("===========================================================================================\n\n")
    report_content.append(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    unique_models = df['model'].unique()
    report_content.append(f"Models included: {', '.join(map(str, unique_models))}\n")
    if 'filename' in df.columns and df['filename'].notna().any():
         report_content.append(f"Data source: Logs from {os.path.basename(df['filename'].dropna().iloc[0])[:12]}... etc.\n\n")
    else:
        report_content.append("Data source: Parsed log files.\n\n")

    report_content.append("This report summarizes model performance for completed 50-epoch runs, including:\n")
    report_content.append("- Different models (e.g., DenseNet121, ResNet50)\n")
    report_content.append("- Standard ERM/HypO (potentially pretrained, 'baseline')\n")
    report_content.append("- Standard ERM/HypO (trained from scratch, 'baseline_scratch')\n")
    report_content.append("- ERM/HypO with MedMNISTC naming convention (typically baseline augmentations, 'medmnistc')\n")
    report_content.append("- ERM/HypO with Custom MedMNISTCAugMix (various Severity/Width settings, potentially pretrained, 'medmnist_c_augmix')\n")
    report_content.append("- ERM/HypO with Custom MedMNISTCAugMix (various Severity/Width settings, trained from scratch, 'medmnist_c_augmix_scratch')\n")
    report_content.append("- ERM/HypO with Plain AugMix (torchvision.transforms.AugMix, 'plain_augmix')\n\n")

    report_content.append("--- Explanation of Metrics ---\n")
    report_content.append("The table below summarizes performance metrics averaged over multiple seeds for each configuration.\n")
    report_content.append("- **model**: Neural network architecture (e.g., 'densenet121', 'resnet50').\n")
    report_content.append("- **base_trial**: Identifier for the core experimental setup (model, loss, augmentation details, scratch status).\n")
    report_content.append("- **loss**: Loss function used ('erm' or 'hypo').\n")
    report_content.append("- **augmentation_strategy**: Type of data augmentation used ('baseline', 'baseline_scratch', 'medmnistc', 'medmnist_c_augmix', 'medmnist_c_augmix_scratch', 'plain_augmix').\n")
    report_content.append("    - 'baseline': Standard training, potentially using pretrained weights.\n")
    report_content.append("    - 'baseline_scratch': Standard training, explicitly from scratch.\n")
    report_content.append("    - 'medmnistc': Runs using MedMNISTC naming (e.g., _trial_..._medmnistc_seed_X), typically baseline augmentations.\n")
    report_content.append("    - 'medmnist_c_augmix': MedMNIST-C AugMix, potentially using pretrained weights.\n")
    report_content.append("    - 'medmnist_c_augmix_scratch': MedMNIST-C AugMix, explicitly from scratch.\n")
    report_content.append("- **augmix_severity**: Severity for 'medmnist_c_augmix' variants (integer or N/A). Not applicable to other strategies.\n")
    report_content.append("- **augmix_mixture_width**: Mixture width for 'medmnist_c_augmix' variants (integer or N/A). Not applicable to other strategies.\n")
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
    report_content.append("- **gap_at_best_id_epoch_mean**: (`best_id_val_acc_mean` - `ood_at_best_id_val_mean`). Gap when selecting by best ID validation.\n")
    report_content.append("- **gap_at_best_ood_epoch_mean**: (`id_val_at_best_ood_mean` - `best_ood_test_acc_mean`). Gap when OOD performance is at its peak.\n")
    report_content.append("- **gap_at_final_epoch_mean**: (`final_epoch_id_val_acc_mean` - `final_epoch_ood_test_acc_mean`). Gap at the end of training.\n\n")

    metrics_to_agg = {
        'best_id_val_acc': ['mean', 'std'], 'best_id_val_epoch': ['mean', 'std'],
        'ood_at_best_id_val': ['mean', 'std'],
        'best_ood_test_acc': ['mean', 'std'], 'best_ood_test_epoch': ['mean', 'std'],
        'id_val_at_best_ood': ['mean', 'std'],
        'final_epoch_id_val_acc': ['mean', 'std'], 'final_epoch_ood_test_acc': ['mean', 'std'],
        'avg_id_val_acc': ['mean', 'std'], 'avg_ood_test_acc': ['mean', 'std'],
        'avg_top5_id_val_acc': ['mean', 'std'], 'avg_top5_ood_test_acc': ['mean', 'std'],
        'actual_epochs_run': ['count'] 
    }
    
    grouping_keys = ['model', 'base_trial', 'loss', 'augmentation_strategy', 'augmix_severity', 'augmix_mixture_width'] 
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
        'model', 'base_trial', 'loss', 'augmentation_strategy', 'augmix_severity', 'augmix_mixture_width', 'num_seeds', 
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
            
    for col_fill_na in ['augmix_severity', 'augmix_mixture_width', 'best_id_val_epoch_mean', 'best_ood_test_epoch_mean']:
        if col_fill_na in report_table_df.columns:
            if '_epoch_mean' in col_fill_na: 
                 def format_epoch_mean(val):
                    if isinstance(val, str) and val not in ['NaN', 'N/A']:
                        try:
                            return f"{round(float(val))}"
                        except ValueError:
                            return val 
                    elif pd.notnull(val) and not isinstance(val, str): 
                        return f"{round(val)}"
                    return "N/A" 
                 report_table_df[col_fill_na] = report_table_df[col_fill_na].apply(format_epoch_mean)
            else: 
                report_table_df[col_fill_na] = report_table_df[col_fill_na].fillna('N/A')


    report_content.append(report_table_df.to_markdown(index=False))
    report_content.append("\n\n")

    report_content.append("--- Key Observations ---\n")
    report_content.append("Report now includes 'Best OOD' checkpoint info, 'Last-Epoch ID Acc', 'Top-5 Averages', and 'ID-OOD Gaps'.\n")


    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(report_content))
        print(f"Extended analysis report saved to: {report_path}")
    except Exception as e:
        print(f"Error writing extended report to {report_path}: {e}")

if __name__ == '__main__':
    main()
