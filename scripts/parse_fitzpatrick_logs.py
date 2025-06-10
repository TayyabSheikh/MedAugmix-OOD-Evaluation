import os
import re
import pandas as pd
import numpy as np
import logging

# Configure basic logging for the parser itself
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(filename)s - %(message)s')

def parse_fitzpatrick_filename(filename):
    """
    Parses the FitzPatrick17k log filename to extract hyperparameters.
    """
    params = {
        'filename': filename,
        'model': None,
        'loss': None,
        'augmentation_strategy': None,
        'augmix_severity': None, 
        'augmix_mixture_width': None, 
        'tv_augmix_severity': None, 
        'tv_augmix_mixture_width': None, 
        'tv_augmix_alpha': None, 
        'plain_medmnistc_collection_source': None,
        'label_partition': None,
        'ood_target': None,
        'epochs_from_filename': None, 
        'seed': None,
        'pretrained': None 
    }

    core_pattern_match = re.search(
        r'(?P<model>resnet50|densenet121)_'
        r'(?P<loss>erm|hypo)_'
        r'(?P<aug_strategy_base>'
            r'baseline_noaug|'
            r'medmnistc_dermamnist_s(?P<sev_mc>\d+)w(?P<wid_mc>\d+)|'
            r'medaugmix_dermamnist_s(?P<sev_ma>\d+)w(?P<wid_ma>\d+)|'
            r'plain_tv_augmix|' 
            r'plain_medmnistc_random' 
        r')_'
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
        
        aug_strategy_base = core_pattern_match.group('aug_strategy_base')
        params['augmentation_strategy'] = aug_strategy_base 

        if 'medmnistc_dermamnist_s' in aug_strategy_base and core_pattern_match.group('sev_mc') and core_pattern_match.group('wid_mc'):
            params['augmix_severity'] = int(core_pattern_match.group('sev_mc'))
            params['augmix_mixture_width'] = int(core_pattern_match.group('wid_mc'))
        elif 'medaugmix_dermamnist_s' in aug_strategy_base and core_pattern_match.group('sev_ma') and core_pattern_match.group('wid_ma'):
            params['augmix_severity'] = int(core_pattern_match.group('sev_ma'))
            params['augmix_mixture_width'] = int(core_pattern_match.group('wid_ma'))
        elif aug_strategy_base == 'plain_tv_augmix':
            pass 
        elif aug_strategy_base == 'plain_medmnistc_random':
            params['plain_medmnistc_collection_source'] = 'dermamnist' 
            pass
            
        params['label_partition'] = int(core_pattern_match.group('lp'))
        params['ood_target'] = core_pattern_match.group('ood')
        params['epochs_from_filename'] = int(core_pattern_match.group('ep')) 
        params['seed'] = int(core_pattern_match.group('seed'))
        params['pretrained'] = True if core_pattern_match.group('pt_status') == 'pt' else False
        
        logging.info(f"Parsed: {filename} -> {params}")
        return params
    else:
        detailed_tv_augmix_match = re.search(
            r'(?P<model>resnet50|densenet121)_'
            r'(?P<loss>erm|hypo)_'
            r'tvaugmix_sev(?P<tv_sev>\d+)_mw(?P<tv_mw>\d+)_alpha(?P<tv_alpha>\d+\.\d+)_'
            r'lp(?P<lp>\d+)_'
            r'ood(?P<ood>\d+)_'
            r'ep(?P<ep>\d+)_' 
            r'seed(?P<seed>\d+)_'
            r'(?P<pt_status>pt|scratch)',
            filename
        )
        if detailed_tv_augmix_match:
            params['model'] = detailed_tv_augmix_match.group('model')
            params['loss'] = detailed_tv_augmix_match.group('loss')
            params['augmentation_strategy'] = 'plain_tv_augmix' 
            params['tv_augmix_severity'] = int(detailed_tv_augmix_match.group('tv_sev'))
            params['tv_augmix_mixture_width'] = int(detailed_tv_augmix_match.group('tv_mw'))
            params['tv_augmix_alpha'] = float(detailed_tv_augmix_match.group('tv_alpha'))
            params['label_partition'] = int(detailed_tv_augmix_match.group('lp'))
            params['ood_target'] = detailed_tv_augmix_match.group('ood')
            params['epochs_from_filename'] = int(detailed_tv_augmix_match.group('ep')) 
            params['seed'] = int(detailed_tv_augmix_match.group('seed'))
            params['pretrained'] = True if detailed_tv_augmix_match.group('pt_status') == 'pt' else False
            logging.info(f"Parsed (detailed tv_augmix): {filename} -> {params}")
            return params

        detailed_plain_medc_match = re.search(
            r'(?P<model>resnet50|densenet121)_'
            r'(?P<loss>erm|hypo)_'
            r'plainmedc_col_(?P<col_source>\w+)_rand_sev_'
            r'lp(?P<lp>\d+)_'
            r'ood(?P<ood>\d+)_'
            r'ep(?P<ep>\d+)_' 
            r'seed(?P<seed>\d+)_'
            r'(?P<pt_status>pt|scratch)',
            filename
        )
        if detailed_plain_medc_match:
            params['model'] = detailed_plain_medc_match.group('model')
            params['loss'] = detailed_plain_medc_match.group('loss')
            params['augmentation_strategy'] = 'plain_medmnistc_random' 
            params['plain_medmnistc_collection_source'] = detailed_plain_medc_match.group('col_source')
            params['label_partition'] = int(detailed_plain_medc_match.group('lp'))
            params['ood_target'] = detailed_plain_medc_match.group('ood')
            params['epochs_from_filename'] = int(detailed_plain_medc_match.group('ep')) 
            params['seed'] = int(detailed_plain_medc_match.group('seed'))
            params['pretrained'] = True if detailed_plain_medc_match.group('pt_status') == 'pt' else False
            logging.info(f"Parsed (detailed plain_medc): {filename} -> {params}")
            return params

        logging.warning(f"Could not parse filename with primary or detailed patterns: {filename}")
        return None


def parse_log_content(filepath):
    metrics = {
        'best_id_val_acc': np.nan, 'best_id_val_bal_acc': np.nan, 'best_id_val_epoch': np.nan,
        'ood_at_best_id_val_acc': np.nan, 'ood_at_best_id_val_bal_acc': np.nan,
        'best_ood_test_bal_acc': np.nan, 'best_ood_test_bal_acc_epoch': np.nan,
        'id_val_bal_acc_at_best_ood_bal_acc': np.nan,
        'ood_test_acc_at_best_ood_bal_acc': np.nan, 
        'id_val_acc_at_best_ood_bal_acc': np.nan, 
        'final_epoch_id_val_acc': np.nan, 'final_epoch_id_val_bal_acc': np.nan,
        'final_epoch_ood_test_acc': np.nan, 'final_epoch_ood_test_bal_acc': np.nan,
        'final_epoch_number': np.nan,
        'epochs_from_args': np.nan, 
        'is_complete': False 
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()

            epochs_arg_match = re.search(r"'epochs': (\d+)", content)
            declared_epochs_from_args = 0
            if epochs_arg_match:
                declared_epochs_from_args = int(epochs_arg_match.group(1))
                metrics['epochs_from_args'] = declared_epochs_from_args 
            
            match_best_id = re.search(
                r'Final Best ID Val Acc: (\d+\.\d+) \(Bal: (\d+\.\d+)\) \(Epoch (\d+)\), OOD Test Acc at this epoch: (\d+\.\d+) \(Bal: (\d+\.\d+)\)',
                content)
            if match_best_id:
                metrics['best_id_val_acc'] = float(match_best_id.group(1))
                metrics['best_id_val_bal_acc'] = float(match_best_id.group(2))
                metrics['best_id_val_epoch'] = int(match_best_id.group(3))
                metrics['ood_at_best_id_val_acc'] = float(match_best_id.group(4))
                metrics['ood_at_best_id_val_bal_acc'] = float(match_best_id.group(5))

            match_best_ood_bal = re.search(
                r'Final Best OOD Test Bal Acc: (\d+\.\d+) \(Epoch (\d+)\), ID Val Bal Acc at this epoch: (\d+\.\d+)',
                content)
            if match_best_ood_bal:
                metrics['best_ood_test_bal_acc'] = float(match_best_ood_bal.group(1))
                metrics['best_ood_test_bal_acc_epoch'] = int(match_best_ood_bal.group(2))
                metrics['id_val_bal_acc_at_best_ood_bal_acc'] = float(match_best_ood_bal.group(3))
                
                best_ood_epoch_for_std_search = str(metrics['best_ood_test_bal_acc_epoch'])
                match_std_at_best_ood_bal_epoch = re.search(
                    r'Final Best OOD Test Acc \(Standard\): (\d+\.\d+) \(Epoch ' + re.escape(best_ood_epoch_for_std_search) + r'\), ID Val Acc at this epoch: (\d+\.\d+)',
                    content
                )
                if match_std_at_best_ood_bal_epoch:
                    metrics['ood_test_acc_at_best_ood_bal_acc'] = float(match_std_at_best_ood_bal_epoch.group(1))
                    metrics['id_val_acc_at_best_ood_bal_acc'] = float(match_std_at_best_ood_bal_epoch.group(2))

            match_final_epoch = re.search(
                r'Final Epoch \((\d+)\) ID Val Acc: (\d+\.\d+) \(Bal: (\d+\.\d+)\), OOD Test Acc: (\d+\.\d+) \(Bal: (\d+\.\d+)\)',
                content)
            if match_final_epoch:
                metrics['final_epoch_number'] = int(match_final_epoch.group(1))
                metrics['final_epoch_id_val_acc'] = float(match_final_epoch.group(2))
                metrics['final_epoch_id_val_bal_acc'] = float(match_final_epoch.group(3))
                metrics['final_epoch_ood_test_acc'] = float(match_final_epoch.group(4))
                metrics['final_epoch_ood_test_bal_acc'] = float(match_final_epoch.group(5))

            if declared_epochs_from_args > 0 and not pd.isna(metrics['final_epoch_number']) and \
               metrics['final_epoch_number'] == (declared_epochs_from_args - 1):
                metrics['is_complete'] = True
            
    except FileNotFoundError:
        logging.error(f"Log file not found: {filepath}")
    except Exception as e:
        logging.error(f"Error parsing content of {filepath}: {e}")
    return metrics

def main():
    log_dir_base = 'hypo_impl/scripts/epoch_summary_logs/'
    dataset_subdir = 'fitzpatrick17k' 
    log_dir = os.path.join(log_dir_base, dataset_subdir)
    
    all_results = []
    completed_run_results = [] 

    logging.info(f"Scanning directory: {os.path.abspath(log_dir)}")
    if not os.path.isdir(log_dir):
        logging.error(f"Log directory not found: {log_dir}")
        return

    for filename in sorted(os.listdir(log_dir)):
        if filename.endswith('_epoch_summary.log'): 
            filepath = os.path.join(log_dir, filename)
            logging.info(f"Processing file: {filename}")
            file_params = parse_fitzpatrick_filename(filename)
            
            if file_params:
                log_metrics = parse_log_content(filepath)
                
                actual_epochs_val = file_params.get('epochs_from_filename')
                if pd.isna(actual_epochs_val) and not pd.isna(log_metrics.get('epochs_from_args')):
                    actual_epochs_val = log_metrics['epochs_from_args']
                
                combined_data = {**file_params, **log_metrics, 'actual_epochs_run': actual_epochs_val}
                all_results.append(combined_data)

                expected_epochs_for_completion = 50 
                if log_metrics.get('is_complete') and \
                   (not pd.isna(log_metrics.get('epochs_from_args')) and log_metrics.get('epochs_from_args') == expected_epochs_for_completion):
                    completed_run_results.append(combined_data)
    
    if not completed_run_results: 
        logging.warning("No COMPLETED log files were successfully parsed for the report.")
        if all_results:
            df_all_parsed = pd.DataFrame(all_results)
            debug_csv_path = os.path.join('hypo_impl/scripts/', f'debug_all_parsed_{dataset_subdir}_logs.csv')
            try:
                df_all_parsed.to_csv(debug_csv_path, index=False)
                logging.info(f"Debug CSV with ALL parsed files saved to: {debug_csv_path}")
            except Exception as e:
                logging.error(f"Error saving debug CSV: {e}")
        return

    df = pd.DataFrame(completed_run_results) 
    
    columns_ordered = [
        'filename', 'model', 'loss', 'augmentation_strategy', 
        'augmix_severity', 'augmix_mixture_width', 
        'tv_augmix_severity', 'tv_augmix_mixture_width', 'tv_augmix_alpha',
        'plain_medmnistc_collection_source',
        'label_partition', 'ood_target', 
        'epochs_from_filename', 'epochs_from_args', 'actual_epochs_run', 
        'seed', 'pretrained',
        'is_complete', 'final_epoch_number',
        'best_id_val_acc', 'best_id_val_bal_acc', 'best_id_val_epoch',
        'ood_at_best_id_val_acc', 'ood_at_best_id_val_bal_acc',
        'best_ood_test_bal_acc', 'best_ood_test_bal_acc_epoch',
        'id_val_bal_acc_at_best_ood_bal_acc',
        'ood_test_acc_at_best_ood_bal_acc',
        'id_val_acc_at_best_ood_bal_acc',
        'final_epoch_id_val_acc', 'final_epoch_id_val_bal_acc',
        'final_epoch_ood_test_acc', 'final_epoch_ood_test_bal_acc'
    ]
    df_ordered = df[[col for col in columns_ordered if col in df.columns]]

    logging.info(f"\n--- Parsed Log Summary (DataFrame Head of COMPLETED runs) ---")
    print(df_ordered.head().to_string())
    
    csv_filename = f'parsed_completed_{dataset_subdir}_logs_summary.csv' 
    output_csv_path = os.path.join('hypo_impl/scripts/', csv_filename) 
    try:
        df_ordered.to_csv(output_csv_path, index=False)
        logging.info(f"Summary CSV of COMPLETED runs saved to: {output_csv_path}")
    except Exception as e:
        logging.error(f"Error saving CSV to {output_csv_path}: {e}")

    report_path = os.path.join('hypo_impl/scripts/', f'{dataset_subdir}_experiment_report.txt') 
    generate_analysis_report(df_ordered, report_path, dataset_name=dataset_subdir)

def generate_analysis_report(df, report_path, dataset_name="fitzpatrick17k"):
    if df.empty:
        logging.warning("DataFrame for report is empty, skipping report generation.")
        return

    report_content = []
    report_content.append(f"Experiment Analysis Report for {dataset_name} (Completed Runs)\n") 
    report_content.append("=" * (len(report_content[0]) -1) + "\n")
    report_content.append(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report_content.append(f"Total completed runs parsed: {len(df)}\n") 
    if 'seed' in df.columns:
        report_content.append(f"Unique seeds found: {df['seed'].nunique()}\n")
    if 'model' in df.columns:
        report_content.append(f"Models included: {', '.join(map(str, df['model'].unique()))}\n")
    if 'augmentation_strategy' in df.columns:
        report_content.append(f"Augmentation strategies: {', '.join(map(str, df['augmentation_strategy'].unique()))}\n")
    if 'actual_epochs_run' in df.columns: 
        report_content.append(f"Actual epochs run (mode): {df['actual_epochs_run'].mode().to_list() if not df['actual_epochs_run'].mode().empty else 'N/A'}\n")
    report_content.append("\n")

    report_content.append("--- Explanation of Key Metrics (relevant to FitzPatrick17k logs) ---\n")
    report_content.append("- **actual_epochs_run**: Number of epochs the training actually ran for, derived from filename or log content.\n") 
    report_content.append("- **best_id_val_acc**: Highest In-Distribution (ID) validation accuracy (standard).\n")
    report_content.append("- **best_id_val_bal_acc**: Balanced accuracy on ID validation set at the epoch of `best_id_val_acc`.\n")
    report_content.append("- **best_id_val_epoch**: Epoch number where best ID validation accuracy occurred.\n")
    report_content.append("- **best_ood_test_bal_acc**: Highest Out-of-Distribution (OOD) balanced test accuracy.\n")
    report_content.append("- **best_ood_test_bal_acc_epoch**: Epoch number where best OOD balanced test accuracy occurred.\n")
    report_content.append("\n")

    grouping_keys = ['model', 'loss', 'augmentation_strategy', 
                     'augmix_severity', 'augmix_mixture_width', 
                     'tv_augmix_severity', 'tv_augmix_mixture_width', 'tv_augmix_alpha',
                     'plain_medmnistc_collection_source',
                     'label_partition', 'ood_target', 'pretrained', 'actual_epochs_run'] 
    
    valid_grouping_keys = [key for key in grouping_keys if key in df.columns and df[key].notna().any()]

    metrics_to_agg_list = [
        'best_id_val_acc', 'best_id_val_bal_acc', 'best_id_val_epoch',
        'ood_at_best_id_val_acc', 'ood_at_best_id_val_bal_acc',
        'best_ood_test_bal_acc', 'best_ood_test_bal_acc_epoch',
        'id_val_bal_acc_at_best_ood_bal_acc',
        'ood_test_acc_at_best_ood_bal_acc',
        'id_val_acc_at_best_ood_bal_acc',
        'final_epoch_id_val_bal_acc', 'final_epoch_ood_test_bal_acc'
    ]
    valid_metrics_to_agg = [m for m in metrics_to_agg_list if m in df.columns]
    agg_dict = {metric: ['mean', 'std'] for metric in valid_metrics_to_agg} 
    
    if not agg_dict and 'filename' in df.columns : 
        agg_dict = {'filename': ['count']}
    elif not agg_dict: 
         agg_dict = {}

    if valid_grouping_keys:
        grouped = df.groupby(valid_grouping_keys, dropna=False) 
        if agg_dict: 
            summary_df = grouped.agg(agg_dict)
            summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        else: 
            summary_df = pd.DataFrame(index=grouped.groups.keys())

        summary_df['num_seeds_completed'] = grouped.size().values 
        summary_df = summary_df.reset_index()
    else: 
        summary_df = pd.DataFrame()
        if valid_metrics_to_agg:
            single_row_data = {f"{m}_mean": df[m].mean() for m in valid_metrics_to_agg if m in df.columns}
            single_row_data.update({f"{m}_std": df[m].std() for m in valid_metrics_to_agg if m in df.columns})
            summary_df = pd.DataFrame([single_row_data])
        summary_df['num_seeds_completed'] = len(df)
        for key in grouping_keys: 
            if key in df.columns:
                 summary_df[key] = df[key].iloc[0] if not df.empty else 'N/A'

    report_content.append("--- Summary of Results (Mean ± Std over Seeds) ---\n")
    
    report_columns_means = [f"{m}_mean" for m in valid_metrics_to_agg]
    report_columns_stds = [f"{m}_std" for m in valid_metrics_to_agg]
    
    display_columns = valid_grouping_keys + ['num_seeds_completed'] 
    
    for mean_col, std_col in zip(report_columns_means, report_columns_stds):
        if mean_col in summary_df.columns:
            display_columns.append(mean_col)
        if std_col in summary_df.columns:
            display_columns.append(std_col)
            
    seen = set()
    display_columns = [x for x in display_columns if not (x in seen or seen.add(x))]
            
    report_table_df = summary_df[[col for col in display_columns if col in summary_df.columns]].copy()

    for col in report_table_df.columns:
        if '_mean' in col:
            if 'epoch_mean' in col: # Format epoch means as integers
                 report_table_df[col] = report_table_df[col].apply(lambda x: f"{int(round(x))}" if pd.notnull(x) else "NaN")
            else:
                 report_table_df[col] = report_table_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")
        elif '_std' in col:
            if 'epoch_std' in col: # Format epoch stds with fewer decimals if desired, or as is
                 report_table_df[col] = report_table_df[col].apply(lambda x: f"±{x:.2f}" if pd.notnull(x) else "")
            else:
                 report_table_df[col] = report_table_df[col].apply(lambda x: f"±{x:.4f}" if pd.notnull(x) else "")
        elif col in ['augmix_severity', 'augmix_mixture_width', 'tv_augmix_severity', 'tv_augmix_mixture_width', 'tv_augmix_alpha', 'actual_epochs_run']:
            if col in valid_grouping_keys or col == 'actual_epochs_run': 
                 report_table_df[col] = report_table_df[col].fillna("N/A")
                 if col == 'actual_epochs_run' and pd.api.types.is_numeric_dtype(report_table_df[col].dtype): 
                     report_table_df[col] = report_table_df[col].apply(lambda x: int(x) if pd.notnull(x) and x != "N/A" else "N/A")

    sort_metric = 'best_ood_test_bal_acc_mean'
    if sort_metric in report_table_df.columns:
        report_table_df = report_table_df.sort_values(by=sort_metric, ascending=False)

    report_content.append(report_table_df.to_markdown(index=False))
    report_content.append("\n\n")

    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(report_content))
        logging.info(f"Analysis report saved to: {report_path}")
    except Exception as e:
        logging.error(f"Error writing report to {report_path}: {e}")

if __name__ == '__main__':
    main()
