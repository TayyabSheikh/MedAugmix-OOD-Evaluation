import argparse
import os
import logging
import pprint
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, accuracy_score

# Assuming utils and dataloaders are accessible from this script's location
from utils import set_model
from dataloader.camelyon17_wilds import get_camelyon17_dataloaders

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Evaluate OOD Baselines on Camelyon17')
parser.add_argument('--ckpt_path', required=True, type=str, help='Path to the trained ERM model checkpoint (.pth.tar)')
parser.add_argument('--in-dataset', default="camelyon17", type=str, help='In-distribution dataset name', choices=['camelyon17']) # Currently only supports camelyon17
parser.add_argument('--wilds_root_dir', default="../data", type=str, help='Root directory for WILDS datasets (relative to script location).')
parser.add_argument('--model', default='resnet18', type=str, help='Model architecture matching the checkpoint.')
parser.add_argument('--head', default='mlp', type=str, help='Head type matching the checkpoint.') # Important if model structure depends on it
parser.add_argument('--feat_dim', default=128, type=int, help='Feature dimension (if needed by model structure).')
parser.add_argument('--gpu', default=0, type=int, help='Which GPU to use.')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for evaluation.')
parser.add_argument('--prefetch', type=int, default=4, help='Number of dataloader workers.')
parser.add_argument('--baseline_methods', nargs='+', default=['msp', 'energy'], help='List of baseline methods to evaluate (e.g., msp energy).')
parser.add_argument('--output_dir', default="./baseline_results", type=str, help='Directory to save scores and metrics.')
parser.add_argument('--normalize', action='store_true', help='Normalize penultimate features (match training args if used).') # Match ERM training args

args = parser.parse_args()

# --- Basic Setup ---
torch.cuda.set_device(args.gpu)
os.makedirs(args.output_dir, exist_ok=True)

# --- Logging Setup ---
log_file = os.path.join(args.output_dir, f"eval_{'_'.join(args.baseline_methods)}_{os.path.basename(args.ckpt_path).replace('.pth.tar','')}.log")
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
fileHandler = logging.FileHandler(log_file, mode='w')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.addHandler(fileHandler)
log.addHandler(streamHandler)
log.info("--- Starting Baseline Evaluation ---")
log.info("Arguments:")
log.info(pprint.pformat(vars(args)))

# --- Helper Functions ---
def to_np(x): return x.data.cpu().numpy()

def calculate_msp_score(logits):
    """Calculates Maximum Softmax Probability score."""
    probs = F.softmax(logits, dim=1)
    msp = torch.max(probs, dim=1)[0]
    return to_np(msp)

def calculate_energy_score(logits, T=1.0):
    """Calculates Energy score."""
    # Note: Original paper uses T=1. Adjust if needed.
    energy = -T * torch.logsumexp(logits / T, dim=1)
    return to_np(energy)

def get_fpr_at_tpr(fpr, tpr, threshold=0.95):
    """Helper to find FPR at a specific TPR threshold."""
    if all(tpr < threshold):
        return 1.0 # TPR never reaches threshold
    idx = np.searchsorted(tpr, threshold)
    return fpr[idx]

# --- Main Evaluation Logic ---
def evaluate_baselines():
    log.info(f"Loading model: {args.model}, Head: {args.head}")
    # Need to pass necessary args for set_model based on its requirements
    # Crucially, n_cls needs to be set correctly for the model's final layer
    if args.in_dataset == "camelyon17":
        args.n_cls = 2
    else:
        # Add other dataset class counts if expanding later
        raise ValueError(f"Dataset {args.in_dataset} not configured for n_cls.")

    model = set_model(args) # Assumes set_model creates the correct architecture

    log.info(f"Loading checkpoint from: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=f'cuda:{args.gpu}')
    # Load only the state_dict, handle potential mismatches if HypO keys exist
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint root is the state_dict (less common)
        model.load_state_dict(checkpoint)
    log.info("Checkpoint loaded successfully.")
    model.eval()

    log.info("Loading dataloaders...")
    # Note: We need both ID validation and OOD test sets for OOD detection evaluation
    # The 'val' split in Camelyon17 WILDS is typically used as the ID test set.
    # The 'test' split is the OOD test set.
    _, id_loader, ood_loader = get_camelyon17_dataloaders(
        root_dir=args.wilds_root_dir,
        batch_size=args.batch_size,
        num_workers=args.prefetch
    )
    if id_loader is None or ood_loader is None:
        log.error("Failed to load dataloaders. Exiting.")
        return

    all_scores = {method: {'id': [], 'ood': []} for method in args.baseline_methods}
    id_targets = []
    id_preds = []

    log.info("Processing ID data (val split) for Accuracy and Scores...")
    with torch.no_grad():
        for i, (input, target, metadata) in enumerate(id_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            # Get logits from the ERM model
            penultimate = model.encoder(input).squeeze()
            if args.normalize:
                penultimate = F.normalize(penultimate, dim=1)
            logits = model.head(penultimate) # Assumes head gives logits

            # Calculate scores for requested methods
            if 'msp' in args.baseline_methods:
                all_scores['msp']['id'].extend(calculate_msp_score(logits))
            # Get predictions for accuracy calculation
            pred = logits.data.max(1)[1]
            id_preds.extend(to_np(pred))
            id_targets.extend(to_np(target))

            if 'energy' in args.baseline_methods:
                all_scores['energy']['id'].extend(calculate_energy_score(logits))
            # Add other methods here (e.g., Mahalanobis)

            if (i + 1) % 100 == 0:
                log.info(f"Processed ID batch {i+1}/{len(id_loader)}")

    # Calculate ID Accuracy
    id_accuracy = accuracy_score(id_targets, id_preds)
    log.info(f"ID Validation Accuracy: {id_accuracy:.4f}")


    log.info("Processing OOD data (test split) for Scores and Accuracy...") # Updated log message
    ood_preds = [] # Added for OOD accuracy
    ood_targets = [] # Added for OOD accuracy
    with torch.no_grad():
        for i, (input, target, metadata) in enumerate(ood_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target_np = to_np(target) # Store target before moving to GPU if needed later
            # Get logits from the ERM model
            penultimate = model.encoder(input).squeeze()
            if args.normalize:
                penultimate = F.normalize(penultimate, dim=1)
            logits = model.head(penultimate) # Assumes head gives logits

            # Get predictions for OOD accuracy
            pred = logits.data.max(1)[1]
            ood_preds.extend(to_np(pred))
            ood_targets.extend(target_np) # Use the stored numpy target

            # Calculate scores for requested methods
            if 'msp' in args.baseline_methods:
                all_scores['msp']['ood'].extend(calculate_msp_score(logits))
            if 'energy' in args.baseline_methods:
                all_scores['energy']['ood'].extend(calculate_energy_score(logits))
            # Add other methods here

            if (i + 1) % 100 == 0:
                log.info(f"Processed OOD batch {i+1}/{len(ood_loader)}")

    # Calculate OOD Generalization Accuracy
    ood_accuracy = accuracy_score(ood_targets, ood_preds) # Added
    log.info(f"OOD Generalization Accuracy (Test Set): {ood_accuracy:.4f}") # Added

    # --- Combine Scores and Calculate OOD Detection Metrics ---
    log.info("Calculating OOD detection metrics...")
    results = {}
    for method in args.baseline_methods:
        log.info(f"--- Method: {method} ---")
        scores_id = np.array(all_scores[method]['id'])
        scores_ood = np.array(all_scores[method]['ood'])

        # Check for NaN/Inf scores
        if np.isnan(scores_id).any() or np.isinf(scores_id).any():
            log.warning(f"NaN/Inf detected in ID scores for method {method}. Skipping metrics.")
            continue
        if np.isnan(scores_ood).any() or np.isinf(scores_ood).any():
            log.warning(f"NaN/Inf detected in OOD scores for method {method}. Skipping metrics.")
            continue

        scores = np.concatenate([scores_id, scores_ood])
        # Labels for OOD detection: 1 for ID (positive), 0 for OOD (negative)
        # This convention matches the HypO evaluation script for easier comparison
        labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])

        # OOD scores need consistent interpretation: Higher score = More likely ID.
        # For Energy score, lower is more likely ID, so we negate it.
        # or higher (e.g., MSP). We need consistent direction for metrics.
        # Higher score = More likely ID.
        # For MSP, higher is more likely ID, so it's correct.
        eval_scores = scores.copy() # Use a copy for metric calculation
        if method == 'energy':
            eval_scores = -eval_scores # Lower energy -> higher score for consistent metric interpretation

        try:
            # AUROC
            auroc = roc_auc_score(labels, eval_scores)

            # AUPR (In vs Out) - PR curve where ID is positive class (label=1)
            aupr_in = average_precision_score(labels, eval_scores, pos_label=1)

            # AUPR (Out vs In) - PR curve where OOD is positive class (label=0)
            # Invert labels (0->1, 1->0) and scores (higher score = more likely OOD)
            aupr_out = average_precision_score(1 - labels, -eval_scores, pos_label=1)

            # FPR@95TPR (TPR is for the ID class, label=1)
            fpr, tpr, _ = roc_curve(labels, eval_scores, pos_label=1)
            fpr_at_tpr95 = get_fpr_at_tpr(fpr, tpr, threshold=0.95)

            log.info(f"AUROC: {auroc:.4f}")
            log.info(f"AUPR (In): {aupr_in:.4f}")
            log.info(f"AUPR (Out): {aupr_out:.4f}")
            log.info(f"FPR@95TPR: {fpr_at_tpr95:.4f}")
            results[method] = {
                'auroc': auroc,
                'aupr_in': aupr_in,
                'aupr_out': aupr_out,
                'fpr95': fpr_at_tpr95
            }

            # Save raw scores (before potential negation) and labels (ID=1, OOD=0)
            np.savez(os.path.join(args.output_dir, f"{method}_scores_{os.path.basename(args.ckpt_path).replace('.pth.tar','')}.npz"),
                     scores=scores, labels=labels) # Save original scores and labels (ID=1)

            # Save metrics to text file
            metrics_path = os.path.join(args.output_dir, f"ood_metrics_{method}_{os.path.basename(args.ckpt_path).replace('.pth.tar','')}.txt")
            with open(metrics_path, 'w') as f:
                 f.write(f"Checkpoint: {args.ckpt_path}\n")
                 f.write(f"Method: {method}\n")
                 f.write(f"ID Validation Accuracy: {id_accuracy:.6f}\n")
                 f.write(f"OOD Generalization Accuracy: {ood_accuracy:.6f}\n") # Added line
                 f.write(f"AUROC: {auroc:.6f}\n")
                 f.write(f"AUPR (In): {aupr_in:.6f}\n")
                 f.write(f"AUPR (Out): {aupr_out:.6f}\n")
                 f.write(f"FPR@95TPR: {fpr_at_tpr95:.6f}\n")
            log.info(f"Metrics saved to: {metrics_path}")

        except Exception as e:
            log.error(f"Error calculating metrics for {method}: {e}")

    log.info("--- Evaluation Finished ---")
    return results

if __name__ == '__main__':
    evaluate_baselines()
