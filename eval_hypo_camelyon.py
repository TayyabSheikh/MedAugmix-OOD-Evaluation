import argparse
import math
import os
import time
from datetime import datetime
import logging
# import tensorboard_logger as tb_logger
import pprint

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import numpy as np
import pandas as pd # For FPR calculation helper
from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score

# from make_datasets_cifar import * # Not needed for Camelyon17

from sklearn.metrics import accuracy_score

from utils import (CompLoss, DisLoss, DisLPLoss, set_loader_small, set_loader_ImageNet, set_model)
from dataloader.camelyon17_wilds import get_camelyon17_dataloaders # Added import

parser = argparse.ArgumentParser(description='Eval HYPO for OOD Detection on Camelyon17') # Updated description
parser.add_argument('--gpu', default=0, type=int, help='which GPU to use') # Changed default GPU
parser.add_argument('--seed', default=4, type=int, help='random seed')
parser.add_argument('--w', default=2, type=float, # Not directly used in eval, but kept for consistency if DisLoss needs it
                    help='loss scale')
parser.add_argument('--proto_m', default= 0.99, type=float, # Not directly used in eval, but kept for consistency if DisLoss needs it
                   help='weight of prototype update')
parser.add_argument('--feat_dim', default = 128, type=int,
                    help='feature dim')
# Added 'camelyon17' to choices and removed others for now
parser.add_argument('--in-dataset', default="camelyon17", type=str, help='ID dataset name', choices=['camelyon17'])
# Added argument for WILDS data root, adjust default relative path if needed
parser.add_argument('--wilds_root_dir', default="../data", type=str, help='Root directory for WILDS datasets (relative to script).')
parser.add_argument('--model', default='resnet18', type=str, help='model architecture: [resnet18, resnet50, etc.]')
parser.add_argument('--head', default='mlp', type=str, help='either mlp or linear head')
# Removed loss argument as it's implicitly hypo for this script

# Replaced ckpt_name and ckpt_loc with ckpt_path
parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Full path to the model checkpoint (.pth.tar file)')

parser.add_argument('-b', '--batch_size', default= 128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--print-freq', '-p', default=100, type=int, # Increased print frequency for eval
                    help='print frequency (default: 100)')
parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
# HypO evaluation inherently uses normalized features from the head
parser.add_argument('--normalize', action='store_true', default=True, # Set default True for HypO eval
                        help='normalize feat embeddings (should be True for HypO)')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# Removed target_domain and cortype as they are not relevant for Camelyon17 OOD evaluation here

parser.set_defaults(bottleneck=True)
# Augment should likely be False for evaluation
parser.set_defaults(augment=False)

args = parser.parse_args()
torch.cuda.set_device(args.gpu)

state = {k: v for k, v in args._get_kwargs()}

date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Extract a base name from the checkpoint path for the log directory
ckpt_filename = os.path.basename(args.ckpt_path)
ckpt_dirname = os.path.basename(os.path.dirname(args.ckpt_path))
# Sanitize dirname and filename for log path
safe_dirname = "".join([c if c.isalnum() else "_" for c in ckpt_dirname])
safe_filename = "".join([c if c.isalnum() else "_" for c in ckpt_filename.replace('.pth.tar', '')])

log_base_name = f"{date_time}_hypo_eval_{safe_dirname}_{safe_filename}" # Added hypo_eval prefix
args.log_directory = f"logs/eval/{args.in_dataset}/{log_base_name}/"

if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory, exist_ok=True) # Added exist_ok=True


#init log
log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s : %(message)s')
fileHandler = logging.FileHandler(os.path.join(args.log_directory, "eval_hypo_camelyon_info.log"), mode='w') # Updated log filename
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.setLevel(logging.INFO) # Set level to INFO for eval
log.addHandler(fileHandler)
log.addHandler(streamHandler)

log.info(f"--- Camelyon17 HypO Evaluation Arguments ---\n{pprint.pformat(state)}") # Log args to INFO level
log.info(f"Evaluating checkpoint: {args.ckpt_path}")
log.info(f"Log directory: {args.log_directory}")


if args.in_dataset == "camelyon17": # Added camelyon17 class count
    args.n_cls = 2
else:
    # This script is specific to camelyon17 now
    raise ValueError(f"This script only supports --in-dataset camelyon17, got: {args.in_dataset}")


#set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


def to_np(x): return x.data.cpu().numpy()

# --- FPR@95TPR Calculation Helper ---
# From https://github.com/hendrycks/outlier-exposure/blob/master/utils/metrics.py
# and https://github.com/facebookresearch/odin/blob/master/code/calData.py
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # slice the values according to the recall level
    # stop_index = np.max(np.where(y_true)) # Get the last occurrence of the positive class - this is not needed
    threshold_idx = np.argmin(np.abs(np.cumsum(y_true)/y_true.sum() - recall_level))

    tpr = np.cumsum(y_true)[threshold_idx] / y_true.sum()
    fpr = np.sum(np.logical_not(y_true[:threshold_idx+1])) / np.sum(np.logical_not(y_true))

    return fpr


# --- Dataloader Setup ---
if args.in_dataset == 'camelyon17':
    # Use the Camelyon17 dataloader function
    # We need val (ID) and test (OOD) splits for evaluation.
    _, val_loader, test_loader = get_camelyon17_dataloaders(
        root_dir=args.wilds_root_dir,
        batch_size=args.batch_size,
        num_workers=args.prefetch
        # eval_mode=True # Removed incorrect argument
    )
    if val_loader is None or test_loader is None:
         log.error(f"Failed to load Camelyon17 dataset from {args.wilds_root_dir}. Exiting.")
         exit() # Exit if dataloaders failed
else:
    # Should not happen due to argument choices constraint
    log.error(f"Dataset {args.in_dataset} not supported by this script.")
    exit()


log.info(f"ID Val dataset size: {len(val_loader.dataset)}")
log.info(f"OOD Test dataset size: {len(test_loader.dataset)}")


def main():

    model = set_model(args)

    # Load checkpoint using the full path
    log.info(f"Loading model and criterion state from: {args.ckpt_path}")
    if not os.path.isfile(args.ckpt_path):
        log.error(f"Checkpoint file not found: {args.ckpt_path}")
        exit()
    checkpoint = torch.load(args.ckpt_path, map_location='cpu') # Load to CPU first

    # Load model state dict
    # Handle potential DataParallel wrapping
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        log.info("Removing 'module.' prefix from state_dict keys")
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.cuda() # Move model to GPU

    # Initialize and load DisLoss criterion state dict
    # Use val_loader for initialization during evaluation
    criterion_dis = DisLoss(args, model, val_loader, temperature=args.temp).cuda()
    if 'dis_state_dict' in checkpoint:
        criterion_dis.load_state_dict(checkpoint['dis_state_dict'])
        log.info("Loaded DisLoss criterion state (prototypes) from checkpoint.")
    else:
        log.error("DisLoss criterion state ('dis_state_dict') not found in checkpoint. Cannot evaluate HypO scores. Exiting.")
        exit()


    model.eval()
    criterion_dis.eval() # Ensure criterion is also in eval mode

    log.info("--- Starting Evaluation ---")

    id_scores = []
    id_targets = []
    id_preds = []
    ood_scores = []
    # ood_targets = [] # OOD targets are not used for OOD detection metrics but good for sanity check
    ood_preds = [] # Added for OOD accuracy
    ood_targets_for_acc = [] # Added for OOD accuracy

    # --- In-Distribution (ID Val) Evaluation ---
    log.info("Processing In-Distribution (Validation) Data...")
    with torch.no_grad():
        for i, (data, target, metadata) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()

            # Get HypO features (normalized output of the head)
            features = model.forward(data) # Assumes model.forward() gives normalized features

            # Calculate similarity to prototypes
            feat_dot_prototype = torch.div(torch.matmul(features, criterion_dis.prototypes.T), args.temp)

            # HypO OOD score: max similarity score
            hypo_score, pred = torch.max(feat_dot_prototype, dim=1)

            id_scores.extend(to_np(hypo_score))
            id_targets.extend(to_np(target))
            id_preds.extend(to_np(pred))

            if (i + 1) % args.print_freq == 0:
                log.info(f"ID Batch [{i+1}/{len(val_loader)}]")

    id_accuracy = accuracy_score(id_targets, id_preds)
    log.info(f"ID Validation Accuracy: {id_accuracy:.4f}")

    # --- Out-of-Distribution (OOD Test) Evaluation ---
    log.info("Processing Out-of-Distribution (Test) Data...")
    with torch.no_grad():
        for i, (data, target, metadata) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            # Get HypO features
            features = model.forward(data)

            # Calculate similarity to prototypes
            feat_dot_prototype = torch.div(torch.matmul(features, criterion_dis.prototypes.T), args.temp)

            # HypO OOD score: max similarity score
            hypo_score, pred = torch.max(feat_dot_prototype, dim=1) # Get prediction as well

            ood_scores.extend(to_np(hypo_score))
            # ood_targets.extend(to_np(target)) # Store OOD targets if needed later
            ood_preds.extend(to_np(pred)) # Added
            ood_targets_for_acc.extend(to_np(target)) # Added

            if (i + 1) % args.print_freq == 0:
                log.info(f"OOD Batch [{i+1}/{len(test_loader)}]")


    # --- Calculate OOD Metrics ---
    log.info("--- Calculating OOD Metrics ---")

    # Calculate OOD Generalization Accuracy
    ood_accuracy = accuracy_score(ood_targets_for_acc, ood_preds) # Added
    log.info(f"OOD Generalization Accuracy (Test Set): {ood_accuracy:.4f}") # Added

    scores = np.concatenate((id_scores, ood_scores))
    # Labels: 1 for ID (positive class), 0 for OOD (negative class)
    labels = np.concatenate((np.ones_like(id_scores), np.zeros_like(ood_scores)))

    # AUROC
    auroc = roc_auc_score(labels, scores)
    log.info(f"AUROC: {auroc:.4f}")

    # AUPR (In vs Out) - PR curve where ID is positive class
    aupr_in = average_precision_score(labels, scores, pos_label=1)
    log.info(f"AUPR (In): {aupr_in:.4f}")

    # AUPR (Out vs In) - PR curve where OOD is positive class
    aupr_out = average_precision_score(1 - labels, -scores, pos_label=1) # Invert labels and scores
    log.info(f"AUPR (Out): {aupr_out:.4f}")

    # FPR@95TPR
    fpr95 = fpr_and_fdr_at_recall(labels, scores, recall_level=0.95, pos_label=1)
    log.info(f"FPR@95TPR: {fpr95:.4f}")

    # --- Save results ---
    results_path = os.path.join(args.log_directory, "ood_metrics_hypo.txt") # Updated results filename
    with open(results_path, 'w') as f:
        f.write(f"Checkpoint: {args.ckpt_path}\n")
        f.write(f"ID Validation Accuracy: {id_accuracy:.6f}\n")
        f.write(f"OOD Generalization Accuracy: {ood_accuracy:.6f}\n") # Added line
        f.write(f"AUROC: {auroc:.6f}\n")
        f.write(f"AUPR (In): {aupr_in:.6f}\n")
        f.write(f"AUPR (Out): {aupr_out:.6f}\n")
        f.write(f"FPR@95TPR: {fpr95:.6f}\n")
    log.info(f"OOD metrics saved to: {results_path}")

    log.info("--- Evaluation Finished ---")


if __name__ == '__main__':
    main()
