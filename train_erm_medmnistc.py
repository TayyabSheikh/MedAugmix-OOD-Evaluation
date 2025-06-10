import argparse
import math
import os
import time
from datetime import datetime
import logging
import tensorboard_logger as tb_logger
import pprint

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import numpy as np


import wandb
wandb.login()

from sklearn.metrics import accuracy_score

from utils import (CompLoss, CompNGLoss, DisLoss, DisLPLoss,
                AverageMeter, adjust_learning_rate, warmup_learning_rate,
                set_loader_small, set_loader_ImageNet, set_model)
# Import the new dataloader for MedMNIST-C in AugMix
from dataloader.camelyon17_medmnistc_in_augmix import get_camelyon17_medmnistc_in_augmix_dataloaders

# Keep standard loader import for other datasets if needed by set_loader_small
from dataloader.camelyon17_wilds import get_camelyon17_dataloaders # This might be redundant if only Camelyon17 is used by this script

# Updated description for ERM
parser = argparse.ArgumentParser(description='Script for training with ERM using MedMNIST-C augmentations (and optionally AugMix) for Camelyon17')
parser.add_argument('--gpu', default=6, type=int, help='which GPU to use')
parser.add_argument('--seed', default=4, type=int, help='random seed')  # original 4
parser.add_argument('--w', default=2, type=float,
                    help='loss scale')
parser.add_argument('--proto_m', default= 0.95, type=float,
                   help='weight of prototype update')
parser.add_argument('--feat_dim', default = 128, type=int,
                    help='feature dim')
# Added 'camelyon17' to choices
parser.add_argument('--in-dataset', default="camelyon17", type=str, help='in-distribution dataset', choices=['PACS', 'VLCS', 'CIFAR-10', 'CIFAR-100', 'ImageNet-100', 'OfficeHome', 'terra_incognita', 'camelyon17'])
parser.add_argument('--id_loc', default="datasets/CIFAR10", type=str, help='location of in-distribution dataset (used for non-WILDS datasets)')
# Added argument for WILDS data root
parser.add_argument('--wilds_root_dir', default="./data", type=str, help='Root directory for WILDS datasets.')
parser.add_argument('--model', default='resnet18', type=str, help='model architecture: [resnet18, wrt40, wrt28, wrt16, densenet100, resnet50, resnet34]')
parser.add_argument('--head', default='mlp', type=str, help='either mlp or linear head')
# Changed default loss to 'erm'
parser.add_argument('--loss', default = 'erm', type=str, choices = ['hypo', 'erm'],
                    help='name of experiment (should be erm for this script)')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
parser.add_argument('--save-epoch', default=100, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default= 32, type=int, #512 # batch-size
                    help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', default=5e-4, type=float,
                    help='initial learning rate')
# if linear lr schedule
parser.add_argument('--lr_decay_epochs', type=str, default='100,150,180',
                        help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
# if cosine lr schedule
parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
parser.add_argument('--normalize', action='store_true',
                        help='normalize feat embeddings')
# Renamed AugMix flag for clarity and added severity/width
parser.add_argument('--use_med_augmix', action='store_true',
                        help='Apply AugMix with MedMNIST-C operations to the training data')
parser.add_argument('--augmix_severity', type=int, default=3,
                        help='Severity for AugMix operations (1-10)')
parser.add_argument('--augmix_mixture_width', type=int, default=3,
                        help='Mixture width for AugMix')

# add pacs specific
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--target_domain', type=str, default='cartoon')

# debug
parser.add_argument('--use_domain', type=bool, default=False, help='whether to use in-domain negative pairs in compactness loss')
parser.add_argument('--mode', default='online', choices = ['online','disabled'], help='whether disable wandb logging')
parser.add_argument('--model_pretrained', default=True, type=lambda x: (str(x).lower() == 'true'), help='Load pretrained model weights (default: True)')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()
torch.cuda.set_device(args.gpu)
state = {k: v for k, v in args._get_kwargs()}

date_time = datetime.now().strftime("%d_%m_%H:%M")

#processing str to list for linear lr scheduling
args.lr_decay_epochs = [int(step) for step in args.lr_decay_epochs.split(',')]


# Construct base name parts
base_name_parts = [
    date_time, args.loss, args.model,
    f"lr_{args.learning_rate}", f"cosine_{args.cosine}",
    f"bsz_{args.batch_size}", f"head_{args.head}", f"wd_{args.w}",
    f"{args.epochs}", f"{args.feat_dim}", f"trial_{args.trial}",
    f"temp_{args.temp}", args.in_dataset, f"pm_{args.proto_m}"
]

# Add target domain if relevant (for non-ImageNet/CIFAR/Camelyon)
if args.in_dataset not in ["ImageNet-100", 'CIFAR-10', 'CIFAR-100', 'camelyon17']:
    base_name_parts.insert(7, f"td_{args.target_domain}") # Insert after bsz

# Add MedMNIST-C suffix if dataset is Camelyon17 (this script always uses it)
if args.in_dataset == 'camelyon17':
    base_name_parts.append("medmnistc")

args.name = "_".join(base_name_parts)


# Adjusted directory logic slightly for clarity (using f-strings)
args.log_directory = f"logs/{args.in_dataset}/{args.name}/"
args.model_directory = f"checkpoints/{args.in_dataset}/{args.name}/" # Using relative path


args.tb_path = f'./save/hypo/{args.in_dataset}_tensorboard'
if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory, exist_ok=True) # Added exist_ok=True
if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)
args.tb_folder = os.path.join(args.tb_path, args.name)
if not os.path.isdir(args.tb_folder):
    os.makedirs(args.tb_folder)

#save args
with open(os.path.join(args.log_directory, 'train_args.txt'), 'w') as f:
    f.write(pprint.pformat(state))

# --- Setup Logging ---
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG) # Set root logger level

# Formatter
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
summary_formatter = logging.Formatter('%(asctime)s : %(message)s') # Simpler for summary

# 1. Detailed Log Handler (File: train_info.log)
detail_log_path = os.path.join(args.log_directory, "train_info.log")
fileHandler = logging.FileHandler(detail_log_path, mode='w')
fileHandler.setFormatter(formatter)
fileHandler.setLevel(logging.DEBUG) # Capture everything
log.addHandler(fileHandler)

# 2. Console Handler (Stream)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.DEBUG) # Show debug messages (including per-batch) on console
log.addHandler(streamHandler)

# 3. Epoch Summary Log Handler (File: <run_name>_epoch_summary.log)
summary_log_dir = f"epoch_summary_logs/{args.in_dataset}/"
os.makedirs(summary_log_dir, exist_ok=True)
summary_log_path = os.path.join(summary_log_dir, f"{args.name}_epoch_summary.log")
summaryFileHandler = logging.FileHandler(summary_log_path, mode='w')
summaryFileHandler.setFormatter(summary_formatter)
summaryFileHandler.setLevel(logging.INFO) # Only capture INFO level messages for summary
log.addHandler(summaryFileHandler)
# --- End Logging Setup ---

log.debug(f"Detailed log: {detail_log_path}")
log.debug(f"Epoch summary log: {summary_log_path}")
log.info(f"--- Training Arguments (MedMNIST-C Augmented) ---\n{pprint.pformat(state)}") # Log args to INFO level for summary file

if args.in_dataset == "CIFAR-10":
    args.n_cls = 10
elif args.in_dataset == "PACS":
    args.n_cls = 7
elif args.in_dataset == "VLCS":
    args.n_cls = 5
elif args.in_dataset == "OfficeHome":
    args.n_cls = 65
elif args.in_dataset == 'terra_incognita':
    args.n_cls = 10
elif args.in_dataset in ["CIFAR-100", "ImageNet-100"]:
    args.n_cls = 100
elif args.in_dataset == "camelyon17": # Added camelyon17 class count
    args.n_cls = 2

#set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
log.debug(f"{args.name}")

# warm-up for large-batch training
if args.batch_size > 256:
    args.warm = True
if args.warm:
    args.warmup_from = 0.001
    args.warm_epochs = 10
    if args.cosine:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
    else:
        args.warmup_to = args.learning_rate

def to_np(x): return x.data.cpu().numpy()

def main():
    tb_log = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # Adjust wandb project name for ERM runs
    wandb_project_name = "erm-camelyon17-medmnistc" if args.in_dataset == 'camelyon17' else "erm"
    if args.use_med_augmix: # Corrected to use_med_augmix
        wandb_project_name += "-augmix"

    wandb.init(
        # Set the project where this run will be logged
        project=wandb_project_name,
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=args.name,
        # Track hyperparameters and run metadata
        mode=args.mode,
        config=args)

    # Modified DataLoader selection
    if args.in_dataset == "ImageNet-100":
        train_loader, val_loader, test_loader = set_loader_ImageNet(args)
    elif args.in_dataset == 'camelyon17':
        # Use the new MedMNIST-C-in-AugMix dataloader function
        log.info("Using MedMNIST-C-in-AugMix dataloader for Camelyon17 training.")
        train_loader, val_loader, test_loader = get_camelyon17_medmnistc_in_augmix_dataloaders(
            root_dir=args.wilds_root_dir,
            batch_size=args.batch_size,
            num_workers=args.prefetch,
            corruption_dataset_name="bloodmnist", # Hardcoded to bloodmnist
            use_med_augmix=args.use_med_augmix, # Pass the new flag
            augmix_severity=args.augmix_severity,
            augmix_mixture_width=args.augmix_mixture_width
        )
        if train_loader is None:
             log.error(f"Failed to load MedMNIST-C-in-AugMix augmented Camelyon17 dataset from {args.wilds_root_dir}. Exiting.")
             return # Exit if dataloaders failed
    elif args.in_dataset in ['CIFAR-10', 'CIFAR-100', 'PACS', 'VLCS', 'OfficeHome', 'terra_incognita']:
        # Fallback to original loaders for other datasets
        train_loader, val_loader, test_loader = set_loader_small(args)
    else:
        log.error(f"Dataset {args.in_dataset} not supported yet for loader selection.")
        return # Exit for unsupported dataset


    model = set_model(args)

    # Define loss functions - For ERM, only CE is strictly needed for training loop
    criterion_ce = torch.nn.CrossEntropyLoss().cuda() # Standard Cross-Entropy for ERM
    criterion_comp = None # Not used in ERM training loop
    criterion_dis = None # Not used in ERM training loop

    # Add a check to ensure the script is run with --loss erm
    if args.loss != 'erm':
        log.warning(f"Warning: This script is intended for ERM training, but --loss is set to {args.loss}. Proceeding with ERM loss calculation.")
        args.loss = 'erm' # Force loss type to ERM for consistency within this script


    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    acc_best = 0.0
    acc_best_id = 0.0
    acc_test_best = 0.0
    acc_test_best_id = 0.0
    epoch_test_best = 0.0
    epoch_val_best = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        # train for one epoch
        # Pass criterion_ce to the training function
        train_sloss, train_uloss, train_dloss, acc, acc_cor= train_hypo(args, train_loader, val_loader, test_loader, model, criterion_comp, criterion_dis, criterion_ce, optimizer, epoch, log)

        # Log appropriate loss based on mode
        if args.loss == 'hypo':
            tb_log.log_value('train_comp_loss', train_uloss, epoch) # Use train_uloss for comp loss
            tb_log.log_value('train_dis_loss', train_dloss, epoch) # Use train_dloss for dis loss
            wandb.log({'Comp Loss Ep': train_uloss,'Dis Loss Ep': train_dloss })
        elif args.loss == 'erm':
            tb_log.log_value('train_ce_loss', train_sloss, epoch) # Use train_sloss for CE loss (as returned by train_hypo)
            wandb.log({'CE Loss Ep': train_sloss})

        # wandb.log({'Comp Loss Ep': train_uloss,'Dis Loss Ep': train_dloss }) # Duplicated log? Removed.
        tb_log.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        wandb.log({'current lr': optimizer.param_groups[0]['lr'], 'acc':acc, 'acc cor': acc_cor})

        # Log epoch summary results to the specific summary log file
        log.info(f"Epoch: {epoch}, ID Val Acc: {acc:.6f}, OOD Test Acc: {acc_cor:.6f}")

        # save checkpoint
        if acc >= acc_best_id:
            acc_best = acc_cor
            acc_best_id = acc
            epoch_val_best = epoch
            wandb.log({'val best ood acc': acc_best, 'val best id acc':acc_best_id})
            print('best accuracy {} at epoch {}'.format(acc_best_id, epoch))
            print('accuracy cor {} at epoch {}'.format(acc_best, epoch))
            # --- Prepare Checkpoint Data ---
            save_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
            }
            # Only save HypO specific states if training with HypO loss
            if args.loss == 'hypo' and criterion_dis is not None and criterion_comp is not None:
                save_dict['dis_state_dict'] = criterion_dis.state_dict()
                save_dict['uni_state_dict'] = criterion_comp.state_dict()
            # --- Save Checkpoint ---
            save_checkpoint(args, save_dict, epoch + 1, save_best=True)

        # Note: acc_cor (OOD accuracy) might not be the best metric to track for ERM best model saving.
        # The current logic saves based on 'acc' (ID validation accuracy), which is appropriate for ERM.
        if acc_cor >= acc_test_best:
            acc_test_best = acc_cor
            acc_test_best_id = acc
            epoch_test_best = epoch
            wandb.log({'test best ood acc': acc_test_best, 'test best id acc':acc_test_best_id})
            print('best test accuracy {} at epoch {}'.format(acc_test_best, epoch))

    print('total val best ood accuracy {} id accuracy {} at epoch {}'.format(acc_best, acc_best_id, epoch_val_best))
    print('total test best ood accuracy {} id accuracy {} at epoch {}'.format(acc_test_best, acc_test_best_id, epoch_test_best))
    print('last epoch ood accuracy {} id accuracy {} at epoch {}'.format(acc_cor, acc, epoch))

    # --- Log Final Summary ---
    log.info("--- Training Summary ---")
    log.info(f"Best ID Val Acc: {acc_best_id:.6f} (Epoch {epoch_val_best})")
    log.info(f"OOD Test Acc at Best ID Val Epoch: {acc_best:.6f}")
    # Note: 'test best' tracks the epoch with the highest OOD test accuracy, which might differ from the best ID val epoch.
    log.info(f"Best OOD Test Acc: {acc_test_best:.6f} (Epoch {epoch_test_best})")
    log.info(f"ID Val Acc at Best OOD Test Epoch: {acc_test_best_id:.6f}")
    log.info(f"Final Epoch ({epoch}) ID Val Acc: {acc:.6f}")
    log.info(f"Final Epoch ({epoch}) OOD Test Acc: {acc_cor:.6f}")
    # --- End Log Final Summary ---


    summary_metrics = {
        'val best ood accuracy': acc_best,
        'val best id accuracy': acc_best_id,
        'val best epoch': epoch_val_best,
        'test best ood accuracy': acc_test_best,
        'test best id accuracy': acc_test_best_id,
        'test best epoch': epoch_test_best,
        'last ood accuracy': acc_cor,
        'last id accuracy': acc,
        'last epoch': epoch
    }

    for metric_name, metric_value in summary_metrics.items():
        wandb.summary[metric_name] = metric_value

# Added criterion_ce to function arguments
def train_hypo(args, train_loader, val_loader, test_loader, model, criterion_comp, criterion_dis, criterion_ce, optimizer, epoch, log):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    supcon_losses = AverageMeter()
    comp_losses = AverageMeter()
    dis_losses = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    for i, values in enumerate(train_loader):
        # Adjusted data unpacking for WILDS (returns x, y, metadata)
        if args.in_dataset == 'camelyon17':
             input, target, metadata = values # Unpack WILDS tuple
             # Access domain ID using tensor indexing (trying index 0 based on error size 4)
             # Domain info might not be strictly needed if not used in loss, but kept for consistency
             try:
                 domain = metadata[:, 0]
             except IndexError:
                 domain = None # Handle cases where metadata might be different
        elif len(values) == 3: # Original handling for other datasets like PACS
            input, target, domain = values
        elif len(values) == 2: # Original handling for datasets like CIFAR
            input, target = values
            domain = None
        else:
            log.warning(f"Unexpected data format from loader: {len(values)} items.")
            continue # Skip batch if format is unknown

        warmup_learning_rate(args, epoch, i, len(train_loader), optimizer)
        bsz = target.shape[0]

        input = input.cuda()
        target = target.cuda()

        penultimate = model.encoder(input).squeeze()

        if args.normalize: # default: False - Normalize penultimate features if requested
            penultimate= F.normalize(penultimate, dim=1)

        # --- Loss Calculation ---
        if args.loss == 'hypo':
            features = model.head(penultimate)
            features = F.normalize(features, dim=1) # HypO uses normalized features from head
            # Ensure criterion_dis and criterion_comp are initialized
            if criterion_dis is None or criterion_comp is None:
                 raise ValueError("HypO criteria not initialized for hypo loss.")
            dis_loss = criterion_dis(features, target)
            comp_loss = criterion_comp(features, criterion_dis.prototypes, target, None) # Domain is None here, adjust if needed
            loss = args.w * comp_loss + dis_loss
            # Update HypO loss meters
            dis_losses.update(dis_loss.item(), bsz) # Use item() and bsz
            comp_losses.update(comp_loss.item(), bsz) # Use item() and bsz
        elif args.loss == 'erm':
            # For ERM, assume model.head provides raw logits suitable for CrossEntropy
            logits = model.head(penultimate) # Get raw logits from head
            loss = criterion_ce(logits, target)
            # Update standard loss meter
            losses.update(loss.item(), bsz) # Use standard loss meter
            # Set HypO losses to 0 for consistent logging/return values if needed elsewhere
            dis_loss = torch.tensor(0.0)
            comp_loss = torch.tensor(0.0)
            dis_losses.update(dis_loss.item(), bsz)
            comp_losses.update(comp_loss.item(), bsz)
        else:
             raise ValueError(f"Unsupported loss type: {args.loss}")

        # Common backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Correctly indented block after optimizer step
        batch_time.update(time.time() - end)
        end = time.time()
        # Restore per-batch console logging (also goes to file via logger setup)
        if i % args.print_freq == 0:
             # Log relevant loss based on mode
             if args.loss == 'hypo':
                 log.debug('Epoch: [{0}][{1}/{2}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Dis Loss {dloss.val:.4f} ({dloss.avg:.4f})\t'
                     'Comp Loss {uloss.val:.4f} ({uloss.avg:.4f})\t'.format(
                         epoch, i, len(train_loader), batch_time=batch_time, dloss=dis_losses, uloss = comp_losses))
             elif args.loss == 'erm':
                 log.debug('Epoch: [{0}][{1}/{2}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'CE Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                         epoch, i, len(train_loader), batch_time=batch_time, loss=losses))


         # Keep wandb logging per step if desired (or move to end of epoch)
        if i % args.print_freq == 0: # Log wandb less frequently too if needed
            if args.loss == 'hypo':
                wandb.log({'Dis Loss' : dis_losses.val, 'Comp Loss' :  comp_losses.val})
            elif args.loss == 'erm':
                wandb.log({'CE Loss' : losses.val})


    model.eval()
    with torch.no_grad():
        accuracies = []
        for i, values in enumerate(val_loader):
            # Adjusted data unpacking for WILDS in validation loop
            if args.in_dataset == 'camelyon17':
                 input, target, metadata = values
                 # Domain info might not be needed for eval
                 # domain = metadata[:, 0]
            elif len(values) == 3:
                input, target, domain = values
            elif len(values) == 2:
                input, target = values
                domain = None
            else:
                log.warning(f"Unexpected data format from val_loader: {len(values)} items.")
                continue
            # Correctly indented processing steps (outside the if/elif/else)
            input = input.cuda()
            target = target.cuda()

            # --- Get Model Output & Prediction for Evaluation ---
            if args.loss == 'hypo':
                 # HypO evaluation uses normalized features and dot product with prototypes
                 if criterion_dis is None:
                      raise ValueError("criterion_dis is None during HypO evaluation")
                 features = model.forward(input) # Assumes model.forward gives normalized features for HypO
                 feat_dot_prototype = torch.div(torch.matmul(features, criterion_dis.prototypes.T), args.temp)
                 # for numerical stability
                 logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
                 logits = feat_dot_prototype - logits_max.detach() # Logits derived from HypO prototypes
                 pred = logits.data.max(1)[1]
            elif args.loss == 'erm':
                 # ERM evaluation uses raw logits from the head
                 penultimate = model.encoder(input).squeeze()
                 if args.normalize: # Apply normalization if specified, even for ERM features before head
                     penultimate= F.normalize(penultimate, dim=1)
                 logits = model.head(penultimate) # Assumes model.head gives raw logits
                 pred = logits.data.max(1)[1]
            else:
                 raise ValueError(f"Unsupported loss type for evaluation: {args.loss}")

            accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

    acc = sum(accuracies) / len(accuracies) if len(accuracies) > 0 else 0.0 # Handle empty loader case

    if test_loader is None:
        acc_cor = 0.
    else:
        with torch.no_grad():
            accuracies_cor = []
            for i, values in enumerate(test_loader):
                # Adjusted data unpacking for WILDS in test loop
                if args.in_dataset == 'camelyon17':
                     input, target, metadata = values
                     # Domain info might not be needed for eval
                     # domain = metadata[:, 0]
                elif len(values) == 3:
                    input, target, domain = values
                elif len(values) == 2:
                    input, target = values
                    domain = None
                else:
                    log.warning(f"Unexpected data format from test_loader: {len(values)} items.")
                    continue
                # Correctly indented processing steps (outside the if/elif/else)
                input = input.cuda()
                target = target.cuda()

                # --- Get Model Output & Prediction for Evaluation (Test Loop) ---
                if args.loss == 'hypo':
                     if criterion_dis is None:
                          raise ValueError("criterion_dis is None during HypO evaluation")
                     features = model.forward(input)
                     feat_dot_prototype = torch.div(torch.matmul(features, criterion_dis.prototypes.T), args.temp)
                     logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
                     logits = feat_dot_prototype - logits_max.detach()
                     pred = logits.data.max(1)[1]
                elif args.loss == 'erm':
                     penultimate = model.encoder(input).squeeze()
                     if args.normalize:
                         penultimate= F.normalize(penultimate, dim=1)
                     logits = model.head(penultimate)
                     pred = logits.data.max(1)[1]
                else:
                     raise ValueError(f"Unsupported loss type for evaluation: {args.loss}")

                accuracies_cor.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

        acc_cor = sum(accuracies_cor) / len(accuracies_cor) if len(accuracies_cor) > 0 else 0.0 # Handle empty loader case

    # measure elapsed time
    # Return appropriate losses based on mode
    if args.loss == 'hypo':
        return supcon_losses.avg, comp_losses.avg, dis_losses.avg, acc, acc_cor
    elif args.loss == 'erm':
        # Return overall loss average for ERM, placeholders for others
        return losses.avg, 0.0, 0.0, acc, acc_cor # Now acc/acc_cor are calculated correctly for ERM too
    else:
        # Should not happen due to earlier check
        return 0.0, 0.0, 0.0, acc, acc_cor



def save_checkpoint(args, state, epoch, save_best = False):
    """Saves checkpoint to disk"""
    if save_best:
        filename = os.path.join(args.model_directory, 'checkpoint_max.pth.tar') # Use os.path.join
    else:
        filename = os.path.join(args.model_directory, f'checkpoint_{epoch}.pth.tar') # Use os.path.join
    torch.save(state, filename)


if __name__ == '__main__':
    main()
