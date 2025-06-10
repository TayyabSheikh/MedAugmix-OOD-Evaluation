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

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from utils import (CompLoss, CompNGLoss, DisLoss, DisLPLoss,
                AverageMeter, adjust_learning_rate, warmup_learning_rate,
                set_loader_small, set_loader_ImageNet, set_model)
from dataloader.fitzpatrick17k_dataloaders_utils import get_fitzpatrick17k_dataloaders

parser = argparse.ArgumentParser(description='Script for training with HYPO on Fitzpatrick17k')
parser.add_argument('--gpu', default=6, type=int, help='which GPU to use')
parser.add_argument('--seed', default=0, type=int, help='random seed') 
parser.add_argument('--w', default=2, type=float,
                    help='loss scale')
parser.add_argument('--proto_m', default= 0.95, type=float,
                   help='weight of prototype update')
parser.add_argument('--feat_dim', default = 128, type=int,
                    help='feature dim')
parser.add_argument('--in-dataset', default="fitzpatrick17k", type=str, help='In-distribution dataset (should be fitzpatrick17k for this script)', choices=['fitzpatrick17k'])
parser.add_argument('--wilds_root_dir', default="./data/finalfitz17k", type=str, help='Root directory for the Fitzpatrick17k dataset.')
parser.add_argument('--model', default='resnet50', type=str, help='model architecture: [resnet18, resnet50, densenet121]')
parser.add_argument('--head', default='mlp', type=str, help='either mlp or linear head')
parser.add_argument('--loss', default = 'erm', type=str, choices = ['hypo', 'erm'],
                    help='name of experiment')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
parser.add_argument('--save-epoch', default=100, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default= 32, type=int, 
                    help='mini-batch size (default: 32)')
parser.add_argument('--learning_rate', default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_decay_epochs', type=str, default='30,40',
                        help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
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
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--target_domain', type=str, default='56', help='Target domain FST group for OOD test (e.g., 12, 34, 56)')
parser.add_argument('--use_domain', type=bool, default=False, help='whether to use in-domain negative pairs in compactness loss')
parser.add_argument('--mode', default='online', choices = ['online','disabled'], help='whether disable wandb logging')
parser.add_argument('--model_pretrained', default=True, type=lambda x: (str(x).lower() == 'true'), help='Load pretrained model weights (default: True)')
parser.add_argument('--label_partition', type=int, default=3, choices=[3, 9, 114], help='Label partition for Fitzpatrick17k (3, 9, or 114 classes)')

# MedAugMix (custom) specific arguments
parser.add_argument('--use_med_augmix', action='store_true', help='Apply custom MedMNIST-C based AugMix operations')
parser.add_argument('--augmix_corruption_dataset', type=str, default='dermamnist', help='MedMNIST dataset for MedMNISTCAugMix corruptions')
parser.add_argument('--augmix_severity', type=int, default=3, help='Severity for MedMNISTCAugMix operations (1-5)')
parser.add_argument('--augmix_mixture_width', type=int, default=3, help='Mixture width for MedMNISTCAugMix')

# Torchvision AugMix (Plain Augmix) specific arguments
parser.add_argument('--use_torchvision_augmix', action='store_true', help='Apply torchvision.transforms.AugMix.')
parser.add_argument('--tv_augmix_severity', type=int, default=3, help='Severity for torchvision AugMix.')
parser.add_argument('--tv_augmix_mixture_width', type=int, default=3, help='Mixture width for torchvision AugMix.')
parser.add_argument('--tv_augmix_alpha', type=float, default=1.0, help='Alpha for Dirichlet distribution in torchvision AugMix.')

# Plain MedMNISTC (single random corruption from a collection with random severity) specific arguments
parser.add_argument('--use_plain_medmnistc', action='store_true', help='Apply a single random MedMNIST-C corruption from a specified collection with random severity.')
parser.add_argument('--plain_medmnistc_collection_source', type=str, default='dermamnist', help='MedMNIST collection for plain MedMNISTC (e.g., dermamnist).')
# The specific corruption name and severity are now handled randomly inside the dataloader if use_plain_medmnistc is True.
# So, --plain_medmnistc_corruption_name and --plain_medmnistc_severity args are removed from parser.

parser.add_argument('--augment', action='store_true', help='Enable basic augmentations (RandomResizedCrop, Flip, ColorJitter) if no other AugMix/MedMNISTC is used. Default is False.')
parser.set_defaults(bottleneck=True, augment=False) 

args = parser.parse_args()
torch.cuda.set_device(args.gpu) 
state = {k: v for k, v in args._get_kwargs()}

date_time = datetime.now().strftime("%y_%m_%d_%H%M") 

args.lr_decay_epochs = [int(step) for step in args.lr_decay_epochs.split(',')]

name_parts = [
    date_time, args.loss, args.model,
    f"lr{args.learning_rate}", f"cos{args.cosine}",
    f"bsz{args.batch_size}", args.head, f"w{args.w}",
    f"ep{args.epochs}", f"fd{args.feat_dim}", args.trial,
    f"t{args.temp}", args.in_dataset, f"lp{args.label_partition}"
]
if args.target_domain: 
    name_parts.append(f"ood{args.target_domain}")

if args.use_plain_medmnistc:
    name_parts.append("plainmedc")
    name_parts.append(f"col_{args.plain_medmnistc_collection_source}") # Indicate collection source
    name_parts.append("rand_sev") # Indicate random severity
elif args.use_torchvision_augmix:
    name_parts.append("tvaugmix")
    name_parts.append(f"sev{args.tv_augmix_severity}")
    name_parts.append(f"mw{args.tv_augmix_mixture_width}")
    name_parts.append(f"alpha{args.tv_augmix_alpha}")
elif args.use_med_augmix:
    name_parts.append("medaugmix")
    name_parts.append(f"sev{args.augmix_severity}")
    name_parts.append(f"mw{args.augmix_mixture_width}")
    name_parts.append(f"cds_{args.augmix_corruption_dataset}") 
elif args.augment: 
    name_parts.append("baseaug")

name_parts.append(f"pm{args.proto_m}")
if not args.model_pretrained:
    name_parts.append("scratch")
args.name = "_".join(name_parts)

args.log_directory = f"logs/{args.in_dataset}/{args.name}/"
args.model_directory = f"checkpoints/{args.in_dataset}/{args.name}/" 

args.tb_path = f'./save/hypo/{args.in_dataset}_tensorboard' 
if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory, exist_ok=True)
if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)
args.tb_folder = os.path.join(args.tb_path, args.name)
if not os.path.isdir(args.tb_folder):
    os.makedirs(args.tb_folder)

with open(os.path.join(args.log_directory, 'train_args.txt'), 'w') as f:
    f.write(pprint.pformat(state))

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG) 
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
summary_formatter = logging.Formatter('%(asctime)s : %(message)s')
detail_log_path = os.path.join(args.log_directory, "train_info.log")
fileHandler = logging.FileHandler(detail_log_path, mode='w')
fileHandler.setFormatter(formatter)
fileHandler.setLevel(logging.DEBUG) 
log.addHandler(fileHandler)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.INFO) 
log.addHandler(streamHandler)
summary_log_dir = f"epoch_summary_logs/{args.in_dataset}/"
os.makedirs(summary_log_dir, exist_ok=True)
summary_log_path = os.path.join(summary_log_dir, f"{args.name}_epoch_summary.log")
summaryFileHandler = logging.FileHandler(summary_log_path, mode='w')
summaryFileHandler.setFormatter(summary_formatter)
summaryFileHandler.setLevel(logging.INFO) 
log.addHandler(summaryFileHandler)

log.debug(f"Detailed log: {detail_log_path}")
log.debug(f"Epoch summary log: {summary_log_path}")
log.info(f"--- Training Arguments ---\n{pprint.pformat(state)}")

if args.in_dataset == "fitzpatrick17k":
    args.n_cls = args.label_partition
else:
    log.error(f"This script is configured for fitzpatrick17k, but in-dataset is {args.in_dataset}.")
    exit() 

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
log.debug(f"Run Name: {args.name}")

if args.batch_size > 256 and args.epochs > 20: 
    args.warm = True
if args.warm:
    args.warmup_from = 0.001 
    args.warm_epochs = min(10, args.epochs // 5) 
    if args.cosine:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
    else:
        args.warmup_to = args.learning_rate
        
def to_np(x): return x.data.cpu().numpy()

def main():
    tb_log = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    wandb_project_name = f"hypo-{args.in_dataset}"
    if args.loss == 'erm':
        wandb_project_name = f"erm-{args.in_dataset}"
    
    wandb.init(project=wandb_project_name, name=args.name, mode=args.mode, config=args)

    log.info(f"Loading Fitzpatrick17k from: {args.wilds_root_dir} with label_partition: {args.label_partition}, OOD Target Domain (FST Group for Test): {args.target_domain}")
    train_loader, val_loader, test_loader = get_fitzpatrick17k_dataloaders(
        root_dir=args.wilds_root_dir,
        batch_size=args.batch_size,
        num_workers=args.prefetch,
        label_partition=args.label_partition,
        target_domain_ood_test=args.target_domain, 
        augment_train=args.augment, 
        use_med_augmix=args.use_med_augmix,
        augmix_corruption_dataset=args.augmix_corruption_dataset,
        augmix_severity=args.augmix_severity,
        augmix_mixture_width=args.augmix_mixture_width,
        use_torchvision_augmix=args.use_torchvision_augmix,
        tv_augmix_severity=args.tv_augmix_severity,
        tv_augmix_mixture_width=args.tv_augmix_mixture_width,
        tv_augmix_alpha=args.tv_augmix_alpha,
        use_plain_medmnistc=args.use_plain_medmnistc,
        plain_medmnistc_collection_source=args.plain_medmnistc_collection_source
        # Removed plain_medmnistc_corruption_name and plain_medmnistc_severity from this call
    )
    log.info(f"Fitzpatrick17k DataLoaders: Train: {len(train_loader.dataset) if train_loader else 0}, "
             f"ID Val: {len(val_loader.dataset) if val_loader else 0}, "
             f"OOD Test (target_domain {args.target_domain}): {len(test_loader.dataset) if test_loader else 0}")
    
    if hasattr(train_loader.dataset, 'num_actual_classes') and args.n_cls != train_loader.dataset.num_actual_classes:
        log.warning(f"Overriding args.n_cls from {args.n_cls} to {train_loader.dataset.num_actual_classes} based on actual classes found in data.")
        args.n_cls = train_loader.dataset.num_actual_classes
        wandb.config.update({'n_cls': args.n_cls}, allow_val_change=True)

    model = set_model(args)
    criterion_ce = torch.nn.CrossEntropyLoss().cuda()
    criterion_comp = None
    criterion_dis = None

    if args.loss == 'hypo':
        criterion_comp = CompLoss(args, temperature=args.temp, use_domain = args.use_domain).cuda()
        criterion_dis = DisLoss(args, model, val_loader, temperature=args.temp).cuda() 
    elif args.loss != 'erm':
        raise ValueError(f"Unsupported loss type: {args.loss}")

    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate,
                                momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

    acc_best_id_val = 0.0 
    ood_acc_at_best_id_val = 0.0 
    epoch_best_id_val = 0
    bal_acc_best_id_val = 0.0 
    ood_bal_acc_at_best_id_val = 0.0 

    best_ood_test_acc = 0.0 
    id_val_acc_at_best_ood = 0.0 
    epoch_best_ood_test = 0
    best_ood_bal_acc = 0.0 
    id_val_bal_acc_at_best_ood = 0.0 
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        
        train_ce_loss, train_comp_loss, train_dis_loss, current_id_val_acc, current_ood_test_acc, current_id_val_bal_acc, current_ood_test_bal_acc = train_hypo_fitzpatrick(
            args, train_loader, val_loader, test_loader, model, 
            criterion_comp, criterion_dis, criterion_ce, optimizer, epoch, log
        )

        wandb_log_data = {'Epoch': epoch, 'Learning Rate': optimizer.param_groups[0]['lr'],
                          'ID Val Acc': current_id_val_acc, 'OOD Test Acc': current_ood_test_acc,
                          'ID Val Bal Acc': current_id_val_bal_acc, 'OOD Test Bal Acc': current_ood_test_bal_acc}
        if args.loss == 'hypo':
            tb_log.log_value('train_comp_loss', train_comp_loss, epoch)
            tb_log.log_value('train_dis_loss', train_dis_loss, epoch)
            wandb_log_data.update({'Train Comp Loss': train_comp_loss, 'Train Dis Loss': train_dis_loss})
        elif args.loss == 'erm':
            tb_log.log_value('train_ce_loss', train_ce_loss, epoch)
            wandb_log_data.update({'Train CE Loss': train_ce_loss})
        wandb.log(wandb_log_data)

        log.info(f"Epoch: {epoch}, ID Val Acc: {current_id_val_acc:.4f} (Bal: {current_id_val_bal_acc:.4f}), OOD Test Acc (FSG {args.target_domain}): {current_ood_test_acc:.4f} (Bal: {current_ood_test_bal_acc:.4f})")

        if current_id_val_acc >= acc_best_id_val:
            acc_best_id_val = current_id_val_acc
            ood_acc_at_best_id_val = current_ood_test_acc
            bal_acc_best_id_val = current_id_val_bal_acc 
            ood_bal_acc_at_best_id_val = current_ood_test_bal_acc
            epoch_best_id_val = epoch
            
            wandb.log({'Best ID Val Acc': acc_best_id_val, 
                       'OOD Test Acc at Best ID Val': ood_acc_at_best_id_val,
                       'Best ID Val Bal Acc': bal_acc_best_id_val,
                       'OOD Test Bal Acc at Best ID Val': ood_bal_acc_at_best_id_val,
                       'Best ID Val Epoch': epoch_best_id_val})
            log.info(f"New Best ID Val Acc: {acc_best_id_val:.4f} (Bal: {bal_acc_best_id_val:.4f}) at Epoch {epoch_best_id_val} (OOD Test Acc: {ood_acc_at_best_id_val:.4f}, Bal: {ood_bal_acc_at_best_id_val:.4f})")
            save_dict = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'opt_state_dict': optimizer.state_dict()}
            if args.loss == 'hypo':
                save_dict['dis_state_dict'] = criterion_dis.state_dict()
                save_dict['uni_state_dict'] = criterion_comp.state_dict()
            save_checkpoint(args, save_dict, epoch + 1, save_best=True)

        if current_ood_test_bal_acc >= best_ood_bal_acc:
            best_ood_bal_acc = current_ood_test_bal_acc
            id_val_bal_acc_at_best_ood = current_id_val_bal_acc
            best_ood_test_acc = current_ood_test_acc 
            id_val_acc_at_best_ood = current_id_val_acc
            epoch_best_ood_test = epoch 
            
            wandb.log({'Best OOD Test Bal Acc': best_ood_bal_acc,
                       'ID Val Bal Acc at Best OOD Test Bal Acc': id_val_bal_acc_at_best_ood,
                       'Best OOD Test Acc (at Best OOD Bal Acc)': best_ood_test_acc,
                       'ID Val Acc (at Best OOD Bal Acc)': id_val_acc_at_best_ood,
                       'Best OOD Test Bal Acc Epoch': epoch_best_ood_test})
            log.info(f"New Best OOD Test Bal Acc: {best_ood_bal_acc:.4f} at Epoch {epoch_best_ood_test} (ID Val Bal Acc: {id_val_bal_acc_at_best_ood:.4f})")

    log.info("--- Training Summary ---")
    log.info(f"Final Best ID Val Acc: {acc_best_id_val:.6f} (Bal: {bal_acc_best_id_val:.6f}) (Epoch {epoch_best_id_val}), OOD Test Acc at this epoch: {ood_acc_at_best_id_val:.6f} (Bal: {ood_bal_acc_at_best_id_val:.6f})")
    log.info(f"Final Best OOD Test Acc (Standard): {best_ood_test_acc:.6f} (Epoch {epoch_best_ood_test}), ID Val Acc at this epoch: {id_val_acc_at_best_ood:.6f}") 
    log.info(f"Final Best OOD Test Bal Acc: {best_ood_bal_acc:.6f} (Epoch {epoch_best_ood_test}), ID Val Bal Acc at this epoch: {id_val_bal_acc_at_best_ood:.6f}")
    log.info(f"Final Epoch ({epoch}) ID Val Acc: {current_id_val_acc:.6f} (Bal: {current_id_val_bal_acc:.6f}), OOD Test Acc: {current_ood_test_acc:.6f} (Bal: {current_ood_test_bal_acc:.6f})")

    summary_metrics = {
        'final_best_id_val_acc': acc_best_id_val,
        'final_ood_test_acc_at_best_id_val': ood_acc_at_best_id_val,
        'final_best_id_val_bal_acc': bal_acc_best_id_val,
        'final_ood_test_bal_acc_at_best_id_val': ood_bal_acc_at_best_id_val,
        'final_best_id_val_epoch': epoch_best_id_val,
        
        'final_best_ood_test_acc': best_ood_test_acc, 
        'final_id_val_acc_at_best_ood_test': id_val_acc_at_best_ood, 
        'final_best_ood_bal_acc': best_ood_bal_acc,
        'final_id_val_bal_acc_at_best_ood_bal_acc': id_val_bal_acc_at_best_ood,
        'final_best_ood_test_epoch': epoch_best_ood_test, 
        
        'final_last_epoch_id_val_acc': current_id_val_acc,
        'final_last_epoch_ood_test_acc': current_ood_test_acc,
        'final_last_epoch_id_val_bal_acc': current_id_val_bal_acc,
        'final_last_epoch_ood_test_bal_acc': current_ood_test_bal_acc,
        'final_last_epoch': epoch
    }
    for metric_name, metric_value in summary_metrics.items():
        wandb.summary[metric_name] = metric_value

def train_hypo_fitzpatrick(args, train_loader, val_loader, ood_test_loader, model, criterion_comp, criterion_dis, criterion_ce, optimizer, epoch, log):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    ce_losses = AverageMeter() 
    comp_losses = AverageMeter()
    dis_losses = AverageMeter()

    model.train()
    end = time.time()
    for i, values in enumerate(train_loader):
        input_data, target_labels, domain_info = values 
        
        warmup_learning_rate(args, epoch, i, len(train_loader), optimizer)
        bsz = target_labels.shape[0]

        input_data = input_data.cuda()
        target_labels = target_labels.cuda()
        if domain_info is not None and isinstance(domain_info, torch.Tensor): 
            domain_info = domain_info.cuda()

        penultimate = model.encoder(input_data).squeeze()

        if args.normalize: 
            penultimate= F.normalize(penultimate, dim=1)

        if args.loss == 'hypo':
            features = model.head(penultimate)
            features = F.normalize(features, dim=1) 
            if criterion_dis is None or criterion_comp is None:
                 raise ValueError("HypO criteria not initialized for hypo loss.")
            dis_loss_val = criterion_dis(features, target_labels)
            current_domain_info_for_loss = domain_info if args.use_domain else None
            comp_loss_val = criterion_comp(features, criterion_dis.prototypes, target_labels, current_domain_info_for_loss)
            loss = args.w * comp_loss_val + dis_loss_val
            
            dis_losses.update(dis_loss_val.item(), bsz) 
            comp_losses.update(comp_loss_val.item(), bsz) 
            ce_losses.update(0, bsz) 
        elif args.loss == 'erm':
            logits = model.head(penultimate) 
            loss = criterion_ce(logits, target_labels)
            ce_losses.update(loss.item(), bsz) 
            dis_losses.update(0, bsz) 
            comp_losses.update(0, bsz)
        else:
             raise ValueError(f"Unsupported loss type: {args.loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            if args.loss == 'hypo':
                 log.debug(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                           f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           f'Dis Loss {dis_losses.val:.4f} ({dis_losses.avg:.4f})\t'
                           f'Comp Loss {comp_losses.val:.4f} ({comp_losses.avg:.4f})')
                 wandb.log({'Step Dis Loss': dis_losses.val, 'Step Comp Loss': comp_losses.val})
            elif args.loss == 'erm':
                 log.debug(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                           f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           f'CE Loss {ce_losses.val:.4f} ({ce_losses.avg:.4f})')
                 wandb.log({'Step CE Loss': ce_losses.val})

    model.eval()
    id_val_acc = 0.0
    ood_test_acc = 0.0
    id_val_bal_acc = 0.0
    ood_test_bal_acc = 0.0

    with torch.no_grad():
        all_preds_id_val = []
        all_targets_id_val = []
        for i, values in enumerate(val_loader): 
            input_data, target_labels, _ = values 
            input_data = input_data.cuda()
            target_labels = target_labels.cuda()

            if args.loss == 'hypo':
                 if criterion_dis is None: raise ValueError("criterion_dis is None during HypO evaluation")
                 features = model.forward(input_data)
                 feat_dot_prototype = torch.div(torch.matmul(features, criterion_dis.prototypes.T), args.temp)
                 logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
                 logits = feat_dot_prototype - logits_max.detach()
            elif args.loss == 'erm':
                 penultimate = model.encoder(input_data).squeeze()
                 if args.normalize: penultimate= F.normalize(penultimate, dim=1)
                 logits = model.head(penultimate)
            else:
                 raise ValueError(f"Unsupported loss type for evaluation: {args.loss}")
            
            pred = logits.data.max(1)[1]
            all_preds_id_val.extend(list(to_np(pred)))
            all_targets_id_val.extend(list(to_np(target_labels)))

        if len(all_targets_id_val) > 0:
            id_val_acc = accuracy_score(all_targets_id_val, all_preds_id_val)
            id_val_bal_acc = balanced_accuracy_score(all_targets_id_val, all_preds_id_val)
        else:
            id_val_acc = 0.0
            id_val_bal_acc = 0.0
            
        if ood_test_loader is not None:
            all_preds_ood = []
            all_targets_ood = []
            for i, values in enumerate(ood_test_loader):
                input_data, target_labels, _ = values
                input_data = input_data.cuda()
                target_labels = target_labels.cuda()

                if args.loss == 'hypo':
                     if criterion_dis is None: raise ValueError("criterion_dis is None during HypO evaluation")
                     features = model.forward(input_data)
                     feat_dot_prototype = torch.div(torch.matmul(features, criterion_dis.prototypes.T), args.temp)
                     logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
                     logits = feat_dot_prototype - logits_max.detach()
                elif args.loss == 'erm':
                     penultimate = model.encoder(input_data).squeeze()
                     if args.normalize: penultimate= F.normalize(penultimate, dim=1)
                     logits = model.head(penultimate)
                else:
                     raise ValueError(f"Unsupported loss type for evaluation: {args.loss}")
                
                pred = logits.data.max(1)[1]
                all_preds_ood.extend(list(to_np(pred)))
                all_targets_ood.extend(list(to_np(target_labels)))

            if len(all_targets_ood) > 0:
                ood_test_acc = accuracy_score(all_targets_ood, all_preds_ood)
                ood_test_bal_acc = balanced_accuracy_score(all_targets_ood, all_preds_ood)
            else:
                ood_test_acc = 0.0
                ood_test_bal_acc = 0.0
        else: 
            ood_test_acc = 0.0
            ood_test_bal_acc = 0.0

    if args.loss == 'hypo':
        return ce_losses.avg, comp_losses.avg, dis_losses.avg, id_val_acc, ood_test_acc, id_val_bal_acc, ood_test_bal_acc
    elif args.loss == 'erm':
        return ce_losses.avg, comp_losses.avg, dis_losses.avg, id_val_acc, ood_test_acc, id_val_bal_acc, ood_test_bal_acc

def save_checkpoint(args, state, epoch, save_best = False):
    """Saves checkpoint to disk"""
    filename = os.path.join(args.model_directory, 
                            'checkpoint_max.pth.tar' if save_best else f'checkpoint_{epoch}.pth.tar')
    torch.save(state, filename)

if __name__ == '__main__':
    main()
