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

parser = argparse.ArgumentParser(description='Script for training with HYPO')
parser.add_argument('--gpu', default=6, type=int, help='which GPU to use')
parser.add_argument('--seed', default=4, type=int, help='random seed')  # original 4
parser.add_argument('--w', default=2, type=float,
                    help='loss scale')
parser.add_argument('--proto_m', default= 0.95, type=float,
                   help='weight of prototype update')
parser.add_argument('--feat_dim', default = 128, type=int,
                    help='feature dim')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset', choices=['PACS', 'VLCS', 'CIFAR-10', 'CIFAR-100', 'ImageNet-100', 'OfficeHome', 'terra_incognita'])
parser.add_argument('--id_loc', default="datasets/CIFAR10", type=str, help='location of in-distribution dataset')
parser.add_argument('--model', default='resnet18', type=str, help='model architecture: [resnet18, wrt40, wrt28, wrt16, densenet100, resnet50, resnet34]') 
parser.add_argument('--head', default='mlp', type=str, help='either mlp or linear head')
parser.add_argument('--loss', default = 'hypo', type=str, choices = ['hypo', 'erm'],
                    help='name of experiment')
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

# add pacs specific
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--target_domain', type=str, default='cartoon')

# debug
parser.add_argument('--use_domain', type=bool, default=False, help='whether to use in-domain negative pairs in compactness loss')
parser.add_argument('--mode', default='online', choices = ['online','disabled'], help='whether disable wandb logging')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()
torch.cuda.set_device(args.gpu) 
state = {k: v for k, v in args._get_kwargs()}

date_time = datetime.now().strftime("%d_%m_%H:%M")

#processing str to list for linear lr scheduling
args.lr_decay_epochs = [int(step) for step in args.lr_decay_epochs.split(',')]


if args.in_dataset == "ImageNet-100" or args.in_dataset == 'CIFAR-10':
    args.name = (f"{date_time}_{args.loss}_{args.model}_lr_{args.learning_rate}_cosine_"
        f"{args.cosine}_bsz_{args.batch_size}_head_{args.head}_wd_{args.w}_{args.epochs}_{args.feat_dim}_"
        f"trial_{args.trial}_temp_{args.temp}_{args.in_dataset}_pm_{args.proto_m}")

else:
    args.name = (f"{date_time}_{args.loss}_std_{args.model}_lr_{args.learning_rate}_cosine_"
        f"{args.cosine}_bsz_{args.batch_size}_td_{args.target_domain}_head_{args.head}_wd_{args.w}_{args.epochs}_{args.feat_dim}_"
        f"trial_{args.trial}_temp_{args.temp}_{args.in_dataset}_pm_{args.proto_m}")

args.log_directory = "logs/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name= args.name)

args.model_directory = "/nobackup2/yf/checkpoints/hypo_cr/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name= args.name)


args.tb_path = './save/hypo/{}_tensorboard'.format(args.in_dataset)
if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory)
if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)
args.tb_folder = os.path.join(args.tb_path, args.name)
if not os.path.isdir(args.tb_folder):
    os.makedirs(args.tb_folder)

#save args
with open(os.path.join(args.log_directory, 'train_args.txt'), 'w') as f:
    f.write(pprint.pformat(state))

#init log
log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s : %(message)s')
fileHandler = logging.FileHandler(os.path.join(args.log_directory, "train_info.log"), mode='w')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.addHandler(fileHandler)
log.addHandler(streamHandler) 

log.debug(state)

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

    wandb.init(
        # Set the project where this run will be logged
        project="hypo",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=args.name, 
        # Track hyperparameters and run metadata
        mode=args.mode,
        config=args)

    if args.in_dataset == "ImageNet-100":
        train_loader, val_loader, test_loader = set_loader_ImageNet(args)
    else:
        if args.in_dataset == 'CIFAR-10' or args.in_dataset == 'CIFAR-100':
            train_loader, val_loader, test_loader = set_loader_small(args)
        else:
            train_loader, val_loader, test_loader = set_loader_small(args)


    model = set_model(args)

    criterion_comp = CompLoss(args, temperature=args.temp, use_domain = args.use_domain).cuda()
    criterion_dis = DisLoss(args, model, val_loader, temperature=args.temp).cuda()

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
        train_sloss, train_uloss, train_dloss, acc, acc_cor= train_hypo(args, train_loader, val_loader, test_loader, model, criterion_comp, criterion_dis, optimizer, epoch, log)

        tb_log.log_value('train_uni_loss', train_uloss, epoch)
        tb_log.log_value('train_dis_loss', train_dloss, epoch)
        wandb.log({'Comp Loss Ep': train_uloss,'Dis Loss Ep': train_dloss })
        tb_log.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        wandb.log({'current lr': optimizer.param_groups[0]['lr'], 'acc':acc, 'acc cor': acc_cor})
        
        # save checkpoint
        if acc >= acc_best_id:
            acc_best = acc_cor
            acc_best_id = acc
            epoch_val_best = epoch
            wandb.log({'val best ood acc': acc_best, 'val best id acc':acc_best_id})
            print('best accuracy {} at epoch {}'.format(acc_best_id, epoch))
            print('accuracy cor {} at epoch {}'.format(acc_best, epoch))
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'dis_state_dict': criterion_dis.state_dict(),
                'uni_state_dict': criterion_comp.state_dict(),
            }, epoch + 1, save_best=True)  

        if acc_cor >= acc_test_best:
            acc_test_best = acc_cor
            acc_test_best_id = acc
            epoch_test_best = epoch
            wandb.log({'test best ood acc': acc_test_best, 'test best id acc':acc_test_best_id})
            print('best test accuracy {} at epoch {}'.format(acc_test_best, epoch))

    print('total val best ood accuracy {} id accuracy {} at epoch {}'.format(acc_best, acc_best_id, epoch_val_best))
    print('total test best ood accuracy {} id accuracy {} at epoch {}'.format(acc_test_best, acc_test_best_id, epoch_test_best))
    print('last epoch ood accuracy {} id accuracy {} at epoch {}'.format(acc_cor, acc, epoch))

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

def train_hypo(args, train_loader, val_loader, test_loader, model, criterion_comp, criterion_dis, optimizer, epoch, log): 
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    supcon_losses = AverageMeter()
    comp_losses = AverageMeter()
    dis_losses = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    for i, values in enumerate(train_loader):
        if len(values) == 3:
            input, target, domain = values
        elif len(values) == 2:
            input, target = values
            domain = None 
        
        warmup_learning_rate(args, epoch, i, len(train_loader), optimizer)
        bsz = target.shape[0]

        input = input.cuda()
        target = target.cuda()

        penultimate = model.encoder(input).squeeze()

        if args.normalize: # default: False 
            penultimate= F.normalize(penultimate, dim=1)
        features= model.head(penultimate)
        features= F.normalize(features, dim=1)

        dis_loss = criterion_dis(features, target) 
        comp_loss = criterion_comp(features, criterion_dis.prototypes, target, None)

        loss = args.w * comp_loss + dis_loss

        dis_losses.update(dis_loss.data, input.size(0))
        comp_losses.update(comp_loss.data, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0: 

            log.debug('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Dis Loss {dloss.val:.4f} ({dloss.avg:.4f})\t'
                'Comp Loss {uloss.val:.4f} ({uloss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, dloss=dis_losses, uloss = comp_losses))

            wandb.log({'Dis Loss' : dis_losses.val, 'Comp Loss' :  comp_losses.val})


    model.eval()
    with torch.no_grad():
        accuracies = []
        for i, values in enumerate(val_loader):
            if len(values) == 3:
                input, target, domain = values
            elif len(values) == 2:
                input, target = values
                domain = None 
            input = input.cuda()
            target = target.cuda()

            features = model.forward(input)
            feat_dot_prototype = torch.div(torch.matmul(features, criterion_dis.prototypes.T), args.temp)

            # for numerical stability
            logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
            logits = feat_dot_prototype - logits_max.detach()

            pred = logits.data.max(1)[1]

            accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

    acc = sum(accuracies) / len(accuracies)

    if test_loader is None:
        acc_cor = 0.
    else:
        with torch.no_grad():
            accuracies_cor = []
            for i, values in enumerate(test_loader):
                if len(values) == 3:
                    input, target, domain = values
                elif len(values) == 2:
                    input, target = values
                    domain = None 
                input = input.cuda()
                target = target.cuda()

                features = model.forward(input)
                feat_dot_prototype = torch.div(torch.matmul(features, criterion_dis.prototypes.T), args.temp)

                # for numerical stability
                logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
                logits = feat_dot_prototype - logits_max.detach()

                pred = logits.data.max(1)[1]
                accuracies_cor.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

        acc_cor = sum(accuracies_cor) / len(accuracies_cor)

    # measure elapsed time
    return supcon_losses.avg, comp_losses.avg, dis_losses.avg, acc, acc_cor



def save_checkpoint(args, state, epoch, save_best = False):
    """Saves checkpoint to disk"""
    if save_best:
        filename = args.model_directory + 'checkpoint_max.pth.tar'
    else:
        filename = args.model_directory + f'checkpoint_{epoch}.pth.tar'
    torch.save(state, filename)


if __name__ == '__main__':
    main()
