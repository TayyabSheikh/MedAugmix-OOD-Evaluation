from __future__ import print_function

import math
import os

from torch import nn
from models.resnet import SupCEHeadResNet


import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.transforms as transforms

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TwoCrop:
    def __init__(self):
        self.transform = 0
    def __call__(self, x):
        return [x, x]



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, labels, topk=(1, 5)):
    """Computes the top-k accuracy for a given set of model outputs and labels.
    
    Args:
        output (torch.Tensor): The model outputs, with shape (batch_size, num_classes).
        labels (torch.Tensor): The true labels, with shape (batch_size,).
        topk (tuple, optional): The top-k values for which to compute the accuracy. Default is (1, 5).
    
    Returns:
        list: A list of the top-k accuracies.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def set_loader_small(args, eval = False, batch_size = None, img_size = 32):
    root = args.id_loc
    if batch_size is None:
        batch_size = args.batch_size
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    # data augmentations for supcon                                     
    train_transform_supcon = transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "CIFAR-10":
        # Data loading code
        if eval: 
            dataset = datasets.CIFAR10(root, train=True, download=True, transform=transform_test)
            if args.subset: 
                dataset = torch.utils.data.Subset(dataset , np.random.choice(len(dataset), 20000, replace=False))
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
        else:
            train_data = datasets.CIFAR10(root, train=True, download=True,
                            transform=train_transform_supcon)

            train_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=args.batch_size, shuffle=True, **kwargs)          

    
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root, train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root, train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        if eval: 
            dataset = datasets.CIFAR100(root, train=True, download=True, transform=transform_test)
            if args.subset: 
                dataset = torch.utils.data.Subset(dataset , np.random.choice(len(dataset), 20000, replace=False))
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR100(root, train=True, download=True,
                             transform=train_transform_supcon),
                    batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root, train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root, train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.in_dataset == "ImageNet-100":
        traindir = os.path.join('./ImageNet-100', 'train')
        valdir = os.path.join('./ImageNet-100', 'val')

        mean = [0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        train_data = datasets.ImageFolder(
            traindir,
            TwoCropTransform(transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))
            )

        val_data = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))      

        from dataloader.corimagenetLoader import CorIMAGENETDataset as Dataset
        test_data = Dataset('test', cortype)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    elif args.in_dataset == "PACS":
        print('loading PACS')
        from dataloader.PACSLoader import PACSDataset as Dataset

        train_data = Dataset('train', args.target_domain)
        val_data = Dataset('val', args.target_domain)
        test_data = Dataset('test', args.target_domain)
   
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    elif args.in_dataset == "OfficeHome":
        print('loading OfficeHome')
        from dataloader.OfficeHomeLoader import OfficeHomeDataset as Dataset

        train_data = Dataset('train', args.target_domain)
        val_data = Dataset('val', args.target_domain)
        test_data = Dataset('test', args.target_domain)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    elif args.in_dataset == "VLCS":
        print('loading VLCS')
        from dataloader.VLCSLoader import VLCSDataset as Dataset

        train_data = Dataset('train', args.target_domain)
        val_data = Dataset('val', args.target_domain)
        test_data = Dataset('test', args.target_domain)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    elif args.in_dataset == "terra_incognita":
        print('loading terra incognita')
        from dataloader.TerraLoader import TerraDataset as Dataset

        train_data = Dataset('train', args.target_domain)
        val_data = Dataset('val', args.target_domain)
        test_data = Dataset('test', args.target_domain)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

def set_loader_ImageNet(args, eval = False, batch_size = None):
    #root = args.id_loc
    root = './ImageNet-100'
    if batch_size is None:
        batch_size = args.batch_size
    # for ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_transform_supcon = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.4, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if batch_size is not None:
        args.batch_size = batch_size

    # Data loading code
    if eval: 
        dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform_test)
        if args.subset: 
            dataset = torch.utils.data.Subset(dataset , np.random.choice(len(dataset), 20000, replace=False))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    else:
        dataset = datasets.ImageFolder(os.path.join(root, 'train'),
                    transform=train_transform_supcon)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(root, 'val'),transform=transform_test),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    from dataloader.corimagenetLoader import CorIMAGENETDataset as Dataset
    cortype='gaussian_noise'
    test_data = Dataset('test', cortype)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader


def set_model(args):
    
    # create model
    model = SupCEHeadResNet(args)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    model = model.cuda()


    return model

def sample_estimator(model, classifier, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        penultimate, out_features = model.encoder.feature_list(data)
        output = classifier(penultimate)
        # output, out_features = model.module.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
        #TEMP
        # out_features[-1] = out_features[i] / out_features[i].norm(p=2, dim=1, keepdim=True)
        out_features[-1] = F.normalize(out_features[-1], dim=1)
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


def estimate_dataset_mean_std(name = 'cifar10'):
    data = datasets.CIFAR10(root='./datasets/cifar10', train=True, download=True,
                    transform=transforms.ToTensor()).data
    data = data.astype(np.float32)/255.

    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:,:,:,i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))

if __name__ == '__main__':
    estimate_dataset_mean_std()
