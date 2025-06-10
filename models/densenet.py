import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, growthRate = 12, depth = 100, reduction = 0.5, bottleneck = True):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        # self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        # out = F.log_softmax(self.fc(out))
        return out

    def intermediate_forward(self, x, layer_index):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.relu(self.bn1(out))
        # out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        # out = F.log_softmax(self.fc(out))
        return out

    def feature_list(self, x):
        out_list = []
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.relu(self.bn1(out))
        out_list.append(out)
        # out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        # out = F.log_softmax(self.fc(out))
        return out_list

model_dict = {
    'densenet100': [DenseNet, 342],
    'densenet121': [None, 1024], # Add densenet121, func is None as loaded from torchvision
}

# Need to import torchvision models here
import torchvision.models as tv_models

class SupCEHeadDenseNet(nn.Module):
    """encoder + head for DenseNet models"""
    # Added args to pass model name, head type, feat_dim, n_cls etc.
    def __init__(self, args, multiplier=1):
        super(SupCEHeadDenseNet, self).__init__()
        name = args.model # Get model name from args
        head = args.head
        feat_dim = args.feat_dim
        num_classes = args.n_cls # Use n_cls from args

        if name not in model_dict:
            raise ValueError(f"Unsupported DenseNet model name: {name}")

        model_fun, dim_in = model_dict[name]
        self.multiplier = multiplier

        # Load encoder based on model name
        # Determine if pretrained weights should be used.
        # Defaults to True if 'model_pretrained' is not in args.
        use_pretrained = getattr(args, 'model_pretrained', True)

        if name == 'densenet121':
            print(f"Loading DenseNet121. Pretrained: {use_pretrained}")
            tv_model = tv_models.densenet121(pretrained=use_pretrained)
            dim_in = tv_model.classifier.in_features # Should be 1024
            self.encoder = tv_model.features
            self.encoder.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
            print(f"Using torchvision pretrained DenseNet-121. Feature dim: {dim_in}")
        elif model_fun is not None: # Handle custom DenseNet (e.g., densenet100)
            self.encoder = model_fun()
            print(f"Using custom DenseNet: {name}. Feature dim: {dim_in}")
        else:
             raise ValueError(f"Model function not found for {name}")

        # self.fc = nn.Linear(dim_in, num_classes) # Not used directly?

        # Initialize head
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        

    def forward(self, x):
        feat = self.encoder(x)
        # Flatten the output of the encoder before the head
        if feat.dim() > 2:
            feat = torch.flatten(feat, 1)
        else:
            feat = feat.squeeze() # Ensure it's 2D if already flattened

        unnorm_features = self.head(feat)
        features = F.normalize(unnorm_features, dim=1)
        return features

    # Keep intermediate_forward if needed, but ensure it works with the new structure
    # def intermediate_forward(self, x, layer_index):
    #     # This might need adjustment depending on how intermediate features are defined/used
    #     if layer_index == 0:
    #         # For torchvision densenet, intermediate might be harder to get cleanly
    #         # For custom densenet, use its method if available
    #         if hasattr(self.encoder, 'intermediate_forward'):
    #              return self.encoder.intermediate_forward(x, layer_index)
    #         else:
    #              # Fallback or raise error
    #              return self.encoder(x) # Return final features if intermediate not defined
    #     elif layer_index == 1:
    #         feat = self.forward(x) # Get final normalized features
    #         # Applying multiplier here seems inconsistent with SupCEHeadResNet, maybe remove?
    #         # feat = self.multiplier * feat
    #         return feat
    #     else:
    #          raise ValueError(f"Unsupported layer_index: {layer_index}")
