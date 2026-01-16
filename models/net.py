from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import sys
import math
import random
import cv2
import time
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import glob
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, CenterCrop
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler, SGD

print("pytorch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("backends cudnn version:", torch.backends.cudnn.version())
print("GPU Type:", torch.cuda.get_device_name(0))
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path)


def dataset_path(dataset_name):
    if dataset_name == "ImageNet2012":
        return "../../../dataset/ImageNet2012/"
    elif dataset_name == "CUB200":
        return "../../../dataset/CUB200_2011/"
    elif dataset_name == "cotton":
        return "../../../dataset/cotton/"
    elif dataset_name == "soybean":
        return "../../../dataset/soybean/"
    elif dataset_name == "StanfordCars":
        return "../../../dataset/StanfordCars/"
    elif dataset_name == "AirCraft":
        return "../../../dataset/AirCraft/"
    elif dataset_name == "StanfordDogs":
        return "../../../dataset/Stanford_Dogs/"
    elif dataset_name == "flower":
        return "../../../dataset/flower/"
    elif dataset_name == "CIFAR10":
        return "../../../dataset/CIFAR10/"
    elif dataset_name == "CIFAR100":
        return "../../../dataset/CIFAR100/"
    else:
        return "../../../dataset/" + dataset_name + "/"

def generate_optimizer(model, optimizer_name="SGD"):
    from torch.optim import SGD, RMSprop, Adagrad, Adam, AdamW
    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(model.parameters(), lr=2e-4, weight_decay=1e-5)
    elif optimizer_name == "Adagrad":
        optimizer = Adagrad(model.parameters(), lr=2e-4, weight_decay=1e-5)
    elif optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    elif optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    else:
        raise NotImplementedError

    return optimizer


def generate_scheduler(optimizer, schedulr_name="StepLR"):
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau, MultiStepLR, \
        CyclicLR, LambdaLR, CosineAnnealingWarmRestarts
    if schedulr_name == "StepLR":
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    elif schedulr_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer)
    elif schedulr_name == "ExponentialLR":
        scheduler = ExponentialLR(optimizer)
    elif schedulr_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer)
    elif schedulr_name == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
    elif schedulr_name == "CyclicLR":
        scheduler = CyclicLR(optimizer)
    elif schedulr_name == "LambdaLR":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 ** epoch)
    elif schedulr_name == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer)
    else:
        raise NotImplementedError
    return scheduler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-set", default="ImageNet2012", type=str, help="Dataset name")
    parser.add_argument("--dataset-path", default="../../../dataset/ImageNet2012/", type=str)
    parser.add_argument('--use-cuda', default=True, action='store_true')
    parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"])
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args
class ChannelAttention(nn.Module):
    def __init__(self, input_channels,reduce_rate=4):
        super(ChannelAttention,self).__init__()
        self.mlp=nn.Sequential(
            nn.Linear(input_channels,input_channels//reduce_rate, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(input_channels//reduce_rate,input_channels,bias=False)
        )

    def forward(self,x):
        bs, channel, length, hegiht, width = x.shape
        y = F.adaptive_avg_pool3d(x, (1,1,1))
        y = y.contiguous().view(bs, -1)
        y = self.mlp(y)
        scale = torch.sigmoid(y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return x*scale
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.spatial_cov = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm3d(1, eps=1e-5, momentum=0.01, affine=True),
        )

    def forward(self, x):
        # x(bs, channel, lenght, height,width)
        bs, channel, length, height, width = x.shape
        temp = torch.max(x, dim=1, keepdim=True)[0]
        spatial_feature = torch.cat([torch.max(x, dim=1)[0].unsqueeze(dim=1), torch.mean(x, dim=1).unsqueeze(dim=1)], dim=1)
        x_out = self.spatial_cov(spatial_feature)
        scale = torch.sigmoid(x_out)
        return x*scale

class AttentionModule(nn.Module):
    def __init__(self, input_channel=32, kernel_size=3, no_spatial=False):
        super(AttentionModule, self).__init__()
        self.no_spatial = no_spatial
        self.channel_attn = ChannelAttention(input_channel, reduce_rate=4)
        self.spatial_attn = SpatialAttention(kernel_size=kernel_size)

    def forward(self,x):
        y = self.channel_attn(x)
        if self.no_spatial==False:
            y = self.spatial_attn(y)
        return y

class TransitionModule(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(TransitionModule, self).__init__()
        self.conv = nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm3d(num_output_features)
        self.activate = nn.ReLU(inplace=False)

        self.pool = nn.Conv3d(num_output_features, num_output_features, kernel_size=2, stride=2)
        self.norm2 = nn.BatchNorm3d(num_output_features)

    def forward(self, x):
        y = self.conv(x)
        y = self.norm1(y)
        y = self.activate(y)

        y = self.pool(y)
        y = self.norm2(y)
        y = self.activate(y)
        return y


class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=2, num_init_channels=32, num_classes=4):
        super(FeatureExtractor,self).__init__()
        self.conv1 = nn.Conv3d(input_channels,num_init_channels,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=False)
        self.norm1 = nn.BatchNorm3d(num_init_channels)
        self.activate = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(num_init_channels,num_init_channels,kernel_size=(3,3,3),stride=1,padding=(1,1,1),bias=False)
        self.norm2 = nn.BatchNorm3d(num_init_channels)

        self.conv3 = nn.Conv3d(num_init_channels,num_init_channels,kernel_size=(3,3,3),stride=1,padding=(1,1,1),bias=False)
        self.norm3 = nn.BatchNorm3d(num_init_channels)

        self.downsample = nn.Sequential(nn.Conv3d(input_channels,num_init_channels, kernel_size=1,stride=1,padding=0),
                                      nn.BatchNorm3d(num_init_channels))

    def forward(self, x):
        residual = self.downsample(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.activate(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.activate(y)

        y = self.conv3(y)

        z = self.norm3(y)
        z = z+residual
        z = self.activate(z)
        return y, z

class ConvolutionUnit(nn.Module):
    def __init__(self, num_init_channels=32, growth_rate=16, bn_size=4, drop_rate=0):
        # num_input_features, growth_rate, bn_size, drop_rate
        super(ConvolutionUnit,self).__init__()
        self.conv1 = nn.Conv3d(num_init_channels, bn_size*growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(bn_size*growth_rate)
        self.activate = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(growth_rate)
        self.drop_rate = drop_rate

    def forward(self,x):
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.activate(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.activate(y)

        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training)
        z = torch.cat([residual, y], dim=1)

        return z


class ConvolutionBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(ConvolutionBlock, self).__init__()
        layer_list = []
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = ConvolutionUnit(num_input_features+i*growth_rate, growth_rate, bn_size, drop_rate)
            layer_list.append(layer)
        self.layers = nn.ModuleList(layer_list)

    def forward(self,x):
        y = x
        for i, layer in enumerate(self.layers):
            y = layer(y)
        return y


class HDAN(nn.Module):
    def __init__(self, input_dim=2, growth_rate=16, block_config=(4,4,4,4),num_init_features=32, bn_size=4, drop_rate=0,num_classes=4):
        super(HDAN,self).__init__()
        self.num_blocks = len(block_config)

        self.feature_extractor = FeatureExtractor(input_dim,num_init_features)

        self.transit_module = TransitionModule(num_init_features, num_init_features)
        num_features = num_init_features
        self.bn_class = nn.BatchNorm3d(num_classes * self.num_blocks + num_init_features)
        self.classifier = nn.Conv3d(num_classes * self.num_blocks + num_init_features, num_classes, kernel_size=1, stride=1,padding=0)
        self.attn = AttentionModule(input_channel=num_init_features,kernel_size=3, no_spatial=False)

        self.dense_blocks = nn.ModuleList([])
        self.upsampling_blocks = nn.ModuleList([])
        self.transit_blocks = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])
        for i, num_layers in enumerate(block_config):
            block = ConvolutionBlock(num_layers, num_features, bn_size,growth_rate, drop_rate)
            num_features = num_features + num_layers * growth_rate

            attention_module = AttentionModule(input_channel=num_features, kernel_size=3, no_spatial=False)
            self.attn_blocks.append(attention_module)

            self.dense_blocks.append(block)

            up_block = nn.ConvTranspose3d(num_features, num_classes, kernel_size=2 ** (i + 1) + 2, stride=2 ** (i + 1), padding=1, groups=num_classes, bias=False)
            self.upsampling_blocks.append(up_block)

            if i != self.num_blocks-1:
                trans_function = TransitionModule(num_input_features=num_features, num_output_features=num_features // 2)
                self.transit_blocks.append(trans_function)
                num_features = num_features // 2

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        first_feature, first_feature_bn = self.feature_extractor(x)
        first_feature = self.attn(first_feature_bn)
        up_output_list = []
        up_output_list.append(first_feature)

        out = self.transit_module(first_feature_bn)
        for i in range(self.num_blocks):
            out_dense = self.dense_blocks[i](out)
            out_dense = self.attn_blocks[i](out_dense)
            temp = self.upsampling_blocks[i](out_dense)

            up_output_list.append(temp)
            if i < self.num_blocks-1:
                out = self.transit_blocks[i](out_dense)

        up_map = torch.cat(up_output_list,dim=1)

        up_map = self.bn_class(up_map)
        up_map = F.relu(up_map)
        result = self.classifier(up_map)
        return result


def test(device):
    input = torch.randn(1, 3, 32, 32).to(device)
    temp1 = torch.max(input, dim=1)[0]
    print(temp1.shape)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.gpu_id = int(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 2, 32, 64, 64).to(device)

    model = HDAN().to(device)
    output = model(input)
    print(output.shape)
    print("Finish!")
