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

from config import config
import nibabel as nib
import glob
import SimpleITK as sitk
from utils.metrics import dice

def check_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def dataset_path(dataset_name):
    if dataset_name == "ImageNet2012":
        return "../../../../Datasets/ImageNet2012/"
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
    elif dataset_name == "PertermInfant":
        return "../../../dataset/"+dataset_name+"/data/"
    else:
        return "../../../dataset/" + dataset_name + "/"


def read_mri(datapath, dtype=np.float32):
    nib_obj = nib.load(datapath)
    ndarray = nib_obj.get_fdata().astype(dtype)
    return ndarray

def read_med_image (file_path, dtype=np.float32):
    import SimpleITK as sitk
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    return img_np, img_stk
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-set", default="PretermInfant", type=str, help="Dataset name")
    parser.add_argument("--dataset-path", default="", type=str)
    parser.add_argument('--use-cuda', default=True, action='store_true')
    parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"])
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args


if __name__ == "__main__":
    torch.cuda.empty_cache()

    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.dataset_path = dataset_path(args.data_set)
    args.gpu_id = int(args.gpu)

    file_path = "./PretermInfant_dataset/iSeg2019/iSeg-2019-Validation/subject-11-label.hdr"
    img_ndarray, img_stk = read_med_image(file_path)
    print(img_ndarray.shape)

    print("Finish!")
