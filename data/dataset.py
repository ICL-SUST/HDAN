from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import logging
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
import nibabel as nib
import glob
import torch
import pandas
import numpy
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, CenterCrop
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler, SGD
from torchvision.datasets import VisionDataset

print("pytorch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("backends cudnn version:", torch.backends.cudnn.version())
print("GPU Type:", torch.cuda.get_device_name(0))
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
logger = logging.getLogger()
FILEPATHKEY = "9dof_2mm_vol"


def check_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def transform_function(resolution=224, is_train=False):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop((resolution, resolution), scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    to_pil = transforms.ToPILImage()
    trans_test = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if is_train:
        return trans_train
    else:
        return transform_test


def read_image(img_path):
    """
    Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process.
    """
    got_img = False
    import os
    from PIL import Image
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    img = None
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
    return img


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
    elif dataset_name=="PretermInfant":
        return "../../../dataset/"+dataset_name+"/data/"
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

class MRI(VisionDataset):
    @staticmethod
    def get_path(root, path):
        if path == "/" or root is None:
            return path
        return os.path.join(root, path)

    def __init__(self, root, metadatafile, transform=None, target_transform=None, verify=False, num_sample=-1, random_state=0):
        super(MRI,self).__init__(root, transform=transform, target_transform=target_transform)
        self.df = pandas.read_csv(metadatafile)

        if num_sample > 0:
            self.df = self.df.sample(n=num_sample, random_state=random_state)

        if verify:
            indices = []
            for i, row in self.df.iterrows():
                if not os.path.exists(self.get_path(root, row[FILEPATHKEY])):
                    indices.append(i)
            if indices:
                logger.info(f"Dropping {len(indices)}")
                logger.debug(f"Dropped rows {indices}")
            self.df = self.df.drop(index=indices)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = self.get_path(self.root, row[FILEPATHKEY])
        subject_id = row["subject_id"]
        age = row["age_at_scan"]
        img = nib.load(path).get_fdata()
        img = (img - img.mean()) / img.std()
        scan = img[numpy.newaxis, :, :, :]
        age = age

        if self.transform:
            scan = self.transform(scan)

        if self.target_transform:
            age = self.target_transform(age)

        return numpy.float32(scan), numpy.float32(age), subject_id

    def __len__(self):
        return self.df.shape[0]

class MagneticResonanceImage(VisionDataset):

    def __init__(self, dir, transform=None, target_transform=None):
        super(MagneticResonanceImage,self).__init__()
        self.dir=dir;
        os.listdir(dir)

    def read_mri(self, datapath, dtype=np.float32):
        import nibabel as nib
        nib_obj = nib.load(datapath)
        ndarray = nib_obj.get_fdata().astype(dtype)
        return ndarray


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-set", default="PretermInfant", type=str, help="Dataset name")
    parser.add_argument("--dataset-path", default="./PretermInfant_dataset/PretermInfant/data/", type=str)
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
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu;
    args.gpu_id = int(args.gpu)

    i = 0
    for item in os.listdir(args.dataset_path):
        i = i+1
        image_dir = os.path.join(args.dataset_path, item)
        for image_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, image_name)
            img_nib_obj = nib.load(img_path)
            img_ndarray = img_nib_obj.get_fdata().astype(np.float32)
            print(f"Image shape: {img_ndarray.shape}")
            tensor_data = torch.from_numpy(img_ndarray).unsqueeze(0)
            print(f"Tensor shape for PyTorch: {tensor_data.shape}")
        if i > 3:
            break
    print("Finish!")
