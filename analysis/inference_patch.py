from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import argparse
import numpy as np
import torch

from models.net import HDAN
import nibabel as nib
import glob
import SimpleITK as sitk
from utils.metrics import dice

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
    to_pil = transforms.ToPILImage();
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-set", default="PretermInfant", type=str, help="Dataset name")
    parser.add_argument("--dataset-path", default="../../../dataset/PretermInfant/", type=str)
    parser.add_argument('--use-cuda', default=True, action='store_true')
    parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"])
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args


def read_mri(datapath, dtype=np.float32):
    nib_obj = nib.load(datapath)
    ndarray = nib_obj.get_fdata().astype(dtype)
    return ndarray


def convert_label_submit(label_img):
    # ndarray label_img (chane, height, width)
    label_processed = np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice = label_img[:, :, i]
        label_slice[label_slice == 1] = 10
        label_slice[label_slice == 2] = 150
        label_slice[label_slice == 3] = 250
        label_processed[:, :, i] = label_slice
    return label_processed


def save_hdr(whole_pred, test_path, name="s0007"):
    f_pred = os.path.join(test_path, name + 'label.hdr')
    whole_pred = whole_pred.transpose(0, 2, 1)
    whole_pred = convert_label_submit(whole_pred)
    whole_pred_itk = sitk.GetImageFromArray(whole_pred.astype(np.uint8))


def save_ndarray(ndarray_file,test_path,name="s0007"):
    np.save(os.path.join(test_path, name + '_label.npy'), ndarray_file)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.dataset_path = dataset_path(args.data_set)
    args.gpu_id = int(args.gpu)

    net = HDAN(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4), num_classes=4).to(device)

    checkpoint = os.path.join("./checkpoints/Ours", "BestInfantSeg_ISEG.pth")
    saved_state_dict = torch.load(checkpoint)
    net.load_state_dict(saved_state_dict)
    net.eval()
    print('Checkpoint: ', checkpoint)
    mri_t1 = "./PretermInfant_dataset/PretermInfant/data/s0007/t1.nii.gz"
    mri_t2 = "./PretermInfant_dataset/PretermInfant/data/s0007/t2.nii.gz"
    t1_ndarray = read_mri(mri_t1)
    t2_ndarray = read_mri(mri_t2)

    mask = t1_ndarray > 0
    mask = mask.astype(bool)

    inputs_T1_norm = (t1_ndarray - t1_ndarray[mask].mean()) / t1_ndarray[mask].std()
    inputs_T2_norm = (t2_ndarray - t2_ndarray[mask].mean()) / t2_ndarray[mask].std()

    inputs_T1_norm = inputs_T1_norm[:, :, :, None]
    inputs_T2_norm = inputs_T2_norm[:, :, :, None]
    inputs = np.concatenate((inputs_T1_norm, inputs_T2_norm), axis=3)
    inputs = inputs[None, :, :, :, :]
    image = inputs.transpose(0, 4, 3, 1, 2)
    image = torch.from_numpy(image).float().to(device)
    print(image.shape)
    _, _, C, H, W = image.shape
    crop_size = (32, 64, 64)
    xstep = 8
    ystep = 8
    zstep = 8

    deep_slices = np.arange(0,    C - crop_size[0], xstep)
    height_slices = np.arange(0,  H - crop_size[1], ystep)
    width_slices = np.arange(0,   W - crop_size[2], zstep)

    whole_pred = np.zeros((1,) + (4,) + image.shape[2:])
    count_used = np.zeros((image.shape[2], image.shape[3], image.shape[4])) + 1e-5

    with torch.no_grad():
        for i in range(len(deep_slices)):
            for j in range(len(height_slices)):
                for k in range(len(width_slices)):
                    deep = deep_slices[i]
                    height = height_slices[j]
                    width = width_slices[k]
                    image_crop = image[:, :, deep: deep + crop_size[0], height: height + crop_size[1], width: width + crop_size[2]]

                    outputs = net(image_crop)
                    print(outputs.shape)
                    whole_pred[slice(None), slice(None), deep: deep + crop_size[0], height: height + crop_size[1], width: width + crop_size[2]] += outputs.data.cpu().numpy()
                    count_used[deep: deep + crop_size[0], height: height +crop_size[1], width: width + crop_size[2]] += 1

    whole_pred = whole_pred / count_used
    whole_pred = whole_pred[0, :, :, :, :]
    whole_pred = np.argmax(whole_pred, axis=0)

    test_path = "./ExperimentResults/PretermInfant/"
    check_path(test_path)
    save_ndarray(whole_pred,test_path,name="s0007")
    print("Finish!")
