import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from train import get_args

# Local imports
from config import config
from models.net import HDAN
from data.loader import H5Dataset
from utils.metrics import dice, modified_hausdorff_distance

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


def save_model(model, save_dir):
    import os
    save_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), save_path)
    print("The model is saved at {}".format(save_path))
    a = 3.1415926
    print("{:0.5f}".format(a))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.gpu_id = int(args.gpu)
    save_dir = r"F:\PretermInfant"
    tissue_list = ["background", "cerebrospinal fluid (CSF)", "gray matter(GM)", "white matter(WM)"]
    net = HDAN(num_init_features=32, growth_rate=16,
                                block_config=(4, 4, 4, 4),
                                drop_rate=0.2, num_classes=4).to(device)

    checkpoint = os.path.join(save_dir, "best_metric_model_swin_unetr.pth")
    saved_state_dict = torch.load(checkpoint)
    net.load_state_dict(saved_state_dict)
    net.eval().to(device)
    print('Checkpoint: ', checkpoint)

    mri_data_val = H5Dataset(config.valdata_path, mode='val')
    valloader = DataLoader(mri_data_val, batch_size=1, shuffle=False)

    dsc_list = []
    dice0_list = []
    dice1_list = []
    dice2_list = []
    dice3_list = []
    mhd_mean_list = []
    mhd1_list = []
    mhd2_list = []
    mhd3_list = []

    with torch.no_grad():
        for data_val in valloader:
            images_val, targets_val = data_val
            net.eval()
            images_val = images_val.to(device)
            targets_val = targets_val.to(device)

            outputs_val = net(images_val)
            _, predicted = torch.max(outputs_val.data, 1)
            # ----------Compute dice-----------
            predicted_val = predicted.data.cpu().numpy()
            targets_val = targets_val.data.cpu().numpy()
            dsc = []
            mhd_mean = []
            asd_mean = []
            for i in range(1, config.num_classes):
                dsc_i = dice(predicted_val, targets_val, i)
                mhd_i = modified_hausdorff_distance(predicted_val, targets_val,i)

                if i == 1:
                    dice1_list.append(dsc_i)
                    mhd1_list.append(mhd_i)

                elif i == 2:
                    dice2_list.append(dsc_i)
                    mhd2_list.append(mhd_i)

                elif i == 3:
                    dice3_list.append(dsc_i)
                    mhd3_list.append(mhd_i)

                else:
                    dice0_list.append(dsc_i)

                dsc.append(dsc_i)
                mhd_mean.append(mhd_i)

            dsc = np.mean(dsc)
            dsc_list.append(dsc)

            mhd_mean_list.append(np.mean(mhd_mean))

    DICE1 = np.mean(dice1_list)  # CSF
    DICE2 = np.mean(dice2_list)  # Gray
    DICE3 = np.mean(dice3_list)   # White
    dsc_mean = np.mean(dsc_list)

    MHD1 = np.mean(mhd1_list)
    MHD2 = np.mean(mhd2_list)
    MHD3 = np.mean(mhd3_list)
    MHD_mean = np.mean(mhd_mean_list)

    print("DICE: CSF Mean:{:0.6f};Gray Matter Mean:{:0.6f}; White Matter Mean:{:0.6f};Tissue Mean:{:0.6f}".format(DICE1, DICE2, DICE3, dsc_mean))
    print("MHD:CSF Mean:{:0.6f};Gray Matter Mean:{:0.6f}; White Matter Mean:{:0.6f};Tissue Mean:{:0.6f}".format(MHD1, MHD2,MHD3,MHD_mean))
    print("Finish!")
