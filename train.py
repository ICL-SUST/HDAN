import argparse
import torch.utils.data as dataloader
import torch.nn as nn
import torch.optim as optim

from utils.common import *
import sys, os
sys.path.append(os.getcwd())
from config import config
from models.net import HDAN
from data.loader import H5Dataset
from utils.metrics import dice

def check_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-set", default="PretermInfant", type=str, help="Dataset name")
    parser.add_argument("--dataset-path", default="./PretermInfant_dataset/PreterInfant/", type=str)
    parser.add_argument('--use-cuda', default=True, action='store_true')
    parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"])
    parser.add_argument('--save',default=r"F:\PretermInfant\checkpoints\Proposed", type=str, help="Where the model is saved")
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_dir = args.save
    check_path(save_dir)
    # --------------------------CUDA check-----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -------------initialize Segmentation Model---------------
    model_S = HDAN(input_dim=config.input_dim, num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4),
                       drop_rate=0.2, num_classes=config.num_classes).to(device)
    # --------------Loss---------------------------
    criterion_S = nn.CrossEntropyLoss().to(device)
    # ---------------setup optimizer--------------------------
    optimizer_S = optim.Adam(model_S.parameters(), lr=config.lr_S, weight_decay=6e-4, betas=(0.97, 0.999))
    scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=config.step_size_S, gamma=0.1)
    # ---------------setup data--------------------------
    mri_data_train = H5Dataset(config.traindata_path, crop_size=config.crop_size, mode='train')
    trainloader = dataloader.DataLoader(mri_data_train, batch_size=config.batch_train, shuffle=True)

    mri_data_val = H5Dataset(config.valdata_path, mode='val')
    valloader = dataloader.DataLoader(mri_data_val, batch_size=1, shuffle=False)

    # --------------Start Training and Validation ---------------------------
    print("The training will be implemented for {} epochs".format(config.num_epoch))
    print('Learning Rate  | epoch  | Loss seg| DSC_val')
    loss_seg = 0; best_dsc = 0;
    save_best_path = os.path.join(save_dir, "BestInfantSeg_ISEG1.pth")
    for epoch in range(config.num_epoch):
        # -----------------------Training--------------------------------------
        model_S.train()
        for i, data in enumerate(trainloader):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            optimizer_S.zero_grad()
            outputs = model_S(images)
            loss_seg = criterion_S(outputs, targets)
            loss_seg.backward()
            optimizer_S.step()
        scheduler_S.step()
        # -----------------------Validation------------------------------------
        dsc_list = []
        with torch.no_grad():
            for data_val in valloader:
                images_val, targets_val = data_val
                model_S.eval()
                images_val = images_val.to(device)
                targets_val = targets_val.to(device)

                outputs_val = model_S(images_val)
                _, predicted = torch.max(outputs_val.data, 1)
                # ----------Compute dice-----------
                predicted_val = predicted.data.cpu().numpy()
                targets_val = targets_val.data.cpu().numpy()
                dsc = []
                for i in range(1, config.num_classes):  # ignore Background 0
                    dsc_i = dice(predicted_val, targets_val, i)
                    dsc.append(dsc_i)
                dsc = np.mean(dsc)
                dsc_list.append(dsc)
        # -------------------Debug-------------------------
        dsc_mean = np.mean(dsc_list)
        for param_group in optimizer_S.param_groups:
            print('%0.6f | %6d | %0.5f | %0.5f ' % (param_group['lr'], epoch, loss_seg.data.cpu().numpy(), dsc_mean))

        if dsc_mean > best_dsc:
            best_dsc = dsc_mean
            torch.save(model_S.state_dict(), save_best_path)
            print("The best model has been updated at {} !".format(save_best_path))

        if (epoch % config.step_size_S) == 0 or epoch == (config.num_epoch - 1) or (epoch % 100) == 0:
            save_path = os.path.join(save_dir, '%s_%s.pth' % (str(epoch).zfill(5), config.checkpoint_name))
            torch.save(model_S.state_dict(), save_path)
            print("the {} model has been saved at {}!".format(epoch, save_path))
    print("well done")

