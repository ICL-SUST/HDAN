import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    NormalizeIntensityd,
    ToTensord,
    AsDiscrete,
)
from monai.data import Dataset, decollate_batch
from monai.inferers import sliding_window_inference

# ================= 配置区域 =================
DATA_DIR = r"F:\PretermInfant\PretermInfant_dataset\iSeg2019_hdf5"
TRAIN_DIR = os.path.join(DATA_DIR, "iSeg-2019-Training")
VAL_DIR = os.path.join(DATA_DIR, "iSeg-2019-Validation")

# 参数设置
LR = 1e-4
MAX_EPOCHS = 150
BATCH_SIZE = 1
PATCH_SIZE = (64, 64, 64)
VAL_INTERVAL = 5


# ===========================================

class iSegH5Dataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        with h5py.File(path, 'r') as f:
            # 读取原始数据
            image = np.array(f['data']).astype(np.float32)
            label = np.array(f['label']).astype(np.uint8)

        # --- 暴力维度修正 ---
        image = np.squeeze(image)
        label = np.squeeze(label)

        # 修正 Image: 目标 (2, D, H, W)
        if image.ndim == 4:
            if image.shape[0] == 2:
                pass
            elif image.shape[-1] == 2:
                image = image.transpose(3, 0, 1, 2)
            elif image.shape[1] == 2:
                image = image.transpose(1, 0, 2, 3)
        elif image.ndim == 3:
            image = image[np.newaxis, ...]

        # 修正 Label: 目标 (1, D, H, W)
        if label.ndim == 3:
            label = label[np.newaxis, ...]
        elif label.ndim == 4:
            if label.shape[-1] == 1:
                label = label.transpose(3, 0, 1, 2)

        data_dict = {"image": image, "label": label}
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict


def main():
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.h5")))
    val_files = sorted(glob.glob(os.path.join(VAL_DIR, "*.h5")))
    print(f"Found {len(train_files)} train, {len(val_files)} val files.")

    # 3. 定义数据增强 transforms
    train_transforms = Compose([
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=1,
            neg=1,
            num_samples=1,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])

    # 4. DataLoader
    train_ds = iSegH5Dataset(train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    val_ds = iSegH5Dataset(val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 5. 模型定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 修正：加回 img_size 参数 ===
    # 虽然会有 DeprecationWarning，但必须加，否则你的版本会报错
    model = SwinUNETR(
        img_size=PATCH_SIZE,
        in_channels=2,
        out_channels=4,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # DiceMetric 配置
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    print("Start training...")
    best_metric = -1
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            if isinstance(batch_data, list):
                batch_data = batch_data[0]

            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{MAX_EPOCHS}, Loss: {epoch_loss / step:.4f}")

        # 验证阶段
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()

            # === 定义后处理变换 ===
            post_pred = AsDiscrete(argmax=True, to_onehot=4)
            post_label = AsDiscrete(to_onehot=4)

            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

                    val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 4, model)

                    # === 应用 One-Hot 转换 ===
                    val_outputs_list = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_list = [post_label(i) for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=val_outputs_list, y=val_labels_list)

                # === 获取分类别分数 ===
                # 这部分代码是修复 IndexError 的关键
                dice_scores = dice_metric.aggregate(reduction="mean_batch")
                metric = dice_metric.aggregate(reduction="mean").item()

                dice_metric.reset()

                if dice_scores.numel() > 0:
                    print(
                        f"  >> Val Dice - CSF: {dice_scores[0].item():.4f}, "
                        f"GM: {dice_scores[1].item():.4f}, "
                        f"WM: {dice_scores[2].item():.4f}"
                    )

                if metric > best_metric:
                    best_metric = metric
                    torch.save(model.state_dict(), "best_metric_model_swin_unetr.pth")
                    print(f"  >> New Best Mean Dice: {best_metric:.4f}")


if __name__ == "__main__":
    main()