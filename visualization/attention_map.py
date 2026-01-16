import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from torch.utils.data import DataLoader
from models.net import HDAN
from data.loader import H5Dataset
from config import config

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

HEATMAP_CMAP = 'jet'

plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14

MODEL_PATH = r"F:\PretermInfant\checkpoints\Ours\BestInfantSeg_ISEG1.pth"
SAVE_DIR = "./vis_results_jet"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention_maps = []


def hook_fn(module, input, output):
    attn_map = torch.sigmoid(output)
    attention_maps.append(attn_map.detach().cpu())


def visualize():
    print(f"Loading model from {MODEL_PATH}...")
    model = HDAN(num_init_features=32, growth_rate=16,
                              block_config=(4, 4, 4, 4),
                              drop_rate=0.2, num_classes=4).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    target_layer = model.attn.spatial_attn.spatial_cov
    target_layer.register_forward_hook(hook_fn)

    val_data = H5Dataset(config.valdata_path, mode='val')
    loader = DataLoader(val_data, batch_size=1, shuffle=False)

    print("Running inference...")
    for i, (image, target) in enumerate(loader):
        if i >= 3: break

        image = image.to(device)
        attention_maps.clear()

        output = model(image)

        D = image.shape[2]
        mid_slice = D // 2
        img_slice = image[0, 0, mid_slice, :, :].cpu().numpy()

        if len(attention_maps) > 0:
            attn_vol = attention_maps[0]
            attn_slice = attn_vol[0, 0, mid_slice, :, :].numpy()
        else:
            return

        fig = plt.figure(figsize=(15, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("(a) Original T1 MRI", y=-0.15)
        ax1.imshow(img_slice, cmap='gray')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("(b) Spatial Attention Map", y=-0.15)
        im = ax2.imshow(attn_slice, cmap=HEATMAP_CMAP)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title("(c) Overlay", y=-0.15)
        ax3.imshow(img_slice, cmap='gray')
        ax3.imshow(attn_slice, cmap=HEATMAP_CMAP, alpha=0.5)
        ax3.axis('off')

        ax_cbar = fig.add_subplot(gs[0, 3])
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=12)

        save_path = os.path.join(SAVE_DIR, f"Professional_Jet_Sample_{i}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"Saved: {save_path}")
        plt.close(fig)


if __name__ == "__main__":
    with torch.no_grad():
        visualize()