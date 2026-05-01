import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset import BrainMRIDataset
from boundary_attention_unet_v2 import BoundaryAttentionUNetV2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===== PATHS =====
IMAGE_DIR = r"D:\for_paper\data\images"   # o'zingnikiga mosla
MASK_DIR = r"D:\for_paper\data\masks"     # o'zingnikiga mosla
MODEL_PATH = r"D:\for_paper\best_boundary_attention_unet_v2.pth"   # kerak bo'lsa o'zgartir

# ===== SETTINGS =====
BATCH_SIZE = 1
VAL_RATIO = 0.2
SEED = 42
NUM_SAVE_SAMPLES = 10
THRESHOLD = 0.5


def denormalize_if_needed(x):
    """
    Agar dataset normalize qilingan bo'lsa, shu yerda qaytarish mumkin.
    Hozircha identity.
    """
    return x


def save_prediction_figure(image, true_mask, pred_mask, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("MRI Image")
    axes[0].axis("off")

    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    timestamp = int(time.time())
    output_dir = os.path.join("outputs", f"boundary_attention_unet_v2_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    dataset = BrainMRIDataset(IMAGE_DIR, MASK_DIR)

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BoundaryAttentionUNetV2(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    saved = 0

    with torch.no_grad():
        for idx, (images, masks) in enumerate(val_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float()

            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            mask_logits, boundary_logits = model(images)
            preds = (torch.sigmoid(mask_logits) > THRESHOLD).float()

            image_np = images[0].detach().cpu().squeeze().numpy()
            image_np = denormalize_if_needed(image_np)

            true_mask_np = masks[0].detach().cpu().squeeze().numpy()
            pred_mask_np = preds[0].detach().cpu().squeeze().numpy()

            save_path = os.path.join(output_dir, f"sample_{idx:03d}.png")
            save_prediction_figure(image_np, true_mask_np, pred_mask_np, save_path)

            saved += 1
            if saved >= NUM_SAVE_SAMPLES:
                break

    print(f"Saved Boundary-Attention U-Net V2 results in: {output_dir}")


if __name__ == "__main__":
    main()