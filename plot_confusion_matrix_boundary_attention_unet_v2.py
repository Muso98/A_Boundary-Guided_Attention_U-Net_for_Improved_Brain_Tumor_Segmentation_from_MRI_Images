import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split

from dataset import BrainMRIDataset
from boundary_attention_unet_v2 import BoundaryAttentionUNetV2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGE_DIR = r"D:\for_paper\data\images"
MASK_DIR = r"D:\for_paper\data\masks"
MODEL_PATH = r"D:\for_paper\best_boundary_attention_unet_v2.pth"

BATCH_SIZE = 4
VAL_RATIO = 0.2
SEED = 42
THRESHOLD = 0.5


def compute_confusion_matrix(model, loader, device, threshold=0.5):
    model.eval()

    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).float()

            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            mask_logits, _ = model(images)
            preds = (torch.sigmoid(mask_logits) > threshold).float()

            preds = preds.view(-1)
            masks = masks.view(-1)

            tp += torch.sum((preds == 1) & (masks == 1)).item()
            tn += torch.sum((preds == 0) & (masks == 0)).item()
            fp += torch.sum((preds == 1) & (masks == 0)).item()
            fn += torch.sum((preds == 0) & (masks == 1)).item()

    return tp, tn, fp, fn


def main():
    os.makedirs("outputs", exist_ok=True)

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

    tp, tn, fp, fn = compute_confusion_matrix(model, val_loader, DEVICE, THRESHOLD)

    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=np.int64)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix - Boundary-Attention U-Net")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_path = "outputs/confusion_matrix_boundary_attention_unet_v2.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix to: {save_path}")
    print("\nRaw values:")
    print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")


if __name__ == "__main__":
    main()