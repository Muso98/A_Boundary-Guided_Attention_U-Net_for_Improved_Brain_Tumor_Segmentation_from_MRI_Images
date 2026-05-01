import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from dataset import BrainMRIDataset
from attention_unet import AttentionUNet


def compute_confusion_matrix(model, loader, device, threshold=0.5):
    model.eval()

    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > threshold).float()

            tp += ((preds == 1) & (masks == 1)).sum().item()
            tn += ((preds == 0) & (masks == 0)).sum().item()
            fp += ((preds == 1) & (masks == 0)).sum().item()
            fn += ((preds == 0) & (masks == 1)).sum().item()

    return tp, tn, fp, fn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = BrainMRIDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        image_size=(256, 256),
        augment=False
    )

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(
        torch.load(
            "checkpoints/attention_unet/best_attention_unet.pth",
            map_location=device,
            weights_only=True
        )
    )

    tp, tn, fp, fn = compute_confusion_matrix(model, val_loader, device)

    # Raw confusion matrix
    cm = np.array([
        [tp, fn],
        [fp, tn]
    ], dtype=np.float64)

    # Row-wise normalization
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    labels = ["Brain Tumor", "Non-Tumor"]

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 16}
    )

    plt.title("Normalized Confusion Matrix", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()

    save_path = "outputs/confusion_matrix_attention_unet_normalized.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved normalized confusion matrix to: {save_path}")
    print("\nRaw values:")
    print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")

    print("\nNormalized matrix:")
    print(cm_normalized)


if __name__ == "__main__":
    main()