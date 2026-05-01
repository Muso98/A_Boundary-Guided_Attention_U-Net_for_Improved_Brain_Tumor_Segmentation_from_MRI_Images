import os
import torch
import numpy as np
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
BATCH_SIZE = 4
VAL_RATIO = 0.2
SEED = 42
THRESHOLD = 0.5
EPS = 1e-8


def compute_confusion_matrix(model, loader, device, threshold=0.5):
    model.eval()

    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).float()

            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            mask_logits, boundary_logits = model(images)
            preds = (torch.sigmoid(mask_logits) > threshold).float()

            preds = preds.view(-1)
            masks = masks.view(-1)

            tp += torch.sum((preds == 1) & (masks == 1)).item()
            tn += torch.sum((preds == 0) & (masks == 0)).item()
            fp += torch.sum((preds == 1) & (masks == 0)).item()
            fn += torch.sum((preds == 0) & (masks == 1)).item()

    return tp, tn, fp, fn


def compute_metrics(tp, tn, fp, fn, eps=1e-8):
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1_score = (2 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "iou": iou,
        "dice": dice
    }


def main():
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

    metrics = compute_metrics(tp, tn, fp, fn, EPS)

    print("\nConfusion Matrix (pixel-level):")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")

    print("\nMatrix format:")
    print(np.array([[tn, fp],
                    [fn, tp]]))

    print("\nDerived Metrics:")
    print(f"Accuracy    : {metrics['accuracy']:.6f}")
    print(f"Precision   : {metrics['precision']:.6f}")
    print(f"Recall      : {metrics['recall']:.6f}")
    print(f"Specificity : {metrics['specificity']:.6f}")
    print(f"F1-score    : {metrics['f1_score']:.6f}")
    print(f"IoU         : {metrics['iou']:.6f}")
    print(f"Dice        : {metrics['dice']:.6f}")


if __name__ == "__main__":
    main()