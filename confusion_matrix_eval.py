import torch
import numpy as np
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
        torch.load("checkpoints/attention_unet/best_attention_unet.pth",
                   map_location=device,
                   weights_only=True)
    )

    tp, tn, fp, fn = compute_confusion_matrix(model, val_loader, device)

    print("\nConfusion Matrix (pixel-level):")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")

    cm = np.array([[tn, fp],
                   [fn, tp]])

    print("\nMatrix format:")
    print(cm)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\nDerived Metrics:")
    print(f"Accuracy    : {accuracy:.6f}")
    print(f"Precision   : {precision:.6f}")
    print(f"Recall      : {recall:.6f}")
    print(f"Specificity : {specificity:.6f}")
    print(f"F1-score    : {f1:.6f}")


if __name__ == "__main__":
    main()