import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from dataset import BrainMRIDataset
from model import UNet


def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return (2.0 * intersection + smooth) / (union + smooth)


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def combined_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    d_loss = dice_loss(pred, target)
    return bce + d_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_dir = "data/images"
    mask_dir = "data/masks"

    # Bir xil datasetning 2 xil versiyasi:
    # train uchun augmentation bor
    # validation uchun augmentation yo‘q
    full_dataset = BrainMRIDataset(image_dir, mask_dir, image_size=(256, 256), augment=False)
    train_dataset_aug = BrainMRIDataset(image_dir, mask_dir, image_size=(256, 256), augment=True)
    val_dataset_plain = BrainMRIDataset(image_dir, mask_dir, image_size=(256, 256), augment=False)

    dataset_size = len(full_dataset)

    # Reproducible split
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()

    train_size = int(0.8 * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_dataset_aug, train_indices)
    val_dataset = Subset(val_dataset_plain, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    best_val_dice = 0.0

    os.makedirs("checkpoints/unet_aug", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = combined_loss(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_score(outputs, masks).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"Val Dice: {avg_val_dice:.4f}"
        )

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "checkpoints/unet_aug/best_unet_aug.pth")
            print("Best augmented U-Net model saved.")

    print("Training finished.")
    print("Best Augmented U-Net Val Dice:", best_val_dice)


if __name__ == "__main__":
    main()