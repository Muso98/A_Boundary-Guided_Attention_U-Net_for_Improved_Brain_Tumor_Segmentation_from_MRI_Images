import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from dataset import BrainMRIDataset
from boundary_attention_unet import BoundaryAttentionUNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==== PATHS ====
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"
SAVE_PATH = "best_boundary_attention_unet.pth"

# ==== HYPERPARAMETERS ====
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 50
VAL_RATIO = 0.2
BOUNDARY_WEIGHT = 0.3


def mask_to_boundary(mask):
    """
    mask: [B,1,H,W] binary tensor
    """
    mask = mask.float()
    eroded = 1 - F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
    boundary = mask - eroded
    boundary = (boundary > 0).float()
    return boundary


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


def dice_score_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def train_one_epoch(model, loader, optimizer, bce_loss, dice_loss):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).float()

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        boundary_targets = mask_to_boundary(masks)

        optimizer.zero_grad()

        mask_logits, boundary_logits = model(images)

        seg_loss = bce_loss(mask_logits, masks) + dice_loss(mask_logits, masks)
        boundary_loss = bce_loss(boundary_logits, boundary_targets)

        loss = seg_loss + BOUNDARY_WEIGHT * boundary_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, bce_loss, dice_loss):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).float()

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        boundary_targets = mask_to_boundary(masks)

        mask_logits, boundary_logits = model(images)

        seg_loss = bce_loss(mask_logits, masks) + dice_loss(mask_logits, masks)
        boundary_loss = bce_loss(boundary_logits, boundary_targets)

        loss = seg_loss + BOUNDARY_WEIGHT * boundary_loss
        dice = dice_score_from_logits(mask_logits, masks)

        running_loss += loss.item()
        running_dice += dice

    return running_loss / len(loader), running_dice / len(loader)


def main():
    dataset = BrainMRIDataset(IMAGE_DIR, MASK_DIR)

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BoundaryAttentionUNet(in_channels=1, out_channels=1).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LR)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    best_dice = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, dice_loss)
        val_loss, val_dice = validate(model, val_loader, bce_loss, dice_loss)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print("Best Boundary-Attention U-Net model saved.")

    print("Training finished.")
    print(f"Best Boundary-Attention U-Net Val Dice: {best_dice}")


if __name__ == "__main__":
    main()