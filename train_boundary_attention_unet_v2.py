import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from dataset import BrainMRIDataset
from boundary_attention_unet_v2 import BoundaryAttentionUNetV2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===== PATHS =====
IMAGE_DIR = r"D:\for_paper\data\images"     # <-- o'zingnikiga moslab qo'y
MASK_DIR  = r"D:\for_paper\data\masks"      # <-- o'zingnikiga moslab qo'y

SAVE_PATH = "best_boundary_attention_unet_v2.pth"
PRETRAINED_ATTENTION_PATH = r"D:\for_paper\checkpoints\attention_unet\best_attention_unet.pth"
# ===== HYPERPARAMETERS =====
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 25
VAL_RATIO = 0.2
BOUNDARY_WEIGHT = 0.10
BOUNDARY_DILATION = 3
SEED = 42


def mask_to_boundary(mask, dilation=3):
    """
    mask: [B,1,H,W] binary tensor
    returns thicker boundary targets
    """
    mask = mask.float()

    # Erode-like operation
    eroded = 1 - F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
    boundary = mask - eroded
    boundary = (boundary > 0).float()

    # Thicken the boundary target
    if dilation > 1:
        boundary = F.max_pool2d(boundary, kernel_size=dilation, stride=1, padding=dilation // 2)
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


def compute_total_loss(mask_logits, boundary_logits, masks, boundary_targets,
                       seg_bce, seg_dice, boundary_bce, boundary_dice):
    seg_loss = seg_bce(mask_logits, masks) + seg_dice(mask_logits, masks)

    # Auxiliary boundary loss
    b_loss = boundary_bce(boundary_logits, boundary_targets) + boundary_dice(boundary_logits, boundary_targets)

    total_loss = seg_loss + BOUNDARY_WEIGHT * b_loss
    return total_loss, seg_loss.item(), b_loss.item()


def load_pretrained_attention_weights(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"[INFO] Pretrained checkpoint not found: {ckpt_path}")
        print("[INFO] Training will start from scratch.")
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_dict = model.state_dict()
    matched = {}
    skipped = []

    for k, v in checkpoint.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)

    model_dict.update(matched)
    model.load_state_dict(model_dict)

    print(f"[INFO] Loaded {len(matched)} matching layers from pretrained Attention U-Net.")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} unmatched layers.")


def train_one_epoch(model, loader, optimizer, seg_bce, seg_dice, boundary_bce, boundary_dice):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).float()

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        boundary_targets = mask_to_boundary(masks, dilation=BOUNDARY_DILATION)

        optimizer.zero_grad()

        mask_logits, boundary_logits = model(images)

        loss, _, _ = compute_total_loss(
            mask_logits, boundary_logits, masks, boundary_targets,
            seg_bce, seg_dice, boundary_bce, boundary_dice
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_dice += dice_score_from_logits(mask_logits, masks)

    return running_loss / len(loader), running_dice / len(loader)


@torch.no_grad()
def validate(model, loader, seg_bce, seg_dice, boundary_bce, boundary_dice):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).float()

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        boundary_targets = mask_to_boundary(masks, dilation=BOUNDARY_DILATION)

        mask_logits, boundary_logits = model(images)

        loss, _, _ = compute_total_loss(
            mask_logits, boundary_logits, masks, boundary_targets,
            seg_bce, seg_dice, boundary_bce, boundary_dice
        )

        dice = dice_score_from_logits(mask_logits, masks)

        running_loss += loss.item()
        running_dice += dice

    return running_loss / len(loader), running_dice / len(loader)


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dataset = BrainMRIDataset(IMAGE_DIR, MASK_DIR)

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = BoundaryAttentionUNetV2(in_channels=1, out_channels=1).to(DEVICE)

    # Load pretrained baseline Attention U-Net weights
    load_pretrained_attention_weights(model, PRETRAINED_ATTENTION_PATH)

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    seg_bce = nn.BCEWithLogitsLoss()
    seg_dice = DiceLoss()
    boundary_bce = nn.BCEWithLogitsLoss()
    boundary_dice = DiceLoss()

    best_dice = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer,
            seg_bce, seg_dice, boundary_bce, boundary_dice
        )

        val_loss, val_dice = validate(
            model, val_loader,
            seg_bce, seg_dice, boundary_bce, boundary_dice
        )

        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - "
            f"LR: {current_lr:.6f} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Train Dice: {train_dice:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print("Best Boundary-Attention U-Net V2 model saved.")

    print("Training finished.")
    print(f"Best Boundary-Attention U-Net V2 Val Dice: {best_dice:.6f}")


if __name__ == "__main__":
    main()