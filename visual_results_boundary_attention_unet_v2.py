import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, random_split

from dataset import BrainMRIDataset
from boundary_attention_unet_v2 import BoundaryAttentionUNetV2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = r"D:\for_paper\data\images"
MASK_DIR  = r"D:\for_paper\data\masks"
MODEL_PATH = r"D:\for_paper\best_boundary_attention_unet_v2.pth"

NUM_SAMPLES = 6
VAL_RATIO = 0.2
SEED = 42
THRESHOLD = 0.5


def get_bbox(mask: np.ndarray):
    """
    mask: 2D binary numpy array
    returns (x_min, y_min, width, height) or None
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, y_min, (x_max - x_min + 1), (y_max - y_min + 1)


def draw_overlay(ax, image, gt_mask, pred_mask, show_gt_bbox=True, show_pred_bbox=True):
    ax.imshow(image, cmap="gray")

    # Ground truth contour
    if gt_mask.sum() > 0:
        ax.contour(gt_mask, levels=[0.5], colors="lime", linewidths=1.5)

    # Prediction contour
    if pred_mask.sum() > 0:
        ax.contour(pred_mask, levels=[0.5], colors="red", linewidths=1.5)

    # Ground truth bounding box
    if show_gt_bbox:
        gt_bbox = get_bbox(gt_mask)
        if gt_bbox is not None:
            x, y, w, h = gt_bbox
            rect = Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1.8, linestyle="--")
            ax.add_patch(rect)

    # Prediction bounding box
    if show_pred_bbox:
        pred_bbox = get_bbox(pred_mask)
        if pred_bbox is not None:
            x, y, w, h = pred_bbox
            rect = Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1.8)
            ax.add_patch(rect)

    ax.axis("off")


def main():
    os.makedirs("outputs/visual_results_bbox", exist_ok=True)

    dataset = BrainMRIDataset(IMAGE_DIR, MASK_DIR)

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = BoundaryAttentionUNetV2().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    saved = 0

    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE).float()

            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            pred_logits, _ = model(img)
            pred = (torch.sigmoid(pred_logits) > THRESHOLD).float()

            img_np = img[0].cpu().squeeze().numpy()
            gt_np = mask[0].cpu().squeeze().numpy()
            pred_np = pred[0].cpu().squeeze().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("MRI Image")
            axes[0].axis("off")

            axes[1].imshow(gt_np, cmap="gray")
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis("off")

            axes[2].imshow(pred_np, cmap="gray")
            axes[2].set_title("Predicted Mask")
            axes[2].axis("off")

            draw_overlay(axes[3], img_np, gt_np, pred_np, show_gt_bbox=True, show_pred_bbox=True)
            axes[3].set_title("Overlay + Bounding Boxes")

            plt.tight_layout()
            save_path = os.path.join("outputs/visual_results_bbox", f"sample_{idx:03d}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            saved += 1
            if saved >= NUM_SAMPLES:
                break

    print("Saved visual results with overlay and bounding boxes in outputs/visual_results_bbox/")


if __name__ == "__main__":
    main()