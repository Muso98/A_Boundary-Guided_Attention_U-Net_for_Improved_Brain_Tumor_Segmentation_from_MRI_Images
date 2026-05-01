import os
import time
import torch
import matplotlib.pyplot as plt
from dataset import BrainMRIDataset
from attention_unet import AttentionUNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_path = "checkpoints/attention_unet/best_attention_unet.pth"

    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    dataset = BrainMRIDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        image_size=(256, 256),
        augment=False
    )

    run_name = f"attention_unet_run_{int(time.time())}"
    save_dir = os.path.join("outputs", run_name)
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(5, len(dataset))

    for i in range(num_samples):
        image, mask = dataset[i]

        image_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_input)
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()

        image_np = image.squeeze().numpy()
        mask_np = mask.squeeze().numpy()
        pred_bin = (pred > 0.5).astype(float)

        plt.figure(figsize=(14, 4))

        plt.subplot(1, 4, 1)
        plt.title("MRI Image")
        plt.imshow(image_np, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Ground Truth")
        plt.imshow(mask_np, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_bin, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("Prediction Overlay")
        plt.imshow(image_np, cmap="gray")
        plt.imshow(pred_bin, alpha=0.45, cmap="jet")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"result_{i}.png"), dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Saved Attention U-Net results in: {save_dir}")


if __name__ == "__main__":
    main()