import os
import torch
import matplotlib.pyplot as plt
from dataset import BrainMRIDataset
from model import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BrainMRIDataset("data/images", "data/masks", image_size=(256, 256))
    image, mask = dataset[0]

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("checkpoints/unet/best_unet_dice_0.8113.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device))
        pred = torch.sigmoid(pred).cpu().squeeze().numpy()

    image_np = image.squeeze().numpy()
    mask_np = mask.squeeze().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("MRI Image")
    plt.imshow(image_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred > 0.5, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()