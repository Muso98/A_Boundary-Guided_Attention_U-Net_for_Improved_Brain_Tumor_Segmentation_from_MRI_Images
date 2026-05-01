import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        self.image_files = sorted(
            [
                f for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            ]
        )

        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")

        if augment:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image: {image_name}")

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        mask = (mask > 0.5).astype(np.float32)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)