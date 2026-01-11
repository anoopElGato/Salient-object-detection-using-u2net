"""
Dataset and DataLoader for Salient Object Detection
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SaliencyDataset(Dataset):
    """
    Simplified dataset - assumes images and masks have same names
    """

    def __init__(self, image_folder, mask_folder, img_size=320):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.img_size = img_size

        # Get image files
        self.images = sorted([f for f in os.listdir(image_folder) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        print(f"Loaded {len(self.images)} images from {image_folder}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Load image
        img_path = os.path.join(self.image_folder, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Load mask (try different extensions)
        base_name = os.path.splitext(img_name)[0]
        mask_path = None

        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(self.mask_folder, base_name + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break

        if mask_path is None:
            raise FileNotFoundError(f"No mask found for {img_name}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # H,W,C -> C,H,W
        mask = torch.from_numpy(mask).unsqueeze(0)  # H,W -> 1,H,W

        return image, mask, img_name


def create_dataloaders(train_img_dir, train_mask_dir, 
                       val_img_dir, val_mask_dir,
                       batch_size=8, num_workers=4, img_size=320):
    """
    Create train and validation dataloaders

    Args:
        train_img_dir: Path to training images
        train_mask_dir: Path to training masks
        val_img_dir: Path to validation images
        val_mask_dir: Path to validation masks
        batch_size: Batch size
        num_workers: Number of workers for data loading
        img_size: Image size
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = SaliencyDataset(
        train_img_dir, 
        train_mask_dir,
        img_size=img_size
    )

    val_dataset = SaliencyDataset(
        val_img_dir,
        val_mask_dir,
        img_size=img_size
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
