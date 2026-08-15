import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class LeukemiaSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def _resolve_mask_path(self, img_filename):
        stem = os.path.splitext(img_filename)[0]
        candidates = [
            os.path.join(self.mask_dir, img_filename),
            os.path.join(self.mask_dir, f"{stem}.jpg"),
            os.path.join(self.mask_dir, f"{stem}.png"),
            os.path.join(self.mask_dir, f"{stem}_mask.png"),
            os.path.join(self.mask_dir, f"{stem}_mask.jpg"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Mask not found for image '{img_filename}' in '{self.mask_dir}'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = self._resolve_mask_path(img_filename)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.resize(image, (256, 256)) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = cv2.resize(mask, (256, 256)) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
