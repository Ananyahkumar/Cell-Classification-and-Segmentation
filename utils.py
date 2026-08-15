import cv2
import numpy as np
import os

def generate_mask(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([120, 50, 50])
    upper = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    cv2.imwrite(save_path, mask)

def batch_generate_masks(img_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    for file in sorted(os.listdir(img_dir)):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            stem, _ = os.path.splitext(file)
            save_path = os.path.join(mask_dir, f"{stem}.jpg")
            if not os.path.exists(save_path):
                generate_mask(os.path.join(img_dir, file), save_path)
