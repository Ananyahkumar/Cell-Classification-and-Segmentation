import os
import cv2
import pandas as pd
from PIL import Image

def generate_cell_dataset(masks_dir, labels_df, output_dir, output_labels_file):
    """Segments cells from masks, crops them, and saves them to a new directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Store data for the new labels CSV
    new_labels_data = []
    cell_count = 0
    
    print("Generating new cell dataset from masks...")

    for index, row in labels_df.iterrows():
        mask_filename = row['image_filename']
        mask_image_path = os.path.join(masks_dir, mask_filename)
        original_label = row['label']

        if not os.path.exists(mask_image_path):
            print(f"Warning: Mask file not found: {mask_image_path}")
            continue

        mask_img_bgr = cv2.imread(mask_image_path)
        mask_img_gray = cv2.cvtColor(mask_img_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(mask_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_cell_area = 100
        padding = 5

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_cell_area:
                x, y, w, h = cv2.boundingRect(contour)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(mask_img_gray.shape[1], x + w + padding)
                y2 = min(mask_img_gray.shape[0], y + h + padding)

                cell_roi_gray = mask_img_gray[y1:y2, x1:x2]
                cell_pil_image = Image.fromarray(cell_roi_gray)
                
                # Create a unique filename for the cropped cell
                new_filename = f"cell_{cell_count:04d}_from_{mask_filename}"
                output_path = os.path.join(output_dir, new_filename)
                cell_pil_image.save(output_path)
                
                new_labels_data.append({'image_filename': new_filename, 'label': original_label})
                cell_count += 1
    
    new_labels_df = pd.DataFrame(new_labels_data)
    new_labels_df.to_csv(output_labels_file, index=False)
    print(f"\nGenerated {cell_count} new cell images and saved labels to '{output_labels_file}'.")


if __name__ == "__main__":
    # Define your original directories and files
    MASKS_DIR = "predicted_masks_only"
    LABELS_FILE = "labels.csv"
    
    # Define the new output directory and labels file
    OUTPUT_CELLS_DIR = "cropped_cells_dataset"
    OUTPUT_LABELS_FILE = "cropped_cells_labels.csv"

    # Load the original labels
    if not os.path.exists(LABELS_FILE):
        print(f"Error: Original labels file '{LABELS_FILE}' not found.")
    else:
        original_labels_df = pd.read_csv(LABELS_FILE)
        generate_cell_dataset(MASKS_DIR, original_labels_df, OUTPUT_CELLS_DIR, OUTPUT_LABELS_FILE)