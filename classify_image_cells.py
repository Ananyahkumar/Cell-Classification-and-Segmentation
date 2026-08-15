# (Full code, identical to your last version but with the correct model architecture)
# ... (all imports)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import torchvision.transforms as T
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
import os
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter1d 
from torch.utils.data import Dataset 
import pennylane as qml
import torch.nn as nn
import cv2

# --- 1. Model Definition (Must be IDENTICAL to train_qcnn.py) ---
N_QUBITS = 8
N_CLASSES = 2
N_QLAYERS = 6
IMAGE_SIZE = (50, 50)

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_circuit(inputs, params):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for i in range(N_QLAYERS):
        for j in range(N_QUBITS):
            qml.RY(params[i, j], wires=j)
        for j in range(N_QUBITS - 1):
            qml.CNOT(wires=[j, j + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class HybridModel(nn.Module):
    def __init__(self, q_layer):
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.4),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, N_QUBITS)
        )
        self.q_layer = q_layer
        self.post_net = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        x = self.pre_net(x)
        x = torch.tanh(x) * np.pi
        x = self.q_layer(x)
        return self.post_net(x)

# ... (rest of the functions: segment_cells_from_mask, classify_single_cell, visualize_predictions)

# --- Cell Segmentation Function ---
def segment_cells_from_mask(mask_image_path):
    """
    Loads a mask image, finds contours, and extracts individual cell images
    directly from the provided mask.
    """
    import cv2 # Ensure OpenCV is imported if not already at the top
    from PIL import Image # Ensure PIL is imported if not already at the top
    
    mask_img_bgr = cv2.imread(mask_image_path)
    if mask_img_bgr is None:
        print(f"Error: Could not load mask image at {mask_image_path}")
        return []
    
    mask_img_gray = cv2.cvtColor(mask_img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cell_data = []
    min_cell_area = 100
    padding = 5

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_cell_area:
            x, y, w, h = cv2.boundingRect(contour)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(mask_img_gray.shape[1], x + w + padding)
            y2 = min(mask_img_gray.shape[0], y + h + padding)

            cell_roi_gray = mask_img_gray[y1:y2, x1:x2]
            cell_pil_image = Image.fromarray(cell_roi_gray)
            
            cell_data.append((cell_pil_image, (x, y, w, h)))
    return cell_data

# --- Classification Function ---
def classify_single_cell(cell_pil_image, model, transform):
    """
    Classifies a single cropped cell image using the QCNN model.
    """
    import torch.nn.functional as F
    import torch

    try:
        # Apply transforms and add a batch dimension
        image_tensor = transform(cell_pil_image).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(image_tensor)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)[0]
        return probs.numpy()
    except Exception as e:
        print(f"Error classifying cell: {e}")
        return None

# --- Visualization Function ---
def visualize_predictions(image_to_annotate, cell_results, output_path="outputs/annotated_images"):
    """
    Draws bounding boxes, cell numbers, and predictions on the image.
    Args:
        image_to_annotate (np.array): The OpenCV image (BGR) to draw on.
        cell_results (list): List of (bbox, prediction_text, color) for each cell.
        output_path (str): Directory to save the annotated image.
    """
    import cv2 # Make sure cv2 is imported if it's not at the top
    import os

    os.makedirs(output_path, exist_ok=True)
    annotated_image = image_to_annotate.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    for i, (bbox, prediction_text, color) in enumerate(cell_results):
        x, y, w, h = bbox
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)

        # Prepare text for cell number and prediction
        cell_label = f"{i+1}"
        
        # Determine text position (above the box)
        text_x = x
        text_y = y - 5 if y - 5 > 10 else y + h + 15

        # Draw text
        cv2.putText(annotated_image, cell_label, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
        
    # Save the annotated image
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency())
    output_filename = f"annotated_cells_{timestamp}.png"
    cv2.imwrite(os.path.join(output_path, output_filename), annotated_image)
    print(f"\nAnnotated image saved to '{os.path.join(output_path, output_filename)}'")

# --- Main Execution Block ---
def main():
    # --- Configuration ---
    model_path = "outputs/models/best_qcnn_cell_model.pth" # Point to the new model
    CONFIDENCE_THRESHOLD = 0.60
    
    MASK_ONLY_DIR = r"C:\Users\anany\OneDrive\Desktop\Unet++ final (segmentation)\predicted_masks_only" 

    print(f"--- QCNN Cell-Level Classifier with Visualization ---")
    print(f"Model: {model_path}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")

    if not os.path.exists(model_path):
        print(f"FATAL: Model file not found at '{model_path}'.")
        print("Please run train_qcnn.py to generate the model file first.")
        return
        
    weight_shapes = {"params": (N_QLAYERS, N_QUBITS)}
    q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes=weight_shapes)
    model = HybridModel(q_layer)
    
    print(f"Attempting to load model from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("✅ QCNN Model loaded successfully.\n")

    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    while True:
        mask_filename = input(f"Enter the filename of the U-Net++ mask image (e.g., 'predicted_mask_16.png') to classify cells from (or 'q' to quit): ").strip()
        if mask_filename.lower() == 'q':
            break

        if not mask_filename:
            print("No filename entered. Please try again.")
            continue
            
        mask_image_path = os.path.join(MASK_ONLY_DIR, mask_filename)

        if not os.path.exists(mask_image_path):
            print(f"Error: Mask image file '{mask_image_path}' does not exist. Please check the path and filename.")
            continue

        print(f"\nProcessing mask image: {os.path.basename(mask_image_path)}")
        
        image_for_annotation = cv2.imread(mask_image_path) 
        if image_for_annotation is None:
            print(f"Error: Could not load image for annotation at {mask_image_path}. Skipping visualization.")
            continue
        
        image_for_annotation = np.copy(image_for_annotation) 

        cell_data = segment_cells_from_mask(mask_image_path)
        
        if not cell_data:
            print("No cells detected in the mask or error during segmentation. Check min_cell_area or mask quality.")
            continue

        print(f"Detected {len(cell_data)} cells. Classifying each cell...")
        leukemic_count = 0
        normal_count = 0
        uncertain_count = 0
        visualization_results = [] 

        for i, (cell_pil_image, bbox) in enumerate(cell_data):
            probs = classify_single_cell(cell_pil_image, model, transform)
            
            if probs is not None:
                prob_normal, prob_leukemia = probs[0], probs[1]
                
                prediction = "Uncertain"
                color = (128, 128, 128)
                if prob_leukemia > prob_normal and prob_leukemia > CONFIDENCE_THRESHOLD:
                    prediction = "Leukemia"
                    leukemic_count += 1
                    color = (0, 0, 255)
                elif prob_normal > prob_leukemia and prob_normal > CONFIDENCE_THRESHOLD:
                    prediction = "Normal"
                    normal_count += 1
                    color = (0, 255, 0)
                else:
                    uncertain_count += 1
                
                prob_str = f"[Normal: {prob_normal:.4f}, Leukemia: {prob_leukemia:.4f}]"
                print(f"  Cell {i+1} (bbox {bbox}): Predicted: {prediction:<10} | Probabilities: {prob_str}")
                
                visualization_results.append((bbox, prediction, color))

        print("\n--- Summary for this image ---")
        print(f"  Total cells classified: {len(cell_data)}")
        print(f"  Normal cells: {normal_count}")
        print(f"  Leukemic cells: {leukemic_count}")
        print(f"  Uncertain cells: {uncertain_count}")
        print("-------------------------------\n")
        
        if image_for_annotation is not None:
            visualize_predictions(image_for_annotation, visualization_results)
        else:
            print("Skipping visualization as the image for annotation could not be loaded.")

if __name__ == "__main__":
    main()