import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pennylane as qml
from pennylane import numpy as np
import os
import sys # Import the sys module

# --- 1. Model Definition (Must be IDENTICAL to train_qcnn.py) ---
N_QUBITS = 8
N_CLASSES = 2
N_QLAYERS = 6
IMAGE_SIZE = (50, 50)

# Define the quantum device
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_circuit(inputs, params):
    """The quantum circuit that forms the core of the QCNN."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for i in range(N_QLAYERS):
        for j in range(N_QUBITS):
            qml.RY(params[i, j], wires=j)
        for j in range(N_QUBITS - 1):
            qml.CNOT(wires=[j, j + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class HybridModel(nn.Module):
    """A SIMPLIFIED Hybrid QCNN Model to encourage gradual learning.
       This must be IDENTICAL to the one in train_qcnn.py."""
    def __init__(self, q_layer):
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2), # 50x50 -> 25x25
            nn.Dropout(0.4),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2), # 25x25 -> 12x12
            nn.Dropout(0.4),
            
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, N_QUBITS) # Corrected size for 50x50 input
        )
        self.q_layer = q_layer
        self.post_net = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        x = self.pre_net(x)
        # Scale classical output to a range suitable for angle embedding
        x = torch.tanh(x) * np.pi
        x = self.q_layer(x)
        return self.post_net(x)

# --- 2. Classification Function ---
def classify_image_qcnn(image_path, model, transform):
    """Loads an image, processes it, and returns the model's output probabilities."""
    try:
        # Open image, convert to grayscale
        image = Image.open(image_path).convert("L")
        # Apply transformations and add a batch dimension
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(image_tensor)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)[0]
        return probs.numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- 3. Main Execution Block ---
def main():
    """Main function to load the model and classify images in a directory."""
    # --- Configuration ---
    model_path = "outputs/models/best_qcnn_model.pth"
    images_dir = "predicted_masks_only"
    CONFIDENCE_THRESHOLD = 0.75
    OUTPUT_FILE = "classification_results.txt" # Define the output file name

    # --- Redirect stdout to a file ---
    original_stdout = sys.stdout # Save the original stdout
    with open(OUTPUT_FILE, 'w') as f:
        sys.stdout = f # Redirect stdout to the file
        
        # All subsequent print statements will go to the file
        print(f"--- QCNN Classification Results ---")
        print(f"Model: {model_path}")
        print(f"Image Directory: {images_dir}")
        print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")

        # --- Load Model ---
        if not os.path.exists(model_path):
            print(f"FATAL: Model file not found at '{model_path}'.")
            print("Please run train_qcnn.py to generate the model file first.")
            sys.stdout = original_stdout # Restore stdout before exiting
            return
            
        weight_shapes = {"params": (N_QLAYERS, N_QUBITS)}
        q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes=weight_shapes)
        model = HybridModel(q_layer)
        
        print(f"Attempting to load model from: {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(" QCNN Model loaded successfully.")

        # --- Image Transform ---
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # --- Find and Classify Images ---
        if not os.path.isdir(images_dir):
            print(f"FATAL: Image directory '{images_dir}' not found.")
            sys.stdout = original_stdout # Restore stdout before exiting
            return

        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
        print(f"\nFound {len(image_files)} images. Starting classification...")
        
        for image_file in sorted(image_files):
            image_path = os.path.join(images_dir, image_file)
            probs = classify_image_qcnn(image_path, model, transform)
            
            if probs is not None:
                prob_normal, prob_leukemia = probs[0], probs[1]
                
                if prob_leukemia > prob_normal and prob_leukemia > CONFIDENCE_THRESHOLD:
                    prediction = "Leukemia"
                elif prob_normal > prob_leukemia and prob_normal > CONFIDENCE_THRESHOLD:
                    prediction = "Normal"
                else:
                    prediction = "Uncertain"
                
                prob_str = f"[Normal: {prob_normal:.4f}, Leukemia: {prob_leukemia:.4f}]"
                print(f"File: {image_file:<25} | Predicted: {prediction:<10} | Probabilities: {prob_str}")
                
        print("\n--- Classification Complete ---")

    sys.stdout = original_stdout # Restore stdout to the console
    print(f"\nClassification results saved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()