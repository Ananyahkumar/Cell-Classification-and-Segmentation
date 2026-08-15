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

# --- 0. Dataset Definition (Placeholder) ---
class QCNNDataset(Dataset):
    """Custom Dataset for loading image data."""
    def __init__(self, file_paths, labels, image_size=(50, 50), augment=False):
        self.file_paths = file_paths
        self.labels = labels
        
        # Define transforms
        self.base_transform = T.Compose([
            T.Resize(image_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1]
        ])
        
        # Define augmentation transforms
        self.augment_transform = T.Compose([
            T.Resize(image_size),
            T.Grayscale(num_output_channels=1),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB") # Open as RGB first
        
        transform = self.augment_transform if self.augment else self.base_transform
        image = transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# --- 1. Model & Quantum Circuit Definition ---
N_QUBITS = 8
N_CLASSES = 2
N_QLAYERS = 6

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
    """A SIMPLIFIED Hybrid QCNN Model to encourage gradual learning."""
    def __init__(self, q_layer):
        super().__init__()
        # A less powerful classical network to avoid instant overfitting
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
            nn.Linear(32 * 12 * 12, N_QUBITS) 
        )
        self.q_layer = q_layer
        self.post_net = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        x = self.pre_net(x)
        # Scale classical output to a range suitable for angle embedding
        x = torch.tanh(x) * np.pi
        x = self.q_layer(x)
        return self.post_net(x)

# --- 2. Plotting & Visualization Functions ---
def plot_curves(history, smoothing_span=25, path='outputs/plots'):
    """Saves plots of raw data and its smoothed trendline."""
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame(history)
    df_smooth = df.ewm(span=smoothing_span).mean()

    plt.figure(figsize=(18, 7))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(df_smooth['train_loss'], color='tab:blue', label='Smoothed Train Loss')
    plt.plot(df_smooth['val_loss'], color='tab:orange', label='Smoothed Val Loss')
    plt.title('Training & Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yticks(np.arange(0.2, 0.8, 0.2))


    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(df_smooth['train_acc'], color='tab:green', label='Smoothed Train Accuracy')
    plt.plot(df_smooth['val_acc'], color='tab:red', label='Smoothed Val Accuracy')
    plt.title('Training & Validation Accuracy', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(path, f'training_curves_{int(time.time())}.png'))
    print(f"\n✅ Smoothed training curves saved to '{path}' directory.")

def visualize_samples(image_dir, label_file, num_samples_per_class=4):
    """Loads and displays a few images from each class to check for trivial patterns."""
    print("\n🔍 Visualizing data samples...")
    df = pd.read_csv(label_file)
    labels = df['label'].unique()
    
    # Create a dummy dataset just for visualization
    all_files = [os.path.join(image_dir, fname) for fname in df['image_filename']]
    all_labels = df['label'].values
    vis_dataset = QCNNDataset(all_files, all_labels, augment=False)
    
    fig, axs = plt.subplots(len(labels), num_samples_per_class, figsize=(10, 2*len(labels)+1))
    fig.suptitle("Image Samples Per Class (Check for trivial differences!)", fontsize=16)

    for i, label_class in enumerate(labels):
        indices = [idx for idx, l in enumerate(all_labels) if l == label_class]
        sample_indices = np.random.choice(indices, num_samples_per_class, replace=False)
        
        for j, sample_idx in enumerate(sample_indices):
            img_tensor, _ = vis_dataset[sample_idx]
            ax = axs[i, j] if len(labels) > 1 else axs[j]
            # Un-normalize for visualization: (tensor * std) + mean
            img_to_show = img_tensor.squeeze() * 0.5 + 0.5
            ax.imshow(img_to_show, cmap='gray')
            ax.set_title(f"Class: {label_class}")
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig("outputs/plots/data_samples_check.png")
    print("✅ Saved data samples visualization to 'outputs/plots/data_samples_check.png'")
    plt.show()

# --- 3. Main Training Function ---
def train():
    """Main function to run the training and validation process."""
    # --- Configuration ---
    IMAGE_DIR = "predicted_masks_only"
    LABEL_FILE = "labels.csv"
    MODEL_SAVE_PATH = "outputs/models/best_qcnn_model.pth"
    IMAGE_SIZE = (50, 50)
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 5e-4 
    EARLY_STOPPING_PATIENCE = 15
    GRADIENT_CLIP_NORM = 1.0

    print("🚀 Starting QCNN Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preparation ---
    df = pd.read_csv(LABEL_FILE)
    image_files = [os.path.join(IMAGE_DIR, fname) for fname in df['image_filename']]
    labels = df['label'].values
    train_files, val_files, train_labels, val_labels = train_test_split(
        image_files, labels, test_size=0.2, random_state=42, stratify=labels)

    print("Preparing datasets...")
    print(f"Training on {len(train_files)} images, validating on {len(val_files)} images.")
    train_dataset = QCNNDataset(train_files, train_labels, image_size=IMAGE_SIZE, augment=True)
    val_dataset = QCNNDataset(val_files, val_labels, image_size=IMAGE_SIZE, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Optimizer, and Scheduler Initialization ---
    weight_shapes = {"params": (N_QLAYERS, N_QUBITS)}
    q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes=weight_shapes)
    
    model = HybridModel(q_layer).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # --- Training State Variables ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- The Training Loop ---
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            optimizer.step()
            
            total_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_loss = total_train_loss / train_total
        train_acc = 100 * train_correct / train_total

        # --- Validation Step ---
        model.eval()
        total_val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = total_val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | F1: {f1:.2f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # --- Check for Improvement (using validation loss) and Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"    -> 🎉 New best model saved with validation loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n✋ Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break
            
        scheduler.step()

    total_time = time.time() - start_time
    print(f"\n✅ Training complete! Total time: {total_time:.2f}s")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    
    plot_curves(history)

if __name__ == "__main__":
    # It's highly recommended to run this visualization first to check your data!
    # If the classes look trivially different, the model will get ~100% accuracy.
    try:
        visualize_samples(image_dir="predicted_masks_only", label_file="labels.csv")
    except FileNotFoundError:
        print("\n⚠️  Could not find image/label files for visualization. Skipping.")
    except Exception as e:
        print(f"\n⚠️  An error occurred during visualization: {e}. Skipping.")

    # Proceed to train the model
    train()