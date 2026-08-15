import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
from unetpp import UNetPP
from dataset import LeukemiaSegDataset
from torch import nn
import matplotlib.pyplot as plt
from utils import batch_generate_masks
from scipy.ndimage import gaussian_filter1d

def compute_metrics(preds, masks, eps=1e-7):
    # preds, masks: [B, 1, H, W] float tensors, already on CPU
    preds_bin = (preds > 0.5).float()
    masks_bin = (masks > 0.5).float()
    tp = (preds_bin * masks_bin).sum().item()
    fp = (preds_bin * (1 - masks_bin)).sum().item()
    fn = ((1 - preds_bin) * masks_bin).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return precision, recall, f1, iou

# 1. Prepare Data & Masks
if not os.listdir("data/masks"):
    batch_generate_masks("data/images", "data/masks")

dataset = LeukemiaSegDataset("data/images", "data/masks")
# dataset = Subset(dataset, range(10))
val_fraction = 0.2
val_size = int(len(dataset) * val_fraction)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# 2. Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetPP().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 3. Optionally load checkpoint
start_epoch = 0
checkpoint_path = "outputs/models/unetpp_checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {start_epoch}")

# 4. Training + Validation Loop
num_epochs = 20
train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_precisions, train_recalls, train_f1s, train_ious = [], [], [], []
val_precisions, val_recalls, val_f1s, val_ious = [], [], [], []

for epoch in range(start_epoch, start_epoch + num_epochs):
    # TRAIN
    model.train()
    running_loss, running_correct, running_total = 0, 0, 0
    all_train_preds, all_train_masks = [], []

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds_bin = (preds > 0.5).float()
        running_correct += (preds_bin == masks).sum().item()
        running_total += masks.numel()
        all_train_preds.append(preds.cpu())
        all_train_masks.append(masks.cpu())

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * running_correct / running_total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    # metrics
    train_preds_cat = torch.cat(all_train_preds, dim=0)
    train_masks_cat = torch.cat(all_train_masks, dim=0)
    t_prec, t_rec, t_f1, t_iou = compute_metrics(train_preds_cat, train_masks_cat)
    train_precisions.append(t_prec)
    train_recalls.append(t_rec)
    train_f1s.append(t_f1)
    train_ious.append(t_iou)

    # VALIDATION
    model.eval()
    val_loss_sum, val_correct, val_total = 0, 0, 0
    all_val_preds, all_val_masks = [], []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            val_loss_sum += loss.item()
            preds_bin = (preds > 0.5).float()
            val_correct += (preds_bin == masks).sum().item()
            val_total += masks.numel()
            all_val_preds.append(preds.cpu())
            all_val_masks.append(masks.cpu())

    val_loss = val_loss_sum / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    # metrics
    val_preds_cat = torch.cat(all_val_preds, dim=0)
    val_masks_cat = torch.cat(all_val_masks, dim=0)
    v_prec, v_rec, v_f1, v_iou = compute_metrics(val_preds_cat, val_masks_cat)
    val_precisions.append(v_prec)
    val_recalls.append(v_rec)
    val_f1s.append(v_f1)
    val_ious.append(v_iou)

    print(f"Epoch {epoch+1}/{start_epoch+num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f}%, Prec: {t_prec:.3f}, Rec: {t_rec:.3f}, F1: {t_f1:.3f}, IoU: {t_iou:.3f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}%, Prec: {v_prec:.3f}, Rec: {v_rec:.3f}, F1: {v_f1:.3f}, IoU: {v_iou:.3f}")

    # Save checkpoint
    os.makedirs("outputs/models", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

# 5. Final Model Save
torch.save(model.state_dict(), "outputs/models/unetpp_final.pth")

# 6. Plot Loss & Accuracy
train_losses_sm = gaussian_filter1d(train_losses, sigma=5)
val_losses_sm = gaussian_filter1d(val_losses, sigma=5)
train_accs_sm = gaussian_filter1d(train_accs, sigma=5)
val_accs_sm = gaussian_filter1d(val_accs, sigma=5)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses_sm, label='Train Loss', color='red')
plt.plot(val_losses_sm, label='Val Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs_sm, label='Train Accuracy', color='blue')
plt.plot(val_accs_sm, label='Val Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

os.makedirs("outputs", exist_ok=True)
plt.tight_layout()
plt.savefig("outputs/training_validation_plot_unetpp.png")
plt.show()

