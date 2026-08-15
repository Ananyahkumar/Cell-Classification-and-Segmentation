import torch
import matplotlib.pyplot as plt
from dataset import LeukemiaSegDataset
from unetpp import UNetPP
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directories
output_dir_plots = "predicted_images"
output_dir_masks = "predicted_masks_only"
os.makedirs(output_dir_plots, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)

# Load trained model
model = UNetPP().to(device)
model.load_state_dict(torch.load("outputs/models/unetpp_final.pth", map_location=device))
model.eval()

# Load samples
dataset = LeukemiaSegDataset("data/images", "data/masks")

N = 280
for i in range(N):
    image, true_mask = dataset[i]
    with torch.no_grad():
        pred_mask = model(image.unsqueeze(0).to(device))
    pred_mask = pred_mask.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.permute(1, 2, 0).numpy())
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(true_mask.squeeze().numpy(),cmap="gray")
    axs[1].set_title("True Mask")
    axs[1].axis("off")
    axs[2].imshow(pred_mask > 0.5, cmap="gray")
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")
    plt.tight_layout()
    
    # Save the figure with original, true, and predicted masks
    plot_path = os.path.join(output_dir_plots, f"prediction_{i+1}.png")
    plt.savefig(plot_path)
    plt.close(fig) # Close the figure to free memory

    # Save only the predicted mask
    mask_path = os.path.join(output_dir_masks, f"predicted_mask_{i+1}.png")
    plt.imsave(mask_path, pred_mask > 0.5, cmap="gray")

print(f"Saved {N} prediction plots to '{output_dir_plots}' folder.")
print(f"Saved {N} predicted masks to '{output_dir_masks}' folder.")
