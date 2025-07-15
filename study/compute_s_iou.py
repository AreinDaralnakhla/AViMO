import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

# Define the IoU computation function
def compute_iou(gt_mask, recon_mask):
    intersection = torch.sum((gt_mask * recon_mask) > 0).item()
    union = torch.sum((gt_mask + recon_mask) > 0).item()
    iou = intersection / union if union > 0 else 0
    return iou

# Define paths
gt_masks_dir = "/home/da10546y/PREMIEREMulti/results/subject_1/masks"  # Ground truth masks
model1_masks_dir = "/home/da10546y/PREMIEREMulti/recon_imgs/econ/masks"  # Model 1 masks
model2_masks_dir = "/home/da10546y/PREMIEREMulti/recon_imgs/exavatar/masks"  # Model 2 masks
model3_masks_dir = "/home/da10546y/PREMIEREMulti/recon_imgs/vid2avatar/masks"  # Model 3 masks

# Transform to resize and convert masks to tensors
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match the image size
    transforms.ToTensor()  # Convert to tensor
])

# Load and process ground truth mask for frame 0
gt_mask_path = os.path.join(gt_masks_dir, "0.png")
gt_mask = mask_transform(Image.open(gt_mask_path).convert("L"))
gt_mask = (gt_mask > 0.5).float()  # Binarize the mask

# Load and process Model 1 mask for frame 0
model1_mask_path = os.path.join(model1_masks_dir, "0.png")
model1_mask = mask_transform(Image.open(model1_mask_path).convert("L"))
model1_mask = (model1_mask > 0.5).float()

# Pad Model 1 mask (2 pixels from the top and 2 pixels from the right)
padding = (0, 12, 6, 0)  # (left, right, top, bottom)
model1_mask = torch.nn.functional.pad(model1_mask, padding, mode="constant", value=0)

# Resize the padded mask back to (256, 256)
model1_mask = torch.nn.functional.interpolate(model1_mask.unsqueeze(0), size=(256, 256), mode="nearest").squeeze(0)

# Load and process Model 2 mask for frame 0
model2_mask_path = os.path.join(model2_masks_dir, "0.png")
model2_mask = mask_transform(Image.open(model2_mask_path).convert("L"))
model2_mask = (model2_mask > 0.5).float()

# Load and process Model 3 mask for frame 0
model3_mask_path = os.path.join(model3_masks_dir, "0.png")
model3_mask = mask_transform(Image.open(model3_mask_path).convert("L"))
model3_mask = (model3_mask > 0.5).float()

# Compute IoU for all models
iou_model1 = compute_iou(gt_mask, model1_mask)
iou_model2 = compute_iou(gt_mask, model2_mask)
iou_model3 = compute_iou(gt_mask, model3_mask)

# Visualize the masks with ground truth overlay
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Overlay ground truth on Model 1 mask
overlay_model1 = model1_mask.squeeze().numpy() + gt_mask.squeeze().numpy() * 0.5
axes[0].imshow(overlay_model1, cmap="Reds", alpha=0.7)
axes[0].set_title(f"ECON\nIoU: {iou_model1:.4f}")
axes[0].axis("off")

# Overlay ground truth on Model 2 mask
overlay_model2 = model2_mask.squeeze().numpy() + gt_mask.squeeze().numpy() * 0.5
axes[1].imshow(overlay_model2, cmap="Greens", alpha=0.7)
axes[1].set_title(f"ExAvatar\nIoU: {iou_model2:.4f}")
axes[1].axis("off")

# Overlay ground truth on Model 3 mask
overlay_model3 = model3_mask.squeeze().numpy() + gt_mask.squeeze().numpy() * 0.5
axes[2].imshow(overlay_model3, cmap="Blues", alpha=0.7)
axes[2].set_title(f"Vid2Avatar\nIoU: {iou_model3:.4f}")
axes[2].axis("off")

plt.suptitle("Frame 0")
plt.tight_layout()
plt.show()

# Print IoU scores for frame 0
print(f"Frame 0 IoU Scores:")
print(f"Model 1: {iou_model1:.4f}")
print(f"Model 2: {iou_model2:.4f}")
print(f"Model 3: {iou_model3:.4f}")