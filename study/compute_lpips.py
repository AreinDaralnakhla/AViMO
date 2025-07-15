import lpips
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex')  # Use AlexNet backbone (default)

# Load images
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a fixed size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Load ground truth and reconstructed images
img_gt = transform(Image.open("/home/da10546y/PREMIEREMulti/results/subject_1/frames/0.png")).unsqueeze(0)  # Ground truth
img1 = transform(Image.open("/home/da10546y/PREMIEREMulti/recon_imgs/econ/reprojections/0.png")).unsqueeze(0)  # Model 1
img2 = transform(Image.open("/home/da10546y/PREMIEREMulti/recon_imgs/exavatar/reprojections/0.png")).unsqueeze(0)  # Model 2
img3 = transform(Image.open("/home/da10546y/PREMIEREMulti/recon_imgs/vid2avatar/reprojections/0.png")).unsqueeze(0)  # Model 3

# Load and resize the GT mask
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match the image size
    transforms.ToTensor()  # Convert to tensor
])
gt_mask = mask_transform(Image.open("/home/da10546y/PREMIEREMulti/results/subject_1/masks/0.png").convert("L"))
gt_mask = (gt_mask > 0.5).float()  # Binarize the mask (1 for foreground, 0 for background)
gt_mask = gt_mask.unsqueeze(0)  # Add batch dimension

# Apply the GT mask to the images
img_gt_masked = img_gt * gt_mask
img1_masked = img1 * gt_mask
img2_masked = img2 * gt_mask
img3_masked = img3 * gt_mask

# Compute LPIPS scores
score1 = lpips_model(img1_masked, img_gt_masked)  # Model 1 vs Ground Truth
score2 = lpips_model(img2_masked, img_gt_masked)  # Model 2 vs Ground Truth
score3 = lpips_model(img3_masked, img_gt_masked)  # Model 3 vs Ground Truth

# Print the LPIPS scores
print(f"LPIPS Score (ECON vs GT): {score1.item()}")
print(f"LPIPS Score (ExAvatar vs GT): {score2.item()}")
print(f"LPIPS Score (Vid2Avatar vs GT): {score3.item()}")