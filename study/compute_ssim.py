import os
import cv2
import numpy as np
import torch
import argparse
from skimage.metrics import structural_similarity as ssim

def detect_face(image):
    """
    Detect the face region in the image using OpenCV's Haar cascade.
    Returns the bounding box (x, y, w, h) of the detected face.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None  # No face detected
    # Return the first detected face
    return faces[0]

def compute_ssim_single_frame(gt_path, pred_path, gt_mask_path, face_only=False, save_intermediate=False):
    """
    Compute SSIM for a single frame, optionally focusing on the face region.
    Uses only the ground truth mask for masking.
    """
    # Load ground truth, predicted images, and mask
    gt_img = cv2.imread(gt_path)
    pred_img = cv2.imread(pred_path)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    if gt_img is None or pred_img is None or gt_mask is None:
        raise ValueError("Could not load images or mask for the specified frame.")

    if face_only:
        # Detect face region in the ground truth image
        gray_gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        face_bbox = detect_face(gray_gt_img)
        if face_bbox is None:
            raise ValueError("No face detected in the ground truth image.")

        x, y, w, h = face_bbox

        # Crop the face region from the images and mask
        gt_img = gt_img[y:y+h, x:x+w]
        pred_img = pred_img[y:y+h, x:x+w]
        gt_mask = gt_mask[y:y+h, x:x+w]

    # Apply the ground truth mask to both images
    gt_masked = cv2.bitwise_and(gt_img, gt_img, mask=gt_mask)
    pred_masked = cv2.bitwise_and(pred_img, pred_img, mask=gt_mask)

    # Save or display intermediate masked images
    if save_intermediate:
        cv2.imwrite("gt_masked.png", gt_masked)
        cv2.imwrite("pred_masked.png", pred_masked)
        print("Intermediate masked images saved as 'gt_masked.png' and 'pred_masked.png'.")

    # Convert images to PyTorch tensors and move to GPU
    gt_tensor = torch.tensor(gt_masked, dtype=torch.float32, device="cuda") / 255.0
    pred_tensor = torch.tensor(pred_masked, dtype=torch.float32, device="cuda") / 255.0

    # Compute SSIM using skimage (on CPU, but with preprocessed tensors)
    gt_cpu = gt_tensor.cpu().numpy()
    pred_cpu = pred_tensor.cpu().numpy()
    score, _ = ssim(gt_cpu, pred_cpu, full=True, data_range=1.0, multichannel=True, win_size=3)

    return score

def compute_average_ssim(gt_dir, pred_dir, gt_mask_dir, face_only=False, save_intermediate=False, exclude_frames=None):
    """
    Compute the average SSIM over all images in the directories, excluding specified frames.
    Logs the number of detected faces, skipped frames, and valid frames used for averaging.
    """
    exclude_frames = set(exclude_frames) if exclude_frames else set()
    ssim_values = []
    skipped_frames = []
    detected_faces = 0
    not_detected_faces = 0

    for filename in sorted(os.listdir(gt_dir)):
        # Extract frame number from filename (e.g., "0.png" -> 0)
        frame_number = int(os.path.splitext(filename)[0])
        if frame_number in exclude_frames:
            print(f"Skipping frame {frame_number} as it is in the exclusion list.")
            skipped_frames.append(frame_number)
            continue

        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        gt_mask_path = os.path.join(gt_mask_dir, filename)

        if not os.path.exists(pred_path) or not os.path.exists(gt_mask_path):
            print(f"Warning: Missing files for {filename}. Skipping.")
            skipped_frames.append(frame_number)
            continue

        try:
            score = compute_ssim_single_frame(gt_path, pred_path, gt_mask_path, face_only, save_intermediate)
            ssim_values.append(score)
            detected_faces += 1
        except ValueError as e:
            print(f"Skipping frame {frame_number}: {e}")
            skipped_frames.append(frame_number)
            not_detected_faces += 1

    # Log summary
    print("\n--- Summary ---")
    print(f"Total frames processed: {len(os.listdir(gt_dir))}")
    print(f"Frames excluded (manual exclusion or errors): {len(skipped_frames)}")
    print(f"Frames with detected faces: {detected_faces}")
    print(f"Frames without detected faces: {not_detected_faces}")
    print(f"Valid frames used for SSIM calculation: {len(ssim_values)}")

    if ssim_values:
        return np.mean(ssim_values)
    else:
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SSIM for images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth images.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted images.")
    parser.add_argument("--gt_mask_dir", type=str, required=True, help="Directory containing ground truth masks.")
    parser.add_argument("--face_only", action="store_true", help="Compute SSIM for the face region only.")
    parser.add_argument("--single_frame", type=int, default=None, help="Compute SSIM for a single frame (specify frame number).")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate masked images.")
    parser.add_argument("--exclude_frames", type=int, nargs="*", default=None, help="List of frame numbers to exclude.")

    args = parser.parse_args()

    if args.single_frame is not None:
        # Compute SSIM for a single frame
        frame_number = args.single_frame
        gt_path = os.path.join(args.gt_dir, f"{frame_number}.png")
        pred_path = os.path.join(args.pred_dir, f"{frame_number}.png")
        gt_mask_path = os.path.join(args.gt_mask_dir, f"{frame_number}.png")

        try:
            ssim_score = compute_ssim_single_frame(gt_path, pred_path, gt_mask_path, args.face_only, args.save_intermediate)
            print(f"SSIM for frame {frame_number}: {ssim_score:.4f}")
        except ValueError as e:
            print(f"Error: {e}")
    else:
        # Compute average SSIM over all frames
        avg_ssim = compute_average_ssim(
            args.gt_dir, args.pred_dir, args.gt_mask_dir, args.face_only, args.save_intermediate, args.exclude_frames
        )
        print(f"\nAverage SSIM: {avg_ssim:.4f}")