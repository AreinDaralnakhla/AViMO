import os
import sys
import cv2
import torch
import pickle
import smplx
import numpy as np
from tqdm import tqdm
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
from premiere.functionsCommon import projectPoints3dTo2d


def compute_bounding_box(joints2d, img_h, img_w, padding=0.2):
    x_min, y_min = np.min(joints2d, axis=0)
    x_max, y_max = np.max(joints2d, axis=0)
    w = x_max - x_min
    h = y_max - y_min
    x_min -= w * padding
    x_max += w * padding
    y_min -= h * padding
    y_max += h * padding
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)
    return [x_min, y_min, x_max, y_max]


if len(sys.argv) < 3:
    print("Usage: python injectHandsPkl.py <input_nlf_pkl> <video_path> <output_pkl> ")
    sys.exit(1)

input_pkl_path = sys.argv[1]
video_path = sys.argv[2]
output_pkl_path = sys.argv[3]

# === Load NLF data ===
with open(input_pkl_path, "rb") as f:
    nlf_data = pickle.load(f)

allFrameHumans = nlf_data["allFrameHumans"]

# === Run WiLoR pipeline ===
print("[INFO] Running WiLoR-mini pipeline...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[!] Cannot open video")
    sys.exit(1)

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

wilor_results = []
frame_id = 0
pbar = tqdm(total=len(allFrameHumans), desc="Running WiLoR pipeline")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_id >= len(allFrameHumans):
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    humans = allFrameHumans[frame_id]

    if not humans:
        wilor_results.append([])  # Empty frame
        frame_id += 1
        pbar.update(1)
        continue

    # Compute bounding boxes for left and right hands
    j2d = np.array(humans[0]["j2d_smplx"])
    left_joints = np.vstack([j2d[25:40], j2d[20]])
    right_joints = np.vstack([j2d[40:55], j2d[21]])
    bboxes = np.array([
        compute_bounding_box(left_joints, video_height, video_width),
        compute_bounding_box(right_joints, video_height, video_width)
    ], dtype=np.float32)
    is_right = [0, 1]

    outputs = pipe.predict_with_bboxes(frame_rgb, bboxes, is_right)
    wilor_results.append(outputs)

    frame_id += 1
    pbar.update(1)

pbar.close()
cap.release()

# === Inject WiLoR hand poses into NLF ===
print(f"[INFO] Injecting WiLoR hand poses into {len(wilor_results)} frames...")

for frame_idx in range(len(wilor_results)):
    frame_wilor = wilor_results[frame_idx]
    frame_nlf = allFrameHumans[frame_idx]
    
    if not frame_nlf:
        continue  # Skip empty frames

    human = frame_nlf[0]  # Assume one person per frame
    rotvec = np.array(human["rotvec"])

    # Initialize placeholders for left and right hand poses
    left_hand_pose = np.zeros((15, 3))
    right_hand_pose = np.zeros((15, 3))

    for hand_result in frame_wilor:
        if "wilor_preds" not in hand_result or "hand_pose" not in hand_result["wilor_preds"]:
            print(f"[WARNING] Missing 'hand_pose' key in 'wilor_preds'. Available keys: {hand_result.keys()}")
            continue

        is_right = hand_result["is_right"]
        hand_pose = hand_result["wilor_preds"]["hand_pose"]  # (15, 3)

        if is_right:
            right_hand_pose = hand_pose
        else:
            left_hand_pose = hand_pose

    # Inject the hand poses into SMPL-X rotvec
    rotvec[22:37] = left_hand_pose  # Left hand
    rotvec[37:52] = right_hand_pose  # Right hand

    human["rotvec"] = rotvec

print(f"Injection complete. Saving updated NLF to {output_pkl_path}...")

# === Save updated NLF data ===
with open(output_pkl_path, "wb") as f:
    pickle.dump(nlf_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Done. Output saved to {output_pkl_path}")