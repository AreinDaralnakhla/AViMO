import os
import os.path as osp
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
import json

def load_nlf_params(nlf_pkl_path):
    """
    Load the NLF parameters from the provided .pkl file.
    """
    with open(nlf_pkl_path, "rb") as f:
        nlf_data = pickle.load(f)
    return nlf_data

def prepare_smplx_params(nlf_human):
    """
    Converts one human's NLF parameters into the expected SMPL-X parameter format.
    """
    rotvec = torch.tensor(nlf_human["rotvec"]).float().unsqueeze(0)  # shape (1, 55, 3)
    root_pose = rotvec[:, 0].unsqueeze(1)             # (1, 1, 3)
    body_pose = rotvec[:, 1:22].reshape(1, -1)        # (1, 63)
    lhand_pose = rotvec[:, 22:37].reshape(1, -1)      # (1, 45)
    rhand_pose = rotvec[:, 37:52].reshape(1, -1)      # (1, 45)
    jaw_pose = torch.zeros(1, 3)                      # Zero out jaw pose (1, 3)
    leye_pose = rotvec[:, 53].reshape(1, 3)          # (1, 3)
    reye_pose = rotvec[:, 54].reshape(1, 3)          # (1, 3)

    trans = torch.tensor(nlf_human["transl_pelvis"]).float().unsqueeze(0)  # shape (1, 3)
    shape = torch.tensor(nlf_human["shape"]).float().unsqueeze(0) 

    # Generate random small values for the first 10 expression parameters
    expr = torch.rand(1, 10).float() * 0.01  # Random small values between 0 and 0.01

    # Pad the expression to 50 values
    expr_padded = torch.zeros(1, 50).float()  # Create a tensor of zeros with shape (1, 50)
    expr_padded[:, :10] = expr  # Fill the first 10 values with the random small values

    smplx_params = {
        "root_pose": root_pose.view(-1).tolist(),  # shape (1, 3)
        "body_pose": body_pose.view(-1, 3).tolist(),  # shape (1, 63)
        "jaw_pose": jaw_pose.view(-1).tolist(),  # shape (1, 3)
        "leye_pose": leye_pose.view(-1).tolist(),  # shape (1, 3)
        "reye_pose": reye_pose.view(-1).tolist(),  # shape (1, 3)
        "lhand_pose": lhand_pose.view(-1, 3).tolist(),  # shape (1, 45)
        "rhand_pose": rhand_pose.view(-1, 3).tolist(),  # shape (1, 45)
        "trans": trans.view(-1).tolist(),  # shape (1, 3)
        "shape": shape.view(-1).tolist(),  # shape (1, 10)
        "expr": expr_padded.view(-1).tolist(),  # shape (1, 10)
    }

    return smplx_params

def generate_cam_params(nlf_data, root_path):
    """
    Generate camera parameters for each frame based on video information.
    """
    cam_params_path = osp.join(root_path, "cam_params")
    os.makedirs(cam_params_path, exist_ok=True)

    # Extract video information
    img_width = nlf_data["video_width"]
    img_height = nlf_data["video_height"]
    fov_x = nlf_data["fov_x_deg"]

    # Calculate focal length based on field of view
    focal_length = (img_width / 2) / np.tan(np.radians(fov_x / 2))

    # Generate camera parameters for each frame
    for frame_idx in range(len(nlf_data["allFrameHumans"])):
        cam_params = {
            "R": np.eye(3).tolist(),  # Identity rotation matrix
            "t": [0.0, 0.0, 0.0],  # Zero translation vector
            "focal": [focal_length, focal_length],  # Focal length
            "princpt": [img_width / 2, img_height / 2],  # Principal point (center of the image)
        }

        # Save camera parameters as JSON
        cam_params_save_path = osp.join(cam_params_path, f"{frame_idx}.json")
        with open(cam_params_save_path, "w") as f:
            json.dump(cam_params, f)

def generate_offsets(root_path):
    """
    Generate offsets (face_offset, locator_offset, joint_offset) and save them as JSON files.
    """
    offsets_path = osp.join(root_path, "smplx_offsets")
    os.makedirs(offsets_path, exist_ok=True)

    # Define the sizes for each offset
    face_offset_size = 10475
    locator_offset_size = 55
    joint_offset_size = 55

    # Create zero-filled offsets
    face_offset = [[0.0, 0.0, 0.0] for _ in range(face_offset_size)]
    locator_offset = [[0.0, 0.0, 0.0] for _ in range(locator_offset_size)]
    joint_offset = [[0.0, 0.0, 0.0] for _ in range(joint_offset_size)]

    # Save face_offset.json
    face_offset_path = osp.join(offsets_path, "face_offset.json")
    with open(face_offset_path, "w") as f:
        json.dump(face_offset, f)
    print(f"Created: {face_offset_path}")

    # Save locator_offset.json
    locator_offset_path = osp.join(offsets_path, "locator_offset.json")
    with open(locator_offset_path, "w") as f:
        json.dump(locator_offset, f)
    print(f"Created: {locator_offset_path}")

    # Save joint_offset.json
    joint_offset_path = osp.join(offsets_path, "joint_offset.json")
    with open(joint_offset_path, "w") as f:
        json.dump(joint_offset, f)
    print(f"Created: {joint_offset_path}")

    # Generate shape_param.json
    smplx_params_path = osp.join(root_path, "smplx_params")
    shape_params = []

    # Collect shape parameters from all frames
    for json_file in sorted(os.listdir(smplx_params_path)):
        if json_file.endswith(".json"):
            file_path = osp.join(smplx_params_path, json_file)
            with open(file_path, "r") as f:
                data = json.load(f)
                if "shape" in data:
                    shape_params.append(data["shape"])

    if len(shape_params) == 0:
        print("No shape parameters found in the JSON files.")
        return

    # Compute the average shape parameter and pad to 100 values
    shape_params = np.array(shape_params)
    average_shape = np.mean(shape_params, axis=0)
    padded_shape = np.zeros(100)
    padded_shape[:len(average_shape)] = average_shape

    # Save shape_param.json
    shape_param_path = osp.join(offsets_path, "shape_param.json")
    with open(shape_param_path, "w") as f:
        json.dump(padded_shape.tolist(), f)
    print(f"Created: {shape_param_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare SMPL-X, camera parameters, and offsets from NLF output.")
    parser.add_argument("--in_pkl", type=str, required=True, help="Path to NLF pkl file")
    parser.add_argument("--root_path", type=str, required=True, help="Path to subject folder containing frames")
    args = parser.parse_args()

    # Load NLF parameters
    nlf_data = load_nlf_params(args.in_pkl)

    # Prepare SMPL-X parameters directory
    smplx_params_path = osp.join(args.root_path, "smplx_params")
    os.makedirs(smplx_params_path, exist_ok=True)

    # Process each frame
    for frame_idx, frame_humans in enumerate(tqdm(nlf_data["allFrameHumans"], desc="Processing frames")):
        if len(frame_humans) == 0:
            continue  # Skip frames with no humans

        # Process only the first human in the frame
        human = frame_humans[0]

        # Convert to SMPL-X format
        smplx_param = prepare_smplx_params(human)

        # Save SMPL-X parameters
        smplx_param_save_path = osp.join(smplx_params_path, f"{frame_idx}.json")
        with open(smplx_param_save_path, "w") as f:
            json.dump(smplx_param, f)

    # Generate camera parameters
    generate_cam_params(nlf_data, args.root_path)

    # Generate offsets
    generate_offsets(args.root_path)

if __name__ == "__main__":
    main()