import os
import pickle
import cv2
import numpy as np
import torch
import smplx
from tqdm import tqdm
from smplx import SMPLX


def project_smplx(nlf_pkl, video_path, output_path, output_frames_dir, updated_pkl_path):
    os.makedirs(output_frames_dir, exist_ok=True)  # Create directory for output frames

    # Load the main NLF data (nlf-final-filtered.pkl)
    with open(nlf_pkl, "rb") as f:
        data = pickle.load(f)

    # Load the transl_pelvis values from the nlf.pkl file
    nlf_dir = os.path.dirname(nlf_pkl)
    nlf_pkl_path = os.path.join(nlf_dir, "nlf.pkl")
    if not os.path.exists(nlf_pkl_path):
        raise FileNotFoundError(f"nlf.pkl not found in the same directory as {nlf_pkl}")
    with open(nlf_pkl_path, "rb") as f:
        nlf_data = pickle.load(f)

    # Extract transl_pelvis values from nlf.pkl
    transl_pelvis_map = {}
    for i, frame_humans in enumerate(nlf_data["allFrameHumans"]):
        for j, hdata in enumerate(frame_humans):
            transl_pelvis_map[(i, j)] = hdata["transl_pelvis"]

    w, h, fps = data["video_width"], data["video_height"], data["video_fps"]
    fov_x = data["fov_x_deg"]
    focal = (w / 2) / np.tan(np.radians(fov_x / 2))
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])

    models_path = os.environ["MODELS_PATH"]
    smplx_model_path = os.path.join(models_path, "smplx", "SMPLX_NEUTRAL.npz")

    model = SMPLX(
        model_path=os.path.dirname(smplx_model_path),
        gender='NEUTRAL',
        use_face_contour=False,
        flat_hand_mean=True,
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
        batch_size=1
    ).to("cpu")

    video = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Iterate through frames
    for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = video.read()
        if not ret:
            break

        if i >= len(data["allFrameHumans"]):
            continue

        for j, hdata in enumerate(data["allFrameHumans"][i]):
            # Replace transl_pelvis with the value from nlf.pkl
            if (i, j) in transl_pelvis_map:
                hdata["transl_pelvis"] = transl_pelvis_map[(i, j)]

            # Extract SMPL-X parameters from the human dictionary
            rotvec = torch.tensor(hdata["rotvec"]).reshape(1, 55, 3).to("cpu")
            transl = torch.tensor(hdata["transl_pelvis"]).reshape(1, 3).to("cpu")

            # Apply the Y-axis translation adjustment directly to transl_pelvis
            transl += torch.tensor([0, 0.35, 0]).to("cpu")  # Adjust Y-axis by 0.35
            # transl += torch.tensor([0, 0, -2]).to("cpu")  # Move closer along Z-axis (adjust -0.5 as needed)

            # # Update the dictionary with the adjusted transl_pelvis
            # hdata["transl_pelvis"] = transl.detach().numpy().flatten().tolist()

            betas = torch.tensor(hdata["shape"]).reshape(1, -1).to("cpu")
            expression = torch.tensor(hdata["expression"]).reshape(1, -1).to("cpu")

            # Run SMPL-X model to get vertices dynamically
            output = model(
                global_orient=rotvec[:, 0].unsqueeze(1),
                body_pose=rotvec[:, 1:22].reshape(1, -1),
                left_hand_pose=rotvec[:, 22:37].reshape(1, -1),
                right_hand_pose=rotvec[:, 37:52].reshape(1, -1),
                # jaw_pose=rotvec[:, 52:53],
                leye_pose=rotvec[:, 53:54],
                reye_pose=rotvec[:, 54:55],
                betas=betas,
                expression=expression,
                transl=transl
            )

            # Extract vertices from the model output
            vertices = output.vertices[0].detach().numpy()
            # joints = output.joints[0].detach().numpy()  # SMPL-X joints
            faces = model.faces  # SMPL-X faces

            # Project vertices using camera intrinsics
            verts_proj = vertices @ K.T  # shape (N, 3)
            z_coords = verts_proj[:, 2:3]
            z_coords[z_coords <= 0] = 1e-6  # Prevent division by zero or negative values
            verts_proj = verts_proj[:, :2] / z_coords  # Normalize by Z

            # #project joints similarly
            # joints_proj = joints @ K.T
            # joints_proj = joints_proj[:, :2] / joints_proj[:, 2:3]
            
            # Debug: Check for invalid values
            if np.any(np.isnan(verts_proj)) or np.any(np.isinf(verts_proj)):
                print("[ERROR] Invalid projected vertices:", verts_proj)
                continue

            # Draw the mesh on the frame
            for face in faces:
                for k in range(3):
                    pt1 = tuple(verts_proj[face[k]].astype(int))
                    pt2 = tuple(verts_proj[face[(k + 1) % 3]].astype(int))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)

            # # draw the 2d joints on the frame
            # for joint in joints_proj:
            #     if np.all(np.isfinite(joint)):
            #         cv2.circle(frame, tuple(joint.astype(int)), 3, (0, 0, 255), -1)
                    

        # Save the current frame as an image
        frame_path = os.path.join(output_frames_dir, f"{i}.png")
        cv2.imwrite(frame_path, frame)

        # Write the frame to the video
        out.write(frame)

    video.release()
    out.release()
    print("Saved frames to:", output_frames_dir)
    print("Saved video to:", output_path)

    # Save the updated dictionary to the specified updated_pkl_path
    with open(updated_pkl_path, "wb") as f:
        pickle.dump(data, f)

    print("Updated SMPL-X parameters saved to:", updated_pkl_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 6:
        print("Usage: python render_smplx_mesh.py <nlf_pkl> <video_path> <output_video_path> <output_frames_dir> <updated_pkl_path>")
        sys.exit(1)

    project_smplx(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])