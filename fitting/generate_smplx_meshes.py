import os
import pickle
import numpy as np
import torch
import smplx
from tqdm import tqdm
from smplx import SMPLX

def generate_smplx_meshes(nlf_pkl, output_dir):
    # Ensure output_dir is a valid directory
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    with open(nlf_pkl, "rb") as f:
        data = pickle.load(f)

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

    # Iterate through frames
    for i in tqdm(range(len(data["allFrameHumans"]))):
        for h_idx, hdata in enumerate(data["allFrameHumans"][i]):
            # Extract SMPL-X parameters from the human dictionary
            rotvec = torch.tensor(hdata["rotvec"]).reshape(1, 55, 3).to("cpu")
            transl = torch.tensor(hdata["transl_pelvis"]).reshape(1, 3).to("cpu")  # Already correct
            betas = torch.tensor(hdata["shape"]).reshape(1, -1).to("cpu")
            expression = torch.tensor(hdata["expression"]).reshape(1, -1).to("cpu")

            # Run SMPL-X model to get vertices
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

            # Extract vertices and faces from the model output
            vertices = output.vertices[0].detach().numpy()
            faces = model.faces

            # Save as PLY file
            output_filename = f"{i}_smplx.ply"
            output_path = os.path.join(output_dir, output_filename)
            save_ply(output_path, vertices, faces)

def save_ply(filename, vertices, faces):
    """Save vertices and faces to a PLY file."""
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")
        
        # Write vertices
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python generate_smplx_meshes.py <nlf_pkl> <output_dir>")
        sys.exit(1)

    generate_smplx_meshes(sys.argv[1], sys.argv[2])