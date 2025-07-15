import torch
import numpy as np
import os
import os.path as osp
from scipy.signal import savgol_filter
import json
from glob import glob
from tqdm import tqdm
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix, matrix_to_axis_angle
from pytorch3d.io import save_ply
import cv2
os.environ["PYOPENGL_PLATFORM"] = "egl"
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PointLights,
PerspectiveCameras,
Materials,
SoftPhongShader,
RasterizationSettings,
MeshRenderer,
MeshRendererWithFragments,
MeshRasterizer,
TexturesVertex)
import argparse
import sys
from smplx import SMPLX

def render_mesh(mesh, face, cam_param, bkg, blend_ratio=1.0):
    mesh = torch.FloatTensor(mesh).cuda()[None,:,:]
    face = torch.LongTensor(face.astype(np.int64)).cuda()[None,:,:]
    cam_param = {k: torch.FloatTensor(v).cuda()[None,:] for k,v in cam_param.items()}
    render_shape = (bkg.shape[0], bkg.shape[1]) # height, width

    batch_size, vertex_num = mesh.shape[:2]
    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                principal_point=cam_param['princpt'], 
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0, perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    materials = Materials(
	device='cuda',
	specular_color=[[0.0, 0.0, 0.0]],
	shininess=0.0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
    
    # background masking
    is_bkg = (fragments.zbuf <= 0).float().cpu().numpy()[0]
    render = images[0,:,:,:3].cpu().numpy()
    fg = render * blend_ratio + bkg/255 * (1 - blend_ratio)
    render = fg * (1 - is_bkg) * 255 + bkg * is_bkg
    return render

def render_single_frame(frame_idx, root_path, smplx_params, smplx_model_path, render_save_path):
    # Load SMPL-X faces from the model file
    smplx_model_data = np.load(smplx_model_path, allow_pickle=True)
    faces = smplx_model_data["f"].astype(np.int32)  # Load faces as integers

    # Load SMPL-X parameters for the specified frame
    smplx_param = smplx_params[frame_idx]

    # we shift transl by 0.1 to match the original SMPL-X model
    transl = smplx_param['trans']
    transl[1] += 0.35

    # Move SMPL-X parameters to GPU
    with torch.no_grad():
        output = model(
            global_orient=torch.FloatTensor(smplx_param['root_pose']).view(1, -1).to("cuda"),
            body_pose=torch.FloatTensor(smplx_param['body_pose']).view(1, -1).to("cuda"),
            leye_pose=torch.FloatTensor(smplx_param['leye_pose']).view(1, -1).to("cuda"),
            reye_pose=torch.FloatTensor(smplx_param['reye_pose']).view(1, -1).to("cuda"),
            left_hand_pose=torch.FloatTensor(smplx_param['lhand_pose']).view(1, -1).to("cuda"),
            right_hand_pose=torch.FloatTensor(smplx_param['rhand_pose']).view(1, -1).to("cuda"),
            expression=torch.FloatTensor(smplx_param['expr'][:10]).view(1, -1).to("cuda"),  # Slice the first 10 values
            transl=torch.FloatTensor(smplx_param['trans']).view(1, -1).to("cuda"),
        )

    vert = output.vertices[0].cpu().numpy()  # Move vertices back to CPU for rendering

    # Load camera parameters
    cam_param_path = osp.join(root_path, 'cam_params', str(frame_idx) + '.json')
    with open(cam_param_path) as f:
        cam_param = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}

    # Load the background image
    img_path = osp.join(root_path, 'frames', str(frame_idx) + '.png')
    img = cv2.imread(img_path)

    # Render the SMPL-X mesh
    render = render_mesh(vert, faces, {'focal': cam_param['focal'], 'princpt': cam_param['princpt']}, img, 1.0)

    # Save the rendered image
    render_output_path = osp.join(render_save_path, f"{frame_idx}_smplx.jpg")
    cv2.imwrite(render_output_path, render)
    print(f"Rendered frame {frame_idx} saved to {render_output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True, help="Root path containing SMPL-X parameters and frames")
    args = parser.parse_args()

    root_path = args.root_path

    # Load SMPL-X parameters of all frames
    smplx_param_path_list = glob(osp.join(root_path, 'smplx_params', '*.json'))
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in smplx_param_path_list])
    smplx_params = {}
    for idx in frame_idx_list:
        smplx_param_path = osp.join(root_path, 'smplx_params', str(idx) + '.json')
        with open(smplx_param_path) as f:
            smplx_param = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
        smplx_params[idx] = smplx_param

    # Define the path to the SMPL-X model directory
    smplx_model_path = "/home/da10546y/NLF-GS/fitting/MODELS_DIR/smplx/SMPLX_FEMALE.npz"

    # Initialize the SMPL-X model
    model = SMPLX(
        model_path=osp.dirname(smplx_model_path),
        gender='NEUTRAL',
        use_face_contour=False,
        flat_hand_mean=True,
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
        batch_size=1
    ).to("cuda")  # Use GPU if available

    # Define the render save path
    render_save_path = osp.join(root_path, 'renders')
    os.makedirs(render_save_path, exist_ok=True)

    # Render all frames
    for frame_idx in tqdm(frame_idx_list):
        render_single_frame(frame_idx, root_path, smplx_params, smplx_model_path, render_save_path)