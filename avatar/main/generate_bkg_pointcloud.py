import os
import numpy as np
import cv2
import json
from tqdm import tqdm
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)
import torch
import os.path as osp
from glob import glob
import argparse


def render_smplx_depthmap(smplx_mesh_path, cam_param, render_shape):
    """Render depth map from SMPL-X mesh using PyTorch3D."""
    smplx_vert, smplx_face = load_ply(smplx_mesh_path)
    smplx_vert = smplx_vert.cuda()[None, :, :]
    smplx_face = smplx_face.cuda()[None, :, :]
    cam_param = {k: torch.FloatTensor(v).cuda()[None, :] for k, v in cam_param.items()}

    # Reverse x- and y-axis following PyTorch3D axis direction
    mesh = torch.stack((-smplx_vert[:, :, 0], -smplx_vert[:, :, 1], smplx_vert[:, :, 2]), 2)
    mesh = Meshes(mesh, smplx_face)

    cameras = PerspectiveCameras(
        focal_length=cam_param['focal'],
        principal_point=cam_param['princpt'],
        device='cuda',
        in_ndc=False,
        image_size=torch.LongTensor(render_shape).cuda().view(1, 2)
    )
    raster_settings = RasterizationSettings(
        image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0, perspective_correct=True
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()

    # Render depth map
    with torch.no_grad():
        fragments = rasterizer(mesh)

    depthmap = fragments.zbuf.cpu().numpy()[0, :, :, 0]
    return depthmap


def generate_background_point_cloud(root_path, output_path):
    """Generate background point cloud by processing all frames."""
    # Paths
    depthmaps_dir = osp.join(root_path, 'depth_maps')
    frames_dir = osp.join(root_path, 'frames')
    masks_dir = osp.join(root_path, 'masks')
    smplx_dir = osp.join(root_path, 'meshes')
    cam_params_dir = osp.join(root_path, 'cam_params')

    # Initialize accumulators
    depthmap_save = 0
    color_save = 0
    is_bkg_save = 0

    # Get frame indices
    depthmap_path_list = sorted(glob(osp.join(depthmaps_dir, '*.png')))
    frame_idx_list = sorted([int(osp.basename(x).split('.')[0]) for x in depthmap_path_list])
    img_height, img_width = cv2.imread(depthmap_path_list[0]).shape[:2]

    # Process each frame
    for frame_idx in tqdm(frame_idx_list):
        # Skip frames that are not divisible by 3 (process only a third of the frames)
        if frame_idx % 4 != 0:
            continue

        # Load image
        img_path = osp.join(frames_dir, f'{frame_idx}.png')
        img = cv2.imread(img_path).astype(np.float32)

        # Load depth map from MoGe or Depth-Anything
        depthmap_path = osp.join(depthmaps_dir, f'{frame_idx}.png')
        depthmap = cv2.imread(depthmap_path)[:, :, 0]  # Use original depth values without inversion

        # Load SMPL-X mesh and render depth map
        smplx_mesh_path = osp.join(smplx_dir, f'{frame_idx}_smplx.ply')
        if not osp.isfile(smplx_mesh_path):
            continue
        with open(osp.join(cam_params_dir, f'{frame_idx}.json')) as f:
            cam_param = json.load(f)
        smplx_depthmap = render_smplx_depthmap(smplx_mesh_path, cam_param, (img_height, img_width))
        smplx_is_fg = smplx_depthmap > 0

        # Normalize depth map
        scale = np.abs(depthmap[smplx_is_fg] - depthmap[smplx_is_fg].mean()).mean()
        scale_smplx = np.abs(smplx_depthmap[smplx_is_fg] - smplx_depthmap[smplx_is_fg].mean()).mean()
        depthmap = depthmap / scale * scale_smplx
        depthmap = depthmap - depthmap[smplx_is_fg].mean() + smplx_depthmap[smplx_is_fg].mean()

        # Load mask
        mask_path = osp.join(masks_dir, f'{frame_idx}.png')
        mask = cv2.imread(mask_path)[:, :, 0]
        is_bkg = mask < 0.5

        # Accumulate background points
        depthmap_save += depthmap * is_bkg
        color_save += img * is_bkg[:, :, None]
        is_bkg_save += is_bkg

    # Average accumulated values
    depthmap_save /= (is_bkg_save + 1e-6)
    color_save /= (is_bkg_save[:, :, None] + 1e-6)

    # Save background point cloud
    with open(output_path, 'w') as f:
        for i in tqdm(range(img_height)):
            for j in range(img_width):
                if is_bkg_save[i, j]:
                    x = (j - cam_param['princpt'][0]) / cam_param['focal'][0] * depthmap_save[i, j]
                    y = (i - cam_param['princpt'][1]) / cam_param['focal'][1] * depthmap_save[i, j]
                    z = depthmap_save[i, j]
                    rgb = color_save[i, j]
                    f.write(f"{x} {y} {z} {rgb[0]} {rgb[1]} {rgb[2]}\n")

    print(f"Background point cloud saved to {output_path}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate background point cloud from SMPL-X meshes and depth maps")
    parser.add_argument("--root_path", type=str, required=True, help="Root path containing input data")
    args = parser.parse_args()

    # Paths
    root_path = args.root_path
    output_path = osp.join(root_path, "bkg_point_cloud.txt")

    # Generate background point cloud
    generate_background_point_cloud(root_path, output_path)