import argparse
import os
import os.path as osp
import torch
import numpy as np
import json
from glob import glob
from tqdm import tqdm
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, required=True)
    parser.add_argument('--test_epoch', type=str, required=True)
    parser.add_argument('--motion_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args()

def export_gaussians(assets, save_path, visualize=True):
    import struct

    mean = assets['mean_3d'].detach().cpu().numpy()
    scale = assets['scale'].detach().cpu().numpy()
    rotation = assets['rotation'].detach().cpu().numpy()
    rgb = assets['rgb'].detach().cpu().numpy()  # Keep RGB normalized in [0, 1]
    opacity = assets['opacity'].detach().cpu().numpy()

    # Convert RGB to SH DC components
    constant = 2 * np.sqrt(np.pi)  # Zeroth-order SH constant
    rgb_sh_dc = (rgb - 0.5) * constant  # Apply the formula

    # Set spherical harmonics coefficients to zero
    sh_coeffs = np.zeros((mean.shape[0], 9), dtype=np.float32)  # Example for degree 1 (9 coefficients)

    # Recenter the Gaussian positions
    centroid = np.mean(mean, axis=0)
    mean -= centroid

    if visualize:
        scale = np.exp(scale) * 0.004
        scale = np.log(scale)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {mean.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float sh_dc_red",
        "property float sh_dc_green",
        "property float sh_dc_blue",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "property float f_dc_0",  # Add spherical harmonic coefficients
        "property float f_dc_1",
        "property float f_dc_2",
        "property float f_dc_3",
        "property float f_dc_4",
        "property float f_dc_5",
        "property float f_dc_6",
        "property float f_dc_7",
        "property float f_dc_8",
        "end_header\n"
    ]
    header = "\n".join(header_lines)

    with open(save_path, 'wb') as f:
        f.write(bytearray(header, 'utf-8'))
        for i in range(mean.shape[0]):
            q = rotation[i]
            entry = struct.pack(
                '<3f 3f f 3f 4f 9f',
                mean[i, 0], mean[i, 1], mean[i, 2],
                rgb_sh_dc[i, 0], rgb_sh_dc[i, 1], rgb_sh_dc[i, 2],  # SH DC components
                opacity[i],
                scale[i, 0], scale[i, 1], scale[i, 2],
                q[0], q[1], q[2], q[3],
                *sh_coeffs[i]  # Spherical harmonic coefficients
            )
            f.write(entry)

import struct
import gzip
import numpy as np


def main():
    args = parse_args()
    cfg.set_args(args.subject_id)

    tester = Tester(args.test_epoch)

    root_path = osp.join('..', 'data', cfg.dataset, 'data', cfg.subject_id)
    with open(osp.join(root_path, 'smplx_offsets', 'shape_param.json')) as f:
        shape_param = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_offsets', 'face_offset.json')) as f:
        face_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_offsets', 'joint_offset.json')) as f:
        joint_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_offsets', 'locator_offset.json')) as f:
        locator_offset = torch.FloatTensor(json.load(f))
    smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

    tester.smplx_params = None
    tester._make_model()

    frame_idx_list = sorted([int(osp.basename(x)[:-5]) for x in glob(osp.join(args.motion_path, 'smplx_params', '*.json'))])

    output_dir = args.output_dir or osp.join(args.motion_path, "exported_gaussians")
    os.makedirs(output_dir, exist_ok=True)

    for frame_idx in tqdm(frame_idx_list):
        with open(osp.join(args.motion_path, 'cam_params', f"{frame_idx}.json")) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k, v in json.load(f).items()}
        with open(osp.join(args.motion_path, 'smplx_params', f"{frame_idx}.json")) as f:
            smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k, v in json.load(f).items()}

        with torch.no_grad():
            model = tester.model.module if hasattr(tester.model, "module") else tester.model
            _, human_asset_refined, _, _ = model.human_gaussian(smplx_param, cam_param)

        save_path = osp.join(output_dir, f"splat_{frame_idx:0d}.ply")  # Change file extension to .spz
        export_gaussians(human_asset_refined, save_path)

    print(f"Exported all frames to {output_dir}")
if __name__ == "__main__":
    main()
