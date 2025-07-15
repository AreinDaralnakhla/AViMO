# import argparse
# import os
# import os.path as osp
# import torch
# import json
# from config import cfg
# from base import Tester
# from utils.smpl_x import smpl_x
# from pytorch3d.io import save_ply

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--subject_id', type=str, required=True)
#     parser.add_argument('--test_epoch', type=str, required=True)
#     parser.add_argument('--output_dir', type=str, default='./neutral_mesh')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     cfg.set_args(args.subject_id)

#     # Load identity-specific shape info
#     root_path = osp.join('..', 'data', cfg.dataset, 'data', args.subject_id)
#     with open(osp.join(root_path, 'smplx_optimized', 'shape_param.json')) as f:
#         shape_param = torch.FloatTensor(json.load(f))
#     with open(osp.join(root_path, 'smplx_optimized', 'face_offset.json')) as f:
#         face_offset = torch.FloatTensor(json.load(f))
#     with open(osp.join(root_path, 'smplx_optimized', 'joint_offset.json')) as f:
#         joint_offset = torch.FloatTensor(json.load(f))
#     with open(osp.join(root_path, 'smplx_optimized', 'locator_offset.json')) as f:
#         locator_offset = torch.FloatTensor(json.load(f))
#     smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

#     # Setup model
#     tester = Tester(args.test_epoch)
#     tester.smplx_params = None
#     tester._make_model()

#     # Extract and save upsampled neutral pose mesh
#     output_dir = args.output_dir
#     os.makedirs(output_dir, exist_ok=True)

#     with torch.no_grad():
#         model = tester.model.module if hasattr(tester.model, "module") else tester.model
#         mesh_up, _, _, _ = model.human_gaussian.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)

#     # Save the neutral pose mesh as .ply
#     mesh_save_path = osp.join(output_dir, "neutral_pose_mesh.ply")
#     verts = mesh_up.detach().cpu()
#     faces = torch.tensor(smpl_x.face_upsampled, dtype=torch.int64)
#     save_ply(mesh_save_path, verts=verts, faces=faces)

#     print(f"[✓] Saved the upsampled neutral pose mesh to: {mesh_save_path}")

# if __name__ == "__main__":
#     main()

########################################################################################3
import torch
import argparse
import numpy as np
from config import cfg
from base import Tester
import os
import os.path as osp
import json
import math
from pytorch3d.transforms import axis_angle_to_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    assert args.subject_id, "Please set sequence name"
    assert args.test_epoch, 'Test epoch is required.'
    return args

def export_gaussians_supersplat(assets, save_path, visualize=True):
    import struct

    mean = assets['mean_3d'].detach().cpu().numpy()
    scale = assets['scale'].detach().cpu().numpy()
    rotation = assets['rotation'].detach().cpu().numpy()
    rgb = assets['rgb'].detach().cpu().numpy()
    opacity = assets['opacity'].detach().cpu().numpy()

    # Convert RGB to SH DC components
    constant = 2 * np.sqrt(np.pi)  # Zeroth-order SH constant
    rgb_sh_dc = (rgb - 0.5) * constant

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
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n"
    ]
    header = "\n".join(header_lines)

    with open(save_path, 'wb') as f:
        f.write(bytearray(header, 'utf-8'))
        for i in range(mean.shape[0]):
            q = rotation[i]
            entry = struct.pack(
                '<3f 3f f 3f 4f',
                mean[i, 0], mean[i, 1], mean[i, 2],
                rgb_sh_dc[i, 0], rgb_sh_dc[i, 1], rgb_sh_dc[i, 2],
                opacity[i],
                scale[i, 0], scale[i, 1], scale[i, 2],
                q[0], q[1], q[2], q[3]
            )
            f.write(entry)

def main():
    args = parse_args()
    cfg.set_args(args.subject_id)

    tester = Tester(args.test_epoch)

    # Load SMPL-X ID info
    root_path = osp.join('..', 'data', cfg.dataset, 'data', cfg.subject_id)
    with open(osp.join(root_path, 'smplx_optimized', 'shape_param.json')) as f:
        shape_param = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'face_offset.json')) as f:
        face_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'joint_offset.json')) as f:
        joint_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'locator_offset.json')) as f:
        locator_offset = torch.FloatTensor(json.load(f))

    from utils.smpl_x import smpl_x
    smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

    tester.smplx_params = None
    tester._make_model()

    # Set up neutral SMPL-X parameters
    zero_pose = torch.zeros((3)).float().cuda()
    smplx_param = {
        'root_pose': torch.FloatTensor([math.pi, 0, 0]).cuda(),
        'body_pose': smpl_x.neutral_body_pose.view(-1).cuda(),
        'jaw_pose': zero_pose,
        'leye_pose': zero_pose,
        'reye_pose': zero_pose,
        'lhand_pose': torch.zeros((len(smpl_x.joint_part['lhand']) * 3)).float().cuda(),
        'rhand_pose': torch.zeros((len(smpl_x.joint_part['rhand']) * 3)).float().cuda(),
        'expr': torch.zeros((smpl_x.expr_param_dim)).float().cuda(),
        'trans': torch.FloatTensor((0, 3, 3)).float().cuda()
    }

    render_shape = (1024, 1024)
    cam_param = {
        'R': torch.eye(3).float().cuda(),
        't': torch.zeros((3)).float().cuda(),
        'focal': torch.FloatTensor((1500, 1500)).cuda(),
        'princpt': torch.FloatTensor((render_shape[1] / 2, render_shape[0] / 2)).cuda()
    }

    with torch.no_grad():
        human_asset, human_asset_refined, human_offset, mesh_neutral_pose = tester.model.module.human_gaussian(smplx_param, cam_param)

    save_dir = './neutral_pose'
    os.makedirs(save_dir, exist_ok=True)
    ply_path = osp.join(save_dir, 'neutral_gaussians.ply')

    export_gaussians_supersplat(human_asset, ply_path, visualize=True)
    print(f"✅ Exported neutral Gaussians to: {ply_path}")

if __name__ == "__main__":
    main()


