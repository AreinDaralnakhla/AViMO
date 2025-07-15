# mpjpe.py
# Unified MPJPE comparison and Open3D visualization for ECON, Hand4Whole, ROMP, and NLF

import os
import pickle
import numpy as np
import open3d as o3d
import torch
import cv2
import json
import smplx
from smplfitter.pt.converter import SMPLConverter
os.environ['DATA_ROOT'] = os.path.join(os.environ["MODELS_PATH"], 'smplfitter')
# === Utility Functions ===
# Define indices to keep: exclude hands (20-63) and face (66-126)
body_joint_indices = [i for i in range(127) if i < 20 or (i == 20 or i == 21) or (64 <= i < 66)]

# === Utility Functions ===
def load_ground_truth(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return np.array([frame[0]['j3d_smplx'] for frame in data['allFrameHumans']])

def align_by_pelvis(gt, pred):
    min_frames = min(len(gt), len(pred))
    gt = gt[:min_frames]
    pred = pred[:min_frames]
    offset = gt[:, 0, :] - pred[:, 0, :]
    return pred + offset[:, np.newaxis, :]

def compute_mpjpe(gt, pred, indices=None):
    min_frames = min(len(gt), len(pred))
    gt = gt[:min_frames]
    pred = pred[:min_frames]
    if indices is not None:
        gt = gt[:, indices, :]
        pred = pred[:, indices, :]
    errors = np.linalg.norm(gt - pred, axis=2)
    return np.mean(errors)

def create_sphere_cloud(joints, color):
    spheres = []
    for pt in joints:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mesh_sphere.translate(pt)
        mesh_sphere.paint_uniform_color(color)
        mesh_sphere.compute_vertex_normals()
        spheres.append(mesh_sphere)
    return spheres

def create_lineset_from_joints(joints, connections, color, name):
    points = o3d.utility.Vector3dVector(joints)
    lines = o3d.utility.Vector2iVector(connections)
    colors = o3d.utility.Vector3dVector([color for _ in connections])
    line_set = o3d.geometry.LineSet(points=points, lines=lines)
    line_set.colors = colors
    return line_set

# === Model Loaders ===
# (Same as before, unchanged: load_predicted_econ, load_predicted_hand4whole, load_predicted_romp,
#  load_predicted_nlf, load_comotion_predictions_from_pt)
# [... include all loader definitions from your script here unchanged ...]

# === Model Loaders ===
def load_predicted_econ(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return np.array([data[k][0] for k in sorted(data.keys(), key=int)])

def load_predicted_hand4whole(folder_path, smplx_model, device):
    joints = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')], key=lambda x: int(x.split('.')[0]))
    for jf in files:
        with open(os.path.join(folder_path, jf), 'r') as f:
            d = json.load(f)
        params = {k: torch.tensor(d[k], dtype=torch.float32).reshape(1, -1).to(device) for k in ['root_pose', 'jaw_pose', 'leye_pose', 'reye_pose']}
        for k in ['body_pose', 'lhand_pose', 'rhand_pose']:
            params[k] = torch.tensor(d[k], dtype=torch.float32).reshape(1, -1, 3).to(device)
        shape = torch.tensor(d['shape'], dtype=torch.float32).reshape(1, 10).to(device)
        out = smplx_model(global_orient=params['root_pose'], body_pose=params['body_pose'], left_hand_pose=params['lhand_pose'],
                          right_hand_pose=params['rhand_pose'], jaw_pose=params['jaw_pose'], leye_pose=params['leye_pose'],
                          reye_pose=params['reye_pose'], betas=shape, return_verts=False)
        joints.append(out.joints.detach().cpu().numpy().squeeze())
    return np.array(joints)

def load_predicted_romp(npz_path, smplx_model, converter, device):
    data = np.load(npz_path, allow_pickle=True)['results'].item()
    joints = []
    for k in sorted(data.keys(), key=lambda x: int(x.split('.')[0])):
        d = data[k]
        orient = torch.tensor(d['global_orient']).to(device)
        pose = torch.tensor(d['body_pose']).to(device)
        betas = torch.tensor(d['smpl_betas']).to(device)
        trans = torch.tensor(d['cam_trans']).to(device)
        pose_tensor = torch.cat([orient, pose], dim=1).reshape(1, 24, 3)
        out = converter.convert(pose_tensor, betas, trans)
        smplx_out = smplx_model(global_orient=out['pose_rotvecs'][:, :3],
                                body_pose=out['pose_rotvecs'][:, 3:66].reshape(1, 21, 3),
                                betas=out['shape_betas'], transl=out['trans'], return_verts=False)
        joints.append(smplx_out.joints.detach().cpu().numpy().squeeze())
    return np.array(joints)

def load_predicted_nlf(gt_path, pred_path):
    with open(gt_path, 'rb') as f:
        gt = pickle.load(f)['allFrameHumans']
    with open(pred_path, 'rb') as f:
        pred = pickle.load(f)['allFrameHumans']
    j_gt, j_pred = [], []
    for g, p in zip(gt, pred):
        if g and p:
            j_gt.append(g[0]['j3d_smplx'])
            j_pred.append(p[0]['j3d_smplx'])
    return np.array(j_pred), np.array(j_gt)

def load_comotion_predictions_from_pt(pt_path, smplx_model, smpl2smplx, device):
    """
    Load CoMotion SMPL predictions from .pt and convert to SMPL-X joints.
    """
    data = torch.load(pt_path, map_location=device)
    poses = data["pose"]      # (F, 72)
    betas = data["betas"]     # (F, 10)
    trans = data["trans"]     # (F, 3)
    frame_idxs = data["frame_idx"].cpu().numpy()  # (F,)

    all_joints = []

    for i in range(poses.shape[0]):
        full_pose = poses[i].reshape(1, 24, 3).to(device)
        beta = betas[i].unsqueeze(0).to(device)
        t = trans[i].unsqueeze(0).to(device)

        smplx_params = smpl2smplx.convert(full_pose, beta, t)

        output = smplx_model(
            global_orient=smplx_params['pose_rotvecs'][:, :3],
            body_pose=smplx_params['pose_rotvecs'][:, 3:66].reshape(1, 21, 3),
            betas=smplx_params['shape_betas'],
            transl=smplx_params['trans'],
            return_verts=False
        )

        joints = output.joints.detach().cpu().numpy().squeeze(0)  # (127, 3)
        all_joints.append(joints)

    return np.array(all_joints)  # (F, 127, 3)

# === Visualization with GUI ===
from functools import partial

def visualize_3d_poses(models_dict, skeleton_connections):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Pose Viewer", width=1600, height=1000)
    model_colors = {
        'Ground Truth': [0, 0, 0],  
        'ECON': [1, 0, 0],  
        'ROMP': [0, 1, 0],  
        'Hand4Whole': [0, 0, 1],  
        'NLF': [1, 1, 0], 
        'CoMotion': [0.5, 0, 0.5]  
    }


    geometries, visibility = {}, {}

    for name, joints in models_dict.items():
        filtered_joints = joints[:, body_joint_indices]
        color = model_colors[name]
        line_set = create_lineset_from_joints(filtered_joints[0], skeleton_connections, color, name)  # Fixed
        vis.add_geometry(line_set)
        spheres = create_sphere_cloud(filtered_joints[0], color)
        for sphere in spheres:
            vis.add_geometry(sphere)
        geometries[name] = [line_set] + spheres
        visibility[name] = True

    vis.get_render_option().background_color = np.array([1, 1, 1])
    vis.get_render_option().show_coordinate_frame = False

    print("Legend:")
    for name, color_name in color_names.items():
        print(f"{name}: {color_name}")

    def toggle_visibility(vis, model_name):
        if visibility[model_name]:
            for geom in geometries[model_name]:
                vis.remove_geometry(geom)
            visibility[model_name] = False
        else:
            for geom in geometries[model_name]:
                vis.add_geometry(geom)
            visibility[model_name] = True

    key_mapping = {
        ord('1'): 'Ground Truth',
        ord('2'): 'ECON',
        ord('3'): 'Hand4Whole',
        ord('4'): 'ROMP',
        ord('5'): 'NLF',
        ord('6'): 'CoMotion'
    }
    for key, model_name in key_mapping.items():
        vis.register_key_callback(key, partial(toggle_visibility, model_name=model_name))

    vis.run()
    vis.destroy_window()
    
# === Main ===
if __name__ == "__main__":
    gt_path = "results/subject_1/aist_converted.pkl"
    econ_path = "/home/da10546y/PREMIEREMulti/results/subject_1/final_econ.pkl"
    hand4_path = "/home/da10546y/PREMIEREMulti/results/subject_1/smplx_init"
    romp_path = "/home/da10546y/PREMIEREMulti/results/subject_1/romp_results.npz"
    nlf_pred_path = "/home/da10546y/PREMIEREMulti/results/subject_1/nlf-final-filtered.pkl"
    comotion_path = "/home/da10546y/PREMIEREMulti/results/subject_1/comotion.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/home/da10546y/NLF-GS/fitting/MODELS_DIR"
    smplx_model = smplx.create(model_path, model_type='smplx', gender='neutral', use_pca=False,
                               flat_hand_mean=True, num_betas=10, ext='npz').to(device)
    smpl2smplx = SMPLConverter('smpl', 'neutral', 'smplx', 'neutral').to(device)

    gt = load_ground_truth(gt_path)
    econ = align_by_pelvis(gt, load_predicted_econ(econ_path))
    romp = align_by_pelvis(gt, load_predicted_hand4whole(hand4_path, smplx_model, device))
    h4w = align_by_pelvis(gt, load_predicted_romp(romp_path, smplx_model, smpl2smplx, device))
    nlf, gt_nlf = load_predicted_nlf(gt_path, nlf_pred_path)
    nlf = align_by_pelvis(gt, nlf)
    comotion = align_by_pelvis(gt, load_comotion_predictions_from_pt(comotion_path, smplx_model, smpl2smplx, device))

    connections = [(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),
                   (0,3),(3,6),(6,9),(9,12),(12,15),(12,16),
                   (16,18),(12,17),(17,19), (18,20),(19,21)]  # filtered to exclude hands/face

    print("Econ MPJPE:", compute_mpjpe(gt, econ, indices=body_joint_indices)*1000, "mm")
    print("Hand4Whole MPJPE:", compute_mpjpe(gt, h4w, indices=body_joint_indices)*1000, "mm")
    print("ROMP MPJPE:", compute_mpjpe(gt, romp, indices=body_joint_indices)*1000, "mm")
    print("NLF MPJPE:", compute_mpjpe(gt_nlf, nlf, indices=body_joint_indices)*1000, "mm")
    print("CoMotion MPJPE:", compute_mpjpe(gt, comotion, indices=body_joint_indices)*1000, "mm")

    visualize_3d_poses({
        'Ground Truth': gt,
        'ECON': econ,
        'Hand4Whole': h4w,
        'ROMP': romp,
        'NLF': nlf,
        'CoMotion': comotion
    }, skeleton_connections=connections)