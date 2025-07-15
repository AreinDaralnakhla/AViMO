import os
os.environ['DATA_ROOT'] = os.path.join(os.environ["MODELS_PATH"], 'smplfitter')

import pickle
import json
import sys
import numpy as np
import smplx
import torch
import roma
import socket
import math

from flask import Flask, render_template, request, make_response
from json import JSONEncoder
from scipy.spatial.transform import Rotation as R

models_path = os.environ["MODELS_PATH"]

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

if len(sys.argv) != 2:
    print("Usage: python appVisualization.py <input_pkl>")
    sys.exit(1)

dir_path = os.path.abspath("templates-Visualization")
print(dir_path)
app = Flask(__name__, root_path=dir_path)

global dataPKL
dataPKL = []

file = open(sys.argv[1], 'rb')
allDataPKL = pickle.load(file) 
file.close()

dataPKL = allDataPKL['allFrameHumans']
print("len(dataPKL): ", len(dataPKL))

global allFramesNumber
allFramesNumber = len(dataPKL)

global maxHumans
maxHumans = 0
for i in range(allFramesNumber):
    maxHumans = max(maxHumans, len(dataPKL[i]))
print("maxHumans:", maxHumans)

global useExpression
useExpression = True

print("model_name: ", allDataPKL['model_name'])
if (allDataPKL['model_type'] == "hmr2") or (allDataPKL['model_type'] == "nlf"):
    useExpression = False

global keypointsNumber
keypointsNumber = 127

# Initialize min and max values
min_values = np.array([float('inf'), float('inf'), float('inf')])
max_values = np.array([float('-inf'), float('-inf'), float('-inf')])

global nbTrackId
nbTrackId = 0
for i in range(allFramesNumber):
    for j in range(len(dataPKL[i])):
        for k in range(len(dataPKL[i][j]['j3d_smplx'][0:keypointsNumber])):
            dataPKL[i][j]['j3d_smplx'][k] /= 1.0
        keypoints = np.array(dataPKL[i][j]['j3d_smplx'][0:keypointsNumber])
        min_values = np.minimum(min_values, keypoints.min(axis=0))
        max_values = np.maximum(max_values, keypoints.max(axis=0))
        nbTrackId = max(nbTrackId, dataPKL[i][j]['id'])
nbTrackId = int(nbTrackId) + 1

print("nbTrackId: ", nbTrackId)
print("Min keypoints: ", min_values)
print("Max keypoints: ", max_values)

global allTracks
allTracks = []
allTracksSegments = []
for i in range(nbTrackId):
    allTracks.append([])
    allTracksSegments.append([])
for i in range(allFramesNumber):
    for j in range(len(dataPKL[i])):
        if dataPKL[i][j]['id'] != -1:
            element = (i, j, dataPKL[i][j]['j3d_smplx'][0].tolist(), 
                       np.squeeze(dataPKL[i][j]['transl_pelvis'], axis=0).tolist())
            allTracks[dataPKL[i][j]['id']].append(element)

global currentVertices
currentVertices = []

notempty = 0
while len(dataPKL[notempty]) == 0:
    notempty += 1
print("notempty: ", notempty)

import smplx  # already imported above
model = smplx.create(
    models_path, 'smplx',
    gender='neutral',
    flat_hand_mean=False,
    use_pca=False,
    num_betas=10,
    ext='npz').cuda()

t = torch.zeros(1, 3).cuda()

@app.route("/getInfos")
def getInfos():
    fps = 30
    if "video_fps" in allDataPKL:
        fps = allDataPKL["video_fps"]
    floor_Zoffset = 0
    if "floor_Zoffset" in allDataPKL:
        floor_Zoffset = allDataPKL["floor_Zoffset"]
    floor_angle_deg = 0
    if "floor_angle_deg" in allDataPKL:
        floor_angle_deg = allDataPKL["floor_angle_deg"]
    # --- NEW: Transform allTracks into objects with a poseId and data field ---
    newTracks = []
    for i, track in enumerate(allTracks):
        if len(track) > 0:
            newTracks.append({"poseId": i, "data": track})
    allInfos = {
        "totalFrameNumber": allFramesNumber,
        "notempty": notempty,
        "maxHumans": maxHumans,
        "nbKeyPoints": keypointsNumber,
        "allTracks": newTracks, 
        "video_width": allDataPKL["video_width"],
        "video_height": allDataPKL["video_height"],
        "video_fps": fps, 
        "floor_angle_deg": floor_angle_deg,
        "floor_Zoffset": floor_Zoffset,
        "fileName": sys.argv[1],
        "smplx_faces": model.faces.flatten().tolist()
    }
    jsonData = json.dumps(allInfos)
    return jsonData

@app.route("/getVertices")
def getVertices():
    frame = int(request.args.get('frame'))
    if len(dataPKL[frame]) == 0:
        jsonData = json.loads('{}')
        return jsonData
    else:
        allsmplx = []
        allId = []
        allKeypoints = []
        for humanID in range(len(dataPKL[frame])):
            betas = torch.from_numpy(np.expand_dims(dataPKL[frame][humanID]['shape'], axis=0)).cuda()
            if useExpression:
                expression = torch.from_numpy(np.expand_dims(dataPKL[frame][humanID]['expression'], axis=0)).cuda()
            pose = torch.from_numpy(np.expand_dims(dataPKL[frame][humanID]['rotvec'], axis=0)).cuda()           
            bs = pose.shape[0]
            kwargs_pose = {
                'betas': betas,
                'return_verts': True,
                'pose2rot': True 
            }
            kwargs_pose = {
                'global_orient': pose[:, 0].unsqueeze(1),  # Global orientation
                'body_pose': pose[:, 1:22].reshape(1, -1),  # Body pose
                'left_hand_pose': pose[:, 23:38].reshape(1, 15, 3),  # Left hand pose
                'right_hand_pose': pose[:, 37:52].reshape(1, 15, 3),  # Right hand pose
                'jaw_pose': pose[:, 52:53],  # Uncomment if jaw pose is needed
                'leye_pose': pose[:, 53:54],  # Left eye pose
                'reye_pose': pose[:, 54:55],  # Right eye pose
                'betas': betas,  # Shape parameters
                'expression': expression,  # Expression parameters
                'transl': trans  # Translation
            }
            # Debug: Print hand pose parameters
            print("Left hand pose:", pose[:, 22:37])
            print("Right hand pose:", pose[:, 37:52])

            # Run SMPL-X model to get vertices dynamically
            output = model(**kwargs_pose)

            verts = output.vertices
            j3d = output.joints
            Rmat = roma.rotvec_to_rotmat(pose[:, 0])
            pelvis = j3d[:, [0]]
            j3d = (Rmat.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)
            person_center = j3d[:, [15]]
            vertices = (Rmat.unsqueeze(1) @ (verts - pelvis).unsqueeze(-1)).squeeze(-1)
            trans = torch.from_numpy(dataPKL[frame][humanID]['transl']).cuda()
            trans = trans - person_center
            vertices = vertices + trans 
            vertices = vertices.detach().cpu().numpy().squeeze().flatten().tolist()
            currentVertices.append(vertices)
            j3d = j3d + trans
            j3d = j3d.detach().cpu().numpy().squeeze().tolist()
            allKeypoints.append(j3d)
            allId.append(int(dataPKL[frame][humanID]['id']))
            allsmplx.append(vertices)
 
        allData = (allKeypoints, allId, allsmplx)
        jsonData = json.dumps(allData, cls=NumpyArrayEncoder)
        return jsonData

@app.route("/getMeshes")
def getMeshes():
    frame = int(request.args.get('frame'))
    if len(dataPKL[frame]) == 0:
        jsonData = json.loads('{}')
        return jsonData
    else:
        allsmplx = []
        allId = []
        allKeypoints = []
        for humanID in range(len(dataPKL[frame])):
            allKeypoints.append(dataPKL[frame][humanID]['j3d_smplx'][0:keypointsNumber])
            allId.append(int(dataPKL[frame][humanID]['id']))
        allData = (allKeypoints, allId, allsmplx)
        jsonData = json.dumps(allData, cls=NumpyArrayEncoder)
        return jsonData

@app.route("/get")
def get():
    frame = int(request.args.get('frame'))
    if len(dataPKL[frame]) == 0:
        jsonData = json.loads('{}')
        return jsonData
    else:
        allsmpl3DKeyPoints = []
        allId = []
        for humanID in range(len(dataPKL[frame])):
            smpl3DKeyPoints = dataPKL[frame][humanID]['j3d_smplx'][0:keypointsNumber]
            allsmpl3DKeyPoints.append(smpl3DKeyPoints)
            allId.append(int(dataPKL[frame][humanID]['id']))
        allData = (allsmpl3DKeyPoints, allId)
        jsonData = json.dumps(allData, cls=NumpyArrayEncoder)
        return jsonData

@app.route("/")
def index():
    return render_template("index.html")
 
def find_open_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    _, port = sock.getsockname()
    sock.close()
    return port
 
if __name__ == "__main__":
    print("main")
    dynamic_port = find_open_port()
    app.run(debug=True, port=dynamic_port)
