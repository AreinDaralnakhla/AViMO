import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

if len(sys.argv) != 5:
    print("Usage: python computeCameraProperties.py <moge_pkl> <input_pkl> <fov> <final_json>")
    sys.exit(1)
    
display = False

mogePklName = sys.argv[1]
inputPklName = sys.argv[2]
fov_x_degrees = float(sys.argv[3])
finalJSONName = sys.argv[4]

print ("Reading PKL: ", mogePklName)
with open(mogePklName, 'rb') as file:
    mogePKL = pickle.load(file)


if (fov_x_degrees == 0):
    fov_x_degrees = 0
    for i in range(len(mogePKL)):
        fov_x_degrees += mogePKL[i]['fov_x_degrees']
    fov_x_degrees /= len(mogePKL)

print ("Fov_x: ",fov_x_degrees)

print ("Reading PKL: ", inputPklName)
with open(inputPklName, 'rb') as file:
    dataPKL = pickle.load(file)

allFrameHumans = dataPKL['allFrameHumans']

fov_x_degrees = 0
for i in range(len(mogePKL)):
    fov_x_degrees += mogePKL[i]['fov_x_degrees']
fov_x_degrees /= len(mogePKL)
fov_x_rad = fov_x_degrees * np.pi / 180

angle = 0
for i in range(len(mogePKL)):
    angle += mogePKL[i]['angle']
angle /= len(mogePKL)
angle_rad = angle * np.pi / 180

all3DJoints = []

for i in range(len(allFrameHumans)):
    humans = allFrameHumans[i]
    if len(humans) == 0:
        continue

    for h in humans:
        all3DJoints.append(h['j3d_smplx'])

min_y = 0
all3DJoints = np.array(all3DJoints)
if len(all3DJoints) != 0:
    print(all3DJoints.shape) # resultslike (N, 127, 3)
    rad = np.radians(-angle)
    R_x = np.array([
        [1,           0,            0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad),  np.cos(rad)]
    ])
    all3DJoints = all3DJoints.dot(R_x.T)

    x_coords = all3DJoints[:, :, 2].flatten()
    y_coords = -all3DJoints[:, :, 1].flatten()

    min_y = np.min(y_coords)

    if display:
        plt.scatter(x_coords, y_coords, s=2)
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.title("2D Projection (X,Z)")
        plt.show()
    
allData = {
        'video': dataPKL['video'],
        'fov_x_deg': fov_x_degrees,
        'fov_x_rad': fov_x_rad,
        'video_width': dataPKL['video_width'],
        'video_height': dataPKL['video_height'],
        'video_fps': dataPKL['video_fps'],
        'floor_angle_deg': angle,
        'floor_angle_rad': angle_rad,
        'floor_Zoffset': min_y
}
print(allData)

print ("Writing JSON: ", finalJSONName)
with open(finalJSONName, 'w') as file:
    json.dump(allData, file)
        






        