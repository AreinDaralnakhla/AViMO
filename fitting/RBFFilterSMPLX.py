import sys
import pickle
import torch
import copy
import os
import roma
import smplx
import platform

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import UnivariateSpline

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from premiere.functionsSMPLX import updateHumanFromSMPLX
from premiere.functionsCommon import projectPoints3dTo2d

os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def buildInterpolator ( Xm, Ym, rbfkernel='linear', rbfepsilon=1.0, rbfsmooth=.01, neighbors=400):
    # Compute z-scores to identify outliers in Ym
    mean_Ym = np.mean(Ym)
    std_Ym = np.std(Ym)
    if std_Ym == 0:
        z_scores = np.zeros_like(Ym)
    else:
        z_scores = np.abs((Ym - mean_Ym) / std_Ym)
    threshold = 4  # Threshold for outlier detection

    # Keep only inliers
    inliers = z_scores < threshold
    Xm_filtered = Xm[inliers]
    Ym_filtered = Ym[inliers]

    # Combine Xm_filtered and Ym_filtered into a single array for input to RBFInterpolator
    points = np.column_stack((Xm_filtered, np.zeros_like(Xm_filtered)))  # Assuming 1D data for Xm
    values = Ym_filtered

    # Build the RBF interpolator with filtered data

    if rbfkernel == 'univariatespline':
        interpolator = UnivariateSpline(Xm_filtered, Ym_filtered, k=3, s=rbfsmooth)
    else:
        interpolator = RBFInterpolator(points, values, kernel=rbfkernel, epsilon=rbfepsilon, smoothing=rbfsmooth)

#    if platform.system() == 'Windows':
#        interpolator = RBFInterpolator(points, values, kernel=rbfkernel, epsilon=rbfepsilon, smoothing=rbfsmooth)
#    else:
#        interpolator = RBFInterpolator(points, values, kernel=rbfkernel, epsilon=rbfepsilon, smoothing=rbfsmooth )#  , neighbors=neighbors)
    
    return interpolator

def RBFFilterTrackRotvec(k, dataPKL, track, trackSize, rbfkernel, rbfsmooth, rbfepsilon, ava):
    # Taille réelle de l'intervalle de frames à traiter
    realSize = track[trackSize - 1][0] - track[0][0] + 1
    size = realSize - trackSize

    # Initialiser les tableaux pour les frames manquantes et existantes
    if (size != 0):
        Xmt = np.zeros(size, dtype=float)
    Xm = np.zeros(trackSize, dtype=float)
    
    # Ym va contenir les quaternions pour chaque frame existante
    Ym = np.zeros((trackSize, 4), dtype=float)  # Les quaternions ont 4 composantes (x, y, z, w)

    # Étape 1 : Convertir les vecteurs de rotation en quaternions
    for i in range(trackSize):
        trackRotvec = dataPKL[track[i][0]][track[i][1]]['rotvec'][k]  # Obtenir le vecteur de rotation
        quat = R.from_rotvec(trackRotvec).as_quat()  # Convertir en quaternion
        norm = np.linalg.norm(quat)  # Norme du quaternion
        if (norm != 0):
            quat /= norm  # Normaliser le quaternion

        Xm[i] = float(track[i][0] - track[0][0]) / realSize  # Normaliser la position temporelle
        Ym[i] = quat  # Stocker le quaternion pour interpolation
        if (i > 0):
            if np.dot(Ym[i - 1], Ym[i]) < 0:
                Ym[i] = -Ym[i]

    rbfi = [None, None, None, None]  # Initialiser les interpolateurs RBF pour chaque composante du quaternion
    di = [None, None, None, None]  # Initialiser les valeurs interpolées pour chaque composante du quaternion
    # Étape 2 : Interpolation de chaque composante des quaternions (x, y, z, w)
    for c in range(4):  # Les 4 composantes du quaternion
        Ym_component = Ym[:, c]  # Prendre la composante c de tous les quaternions
        rbfi[c] = buildInterpolator (Xm, Ym_component, rbfkernel, rbfepsilon, rbfsmooth)  # Interpolateur RBF
        
        # If using univariate spline, pass just Xm as 1D; otherwise pass the 2D points
        if rbfkernel == 'univariatespline':
            di[c] = rbfi[c](Xm)
        else:
            newXm = np.column_stack((Xm, np.zeros_like(Xm)))
            di[c] = rbfi[c](newXm)  # Interpolation des valeurs pour les frames présentes

    for i in range(trackSize):
        quat = np.array([di[c][i] for c in range(4)])
        norm = np.linalg.norm(quat)  # Norme du quaternion
        if (norm != 0):
            quat /= norm  # Normaliser le quaternion
            dataPKL[track[i][0]][track[i][1]]['rotvec'][k] = R.from_quat(quat).as_rotvec()  # Remplacer par la valeur interpolée
        else:
            dataPKL[track[i][0]][track[i][1]]['rotvec'][k] = np.zeros(3)

    if (size != 0):
        # Interpolation pour les frames manquantes
        count = 0
        for i in range(realSize):
            if ava[i] == 0:
                Xmt[count] = float(i) / realSize
                count += 1
            
        dit = [None, None, None, None]  # Initialiser les valeurs interpolées pour chaque composante du quaternion
        # Obtenir les valeurs interpolées pour les frames manquantes
        for c in range(4):  # Les 4 composantes du quaternion
            if rbfkernel == 'univariatespline':
                dit[c] = rbfi[c](Xmt)
            else:
                newXmt = np.column_stack((Xmt, np.zeros_like(Xmt)))
                dit[c] = rbfi[c](newXmt)  # Interpolated values for missing frames

        # Mise à jour des valeurs interpolées pour les frames manquantes
        count = 0
        for i in range(realSize):
            if ava[i] == 0:
                interpolated_quat = np.array([dit[c][count] for c in range(4)])
                norm = np.linalg.norm(interpolated_quat)  # Norme du quaternion
                if (norm != 0):
                    interpolated_quat /= norm  # Normaliser le quaternion
                dataPKL[track[0][0] + i][-1]['rotvec'][k] = R.from_quat(interpolated_quat).as_rotvec()  # Stocker le vecteur de rotation interpolé
                count += 1
    return k

def RBFFilterTrackArray ( key, dim, dataPKL, track, trackSize, rbfkernel, rbfsmooth, rbfepsilon, ava ):
    realSize = track[trackSize-1][0] - track[0][0] + 1

    size = realSize - trackSize
    if (size != 0):
        Xmt = np.zeros(size, dtype=float)
    Xm = np.zeros(trackSize, dtype=float)
    Ym = np.zeros(trackSize, dtype=float)
    
    for c in range(dim):
        for i in range(trackSize):
            trackPositions = dataPKL[track[i][0]][track[i][1]][key]
            Xm[i] = float(track[i][0]-track[0][0])/realSize
            Ym[i] = trackPositions[c] 
        rbfi = buildInterpolator (Xm, Ym, rbfkernel, rbfepsilon, rbfsmooth)  # Interpolateur RBF
        
        
        # Evaluate interpolator for existing frames
        if rbfkernel == 'univariatespline':
            di = rbfi(Xm)  # Pass 1D
        else:
            newXm = np.column_stack((Xm, np.zeros_like(Xm)))
            di = rbfi(newXm)   # interpolated values
        for i in range(trackSize):
            dataPKL[track[i][0]][track[i][1]][key][c] = di[i]
        if (size != 0):
            count = 0
            for i in range(realSize):
                if ava[i] == 0:
                    Xmt[count] = float(i)/realSize
                    count += 1

            if rbfkernel == 'univariatespline':
                dit = rbfi(Xmt)
            else:
                newXmt = np.column_stack((Xmt, np.zeros_like(Xmt)))
                dit = rbfi(newXmt)   # interpolated values
            count = 0
            for i in range(realSize):
                if ava[i] == 0:
                    dataPKL[track[0][0]+i][-1][key][c] = dit[count]
                    count += 1
    return

def keypointsBBox3D(keypoints):
    """
    Compute the 3D bounding box of a set of keypoints.

    This function returns the minimum and maximum coordinates among
    all keypoints provided.

    :param keypoints: 3D coordinates of keypoints (N x 3).
    :type keypoints: np.ndarray
    :return: A tuple (min_coords, max_coords), each with shape (3,).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    min_coords = np.min(keypoints, axis=0)
    max_coords = np.max(keypoints, axis=0)
    return min_coords, max_coords

def keypointsBBox2D(keypoints):
    """
    Compute the 2D bounding box of a set of keypoints.

    This function returns the minimum and maximum coordinates among
    all keypoints provided.

    :param keypoints: 2D coordinates of keypoints (N x 2).
    :type keypoints: np.ndarray
    :return: A tuple (min_coords, max_coords), each with shape (2,).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    min_coords = np.min(keypoints, axis=0)
    max_coords = np.max(keypoints, axis=0)
    return min_coords, max_coords

def main():
    if len(sys.argv) != 6:
        print("Usage: python rbffilter-interpolation.py <input_pkl> <output_pkl> <kernel> <smooth> <epsilon>")
        sys.exit(1)

    # Open the pkl file
    fileName  = sys.argv[1] 
    print ("read pkl file: ", fileName)
    file = open(fileName, 'rb')
    dataPKL = pickle.load(file) 
    file.close()

    allFrameHumans = dataPKL['allFrameHumans']

    for frame in allFrameHumans:
        frame[:] = [human for human in frame if human['id'] != -1]

    kernel = sys.argv[3]
    smooth = float(sys.argv[4])
    epsilon = float(sys.argv[5])

    # Calculate the maximum number of humans and the maximum id
    maxHumans = 0
    maxId = 0
    for i in range(len(allFrameHumans)):
        maxHumans = max(maxHumans, len(allFrameHumans[i]))
        for j in range(len(allFrameHumans[i])):
            maxId = max(maxId, allFrameHumans[i][j]['id'])
    print('maxHumans: ', maxHumans)
    print('maxId: ', maxId)

    # Calculate the size of each track
    tracksSize = np.zeros(maxId+1, dtype=int)
    for i in range(len(allFrameHumans)):
        for j in range(len(allFrameHumans[i])):
            if allFrameHumans[i][j]['id'] != -1:
                tracksSize[allFrameHumans[i][j]['id']] += 1
    print('tracksSize: ', tracksSize)
    print ('total frames: ', len(allFrameHumans))
    # Create the tracks
    tracks = []
    for i in range(maxId+1):
        tracks.append(np.zeros((tracksSize[i],2), dtype=int))

    # Create the tracksCurrentPosition
    tracksCurrentPosition = np.zeros(maxId+1, dtype=int)

    for i in range(len(allFrameHumans)):
        for j in range(len(allFrameHumans[i])):
            if (allFrameHumans[i][j]['id'] != -1):
                idToProcess = allFrameHumans[i][j]['id']
                tracks[idToProcess][tracksCurrentPosition[idToProcess]] = [i, j]
                tracksCurrentPosition[idToProcess] += 1

    tracksStart = np.zeros(maxId+1, dtype=int)
    for t in range(len(tracks)):
        if (tracksSize[t] == 0):
            continue
        tracksStart[t] = tracks[t][0][0]
    #print ("Size: ",len(allFrameHumans[0]))

    notempty = 0
    while (notempty < len(allFrameHumans) and (len(allFrameHumans[notempty])==0)):
        notempty += 1
    if (notempty == len(allFrameHumans)):
        notempty = -1
    print("notempty: ",notempty)

    if (notempty != -1):
        models_path = os.environ["MODELS_PATH"]

        useExpression = True
        if (dataPKL['model_type'] == "hmr2") or (dataPKL['model_type'] == "nlf"):
            useExpression = False

        print ('[INFO] Loading SMPLX model')
        gender = 'neutral'
        modelSMPLX = smplx.create(
            models_path, 'smplx',
            gender=gender,
            use_pca=False, flat_hand_mean=True,
            num_betas=10,
            ext='npz').cuda()
        print ('[INFO] SMPLX model loaded')

        nbRotvec = allFrameHumans[notempty][0]['rotvec'].shape[0]
        print ("nbRotvec: ", nbRotvec)

        print()

        for t in range(len(tracks)):
            if (tracksSize[t] == 0):
                continue
            print ("Track: ", t,"/",len(tracks)-1)
            track = tracks[t]
            trackSize = tracksSize[t]
            print ("    trackSize: ",trackSize)
            print ("    trackStart: ",track[0][0])
            print ("    trackEnd: ",track[trackSize-1][0])
            realSize = track[trackSize-1][0] - track[0][0] + 1
            print ("    realSize: ",realSize)
            ava = np.zeros(realSize, dtype=int)
            for i in range(trackSize):
                ava[track[i][0]-track[0][0]] = 1
            count = 0
            for i in range(realSize):
                if ava[i] == 0:
                    human = copy.deepcopy(allFrameHumans[track[0][0]][track[0][1]])
                    allFrameHumans[track[0][0]+i].append(human)
                    count += 1
            print ("    count: ", count)
            pbar = tqdm(total=nbRotvec, unit=' Rotation vector', dynamic_ncols=True, position=0, leave=True)
            for k in range(nbRotvec):
                RBFFilterTrackRotvec (k, allFrameHumans, tracks[t], tracksSize[t], kernel, smooth, epsilon, ava)
                pbar.update(1)
            pbar.close()    
            print ("    rotvec done")
            RBFFilterTrackArray ( 'shape', 10 , allFrameHumans, tracks[t], tracksSize[t], kernel, smooth, epsilon, ava )
            print ("    shape done")
            if useExpression:
                RBFFilterTrackArray ( 'expression', 10 , allFrameHumans, tracks[t], tracksSize[t], kernel, smooth, epsilon, ava )
                print ("    expression done")
            RBFFilterTrackArray ( 'transl', 3 , allFrameHumans, tracks[t], tracksSize[t], kernel, smooth, epsilon, ava )
            print ()
        #print(results)

        print ("Update SMPLX")
        pbar = tqdm(total=len(allFrameHumans), unit=' frames', dynamic_ncols=True, position=0, leave=True)
        for i in range(len(allFrameHumans)):
            allFrameHumans[i] = [human for human in allFrameHumans[i] if human['id'] != -1]
            for j in range(len(allFrameHumans[i])):
                allFrameHumans[i][j] = updateHumanFromSMPLX(allFrameHumans[i][j], modelSMPLX, useExpression)
                proj_2d = projectPoints3dTo2d( allFrameHumans[i][j]['j3d_smplx'], fov=dataPKL['fov_x_deg'], width=dataPKL['video_width'], height=dataPKL['video_height'] )
                allFrameHumans[i][j]['j2d_smplx'] = proj_2d

                # Compute the 3D bounding box from keypoints
                min_coords, max_coords = keypointsBBox3D(allFrameHumans[i][j]['j3d_smplx'])
                allFrameHumans[i][j]['bbox3d'] = [min_coords, max_coords]
                # Compute the 2D bounding box from keypoints
                min_coords, max_coords = keypointsBBox3D(allFrameHumans[i][j]['j2d_smplx'])
                allFrameHumans[i][j]['bbox'] = [min_coords, max_coords]
            pbar.update(1)
        pbar.close()

    # Save the pkl file
    with open(sys.argv[2], 'wb') as handle:
        pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL) 

if __name__ == '__main__':
    main()
