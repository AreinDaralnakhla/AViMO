import cv2
import sys
import pickle
import numpy as np
import av
import matplotlib.pyplot as plt

from premiere.functionsMoge import colorizeDepthImage
from premiere.functionsDepth import unpackDepthImage, packDepthImage, computeMinMaxDepth
from tqdm import tqdm

def align_depth_map(ref_depth, ref_motion_mask, target_depth, motion_mask, depth_mask):
    """
    Aligne la carte de profondeur target_depth sur ref_depth en utilisant uniquement
    les pixels statiques (motion_mask == 0) et valides (depth_mask == True).

    La relation attendue est:
        ref_depth ≈ a * target_depth + b
    où a et b sont estimés par régression linéaire (moindres carrés)
    sur les pixels sélectionnés.

    Parameters:
    -----------
    ref_depth : np.array
        Carte de profondeur de référence.
    target_depth : np.array
        Carte de profondeur à aligner.
    motion_mask : np.array (dtype uint8 ou bool)
        Masque indiquant les zones en mouvement : 1 pour les personnes en mouvement, 0 pour les pixels statiques.
    depth_mask : np.array (dtype bool ou uint8)
        Masque indiquant les pixels valides dans target_depth (True/1 pour les pixels valides, False/0 sinon).

    Returns:
    --------
    aligned_depth : np.array
        Carte de profondeur recalée.
    a : float
        Facteur d'échelle estimé.
    b : float
        Décalage estimé.
    """
    # Créer le masque combiné : on ne considère que les pixels statiques ET valides
    static_mask = (motion_mask == 0)
    ref_static_mask = (ref_motion_mask == 0)
    valid_pixels = ref_static_mask & static_mask & depth_mask.astype(bool)

    # Vérifier qu'il y a suffisamment de pixels pour la régression
    if np.sum(valid_pixels) < 10:
        # S'il y a trop peu de pixels, on retourne la depth_map sans transformation
        return target_depth, 1.0, 0.0

    # Extraire les valeurs pour la régression
    ref_vals = ref_depth[valid_pixels].flatten()
    target_vals = target_depth[valid_pixels].flatten()

    # Estimer a et b par régression linéaire (moindres carrés)
    A = np.vstack([target_vals, np.ones(len(target_vals))]).T
    solution, _, _, _ = np.linalg.lstsq(A, ref_vals, rcond=None)
    a, b = solution

    # Appliquer la transformation sur toute la frame
    aligned_depth = a * target_depth + b
    return aligned_depth, a, b

# Check for proper argument count
if len(sys.argv) != 7:
    print("Usage: python stabilyzeDepthMap.py <input_depth_pkl> <input_depth_video> <input_seg_video> <output_depth_pkl> <output_depth_video> <display: 0 No, 1 Yes>")
    sys.exit(1)

# Parse command-line arguments
inputDepthPklName = sys.argv[1]
inputDepthVideolName = sys.argv[2]
inputSegVideolName = sys.argv[3]
outputDepthPklName = sys.argv[4]
outputDepthVideolName = sys.argv[5]
display = int(sys.argv[6]) == 1

# Open video captures
videoDepth = cv2.VideoCapture(inputDepthVideolName)
videoMask = cv2.VideoCapture(inputSegVideolName)

# Read input pkl with extra data
print("Read input depth pkl: ", inputDepthPklName)
# Read additional pkl for depth info
with open(inputDepthPklName, 'rb') as file:
    videoDepthPKL = pickle.load(file)

# Retrieve video properties
width = int(videoDepth.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoDepth.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(videoDepth.get(cv2.CAP_PROP_FPS))
frames_count = int(videoDepth.get(cv2.CAP_PROP_FRAME_COUNT))

# Check if videos opened properly
if videoDepth.isOpened() == False:
    print('[!] error opening the depth video')
    sys.exit(1)
if videoMask.isOpened() == False:
    print('[!] error opening the mask video')
    sys.exit(1)

outputContainer = av.open(outputDepthVideolName, mode='w')
# Configurer les options du codec avec le multithreading
codec_options = {
    'lossless': '1',
    'preset': 'veryslow',
    'crf': '0',
    'threads': '12',
}
# Ajouter un flux vidéo de sortie avec le codec H.264
outputStream = outputContainer.add_stream('libx264rgb',options=codec_options)
outputStream.width = width
outputStream.height = height
outputStream.pix_fmt = 'rgb24'  # Format pixel standard pour libx264
outputStream.thread_type = 'AUTO'
# Verify stream is properly configured
if not outputStream.codec_context.is_open:
    outputStream.codec_context.open()

videoOutDepthPKL = []

frameCount = 0
pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

ref_depth = None
ref_depth_mask = None

# Main loop over frames
while videoDepth.isOpened():
    # Read frames    
    retDepth, frameDepth = videoDepth.read()
    retMask, frameMask = videoMask.read()

    # End loop if frames cannot be read
    if not retDepth or not retMask:
        break

    # Mask out the background
    frameMaskGray = frameMask[:, :, 0]

    # Retrieve min/max depth from videoDepthPKL for current frame
    min_depth = videoDepthPKL[frameCount][0]
    max_depth = videoDepthPKL[frameCount][1]

    # Convert raw depth frame to visible scale
    depthMap, depthMask = unpackDepthImage(frameDepth, min_depth, max_depth, brg2rgb=True)
    depthMask = ~depthMask

    min_depth, max_depth = computeMinMaxDepth(depthMap, depthMask)
    if (frameCount == 0):
        # Initialize reference depth and mask
        ref_depth = depthMap.copy()
        ref_depth_mask = depthMask.copy()
        frameMaskGray_ref = frameMaskGray.copy()
    else:
        aligned_depth, a, b = align_depth_map(ref_depth, frameMaskGray_ref, depthMap,
                                              frameMaskGray, depthMask)
        depthMap = aligned_depth

    min_depth, max_depth = computeMinMaxDepth(depthMap, depthMask)

    data = [ min_depth, max_depth ]
    videoOutDepthPKL.append(data)
        
    colorImage = packDepthImage(depthMap, depthMask, min_depth, max_depth)

    # Reconvertir l'image OpenCV en un frame AV au format 'yuv444p'
    outframe = av.VideoFrame.from_ndarray(colorImage, format='rgb24')
    outframe = outframe.reformat(format='rgb24')
        # Encoder le frame
    packets = outputStream.encode(outframe)
    for packet in packets:
        outputContainer.mux(packet)
    
    # If display is enabled, show updated depth
    if display:
        visuDepth = colorizeDepthImage(depthMap)
        cv2.imshow('Depth Frame', visuDepth)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    frameCount += 1
    pbar.update(1)

pbar.close()

with open(outputDepthPklName, 'wb') as handle:
    pickle.dump(videoOutDepthPKL, handle, protocol=pickle.HIGHEST_PROTOCOL) 

# Encoder les frames restants
packets = outputStream.encode(None)
for packet in packets:
    outputContainer.mux(packet)
outputStream.close()
outputContainer.close()

# Clean up
videoDepth.release()
videoMask.release()