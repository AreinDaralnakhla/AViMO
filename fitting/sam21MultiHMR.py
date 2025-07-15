# This is a hack to make this script work from outside the root project folder (without requiring install)
import pickle
import sys
import cv2
import os
import av
import os.path as osp
import numpy as np
import torch
from sam2.make_sam import make_sam_from_state_dict
from sam2.demo_helpers.video_data_storage import SAM2VideoObjectResults
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.distance import pdist

models_path = os.environ["MODELS_PATH"]

def initializeVideoMasking ( allFrameHumans, index, frame_idx, memory_per_obj_dict, sammodel, reverse=False ):
    # Generate & store prompt memory encodings for each object as needed
    for human in range(len(allFrameHumans[index])):
        if reverse:
            if allFrameHumans[index][human]['id'] == -1:
                continue
        locNorm = allFrameHumans[index][human]['locNorm']
        obj_prompts = { 'box_tlbr_norm_list': [], 'fg_xy_norm_list': [(locNorm[0], locNorm[1])], 'bg_xy_norm_list': [] }
        # Loop over all sets of prompts for the current frame
        obj_key_name =  str(human)
#        print(f"Generating prompt for object: {obj_key_name} (frame {frame_idx})")
        init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(encoded_imgs_list, **obj_prompts,
                                                                          mask_index_select=2)
        memory_per_obj_dict[obj_key_name].store_prompt_result(frame_idx, init_mem, init_ptr)

def processFrame ( frame, frame_idx, encoded_imgs_list, memory_per_obj_dict, sammodel, colors ):
    # Update tracking using newest frame
    combined_mask_result = np.zeros(frame.shape, dtype=np.uint8)
    combined_mask_id_result = np.zeros(frame.shape, dtype=np.uint8)
    for obj_key_name, obj_memory in memory_per_obj_dict.items():
        obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(encoded_imgs_list, 
                                                                                            **obj_memory.to_dict())
        # Skip storage for bad results (often due to occlusion)
        obj_score = obj_score.item()
        if obj_score < 0:
    #                print(f"Bad object score for {obj_key_name}! Skipping memory storage...")
            continue
        # Store 'recent' memory encodings from current frame (helps track objects with changing appearance)
        # -> This can be commented out and tracking may still work, if object doesn't change much
        obj_memory.store_result(frame_idx, mem_enc, obj_ptr)
        obj_mask = torch.nn.functional.interpolate(
            mask_preds[:, best_mask_idx, :, :],
            size=combined_mask_result.shape[:2],
            mode="bilinear"  # Use nearest-neighbor interpolation
            )            
        obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()
        idsam = int(obj_key_name)
        for j in range(len(allFrameHumans[frame_idx])):
            loc = allFrameHumans[frame_idx][j]['loc']
            # x1, y1 = getCoordinates ( loc[0], loc[1], Oh, factor)
            x1, y1 = loc[0], loc[1]
            if 0 <= int(y1) < obj_mask_binary.shape[0] and 0 <= int(x1) < obj_mask_binary.shape[1]:
                if obj_mask_binary[int(y1), int(x1)] == True:
                    allFrameHumans[frame_idx][j]['idsam'] = idsam
                  #  print (frame_idx, j, idsam)
                    break            
        color = colors[obj_key_name]
        combined_mask_result[obj_mask_binary] = color
        combined_mask_id_result[obj_mask_binary] =  (idsam+1)*(255, 255, 255)

    return combined_mask_id_result, combined_mask_result


if len(sys.argv) < 8:
    print ("Usage: python sam21MultiHMR.py <pkl_path> <video_path> <human> <track_size_min> <output_pkl_path> <output_video_path> <dispersion threshold> <root_path>")
    sys.exit(1)

# Define pathing & device usage
input_pkl_path = sys.argv[1]
video_path = sys.argv[2] 
human = int(sys.argv[3])
trackSizeMin = int(sys.argv[4])
output_pkl_path = sys.argv[5]
output_video_path = sys.argv[6]
dispersionthreshold = float(sys.argv[7])
root_path = sys.argv[8] 


# Open the pkl file
print ("Read pose pkl: ", input_pkl_path)
file = open(input_pkl_path, 'rb')
dataPKL = pickle.load(file) 
file.close()

# Calculate the maximum number of humans and the maximum id
maxHumans = 0
maxId = 0
maxIndex = -1

allFrameHumans = dataPKL['allFrameHumans']

#newAllFrameHumans = copy.deepcopy(allFrameHumans)

for i in range(len(allFrameHumans)):
    currentHumans = len(allFrameHumans[i])
    if currentHumans > maxHumans:
        maxHumans = currentHumans
        maxIndex = i
    for j in range(len(allFrameHumans[i])):
        maxId = max(maxId, allFrameHumans[i][j]['id'])

print('maxHumans: ', maxHumans)
print('maxId: ', maxId)
print('maxIndex: ', maxIndex)

# Calculate the size of each track
tracksSize = np.zeros(maxId+1, dtype=int)
for i in range(len(allFrameHumans)):
    for j in range(len(allFrameHumans[i])):
        tracksSize[allFrameHumans[i][j]['id']] += 1
print('Tracks Size: ', tracksSize)

# Create the tracks
tracks = []
for i in range(maxId+1):
    tracks.append(np.zeros((tracksSize[i],2), dtype=int))

# Create the tracksCurrentPosition
tracksCurrentPosition = np.zeros(maxId+1, dtype=int)

for i in range(len(allFrameHumans)):
    for j in range(len(allFrameHumans[i])):
        idToProcess = allFrameHumans[i][j]['id']
        tracks[idToProcess][tracksCurrentPosition[idToProcess]] = [i, j]
        tracksCurrentPosition[idToProcess] += 1

# We remove the tracks with less than trackSizeMin elements
for t in range(len(tracks)):
    if (tracksSize[t] < trackSizeMin):
        #print (t, tracksSize[t],tracks[t][0][0])
        for i in range(tracksSize[t]):
            allFrameHumans[tracks[t][i][0]][tracks[t][i][1]]['id'] = -1


for i in range(len(allFrameHumans)):
    for j in range(len(allFrameHumans[i])):
        loc = allFrameHumans[i][j]['loc']
        #x1, y1 = getCoordinates ( loc[0], loc[1], Oh, factor)
        x1, y1 = loc[0], loc[1]
        x1 = x1/dataPKL['video_width']
        y1 = y1/dataPKL['video_height']
        allFrameHumans[i][j]['locNorm'] = [x1, y1]

for i in range(len(allFrameHumans)):
    for j in range(len(allFrameHumans[i])):
        locNorm = allFrameHumans[i][j]['locNorm']
        x1 = locNorm[0]
        y1 = locNorm[1]
        if (x1 < 0) or (y1 < 0) or (x1 > 1) or (y1 > 1):
            allFrameHumans[i][j]['id'] = -1


allHumansLoc = []
for i in range(len(allFrameHumans)):
    humansLoc = []
    for j in range(len(allFrameHumans[i])):
        if allFrameHumans[i][j]['id'] == -1:
            continue
        loc = allFrameHumans[i][j]['loc']
        #x1, y1 = getCoordinates ( loc[0], loc[1], Oh, factor)
        x1, y1 = loc[0], loc[1]
        x1 = x1/dataPKL['video_width']
        y1 = y1/dataPKL['video_height']
        humansLoc.append([x1, y1])
        allFrameHumans[i][j]['locNorm'] = [x1, y1]
    allHumansLoc.append(humansLoc)

lengths = []
for i in range(len(allFrameHumans)):
    lengths.append(len(allFrameHumans[i]))

maxLengths = max(lengths)
print ('maxLengths: ', maxLengths)
print ('maxLengthsIndex: ', lengths.index(maxLengths))

maxHumans = 0
for i in range(len(allFrameHumans)):
    currentHumans = 0
    for j in range(len(allFrameHumans[i])):
        if allFrameHumans[i][j]['id'] != -1:
            currentHumans = currentHumans + 1
    if currentHumans > maxHumans:
        maxHumans = currentHumans
        maxIndex = i
        print (i, maxHumans)
    for j in range(len(allFrameHumans[i])):
        maxId = max(maxId, allFrameHumans[i][j]['id'])

print ('maxHumans: ', maxHumans)
print ('maxId: ', maxId)
print ('maxIndex: ', maxIndex)

# Calculate optimal maxIndex


# On parcourt les frames du début vers la fin
for i in range(maxIndex,len(allHumansLoc)):
    # Vérifier si le nombre de danseurs dans cette frame égale maxHumans
    if len(allHumansLoc[i]) == maxHumans and maxHumans > 0:
        # Calculer la distance minimum entre les danseurs
        # allHumansLoc[i] est une liste de [x, y] pour chaque danseur
        coords = np.array(allHumansLoc[i])  # Nx2
        if len(coords) > 1:
            # Calculer toutes les distances paires
            distances = pdist(coords)  # distances entre chaque paire de points
            min_dist = np.min(distances)
        else:
            # S'il n'y a qu'un danseur, la question de dispersion ne se pose pas réellement
            # On considère que la condition de dispersion est triviale
            min_dist = float('inf')

        # Vérifier si la distance minimale entre toutes les têtes est > dispersionthreshold
        if min_dist > dispersionthreshold:
            maxIndex = i
            break  # On prend la première frame satisfaisant la condition

if maxIndex == -1:
    print("Aucune frame ne satisfait les conditions (maxHumans présents et têtes suffisamment séparées).")

maxIndex += 1  # On veut inclure la frame maxIndex

print("maxIndex optimal trouvé :", maxIndex)


for i in range(len(allFrameHumans)):
    for j in range(len(allFrameHumans[i])):
        allFrameHumans[i][j]['idsam'] = -1

# Define image processing config (shared for all video frames)
imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}
# Set up memory storage for tracked objects
# -> Assumes each object is represented by a unique dictionary key (e.g. 'obj1')
# -> This holds both the 'prompt' & 'recent' memory data needed for tracking!
memory_per_obj_dict = defaultdict(SAM2VideoObjectResults.create)

white = (255, 255, 255)
# Define colors for each object
colors = {
    "0": (255, 0, 255),  # Green
    "1": (0, 255, 0),  # Green
    "2": (0, 0, 255),  # Red
    "3": (255, 0, 0),  # Blue
    "4": (255, 255, 0),  # Green
    "5": (0, 255, 255),  # Red
    "6": (255, 0, 255),  # Blue
    "7": (0, 128, 0),  # Green
    "8": (0, 0, 128),  # Red
    "9": (128, 0, 0),  # Blue
    "10": (255, 255, 255),  # Green
    "11": (0, 255, 0),  # Green
    "12": (0, 0, 255),  # Red
    "13": (255, 0, 0),  # Blue
    "14": (255, 255, 0),  # Green
    "15": (0, 255, 255),  # Red
    "16": (255, 0, 255),  # Blue
    "17": (0, 128, 0),  # Green
    "18": (0, 0, 128),  # Red
    "19": (128, 0, 0),  # Blue
}

# Read first frame to check that we can read from the video, then reset playback
vcap = cv2.VideoCapture(video_path)
ok_frame, first_frame = vcap.read()
if not ok_frame:
    raise IOError(f"Unable to read video frames: {video_path}")
vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = vcap.get(cv2.CAP_PROP_FPS)
width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_size = (first_frame.shape[1], first_frame.shape[0])

# Set up model
print("Loading model...")
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16
# Initialiser le modèle SAM 2.1
model_name = os.path.join(models_path, 'sam2', 'sam2.1_hiera_large.pt')

#model_config_dict, sammodel = make_samv2_from_original_state_dict(model_name)
model_config_dict, sammodel = make_sam_from_state_dict(model_name)

sammodel.to(device=device, dtype=dtype)

# Initialize VideoWriter for output video
#fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Codec for lossless video

print ("total_frames: ", total_frames)
print ("frame_size: ", frame_size)

#out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

outputVideoContainer = av.open(output_video_path, mode='w')
# Configurer les options du codec avec le multithreading
codec_options = {
    'lossless': '1',
    'preset': 'veryslow',
    'crf': '0',
    'threads': 'auto',
}
# Ajouter un flux vidéo de sortie avec le codec H.264
outputStream = outputVideoContainer.add_stream('libx264rgb',options=codec_options)
outputStream.width = width
outputStream.height = height
outputStream.pix_fmt = 'rgb24'  # Format pixel standard pour libx264
outputStream.thread_type = 'AUTO'
# Verify stream is properly configured
if not outputStream.codec_context.is_open:
    outputStream.codec_context.open()

previousFrames = []
combined_mask_results = []
combined_id_mask_results = []

pbar = tqdm(total=total_frames, unit=' frames', dynamic_ncols=True, position=0, leave=True)

# Define paths for masks directory
masks_dir = osp.join(root_path, "masks")
os.makedirs(masks_dir, exist_ok=True)

def save_mask_frame(frame_idx, mask_frame):
    """
    Save the mask frame as an image in the masks directory.
    """
    mask_path = osp.join(masks_dir, f"{frame_idx}.png")
    cv2.imwrite(mask_path, mask_frame)

# Process video frames
close_keycodes = {27, ord("q")}  # Esc or q to close
try:
    # First pass: Process all frames up to maxIndex to find initialization frame
    initialization_frame_idx = maxIndex - 1 if maxIndex > 0 else 0
    
    # Second pass: Process all frames sequentially
    for frame_idx in range(total_frames):
        ok_frame, frame = vcap.read()
        if not ok_frame:
            break
            
        # Encode frame data (shared for all objects)
        encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)
        
        # Initialize only at the optimal frame
        if frame_idx == initialization_frame_idx:
            initializeVideoMasking(allFrameHumans, maxIndex, frame_idx, memory_per_obj_dict, sammodel)
        
        # Process the frame
        combined_mask_id_result, combined_mask_result = processFrame(
            frame, frame_idx, encoded_imgs_list, memory_per_obj_dict, sammodel, colors
        )
        
        # Write frame immediately to output
        outframe = av.VideoFrame.from_ndarray(combined_mask_id_result, format='rgb24')
        outframe = outframe.reformat(format='rgb24')
        packets = outputStream.encode(outframe)
        for packet in packets:
            outputVideoContainer.mux(packet)
        
        # Save mask frame
        save_mask_frame(frame_idx, combined_mask_result)
        
        pbar.update(1)
    
    pbar.close()

    with open(output_pkl_path, 'wb') as handle:
        pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL)

finally:
    # Release the VideoWriter
    packets = outputStream.encode(None)
    for packet in packets:
        outputVideoContainer.mux(packet)    
    vcap.release()
    cv2.destroyAllWindows()


    # python .\sam21MultiHMR.py .\pkl\D0-015_896L-clean-track.pkl ..\videos\D0-talawa_technique_intro-Scene-015.mp4 3 5 .\pkl\D0-015_896L-clean-track-seg.pkl seg-track-D0-015.mp4
    # python .\sam21MultiHMR.py .\pkl\T3-2-1024-896L-Clean-Track.pkl "C:\Users\colantoni\Nextcloud\Tmp\Tracking\T3-2-1024.MP4" 3 30 .\pkl\T3-2-1024-896L-Clean-Track-Seg.pkl
    # python .\sam21MultiHMR.py .\pkl\T3-2-1024-896L-Clean-Track.pkl "F:\MyDrive\Tmp\Tracking\T3-2-1024.MP4" 2 30 .\pkl\T3-2-1024-896L-Clean-Track-Seg.pkl
    # python .\sam21MultiHMR.py .\pkl\D0-talawa_technique_intro-Scene-003_896L-clean-track.pkl ..\videos\D0-talawa_technique_intro-Scene-003.mp4 3 1 .\pkl\D0-talawa_technique_intro-Scene-003_896L-clean-track-seg.pkl

# python .\sam21MultiHMR.py .\pkl\T3-2-1024-896L-Clean-Track.pkl "C:\Users\colantoni\Nextcloud\Tmp\Tracking\T3-2-1024.MP4" 3 30 .\pkl\T3-2-1024-896L-Clean-Track-Seg.pkl