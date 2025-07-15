import os
import sys
import cv2
import av
import numpy as np
import torch
import pickle

from tqdm import tqdm
from premiere.functionsMoge import initMoGeModel, computeFloorAngle, computeFov, colorizeDepthImage
from premiere.functionsDepth import packDepthImage, computeMinMaxDepth

models_path = os.environ["MODELS_PATH"]

# Validate command line arguments
if len(sys.argv) != 6:
    print("Usage: python videoAnalysisMoge.py <input_video> <output_video> <output_pkl> <depthmaps_dir> <fov>")
    sys.exit(1)

videoInName = sys.argv[1]
videoOutName = sys.argv[2]
outputPklName = sys.argv[3]
depthmaps_dir = sys.argv[4]
fov_x_degrees = float(sys.argv[5])
useFixedFov = (fov_x_degrees != 0)

# Initialize the MoGe model on the specified device (CUDA))
device_name = 'cuda' 
device, model = initMoGeModel(device_name)

# Open the input video
video = cv2.VideoCapture(videoInName)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if not video.isOpened():
    print('[!] Error opening the video.')
    sys.exit(1)

# Prepare an output container (PyAV) to store the depth images as video
outputContainer = av.open(videoOutName, mode='w')
codec_options = {
    'lossless': '1',
    'preset': 'veryslow',
    'crf': '0',
    'threads': '12',
}
outputStream = outputContainer.add_stream('libx264rgb', options=codec_options)
outputStream.width = width
outputStream.height = height
outputStream.pix_fmt = 'rgb24'
outputStream.thread_type = 'AUTO'

# Ensure the codec context is opened
if not outputStream.codec_context.is_open:
    outputStream.codec_context.open()

print(f"Total number of frames in video: {frames_count}")

# Determine frame indices to process
step_size = 40  # Process every 4th frame to reduce computation
frame_indices = list(range(0, frames_count, step_size))
print(f"Processing {len(frame_indices)} frames out of {frames_count}")

# Create directory to save depth maps
os.makedirs(depthmaps_dir, exist_ok=True)

# Progress bar to track processing
pbar = tqdm(frame_indices, unit=' frames', dynamic_ncols=True, position=0, leave=True)

# This list will hold the metadata for each processed frame
allFrameData = []

# Loop over selected frames
for i in pbar:
    # Set the video position to the current frame index
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = video.read()
    if ret:
        # Convert the frame to a torch tensor (normalized to [0,1])
        image_tensor = torch.tensor(frame / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        print(f"Image tensor shape: {image_tensor.shape}")  # Should be [3, height, width]

        # If the user provided a non-zero FOV, use it; otherwise let MoGe compute or estimate
        if useFixedFov:
            output = model.infer(image_tensor, fov_x=fov_x_degrees)
        else:
            output = model.infer(image_tensor)

        # Extract outputs
        depth = output['depth'].cpu().numpy()        # Depth map
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Depth map min: {depth.min()}, max: {depth.max()}")

        # Save depth map as grayscale image
        depthmap_path = os.path.join(depthmaps_dir, f"{i}.png")
        depth_image = (depth / depth.max() * 255).astype(np.uint8)  # Normalize depth to [0, 255]
        cv2.imwrite(depthmap_path, depth_image)

        intrinsics = output['intrinsics'].cpu().numpy()
        pointCloud3D = output['points'].cpu().numpy()
        mask = output['mask'].cpu().numpy()          # Validity mask

        # Compute floor angle using a subset of points
        la, lb, angle_radians, angle = computeFloorAngle(
            pointCloud3D, mask, max_points=20000, quantile=0.01, displayChart=False
        )

        # Compute or extract the field of view
        fov_x, fov_y = computeFov(intrinsics)
        fov_x_degrees = np.degrees(fov_x)
        fov_y_degrees = np.degrees(fov_y)

        # Compute min/max depth within the mask
        min_depth, max_depth = computeMinMaxDepth(depth, mask)

        # Record metadata for this frame
        data = {
            'min_depth': float(min_depth),
            'max_depth': float(max_depth),
            'fov_x': float(fov_x),
            'fov_y': float(fov_y),
            'fov_x_degrees': float(fov_x_degrees),
            'fov_y_degrees': float(fov_y_degrees),
            'angle': float(angle),
            'angle_radians': float(angle_radians),
            'la': float(la),
            'lb': float(lb),
        }
        allFrameData.append(data)

        # Pack the depth image into a 3-channel RGB array (for writing as a video)
        colorImage = packDepthImage(depth, mask, min_depth, max_depth)

        # Convert to a PyAV VideoFrame and encode
        outframe = av.VideoFrame.from_ndarray(colorImage, format='rgb24')
        outframe = outframe.reformat(format='rgb24')
        packets = outputStream.encode(outframe)
        for packet in packets:
            outputContainer.mux(packet)

pbar.close()

# Save metadata to Pkl file
print('Saving data into Pkl file...')
with open(outputPklName, 'wb') as file:
    pickle.dump(allFrameData, file, protocol=pickle.HIGHEST_PROTOCOL)

# Finalize and close the output video stream
packets = outputStream.encode(None)
for packet in packets:
    outputContainer.mux(packet)
outputStream.close()
outputContainer.close()

video.release()