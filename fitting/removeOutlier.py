import sys
import pickle
import numpy as np

from premiere.functionsCommon import buildTracks, computeMaxId

def removeOutliersInTrack(track, trackSize, allFrameHumans, fps, velocity_thresh_factor=4.0):
    """
    Remove outliers from a track based on velocity thresholds.

    :param track: list of tuples (frame_index, person_index)
    :param trackSize: number of points in the track
    :param allFrameHumans: data structure holding human data per frame
    :param fps: frames per second of the video
    :param velocity_thresh_factor: multiplicative threshold on the MAD-based speed outlier detection
    """
    times = np.zeros(trackSize, dtype=int)
    points3d = np.zeros((trackSize,3), dtype=float)

    # Collect frames (times) and 3D points
    for i in range(trackSize):
        times[i] = track[i][0]
        points3d[i] = allFrameHumans[track[i][0]][track[i][1]]['j3d_smplx'][0]

    # If the track is too short, skip outlier detection
    if trackSize < 3:
        return

    # Compute time deltas in seconds
    # dt[i] = (times[i+1] - times[i]) / fps
    dt_frames = np.diff(times)  
    dt_seconds = dt_frames / fps

    # Handle any zero time deltas if they occur
    valid_dt_mask = dt_frames != 0
    if not np.all(valid_dt_mask):
        # If some dt are zero, you may handle them differently
        # e.g., skip those segments, or set them to a very small non-zero
        # For simplicity here, let's just skip them in velocity calculation
        pass

    # Compute velocities (points3d are in “units” -> e.g., meters if scaled; otherwise the unit is your 3D coordinate system)
    # velocity[i] = (points3d[i+1] - points3d[i]) / (times[i+1] - times[i] in seconds)
    velocities = (points3d[1:] - points3d[:-1]) / dt_seconds[:, None]  # shape = (trackSize-1, 3)
    speed = np.linalg.norm(velocities, axis=1)

    # Compute median and MAD of speed
    median_speed = np.median(speed)
    abs_dev = np.abs(speed - median_speed)
    mad_speed = np.median(abs_dev) + 1e-9  # add a small eps to avoid division by zero

    # Compute a z-score-like metric for outliers
    z_score_like = abs_dev / mad_speed

    # Identify segments (between i and i+1) with large jumps
    large_jumps = z_score_like > velocity_thresh_factor

    # We will mark outliers for the track points
    outlier_mask = np.zeros(trackSize, dtype=bool)

    # Mark outliers in both points around the large jump (or only i+1, depending on preference)
    for i, jump in enumerate(large_jumps):
        if jump:
            # Mark i+1 (and optionally i) as outlier
            outlier_mask[i+1] = True

    # Remove outliers from allFrameHumans
    count = 0
    for i in range(trackSize):
        if outlier_mask[i]:
            frame_idx = track[i][0]
            person_idx = track[i][1]
            allFrameHumans[frame_idx][person_idx]['id'] = -1
            count += 1

    print(f"Removed {count} outliers from track {track[0][1]}")
    return


if len(sys.argv) < 3:
    print("Usage: python removeOutlier.py <input_pkl_path> <output_pkl_path>")
    sys.exit(1)

pkl_path = sys.argv[1]
output_path = sys.argv[2]

print("Read pkl:", pkl_path)
with open(pkl_path, 'rb') as f:
    dataPKL = pickle.load(f)

video_fps = dataPKL['video_fps']
allFrameHumans = dataPKL['allFrameHumans']

maxId = computeMaxId(allFrameHumans)
tracks, tracksSize = buildTracks(allFrameHumans, maxId)

print("Number of tracks:", len(tracks))
print("Number of humans:", maxId)
print("Processing...")

# Remove outliers in each track taking into account the fps
for i, track in enumerate(tracks):
    if tracksSize[i] > 0:
        removeOutliersInTrack(track, tracksSize[i], allFrameHumans, video_fps)

# Save the updated pickle
with open(output_path, 'wb') as handle:
    pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL)
