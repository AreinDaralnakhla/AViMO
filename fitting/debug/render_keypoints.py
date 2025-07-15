import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm


def render_j2d_smplx(nlf_pkl, video_path, output_path, output_frames_dir):
    os.makedirs(output_frames_dir, exist_ok=True)  # Create directory for output frames

    # Load the main NLF data (nlf-final-filtered.pkl)
    with open(nlf_pkl, "rb") as f:
        data = pickle.load(f)

    # Extract video properties
    w, h, fps = data["video_width"], data["video_height"], data["video_fps"]

    # Open the video
    video = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Iterate through frames
    for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = video.read()
        if not ret:
            break

        if i >= len(data["allFrameHumans"]):
            continue

        for j, hdata in enumerate(data["allFrameHumans"][i]):
            # Extract 2D keypoints
            j2d = np.array(hdata.get("j2d_smplx", []))  # 2D keypoints

            # Render 2D keypoints on the frame
            if j2d.size > 0:
                for point in j2d:
                    if len(point) == 2:  # Handle (x, y) format
                        x, y = point
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red circles for 2D points
                    elif len(point) == 3:  # Handle (x, y, confidence) format
                        x, y, conf = point
                        if conf > 0.5:  # Only render points with sufficient confidence
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red circles for 2D points

        # Save the current frame as an image
        frame_path = os.path.join(output_frames_dir, f"{i}.png")
        cv2.imwrite(frame_path, frame)

        # Write the frame to the video
        out.write(frame)

    video.release()
    out.release()
    print("Saved frames to:", output_frames_dir)
    print("Saved video to:", output_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python render_j2d_smplx.py <nlf_pkl> <video_path> <output_video_path> <output_frames_dir>")
        sys.exit(1)

    render_j2d_smplx(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])