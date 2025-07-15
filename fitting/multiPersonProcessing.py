
import os
import sys
import cv2
import pickle
import subprocess
import numpy as np
import shutil

from argparse import ArgumentParser

def getMajorityValue(humansPerFrames):
    if humansPerFrames.size == 0:
        return None
    values, counts = np.unique(humansPerFrames.flatten(), return_counts=True)
    majority_value = values[np.argmax(counts)]
    return majority_value

def getHumanNumber(dataPKL):
    humansPerFrames = np.empty([len(dataPKL['allFrameHumans']), 1],dtype=int)
    for i in range(len(dataPKL['allFrameHumans'])):
        humansPerFrames[i] = len(dataPKL['allFrameHumans'][i])
    humanNumber = getMajorityValue(humansPerFrames)
    maxNbHumans = max(humansPerFrames)
    minNbHumans = min(humansPerFrames)
    return humanNumber, maxNbHumans, minNbHumans

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--directory", type=str, default=None, help="Directory to store the processed files")
    parser.add_argument("--video", type=str, default=None, help="Video file to process")
    parser.add_argument("--root_path", type=str, default=None, help="Root path to output updated nlf and rendered files")
    parser.add_argument("--fov", type=float, default=0, help="Field of view for the 3D pose estimation")
    parser.add_argument("--type", type=str, default="nlf", choices=["multihmr", "nlf"], help="Type of 3D pose estimation to use")
    parser.add_argument("--depthestimator", type=str, default="moge", choices=["vda", "moge"], help="Type of depth estimation to use")
    parser.add_argument("--rbfkernel", type=str, default="linear", choices=["linear", "multiquadric", "univariatespline"], help="RBF kernel to use for the 3D pose estimation filtering")
    parser.add_argument("--rbfsmooth", type=float, default=-1, help="Smoothness for the RBF kernel")
    parser.add_argument("--rbfepsilon", type=float, default=-1, help="Epsilon for the RBF kernel")
    parser.add_argument("--depthmode", type=str, default="pkl", choices=["pkl", "average", "head"], help="Depth mode to use for the 3D pose estimation")
    parser.add_argument("--step", type=int, default=0, help="Step to process (default: 0 for all steps)")
    parser.add_argument("--batchsize", type=int, default=25, help="Batch size for the nlf 3D pose estimation")
    parser.add_argument("--displaymode", action="store_true", help="Display mode activated if this flag is set")
    parser.add_argument("--handestimation", action="store_true", help="Inject hand estimation based on Wilor if this flag is set")
    parser.add_argument("--detectionthreshold", type=float, default=0.3,help="Threshold for detecting the human")
    parser.add_argument("--dispersionthreshold", type=float, default=.1, help="Threshold for human dispersion used for selecting the frame to start the segmentation/tracking")
    parser.add_argument("--only_pose_est", action="store_true", help="Skip depth map generation and 3D mesh generation if this flag is set")
    
    print ("\n############################################################")
    print ("# Arguments")
    print ("############################################################")
    args = parser.parse_args()
    print ("Type: ", args.type)
    print ("Directory: ", args.directory)
    print ("Video: ", args.video)
    print ("Fov: ", args.fov)
    print ("Depthestimator: ", args.depthestimator)
    print ("Rbfsmooth: ", args.rbfsmooth)
    print ("Rbfepsilon: ", args.rbfepsilon)
    print ("Rbfkernel: ", args.rbfkernel)
    print ("Depthmode: ", args.depthmode) 
    print ("Displaymode: ", args.displaymode)
    print ("Step: ", args.step)
    print ("Dispersionthreshold: ", args.dispersionthreshold)
    print ("Detectionthreshold: ", args.detectionthreshold)
    print ("Handestimation: ", args.handestimation)
    print ("Batchsize: ", args.batchsize)
    
    videoFileName = "\""+args.video+"\""

    handEstimation = args.handestimation 
    rbfkernel = args.rbfkernel
    rbfsmooth = args.rbfsmooth
    rbfepsilon = args.rbfepsilon
    fov = args.fov
    dispersionthreshold = args.dispersionthreshold
    detectionThreshold = args.detectionthreshold
    type = args.type
    depthEstimator = args.depthestimator
    only_pose_est = args.only_pose_est

    if args.directory is None:
        print("Please provide a directory")
        sys.exit(1)
    if args.video is None:
        print("Please provide a video")
        sys.exit(1)
    
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        print(f"Created directory: {args.directory}")
    
    print ("\n############################################################")
    print ("# Video information")
    print ("############################################################")
    video = cv2.VideoCapture(args.video)
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video width: ", width)
    print("Video height: ", height)
    print("Video fps: ", fps)
    print("Video frames count: ", frames_count)
    video.release()
    
    if rbfsmooth < 0:
        if (rbfkernel == "linear"):
            if fps > 100:
                rbfsmooth = 0.02
            elif fps > 60:
                rbfsmooth = 0.01
            else:
                rbfsmooth = 0.005
        elif (rbfkernel == "univariatespline"):
            if fps > 60:
                rbfsmooth = 0.5
            else:
                rbfsmooth = 0.25
        elif (rbfkernel == "multiquadric"):
            if fps > 60:
                rbfsmooth = 0.000025
            else:
                rbfsmooth = 0.00001
                
    if rbfepsilon < 0:
        if (rbfkernel == "multiquadric"):
            if fps > 60:
                rbfepsilon = 20
            else:
                rbfepsilon = 25
            
    print("\n############################################################")
    print("# Step 0: Extract Frames from Video")
    print("############################################################")

    # Extract frames from video
    if args.step <= 0:
        print()
        frames_dir = os.path.join(args.root_path, "frames")
        video_path = args.video  # Use the video argument provided when running the script

        # Ensure the frames directory exists
        os.makedirs(frames_dir, exist_ok=True)

        # Open the video file
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            sys.exit(1)

        frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, frame = vidcap.read()
        frame_idx = 0

        # Process each frame
        while success:
            print(f"Extracting frame {frame_idx}/{frame_num}", end="\r")
            frame_path = os.path.join(frames_dir, f"{frame_idx}.png")
            cv2.imwrite(frame_path, frame)
            success, frame = vidcap.read()
            frame_idx += 1

        vidcap.release()
        print(f"\nFrames extracted to: {frames_dir}")

    print("\n############################################################")
    print("# Step 1: Split Frames for Training and Testing")
    print("############################################################")

    if args.step <= 1:
        print()
        frames_dir = os.path.join(args.root_path, "frames")
        output_file_all = os.path.join(args.root_path, "frame_list_all.txt")
        output_file_train = os.path.join(args.root_path, "frame_list_train.txt")
        output_file_test = os.path.join(args.root_path, "frame_list_test.txt")

        if not os.path.exists(frames_dir):
            print(f"Error: Frames directory not found at {frames_dir}")
            sys.exit(1)

        frame_files = os.listdir(frames_dir)
        frame_numbers = [int(os.path.splitext(f)[0]) for f in frame_files if f.endswith('.png')]
        frame_numbers.sort()

        with open(output_file_all, 'w') as f:
            for frame_number in frame_numbers:
                f.write(f"{frame_number}\n")

        print(f"Frame numbers written to {output_file_all}")

        # Select frames at 5 fps from the original 60 fps list
        selected_frames = frame_numbers[::6]  # Select every 6th frame to get 5 fps from 60 fps

        # Split the selected frames into 80% train and 20% test
        split_index = int(len(selected_frames) * 0.8)
        train_frames = selected_frames[:split_index]
        test_frames = selected_frames[split_index:]

        with open(output_file_train, 'w') as f:
            for frame_number in train_frames:
                f.write(f"{frame_number}\n")
        print(f"Train frame numbers written to {output_file_train}")

        with open(output_file_test, 'w') as f:
            for frame_number in test_frames:
                f.write(f"{frame_number}\n")
        print(f"Test frame numbers written to {output_file_test}")

    print ("\n############################################################")
    print ("# Step 2: MoGe analysis")
    print ("############################################################")
    output_moge_pkl = os.path.join(args.directory, "moge.pkl")
    depthmaps_dir = os.path.join(args.root_path, "depth_maps") 
    if args.step <= 2:
        print()
        output_moge_video = os.path.join(args.directory, "videoDepthAnalysis.mp4")
        command_videoAnalysisMoge = "python videoAnalysisMoge.py " + videoFileName + "  " + output_moge_video + " " + output_moge_pkl + " " + depthmaps_dir + " " + str(fov) 
        print("Processing MoGe analysis...")
        print(command_videoAnalysisMoge)
        result = subprocess.run(command_videoAnalysisMoge, shell=True)
        if result.returncode != 0:
            print("\nError in MoGe analysis")
            sys.exit(1)

    print ("\n############################################################")
    print ("# Extract data from MoGe analysis")
    print ("############################################################")
    print()
    print ("Extracting data from MoGe analysis...")
    print ("Reading Pkl file: ", output_moge_pkl)
    with open(output_moge_pkl, 'rb') as file:
        mogePKL = pickle.load(file)
      
    fov_x_degrees = 0
    for i in range(len(mogePKL)):
        fov_x_degrees += mogePKL[i]['fov_x_degrees']
    fov_x_degrees /= len(mogePKL)
    print ("Estimated Fov_x: ",fov_x_degrees)
    
    angle = 0
    for i in range(len(mogePKL)):
        angle += mogePKL[i]['angle']
    angle /= len(mogePKL)
    print ("Angle: ",angle)

    if args.fov > 0:
        fov_x_degrees = args.fov
        print ("Used Fov_x: ",fov_x_degrees)

    if type == "nlf":
        print ("\n############################################################")
        print ("# Step 3: Extract 3D poses with NLF")
        print ("############################################################")
        output_type_pkl = os.path.join(args.directory, type+".pkl")
        if args.step <= 3:
            print()
            command_videoNLF = "python videoNLF.py --video " + videoFileName + " --out_pkl " + output_type_pkl + " --fov " + str(fov_x_degrees) + " --det_thresh " + str(detectionThreshold) + " --batchsize " + str(args.batchsize)
            print("Processing NLF poses estimation...") 
            print(command_videoNLF)
            result = subprocess.run(command_videoNLF, shell=True)
            if result.returncode != 0:
                print("\nError in NLF pose estimation")
                sys.exit(1)    
    else:
        print ("\n############################################################")
        print ("# Step 3: Extract 3D poses with MultiHMR")
        print ("############################################################")
        output_type_pkl = os.path.join(args.directory, type+".pkl")
        if args.step <= 3:
            print()
            command_videoMultiHMR = "python videoMultiHMR.py --video " + videoFileName + " --out_pkl " + output_type_pkl + " --fov " + str(fov_x_degrees)
            print("Processing multiHMR poses estimation...") 
            print(command_videoMultiHMR)
            result = subprocess.run(command_videoMultiHMR, shell=True)
            if result.returncode != 0:
                print("\nError in MultiHMR pose estimation")
                sys.exit(1)  

        
    print ("\n############################################################")
    print ("# Extract total human number")
    print ("############################################################")
    print()
    # Open the pkl file
    print ("Read pkl: ",output_type_pkl)
    file = open(output_type_pkl, 'rb')
    dataPKL = pickle.load(file) 
    file.close()
   
    print("Frames: ", len(dataPKL['allFrameHumans'])) 
    humanNumber, maxNbHumans, minNbHumans = getHumanNumber(dataPKL)
    print('humanNumber: ', humanNumber)
    print('maxNbHumans: ', maxNbHumans)
    print('minNbHumans: ', minNbHumans)    

    output_cleaned_pkl = os.path.join(args.directory, type+"-clean.pkl")
    threshold = 0.5
        
    print ("\n############################################################")
    print ("# Step 4: Clean poses")
    print ("############################################################")
    output_video_json = os.path.join(args.directory, "video.json")
    if args.step <= 4:
        print()
        command_cleanFramesPkl = "python cleanFramesPkl.py " + output_type_pkl + " " + output_cleaned_pkl + " " + str(humanNumber) + " " + str(threshold)
        print("Processing cleaned pkl...") 
        print(command_cleanFramesPkl)
        result = subprocess.run(command_cleanFramesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in cleaning poses")
            sys.exit(1)
        print()
        output_video_json = os.path.join(args.directory, "video.json")
        command_computeCameraProperties = "python computeCameraProperties.py " + output_moge_pkl + " " + output_cleaned_pkl + " " + str(fov_x_degrees) + " " + output_video_json
        print("Processing camera properties...")
        print(command_computeCameraProperties)
        result = subprocess.run(command_computeCameraProperties, shell=True)
        if result.returncode != 0:
            print("\nError in camera properties")
            sys.exit(1)
        print()
        command_injectCameraPropertiesPkl = "python injectCameraPropertiesPkl.py " + output_video_json + " " + output_type_pkl + " " + output_type_pkl
        print("Inject camera properties in", output_type_pkl)
        print(command_injectCameraPropertiesPkl)
        result = subprocess.run(command_injectCameraPropertiesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in camera properties")
            sys.exit(1)
        command_injectCameraPropertiesPkl = "python injectCameraPropertiesPkl.py " + output_video_json + " " + output_cleaned_pkl + " " + output_cleaned_pkl
        print("Inject camera properties in", output_cleaned_pkl)
        print(command_injectCameraPropertiesPkl)
        result = subprocess.run(command_injectCameraPropertiesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in inject camera properties")
            sys.exit(1)

    threshold = 0.5
    if fps > 60:
        threshold = 0.4
    print ("\n############################################################")
    print ("# Step 5: 3D Tracking")
    print ("############################################################")
    output_tracking_pkl = os.path.join(args.directory, type+"-clean-track.pkl")
    if args.step <= 5:
        print()
        command_tracking3DPkl = "python tracking3DPkl.py " + output_cleaned_pkl + " " + output_tracking_pkl + " " + str(threshold)
        print("Processing tracking pkl...") 
        print(command_tracking3DPkl)
        result = subprocess.run(command_tracking3DPkl, shell=True)
        if result.returncode != 0:
            print("\nError in 3D tracking")
            sys.exit(1)
        
    print("\n############################################################")
    print("# Step 6: Add SAM2.1 tracking")
    print("############################################################")
    trackMinSize = 30
    if fps < 50:
        trackMinSize = 5
    output_seg_pkl = os.path.join(args.directory, type + "-clean-track-seg.pkl")
    output_video_segmentation = os.path.join(args.directory, type + "-videoSegmentation.mp4")

    if args.step <= 6:
        print()
        # Command to call the SAM2.1 tracking script
        command_fusionMultiHMRTracking = (
            f"python sam21MultiHMR.py {output_tracking_pkl} {videoFileName} {humanNumber} {trackMinSize} "
            f"{output_seg_pkl} {output_video_segmentation} {dispersionthreshold} {args.root_path}"
        )
        print("Processing fusion...")
        print(command_fusionMultiHMRTracking)
        result = subprocess.run(command_fusionMultiHMRTracking, shell=True)
        if result.returncode != 0:
            print("\nError in SAM2.1 tracking")
            sys.exit(1)
        
    print ("\n############################################################")
    print ("# Step 7: Tracks fusion")
    print ("############################################################")
    output_fusion_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion.pkl")
    output_final_pkl = output_fusion_pkl
    trackMinSize = 30
    if fps < 50:
        trackMinSize = 10
    if args.step <= 7:
        print()
        command_tracksFusion = "python tracksFusion.py " + output_seg_pkl + " " + output_fusion_pkl + " 10"
        print("Processing track fusion...") 
        print(command_tracksFusion)
        result = subprocess.run(command_tracksFusion, shell=True)
        if result.returncode != 0:
            print("\nError in track fusion")
            sys.exit(1)

    print ("\n############################################################")
    print ("# Step 8: Remove outlier in pkl")
    print ("############################################################")
    output_final_outlier_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion-outlier.pkl")
    if args.step <= 8:
        print()
        command_removeoutlier = "python removeOutlier.py " + output_fusion_pkl + " " + output_final_outlier_pkl
        print("Processing Outlier removal...") 
        print(command_removeoutlier)
        result = subprocess.run(command_removeoutlier, shell=True)
        if result.returncode != 0:
            print("\nError in outlier removal")
            sys.exit(1)

    print ("\n############################################################")
    print ("# Step 9: Inject hand estimation based on Wilor in pkl")
    print ("############################################################")
    previous_output_final_outlier_pkl = output_final_outlier_pkl
    output_final_outlier_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion-outlier-handestimation.pkl")
    if args.step <= 9:
        print()
        command_injectHands = "python injectHandsPkl.py " + previous_output_final_outlier_pkl + " " + videoFileName + " " + output_final_outlier_pkl 
        print("Processing hand estimation...")
        print(command_injectHands)
        result = subprocess.run(command_injectHands, shell=True)
        if result.returncode != 0:
            print("\nError in hand estimation")
            sys.exit(1)

    print ("\n############################################################")
    print ("# Step 10: RBF and Filtering")
    print ("############################################################")
    if handEstimation:
        output_final_outlier_filtered_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion-outlier-handestimation-filtered.pkl")
    else:
        output_final_outlier_filtered_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion-outlier-filtered.pkl")
    if args.step <= 10:
        print()
        command_rbfFiltering = "python RBFFilterSMPLX.py " + output_final_outlier_pkl + " " + output_final_outlier_filtered_pkl + " " + rbfkernel + " " + str(rbfsmooth) + " " + str(rbfepsilon) 
        print("Processing RBF and filtering...") 
        print(command_rbfFiltering)
        result = subprocess.run(command_rbfFiltering, shell=True)
        if result.returncode != 0:
            print("\nError in RBF and filtering")
            sys.exit(1)

    print ("\n############################################################")
    print ("# Copy final pkl files")
    print ("############################################################")
    print()
    output_destination_pkl = os.path.join(args.directory, type+"-final.pkl")
    output_destination_filtered_pkl = os.path.join(args.directory, type+"-final-filtered.pkl")

    print("Copying final pkl files...")
    print ("Final pkl")
    print("From: ",output_final_pkl)
    print("To: ",output_destination_pkl)
    shutil.copyfile(output_final_pkl, output_destination_pkl)
    print ("Final filtered pkl")
    print("From: ",output_final_outlier_filtered_pkl)
    print("To: ",output_destination_filtered_pkl)
    shutil.copyfile(output_final_outlier_filtered_pkl, output_destination_filtered_pkl)

    print("\n############################################################")
    print("# Step 11: Render SMPLX")
    print("############################################################")

    # Define paths for rendering
    input_pkl = os.path.join(args.directory, type + "-final-filtered.pkl")
    output_rendered_video = os.path.join(args.root_path, "smplx_rendered.mp4")
    output_rendered_frames_dir = os.path.join(args.root_path, "smplx_rendered")
    updated_pkl_path = os.path.join(args.root_path, "updated-" + type + "-final-filtered.pkl")  # Save updated pkl in root_path

    if args.step <= 11:
        print()
        # Command to call the render script
        command_renderMesh = (
            f"python render_smplx_opencv.py {input_pkl} {args.video} {output_rendered_video} "
            f"{output_rendered_frames_dir} {updated_pkl_path}"
        )
        print("Rendering SMPLX...")
        print(command_renderMesh)
        result = subprocess.run(command_renderMesh, shell=True)
        if result.returncode != 0:
            print("\nError in rendering SMPLX")
            sys.exit(1)

        print("Rendered video saved to:", output_rendered_video)
        print("Rendered frames saved to:", output_rendered_frames_dir)
        print("Updated SMPLX parameters saved to:", updated_pkl_path)
        
    print("\n############################################################")
    print("# Step 12: Prepare SMPLX & Cam Params")
    print("############################################################")

    input_updated_pkl = os.path.join(args.root_path, "updated-"+type+"-final-filtered.pkl")
    output_smplx_params_dir = os.path.join(args.root_path, "smplx_params")
    output_cam_params_dir = os.path.join(args.root_path, "cam_params")

    if args.step <= 12:
        print()
        # Command to call the prepare_params.py script
        command_prepareParams = f"python prepare_params.py --in_pkl {input_updated_pkl} --root_path {args.root_path}"
        print("Preparing SMPLX and camera parameters...")
        print(command_prepareParams)
        result = subprocess.run(command_prepareParams, shell=True)
        if result.returncode != 0:
            print("\nError in preparing SMPLX and camera parameters")
            sys.exit(1)

        print("SMPLX parameters saved to:", output_smplx_params_dir)
        print("Camera parameters saved to:", output_cam_params_dir)

if not args.only_pose_est:
    print("\n############################################################")
    print("# Step 13: Generate 3D SMPLX meshes")
    print("############################################################")

    input_updated_pkl = os.path.join(args.root_path, "updated-"+type+"-final-filtered.pkl")
    output_smplx_mesh_dir = os.path.join(args.root_path, "meshes")

    if args.step <= 13:
        print()
        # Command to call the generate_mesh.py script
        command_generateMesh = f"python generate_smplx_meshes.py {input_updated_pkl} {output_smplx_mesh_dir}"
        print("Generating 3D SMPLX meshes...")
        print(command_generateMesh)
        result = subprocess.run(command_generateMesh, shell=True)
        if result.returncode != 0:
            print("\nError in generating 3D SMPLX meshes")
            sys.exit(1)
        print("3D SMPLX meshes saved to:", output_smplx_mesh_dir)

# python .\multiHMRPipeline.py --directory ../results/T3-2-1024 --video "F:\MyDrive\Tmp\Tracking\T3-2-1024.MP4" --step 6
# python .\multiHMRPipeline.py --video ..\videos\D0-talawa_technique_intro-Scene-003.mp4 --directory ..\results\D0-003\ --step 0