import open3d as o3d
import numpy as np

def visualize_point_cloud(file_path):
    # Load point cloud from file
    points = []
    colors = []
    line_count = 0  # Initialize line count

    with open(file_path, 'r') as f:
        for line in f:
            line_count += 1  # Increment line count
            x, y, z, r, g, b = map(float, line.strip().split())
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])  # Normalize RGB to [0, 1]

    print(f"Total number of lines in the file: {line_count}")  # Output line count

    # Convert to numpy arrays
    points = np.array(points)
    colors = np.array(colors)

    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud Visualization")

# Path to your point cloud file
file_path = "/home/da10546y/NLF-GS/outputs/subject_6/bkg_point_cloud.txt"
visualize_point_cloud(file_path)