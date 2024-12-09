import cv2
import numpy as np
import open3d as o3d  
import os

f_x = 376  # Focal length in x
f_y = 376  # Focal length in y
c_x = 376  # Principal point x
c_y = 240  # Principal point y

depth_maps_dir = "extracted_data/depth_maps"  
output_dir = "extracted_data/point_clouds"  
os.makedirs(output_dir, exist_ok=True)

depth_scale = 1.0 / 255.0 * 20.0  # normalized to [0, 255] with max 20m

depth_map_files = sorted(os.listdir(depth_maps_dir))
for depth_map_file in depth_map_files:
    depth_map_path = os.path.join(depth_maps_dir, depth_map_file)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    depth_map *= depth_scale

    rows, cols = depth_map.shape
    points = []
    for v in range(rows):
        for u in range(cols):
            z = depth_map[v, u]
            if z > 0:  
                x = (u - c_x) * z / f_x
                y = (v - c_y) * z / f_y
                points.append([x, y, z])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    point_cloud_path = os.path.join(output_dir, f"{os.path.splitext(depth_map_file)[0]}.ply")
    o3d.io.write_point_cloud(point_cloud_path, pcd)
    print(f"Point cloud saved to {point_cloud_path}")

if depth_map_files:
    example_pcd_path = os.path.join(output_dir, f"{os.path.splitext(depth_map_files[0])[0]}.ply")
    example_pcd = o3d.io.read_point_cloud(example_pcd_path)
    o3d.visualization.draw_geometries([example_pcd])

print("Point cloud generation complete for all depth maps!")
