import open3d as o3d
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


point_clouds_dir = "extracted_data/point_clouds"  
imu_data_path = "extracted_data/imu_data.csv"  
output_path = "combined_point_cloud.ply"  

def integrate_imu_data(imu_data_path):
    imu_data = pd.read_csv(imu_data_path)
    imu_data = imu_data.iloc[::10].reset_index(drop=True)

    timestamps = imu_data['timestamp'].to_numpy()
    time_deltas = np.diff(timestamps) / 1e9  

    cumulative_rotation = np.eye(3)  # init rotation matrix (identity)
    cumulative_translation = np.zeros(3)  # init translation (origin)
    transforms = [] 

    for i in range(len(time_deltas)):
        delta_t = time_deltas[i]

        angular_velocity = imu_data.iloc[i][["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]].to_numpy()
        angular_velocity[np.abs(angular_velocity) < 1e-3] = 0

        theta = angular_velocity * delta_t 
        theta_magnitude = np.linalg.norm(theta)
        if theta_magnitude > 0:
            axis = theta / theta_magnitude
            rotation_delta = axis_angle_to_rotation_matrix(axis, theta_magnitude)
        else:
            rotation_delta = np.eye(3)

        cumulative_rotation = np.dot(cumulative_rotation, rotation_delta)

        linear_acceleration = imu_data.iloc[i][["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]].to_numpy()
        linear_acceleration[np.abs(linear_acceleration) < 1e-3] = 0

        translation_delta = 0.5 * linear_acceleration * delta_t**2

        cumulative_translation += np.dot(cumulative_rotation, translation_delta)

        transform = np.eye(4)
        transform[:3, :3] = cumulative_rotation
        transform[:3, 3] = cumulative_translation
        transforms.append(transform)

    return transforms



def axis_angle_to_rotation_matrix(axis, angle):
    # rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def visualize_trajectory(imu_transforms):
    trajectory = [transform[:3, 3] for transform in imu_transforms]
    trajectory = np.array(trajectory)
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory (X-Y)")
    plt.plot(trajectory[:, 0], trajectory[:, 2], label="Trajectory (X-Z)")
    plt.xlabel("X-axis (meters)")
    plt.ylabel("Y/Z-axis (meters)")
    plt.legend()
    plt.title("IMU-Based Trajectory")
    plt.grid()
    plt.show()


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    r = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return r

def load_point_clouds(point_clouds_dir):
    point_cloud_files = sorted([f for f in os.listdir(point_clouds_dir) if f.endswith(".ply")])
    point_clouds = []
    for file in point_cloud_files:
        pcd_path = os.path.join(point_clouds_dir, file)
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        # downsample the point cloud using voxel grid
        pcd = pcd.voxel_down_sample(voxel_size=0.05)  
        point_clouds.append(pcd)
    return point_clouds


def integrate_transforms_and_point_clouds(point_clouds, imu_transforms):
    global_pcd = o3d.geometry.PointCloud()
    cumulative_transform = np.eye(4)  # Start at the origin

    for i, (pcd, transform) in enumerate(zip(point_clouds, imu_transforms)):
        cumulative_transform = np.dot(cumulative_transform, transform)
        pcd.transform(cumulative_transform)  
        global_pcd += pcd

        print(f"Integrated point cloud {i + 1}/{len(point_clouds)}")

    return global_pcd

def refine_registration(global_pcd):
    voxel_size = 0.05  
    source = global_pcd
    target = global_pcd.voxel_down_sample(voxel_size=voxel_size)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=voxel_size,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    global_pcd.transform(reg_p2p.transformation)
    return global_pcd


def main():
    imu_transforms = integrate_imu_data(imu_data_path)
    #for i, transform in enumerate(imu_transforms[:10]):  # Check the first 10 transforms
    #    print(f"Transform {i}:\n{transform}")
    #visualize_trajectory(imu_transforms)

    identity_transform = np.eye(4)
    imu_transforms.insert(0, identity_transform)
    point_clouds = load_point_clouds(point_clouds_dir)

    batch_size = 500  
    total_batches = len(point_clouds) // batch_size + 1

    global_pcd = o3d.geometry.PointCloud()

    for batch_idx in range(total_batches):
        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(point_clouds))

        batch_point_clouds = point_clouds[start_idx:end_idx]
        batch_imu_transforms = imu_transforms[start_idx:end_idx]

        batch_pcd = integrate_transforms_and_point_clouds(batch_point_clouds, batch_imu_transforms)
        
        global_pcd += batch_pcd
        global_pcd = refine_registration(global_pcd)

        o3d.io.write_point_cloud(f"global_pcd_batch_{batch_idx + 1}.ply", global_pcd)

    global_pcd = global_pcd.voxel_down_sample(voxel_size=0.05)  # final downsampling
    o3d.io.write_point_cloud(output_path, global_pcd)
    print(f"Final combined point cloud saved to {output_path}")

    o3d.visualization.draw_geometries([global_pcd])



if __name__ == "__main__":
    main()
