import cv2
import numpy as np
import os

# Paths
calibration_file = "calibration/camera_calibration/output/stereo_calibration.npz"
cam0_dir = "extracted_data/cam0_images"  # Input folder for cam0 images
cam1_dir = "extracted_data/cam1_images"  # Input folder for cam1 images
output_dir_cam0 = "extracted_data/rectified_cam0_images"  # Output folder for rectified cam0 images
output_dir_cam1 = "extracted_data/rectified_cam1_images"  # Output folder for rectified cam1 images
output_dir_Q = "calibration/metadata"
os.makedirs(output_dir_cam0, exist_ok=True)
os.makedirs(output_dir_cam1, exist_ok=True)

# Load calibration data
calibration_data = np.load(calibration_file)
camera_matrix_0 = calibration_data["camera_matrix_0"]
dist_coeffs_0 = calibration_data["dist_coeffs_0"]
camera_matrix_1 = calibration_data["camera_matrix_1"]
dist_coeffs_1 = calibration_data["dist_coeffs_1"]
R = calibration_data["R"]
T = calibration_data["T"]

print("Calibration data loaded successfully.")

# Print calibration parameters
print("Camera Matrix 0:", camera_matrix_0)
print("Distortion Coefficients 0:", dist_coeffs_0)
print("Camera Matrix 1:", camera_matrix_1)
print("Distortion Coefficients 1:", dist_coeffs_1)
print("Rotation Matrix (R):", R)
print("Translation Vector (T):", T)

# Get the dimensions of a sample image
image_path = os.path.join(cam0_dir, sorted(os.listdir(cam0_dir))[0])
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Sample image not found at {image_path}")
height, width, channels = image.shape
print(f"Image dimensions: Width={width}, Height={height}")

# Define image size
image_size = (width, height)

# Perform stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    camera_matrix_0, dist_coeffs_0,
    camera_matrix_1, dist_coeffs_1,
    image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0  # Set to 0 for a cropped rectified image, or 1 for full FOV
)

# Generate rectification maps
map1_x, map1_y = cv2.initUndistortRectifyMap(
    camera_matrix_0, dist_coeffs_0, R1, P1, image_size, cv2.CV_32FC1
)
map2_x, map2_y = cv2.initUndistortRectifyMap(
    camera_matrix_1, dist_coeffs_1, R2, P2, image_size, cv2.CV_32FC1
)

# Process all images in the folders
cam0_images = sorted(os.listdir(cam0_dir))
cam1_images = sorted(os.listdir(cam1_dir))

if len(cam0_images) != len(cam1_images):
    raise ValueError("Mismatch in the number of images between cam0 and cam1 folders.")

for img0_name, img1_name in zip(cam0_images, cam1_images):
    img0_path = os.path.join(cam0_dir, img0_name)
    img1_path = os.path.join(cam1_dir, img1_name)

    # Load stereo images
    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    if img0 is None or img1 is None:
        print(f"Skipping pair: {img0_name}, {img1_name} (image could not be loaded)")
        continue

    # Rectify images
    img0_rectified = cv2.remap(img0, map1_x, map1_y, interpolation=cv2.INTER_LINEAR)
    img1_rectified = cv2.remap(img1, map2_x, map2_y, interpolation=cv2.INTER_LINEAR)

    # Save rectified images
    output_path_cam0 = os.path.join(output_dir_cam0, img0_name)
    output_path_cam1 = os.path.join(output_dir_cam1, img1_name)
    cv2.imwrite(output_path_cam0, img0_rectified)
    cv2.imwrite(output_path_cam1, img1_rectified)

    print(f"Rectified {img0_name} and {img1_name}")

# Save Q matrix for depth computation
np.save(os.path.join(output_dir_Q, "Q_matrix.npy"), Q)
print("Q matrix saved for depth computation.")
print("Batch stereo rectification complete.")
