import cv2
import numpy as np
import os

# Paths
cam0_dir = "/home/vboxuser/final_project/EECE5554_Final_Project/src/vi_slam/extracted_data/rectified_cam0_images"
cam1_dir = "/home/vboxuser/final_project/EECE5554_Final_Project/src/vi_slam/extracted_data/rectified_cam1_images"
output_dir = "extracted_data/disparity_maps"
os.makedirs(output_dir, exist_ok=True)

# Ensure directories are not empty
cam0_images = sorted(os.listdir(cam0_dir))
cam1_images = sorted(os.listdir(cam1_dir))

if len(cam0_images) == 0 or len(cam1_images) == 0:
    raise FileNotFoundError("One or both input directories are empty. Please check your rectified image folders.")

if len(cam0_images) != len(cam1_images):
    raise ValueError("Mismatch in the number of images between cam0 and cam1 folders.")

print(f"Found {len(cam0_images)} image pairs for disparity computation.")

# StereoSGBM Parameters
min_disparity = 0
num_disparities = 16 * 8  # Should be divisible by 16
block_size = 10
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,  # Adjusted for more consistent matching
    speckleWindowSize=100,  # Tuned for noise reduction
    speckleRange=3
)

# Process image pairs
for img0_name, img1_name in zip(cam0_images, cam1_images):
    img0_path = os.path.join(cam0_dir, img0_name)
    img1_path = os.path.join(cam1_dir, img1_name)

    # Load stereo images
    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    if img0 is None or img1 is None:
        print(f"Skipping pair: {img0_name}, {img1_name} (failed to load images)")
        continue

    # Ensure image dimensions match
    if img0.shape != img1.shape:
        print(f"Skipping pair: {img0_name}, {img1_name} (image size mismatch)")
        continue

    # Preprocessing: Apply histogram equalization
    img0 = cv2.equalizeHist(img0)
    img1 = cv2.equalizeHist(img1)

    # Optional: Apply Gaussian blur
    img0 = cv2.GaussianBlur(img0, (3, 3), 0)
    img1 = cv2.GaussianBlur(img1, (3, 3), 0)

    # Compute disparity map
    disparity = stereo.compute(img0, img1).astype(np.float32) / 16.0

    # Normalize disparity for visualization
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # Save the normalized disparity map
    output_path = os.path.join(output_dir, f"disparity_{img0_name}")
    cv2.imwrite(output_path, disparity_normalized)

    print(f"Processed {img0_name} and {img1_name}")

print("Disparity computation complete.")
