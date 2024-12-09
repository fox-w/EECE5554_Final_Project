import cv2
import numpy as np
import os

cam0_dir = "extracted_data/cam0_images"
cam1_dir = "extracted_data/cam1_images"
output_dir = "extracted_data/disparity_maps"
os.makedirs(output_dir, exist_ok=True)

cam0_images = sorted(os.listdir(cam0_dir))
cam1_images = sorted(os.listdir(cam1_dir))

min_disparity = 0
num_disparities = 16 * 8  
block_size = 9

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=200,
    speckleRange=2
)

for img0_name, img1_name in zip(cam0_images, cam1_images):
    img0_path = os.path.join(cam0_dir, img0_name)
    img1_path = os.path.join(cam1_dir, img1_name)

    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    # apply histogram equalization to improve contrast
    img0 = cv2.equalizeHist(img0)
    img1 = cv2.equalizeHist(img1)

    # apply gaussian blur to reduce noise
    img0 = cv2.GaussianBlur(img0, (5, 5), 0)
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)

    # compute disparity map
    disparity = stereo.compute(img0, img1).astype(np.float32) / 16.0
    disparity = cv2.medianBlur(disparity.astype(np.uint8), 5)

    # normalize for visualization
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # save map
    output_path = os.path.join(output_dir, f"disparity_{img0_name}")
    cv2.imwrite(output_path, disparity_normalized)

    print(f"Processed {img0_name} and {img1_name}")
