import cv2
import numpy as np
import os

focal_length = 750  
baseline = 0.1093  
max_depth = 20  
max_visual_depth = 20 

disparity_dir = "extracted_data/disparity_maps"
output_dir = "extracted_data/depth_maps"
os.makedirs(output_dir, exist_ok=True)

disparity_files = sorted(os.listdir(disparity_dir))
for disparity_file in disparity_files:
    disparity_path = os.path.join(disparity_dir, disparity_file)
    disparity_map = cv2.imread(disparity_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Crop the disparity map
    crop_x_start = 100  # Start column for the crop (adjust based on your image)
    crop_x_end = disparity_map.shape[1]  # End column for the crop
    cropped_disparity_map = disparity_map[:, crop_x_start:crop_x_end]
    disparity_map = cropped_disparity_map

    valid_disparity_mask = disparity_map > 0
    disparity_map[~valid_disparity_mask] = 0.1  

    depth_map = (focal_length * baseline) / disparity_map
    depth_map[depth_map > max_depth] = 0

    # mask for invalid depth values depth = 0
    invalid_mask = (depth_map == 0).astype(np.uint8)
    depth_map_filled = cv2.inpaint(depth_map.astype(np.float32), invalid_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

    # apply gaussian blur to reduce noise
    depth_map_smoothed = cv2.GaussianBlur(depth_map_filled, (7, 7), 0)

    depth_map_clipped = np.clip(depth_map_smoothed, 0, max_visual_depth)
    depth_map_normalized = cv2.normalize(depth_map_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)

    # depth_map_normalized = cv2.normalize(
    #     max_visual_depth - depth_map_clipped,  # Invert depth values
    #     None,
    #     alpha=0,
    #     beta=255,
    #     norm_type=cv2.NORM_MINMAX,
    # )
    # depth_map_normalized = np.uint8(depth_map_normalized)

    output_path = os.path.join(output_dir, f"depth_{disparity_file}")
    cv2.imwrite(output_path, depth_map_normalized)
    print(f"Processed depth map for {disparity_file}")
