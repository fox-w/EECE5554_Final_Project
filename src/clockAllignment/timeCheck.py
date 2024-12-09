import os
cam0_dir = "extracted_data/cam0_images"
imu_timestamp = 1403636579809262870
cam0_images = sorted(os.listdir(cam0_dir))
closest_image = None
closest_diff = float('inf')

for img_name in cam0_images:
    img_timestamp = int(os.path.splitext(img_name)[0])  
    diff = abs(img_timestamp - imu_timestamp)
    if diff < closest_diff:
        closest_diff = diff
        closest_image = img_name

print(f"closest image: {closest_image}, difference: {closest_diff} ns")
