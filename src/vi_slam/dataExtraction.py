import os
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np

output_dir = "extracted_data"
os.makedirs(output_dir, exist_ok=True)
image_dir_cam0 = os.path.join(output_dir, "cam0_images")
image_dir_cam1 = os.path.join(output_dir, "cam1_images")
imu_file = os.path.join(output_dir, "imu_data.csv")
#leica_file = os.path.join(output_dir, "leica_position.csv")

os.makedirs(image_dir_cam0, exist_ok=True)
os.makedirs(image_dir_cam1, exist_ok=True)

bridge = CvBridge()
bag = rosbag.Bag('/home/vboxuser/final_project/EECE5554_Final_Project/data/MH_01_easy.bag', 'r')

# Counters for limiting to the first 20 images
cam0_count = 0
cam1_count = 0
imu_count = 0
max_images = 5

print("images")
for topic, msg, t in bag.read_messages(topics=['/cam0/image_raw', '/cam1/image_raw']):
    if cam0_count >= max_images and cam1_count >= max_images:
        break  # Exit the loop if both topics have 20 images
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        timestamp = str(t.to_nsec())
        if topic == '/cam0/image_raw':
            cv2.imwrite(os.path.join(image_dir_cam0, f"{timestamp}.png"), cv_image)
            cam0_count += 1
        elif topic == '/cam1/image_raw':
            cv2.imwrite(os.path.join(image_dir_cam1, f"{timestamp}.png"), cv_image)
            cam1_count += 1
    except Exception as e:
        print(f"Error processing image: {e}")

# Extract IMU data
# print("Extracting IMU data...")
# with open(imu_file, 'w') as imu_csv:
#     imu_csv.write("timestamp,angular_velocity_x,angular_velocity_y,angular_velocity_z,"
#                   "linear_acceleration_x,linear_acceleration_y,linear_acceleration_z\n")
#     for topic, msg, t in bag.read_messages(topics=['/imu0']):
#         timestamp = t.to_nsec()
#         imu_csv.write(f"{timestamp},{msg.angular_velocity.x},{msg.angular_velocity.y},{msg.angular_velocity.z},"
#                       f"{msg.linear_acceleration.x},{msg.linear_acceleration.y},{msg.linear_acceleration.z}\n")

#print("lecia pos")
#with open(leica_file, 'w') as leica_csv:
#    leica_csv.write("timestamp,x,y,z\n")
#    for topic, msg, t in bag.read_messages(topics=['/leica/position']):
#        timestamp = t.to_nsec()
#        leica_csv.write(f"{timestamp},{msg.point.x},{msg.point.y},{msg.point.z}\n")

bag.close()
print("Data extraction complete!")
