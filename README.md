# EECE5554_Final_Project

Make sure you have these installed:
sudo apt install ros-noetic-sensor-msgs ros-noetic-geometry-msgs ros-noetic-std-msgs ros-noetic-rospy

sudo apt install ros-noetic-stereo-image-proc

sudo apt install ros-noetic-rviz

Should be able to clone repo and run catkin_make inside of EECE5554_Final_Project

Once you clone EECE5554_Final_Project:
- inside it, create a folder /data and put the bag file inside of it

FOR REFERENCE - These are the topics available in the bag file MH_01_easy.bag:


/cam0/image_raw
/cam1/image_raw
/clock
/imu0
/leica/position
/rosout
/rosout_agg


The method for data analysis will be the following:
1. Horizontal alignment of stereo images and alignment of IMU data with image data on
time axis.
2. Stereo Matching and depth estimation, converting data from stereo images to point cloud.
3. Use position and pose estimations from IMU as an initial guess for ICP or another
algorithm to align point clouds and combine into map.
4. Use transforms from point cloud data to derive the path of the drone through time, and
compare the results to the absolute measurements performed by external sensors.
5. Use Open3D or PCL to visualize point cloud map and drone position.
6. Create figures visualizing differences in stereo-inertial data vs laser tracking data
