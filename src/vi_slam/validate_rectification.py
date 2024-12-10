import cv2
import numpy as np

# Load rectified images
img0 = cv2.imread("/home/vboxuser/final_project/EECE5554_Final_Project/src/vi_slam/extracted_data/rectified_cam0_images/1403636579842242034.png")
img1 = cv2.imread("/home/vboxuser/final_project/EECE5554_Final_Project/src/vi_slam/extracted_data/rectified_cam1_images/1403636579810358543.png")

# Draw horizontal lines
line_spacing = 20
for y in range(0, img0.shape[0], line_spacing):
    cv2.line(img0, (0, y), (img0.shape[1], y), (0, 255, 0), 1)
    cv2.line(img1, (0, y), (img1.shape[1], y), (0, 255, 0), 1)

cv2.imshow("Rectified Cam0", img0)
cv2.imshow("Rectified Cam1", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
