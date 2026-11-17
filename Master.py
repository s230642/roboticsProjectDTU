import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "MyRobot", "Python"))
from MyRobot import MyRobot
from CameraCalibration import CAM_INDEX
from LineTracker import track_line
from TransformTo3D import skeleton_to_world

VIEW_POSE = [0, -4, -87, -88]  # joint angles (deg) where the camera looks down at the line

robot = MyRobot()
if not robot.init_status:
    sys.exit()

#Move to the viewing pose and take an image
robot.move_j(*VIEW_POSE)
cam = cv2.VideoCapture(CAM_INDEX)
for _ in range(5):
    cam.read()
ret, frame = cam.read()
cam.release()

#Get the skeleton of the line
skeleton = track_line(frame)

#Transform to 3D points
points = skeleton_to_world(skeleton, robot.read_ee_position())

#Show the detected line and confirm before moving
frame[skeleton > 0] = (0, 0, 255)
cv2.imshow("detected line (press a key)", frame)
cv2.waitKey()
cv2.destroyAllWindows()

#Drag the pointer along the line
if len(points) == 0:
    print("No line found")
elif input(f"Follow the line ({len(points)} points)? [y/N]: ").strip().lower() == "y":
    robot.follow_path(points)
