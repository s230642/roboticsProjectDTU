import cv2
from LineTracker import track_line
from TransformTo3D import skeleton_to_world

#Load the camera and take an image
cam = cv2.VideoCapture(0)
ret, frame = cam.read()

#Get the skeleton of the line
skeleton = track_line(frame)

#Transform to 3D points
worldPoints = skeleton_to_world(skeleton)

#Preform forwards kinematics

#Release the camera
cam.release()