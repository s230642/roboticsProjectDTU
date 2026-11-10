from MyRobot.Python.MyRobot import *

def forward(deg): return MyRobot.forward(None, deg)

print("forward([30, -44, -120, 0])")
print(forward([30, -44, -120, 0]))
print("---")
print("forward([0, -10, 0, -140])")
print(forward([0, -10, 0, -140]))

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2

cam = cv2.VideoCapture(1)
ok, frame = cam.read()
if not ok:
    print("Could not read from the camera.")
    
cv2.imshow("Frame",frame)
cv2.waitKey()
print("Done")