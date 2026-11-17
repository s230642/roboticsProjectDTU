import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "MyRobot", "Python"))
from MyRobot import MyRobot
from CameraCalibration import CAM_INDEX, load_calibration, pixels_to_table

HOVER_MM = 20.0          # height above the target to approach from and retreat to
TOUCH_OFFSET_MM = 0.0    # final height above the table (raise it to stop just above the table)


def grab_frame(cam):
    for _ in range(5):  # drop buffered frames so the image matches the current arm pose
        cam.read()
    ok, frame = cam.read()
    if not ok:
        raise RuntimeError("Could not read from the camera")
    return frame


def click_point(frame):
    """Show the frame and return the clicked (u, v), or None to retake, or 'quit'."""
    print("  Click the target in the image. Enter = accept, r = retake image, q = quit")
    point = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point[:] = [(x, y)]

    cv2.namedWindow("click to move")
    cv2.setMouseCallback("click to move", on_mouse)
    while True:
        shown = frame.copy()
        if point:
            cv2.drawMarker(shown, point[0], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow("click to move", shown)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 and point:
            return np.array(point[0], dtype=float)
        if key == ord("r"):
            return None
        if key == ord("q"):
            return "quit"


def main():
    f, cx, cy, z_table, T_ee_cam = load_calibration()
    robot = MyRobot()
    if not robot.init_status:
        return
    cam = cv2.VideoCapture(CAM_INDEX)

    try:
        view_q = robot.move_until_confirmed("Move the arm to a pose where the camera looks at the table.")
        while True:
            robot.move_j(*view_q)
            frame = grab_frame(cam)
            T_base_ee = robot.forward(robot.read_joint_angles())

            px = click_point(frame)
            if px is None:
                continue
            if isinstance(px, str):
                break

            hit = pixels_to_table(px, T_base_ee, T_ee_cam, f, cx, cy, z_table)
            if len(hit) == 0:
                print("  That pixel does not look at the table (ray points above the horizon). Click lower.")
                continue
            target = hit[0].copy()
            target[2] += TOUCH_OFFSET_MM
            print(f"  Pixel {px.astype(int)} -> table point {np.round(target, 1)} mm")

            solution = robot.find_pitch([target + [0, 0, HOVER_MM], target])
            if solution is None:
                print("  The arm cannot reach that point. Pick a point closer to the robot.")
                continue
            pitch, (q_hover, q_touch) = solution
            print(f"  Pitch {pitch} deg, joints hover {np.round(q_hover, 1)}, touch {np.round(q_touch, 1)}")

            if input("  Move there? [y/N]: ").strip().lower() != "y":
                continue
            robot.move_j(*q_hover)
            robot.move_j(*q_touch)
            reached = robot.read_ee_position()[:3, 3]
            print(f"  Reached {np.round(reached, 1)} mm (error {np.linalg.norm(reached - target):.1f} mm from target, by FK)")
            input("  Check where the pointer is, then press Enter to go back...")
            robot.move_j(*q_hover)

            if input("  Keep the same viewing pose? [Y/n]: ").strip().lower() == "n":
                view_q = robot.move_until_confirmed("Move the arm to a new viewing pose.")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
