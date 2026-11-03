import os
from collections import deque

import numpy as np

from CameraCalibration import CALIB_FILE, load_calibration, pixels_to_table

# ------------------------------- Default parameters -------------------------------
FOCAL_LENGTH_PX = 800.0          # camera focal length in pixels
CX, CY = 320.0, 240.0            # image center in pixels
Z_TABLE = 0.0                    # table height in the robot base frame (mm)
CAMERA_HEIGHT = 200.0            # camera height above the table, used when no arm pose is given (mm)
T_EE_CAM = np.array([[0, 1, 0, 0],   # camera pose relative to the pointer tip (camera z = pointer direction)
                     [0, 0, 1, 0],
                     [1, 0, 0, 0],
                     [0, 0, 0, 1]], dtype=float)
POINT_SPACING_MM = 5.0           # distance between output points along the line
TARGET_HEIGHT_MM = 0.0           # height above the table to place the output points (0 = touching)

if os.path.exists(CALIB_FILE):
    FOCAL_LENGTH_PX, CX, CY, Z_TABLE, T_EE_CAM = load_calibration()


def order_skeleton(skeleton):
    """Return the skeleton pixels as an ordered Nx2 array of (u, v) along the longest path."""
    pixels = set(zip(*np.nonzero(skeleton)))
    if not pixels:
        return np.empty((0, 2))

    def neighbours(p):
        y, x = p
        return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and (y + dy, x + dx) in pixels]

    def bfs(start):
        parent = {start: None}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for n in neighbours(p):
                if n not in parent:
                    parent[n] = p
                    queue.append(n)
        return p, parent

    end_a, _ = bfs(next(iter(pixels)))
    end_b, parent = bfs(end_a)
    path = [end_b]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return np.array(path, dtype=float)[:, ::-1]


def resample(points, spacing):
    """Resample an ordered polyline so consecutive points are `spacing` apart."""
    if len(points) < 2:
        return points
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    s = np.arange(0, dist[-1], spacing)
    return np.column_stack([np.interp(s, dist, points[:, i]) for i in range(points.shape[1])])


def default_camera_pose(height=CAMERA_HEIGHT):
    """Pointer-tip pose that puts the camera straight down, `height` above the table at x = y = 0."""
    T_base_cam = np.diag([1.0, -1.0, -1.0, 1.0])
    T_base_cam[2, 3] = Z_TABLE + height
    return T_base_cam @ np.linalg.inv(T_EE_CAM)


def skeleton_to_world(skeleton, T_base_ee=None, f=None, cx=None, cy=None, z_table=None,
                      T_ee_cam=None, spacing=POINT_SPACING_MM, target_height=TARGET_HEIGHT_MM):
    """
    Convert a skeleton image from track_line into ordered 3D points in the robot base frame (mm).

    T_base_ee: 4x4 pointer-tip pose from forward kinematics when the image was taken.
               If None, the camera is assumed straight down CAMERA_HEIGHT above the table.
    Returns an Nx3 array, ordered along the line starting at the end closest to the pointer tip.
    """
    f = FOCAL_LENGTH_PX if f is None else f
    cx = CX if cx is None else cx
    cy = CY if cy is None else cy
    z_table = Z_TABLE if z_table is None else z_table
    T_ee_cam = T_EE_CAM if T_ee_cam is None else T_ee_cam
    T_base_ee = default_camera_pose() if T_base_ee is None else T_base_ee

    px = order_skeleton(skeleton)
    if len(px) == 0:
        return np.empty((0, 3))

    points = pixels_to_table(px, T_base_ee, T_ee_cam, f, cx, cy, z_table)
    points = resample(points, spacing)
    points[:, 2] += target_height

    tip = T_base_ee[:3, 3]
    if np.linalg.norm(points[-1, :2] - tip[:2]) < np.linalg.norm(points[0, :2] - tip[:2]):
        points = points[::-1]
    return points


if __name__ == "__main__":
    import cv2
    from LineTracker import track_line

    frame = cv2.imread("test_image_line.png")
    skeleton = track_line(frame)
    points = skeleton_to_world(skeleton)
    print(f"{len(points)} points, first: {np.round(points[0], 1)}, last: {np.round(points[-1], 1)}")
