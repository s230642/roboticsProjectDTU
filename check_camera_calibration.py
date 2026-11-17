"""Per-view errors of the camera calibration, and how the fit changes when each view is left out."""
import json

import numpy as np

from CameraCalibration import CALIB_FILE, pixels_to_table, solve_camera


def view_errors(views, dot, T_ee_cam, f, cx, cy, z_table):
    """Per view: reprojection error (px), dot error on the table (mm), camera height above the table (mm)
    and camera tilt away from straight down (deg)."""
    rows = []
    for T_base_ee, uv in views:
        T = T_base_ee @ T_ee_cam
        x, y, z = (np.linalg.inv(T) @ np.append(dot, 1))[:3]
        px_err = np.hypot(f * x / z + cx - uv[0], f * y / z + cy - uv[1])
        hit = pixels_to_table(uv, T_base_ee, T_ee_cam, f, cx, cy, z_table)
        mm_err = np.linalg.norm(hit[0, :2] - dot[:2]) if len(hit) else float("nan")
        tilt = np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))
        rows.append((px_err, mm_err, T[2, 3] - z_table, tilt))
    return rows


if __name__ == "__main__":
    with open(CALIB_FILE) as fh:
        c = json.load(fh)
    views = [(np.array(v["T_base_ee"]), np.array(v["uv"])) for v in c["views"]]
    dot, cx, cy = np.array(c["dot"]), c["cx"], c["cy"]

    print(f"Saved fit: f {c['f']:.0f} px, camera offset {np.round(np.array(c['T_ee_cam'])[:3, 3], 1)} mm")
    for i, (px, mm, h, tilt) in enumerate(view_errors(views, dot, np.array(c["T_ee_cam"]), c["f"], cx, cy, c["z_table"])):
        tip = views[i][0][:3, 3]
        print(f"  view {i + 1}: pixel {np.round(views[i][1]).astype(int)}, reprojection {px:5.1f} px, "
              f"dot error {mm:4.1f} mm, camera {h:4.0f} mm above table, tilted {tilt:3.0f} deg, "
              f"base angle {np.degrees(np.arctan2(tip[1], tip[0])):4.0f} deg, tip reach {np.hypot(*tip[:2]):4.0f} mm")

    T_ee_cam, f, rms = solve_camera(views, dot, cx, cy, c["f0"], f_weight=100)
    print(f"Refit with f held at the ruler estimate: f {f:.0f} px, rms {rms:.1f} px, camera offset {np.round(T_ee_cam[:3, 3], 1)} mm")
    for i, (px, mm, h, tilt) in enumerate(view_errors(views, dot, T_ee_cam, f, cx, cy, c["z_table"])):
        print(f"  view {i + 1}: reprojection {px:5.1f} px, dot error {mm:4.1f} mm, camera {h:4.0f} mm above table, "
              f"tilted {tilt:3.0f} deg")

    print("Refit without each view:")
    for i in range(len(views)):
        T_ee_cam, f, rms = solve_camera(views[:i] + views[i + 1:], dot, cx, cy, c["f0"])
        print(f"  without view {i + 1}: f {f:4.0f} px, rms {rms:4.1f} px, camera offset {np.round(T_ee_cam[:3, 3], 1)} mm")
