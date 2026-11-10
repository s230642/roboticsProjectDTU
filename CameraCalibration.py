import json
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "MyRobot", "Python"))

CAM_INDEX = 1
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calibration.json")
N_TABLE_TOUCHES = 3
N_VIEWS = 6
VIEW_HINTS = [
    "camera straight down, low above the table",
    "camera straight down, high above the table",
    "camera tilted, dot in the upper half of the image",
    "base rotated to one side, dot near the left edge of the image",
    "base rotated to the other side, dot near the right edge of the image",
    "camera tilted differently, dot near a corner of the image",
]


# ------------------------------- Using the calibration -------------------------------

def load_calibration(path=CALIB_FILE):
    with open(path) as fh:
        c = json.load(fh)
    return c["f"], c["cx"], c["cy"], c["z_table"], np.array(c["T_ee_cam"])


def pixels_to_table(px, T_base_ee, T_ee_cam, f, cx, cy, z_table):
    """Map an Nx2 array of pixels (u, v) to points on the table in the base frame (mm)."""
    px = np.asarray(px, dtype=float).reshape(-1, 2)
    T = T_base_ee @ T_ee_cam
    R, o = T[:3, :3], T[:3, 3]
    rays = np.column_stack([(px[:, 0] - cx) / f, (px[:, 1] - cy) / f, np.ones(len(px))])
    d = rays @ R.T
    t = (z_table - o[2]) / d[:, 2]
    return (o + t[:, None] * d)[t > 0]


# ------------------------------- Solving -------------------------------

def _make_T(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def _residuals(params, views, dot, cx, cy, f0):
    T_ee_cam, f = _make_T(params[:3], params[3:6]), params[6]
    res = []
    for T_base_ee, (u, v) in views:
        x, y, z = (np.linalg.inv(T_base_ee @ T_ee_cam) @ np.append(dot, 1))[:3]
        z = max(z, 1e-3)
        res += [f * x / z + cx - u, f * y / z + cy - v]
    res.append(0.1 * (f - f0))
    return res


def solve_camera(views, dot, cx, cy, f0):
    """Fit T_ee_cam and f so the dot projects onto the clicked pixel in every view."""
    best = None
    for start in Rotation.create_group("O"):
        x0 = np.concatenate([start.as_rotvec(), np.zeros(3), [f0]])
        sol = least_squares(_residuals, x0, args=(views, dot, cx, cy, f0), x_scale="jac")
        if best is None or sol.cost < best.cost:
            best = sol
    T_ee_cam, f = _make_T(best.x[:3], best.x[3:6]), best.x[6]
    rms_px = np.sqrt(np.mean(np.square(best.fun[:-1])))
    return T_ee_cam, f, rms_px


# ------------------------------- User interaction -------------------------------

def click_points(cam, n, instruction, allow_skip=False):
    print("\n" + instruction)
    print(f"  In the camera window: click {n} point(s), Enter = accept, Backspace = clear"
          + (", s = skip" if allow_skip else ""))
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n:
            points.append((x, y))

    cv2.namedWindow("calibration")
    cv2.setMouseCallback("calibration", on_mouse)
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        for p in points:
            cv2.drawMarker(frame, p, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow("calibration", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 and len(points) == n:
            return np.array(points, dtype=float)
        if key == 8:
            points.clear()
        if allow_skip and key == ord("s"):
            return None


def load_state(path=CALIB_FILE):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_state(state, path=CALIB_FILE):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


def ask_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Please type a number.")


# ------------------------------- Calibration steps -------------------------------

def main():
    from MyRobot import MyRobot

    robot = MyRobot()
    if not robot.init_status:
        return
    cam = cv2.VideoCapture(CAM_INDEX)
    ok, frame = cam.read()
    if not ok:
        print("Could not read from the camera.")
        return
    cy, cx = frame.shape[0] / 2, frame.shape[1] / 2

    state = load_state()
    if state:
        print(f"Resuming from {CALIB_FILE} (delete a key from it to redo that step).")
    state["cx"], state["cy"] = cx, cy
    save_state(state)

    try:
        print("\n=== Step 1/5: Table height ===")
        if "z_table" in state:
            print(f"  Already done: {state['z_table']:.1f} mm, skipping.")
        else:
            zs = []
            for i in range(N_TABLE_TOUCHES):
                q = robot.move_until_confirmed(f"Move the arm so the pointer tip just touches the table "
                                         f"(spot {i + 1} of {N_TABLE_TOUCHES}, spread the spots out).")
                zs.append(robot.forward(q)[2, 3])
            state["z_table"] = float(np.mean(zs))
            save_state(state)
            print(f"  Table height: {state['z_table']:.1f} mm (spread between spots: {np.ptp(zs):.1f} mm)")
            if np.ptp(zs) > 5:
                print("  Warning: large spread. The table may not be level, or the FK is inaccurate.")
        z_table = state["z_table"]

        print("\n=== Step 2/5: Focal length (rough estimate) ===")
        if "f0" in state:
            print(f"  Already done: {state['f0']:.0f} px, skipping.")
        else:
            robot.move_until_confirmed("Point the camera straight down, 15-30 cm above the table. "
                                 "Lay the ruler flat on the table through the middle of the image.")
            z0 = ask_float("  With the ruler, measure the distance from the camera lens down to the table (mm): ")
            marks = click_points(cam, 2, "Click two marks on the ruler that are far apart, both near the middle of the image.")
            length = ask_float("  Real distance between the two marks you clicked (mm): ")
            state["f0"] = float(np.linalg.norm(marks[1] - marks[0]) * z0 / length)
            save_state(state)
            print(f"  Focal length estimate: {state['f0']:.0f} px")
        f0 = state["f0"]

        print("\n=== Step 3/5: Reference dot ===")
        if "dot" in state: 
            print(f"  Already done: {np.round(state['dot'], 1)} mm, skipping.")
        else:
            print("Draw a small dark dot on the table (or use a sticker), roughly in the middle of the reachable area.")
            q = robot.move_until_confirmed("Move the arm so the pointer tip touches the center of the dot.")
            state["dot"] = robot.forward(q)[:3, 3].tolist()
            # Views and the solution depend on the dot, so they are stale now.
            for key in ("views", "f", "T_ee_cam"):
                state.pop(key, None)
            save_state(state)
            print(f"  Dot position: {np.round(state['dot'], 1)} mm")
        dot = np.array(state["dot"])

        print(f"\n=== Step 4/5: View the dot from {N_VIEWS} different poses ===")
        state.setdefault("views", [])
        if len(state["views"]) >= N_VIEWS:
            print(f"  Already done: {len(state['views'])} views, skipping.")
        else:
            if state["views"]:
                print(f"  {len(state['views'])} view(s) already saved, continuing.")
            print("The more the poses differ (height, tilt, base rotation, where the dot is in the image), the better.")
            while len(state["views"]) < N_VIEWS:
                n = len(state["views"])
                hint = VIEW_HINTS[n % len(VIEW_HINTS)]
                q = robot.move_until_confirmed(f"View {n + 1}/{N_VIEWS}: move the arm so the camera sees the dot, "
                                         f"{hint}. The pointer must not cover the dot.")
                pt = click_points(cam, 1, "Click the center of the dot (s = skip if it is not visible).", allow_skip=True)
                if pt is not None:
                    state["views"].append({"T_base_ee": robot.forward(q).tolist(), "uv": pt[0].tolist()})
                    # A new view invalidates any previous solution.
                    state.pop("f", None)
                    state.pop("T_ee_cam", None)
                    save_state(state)
        views = [(np.array(v["T_base_ee"]), np.array(v["uv"])) for v in state["views"]]

        print("\n=== Step 5/5: Solving ===")
        if "f" in state and "T_ee_cam" in state:
            print("  Already done, skipping.")
            f, T_ee_cam = state["f"], np.array(state["T_ee_cam"])
        else:
            T_ee_cam, f, rms_px = solve_camera(views, dot, cx, cy, f0)
            f = float(f)
            state["f"], state["T_ee_cam"] = f, T_ee_cam.tolist()
            save_state(state)
            print(f"  Focal length: {f:.0f} px (ruler estimate was {f0:.0f} px)")
            print(f"  Camera position relative to pointer tip: {np.round(T_ee_cam[:3, 3], 1)} mm")
            print(f"  Camera axes in end-effector frame (columns x, y, z):\n{np.round(T_ee_cam[:3, :3], 3)}")
            print(f"  Reprojection error: {rms_px:.1f} px RMS")
            for i, (T_base_ee, uv) in enumerate(views):
                p = pixels_to_table(uv, T_base_ee, T_ee_cam, f, cx, cy, z_table)
                err = np.linalg.norm(p[0, :2] - dot[:2]) if len(p) else float("nan")
                print(f"  View {i + 1}: dot found {err:.1f} mm from where the pointer touched it")
            if rms_px > 10:
                print("  Warning: large error. Check the FK, and that the dot did not move during calibration.")
            print(f"  Saved to {CALIB_FILE}")

        print("\n=== Optional: Check on new poses ===")
        while True:
            q = robot.move_until_confirmed("Move to a new pose where the camera sees the dot (or press Enter, then s, to finish).")
            pt = click_points(cam, 1, "Click the center of the dot (s = finish).", allow_skip=True)
            if pt is None:
                break
            p = pixels_to_table(pt, robot.forward(q), T_ee_cam, f, cx, cy, z_table)
            print(f"  Error: {np.linalg.norm(p[0, :2] - dot[:2]):.1f} mm" if len(p) else "  The ray misses the table.")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
