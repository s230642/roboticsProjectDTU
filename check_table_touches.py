"""Check how well FK agrees with poses where the fingertip touches the table, and fit per-joint offsets."""
import os
import re
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "MyRobot", "Python"))
from MyRobot import MyRobot

TOUCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "touching_table.txt")


def load_touches(path=TOUCH_FILE):
    """Joint angles (deg) of every 'Initial joint angles (deg): [...]' line in the file."""
    with open(path) as fh:
        rows = re.findall(r"joint angles \(deg\): \[([^\]]*)\]", fh.read())
    return np.array([[float(v) for v in row.split()] for row in rows])


def table_heights(robot, Q, offsets=np.zeros(4)):
    """FK fingertip height for each pose, with `offsets` (deg) added to the joint readings."""
    return np.array([robot.forward(q + offsets)[2, 3] for q in Q])


def fit_offset(robot, Q, joint):
    """Extra offset (deg) on one joint that puts all touches at the most equal height."""
    def residual(p):
        offsets = np.zeros(4)
        offsets[joint] = p[0]
        z = table_heights(robot, Q, offsets)
        return z - z.mean()
    return least_squares(residual, [0.0]).x[0]


def report(label, z):
    print(f"{label:28s} spread {np.ptp(z):5.1f} mm, rms {z.std():4.1f} mm, table z {z.mean():6.1f} mm")


if __name__ == "__main__":
    robot = MyRobot.__new__(MyRobot)  # FK only, no connection to the arm
    Q = load_touches()
    print(f"{len(Q)} touches, wrist angles {np.round(Q[:, 3]).astype(int)}")
    report("current FK", table_heights(robot, Q))
    for joint in (1, 2, 3):
        delta = fit_offset(robot, Q, joint)
        offsets = np.zeros(4)
        offsets[joint] = delta
        report(f"joint {joint + 1} offset {delta:+.1f} deg", table_heights(robot, Q, offsets))
    print("To apply an offset of d deg on a joint, subtract d from that joint's value in MyRobot.joint_offsets.")
