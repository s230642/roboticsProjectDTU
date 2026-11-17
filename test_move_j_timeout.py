"""Check that move_j corrects for sag, and raises instead of hanging when the arm cannot get there (no robot needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "MyRobot", "Python"))
from MyRobot import MyRobot


def fake_robot(sag=(0, 0, 0, 0), stuck=False, fail_reads=False):
    """MyRobot without a connection. Each joint settles `sag` deg from its written goal (like holding against gravity).
    stuck: joints never respond to new goals. fail_reads: every read raises."""
    robot = MyRobot.__new__(MyRobot)
    robot.motor_ids = [1, 2, 3, 4]
    robot.joint_offsets = np.full(4, 150.0)
    robot.joint_limits = np.array([[-90, 90], [-180, 180], [-140, 140], [-140, 140]])
    robot.use_smooth_speed_flag = False
    goal = np.full(4, np.nan)

    def write2(dxl_id, addr, ticks):
        if not (stuck and not np.isnan(goal[dxl_id - 1])):
            goal[dxl_id - 1] = MyRobot.ticks_to_deg(ticks) - robot.joint_offsets[dxl_id - 1]

    def read():
        if fail_reads:
            raise IOError("DXL read2 error")
        return goal + sag

    robot._write2, robot.read_joint_angles = write2, read
    return robot, goal


if __name__ == "__main__":
    target = np.array([0.0, -60.0, -60.0, -30.0])
    cases = [("no sag", {}),
             ("joint 2 sags 3 deg, joint 3 sags 1.3 deg", {"sag": (0, -3, 1.3, 0)}),
             ("joint 2 sags 3 deg and ignores new goals", {"sag": (0, -3, 0, 0), "stuck": True}),
             ("every read fails", {"fail_reads": True})]
    for name, kwargs in cases:
        robot, goal = fake_robot(**kwargs)
        try:
            robot.move_j(*target, timeout_s=2.0)
            reached = goal + kwargs.get("sag", 0)
            print(f"{name}: reached {np.round(reached, 1)}, error {np.round(reached - target, 2)} deg")
        except TimeoutError as e:
            print(f"{name}: TimeoutError -> {e}")
