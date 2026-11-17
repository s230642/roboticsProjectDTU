"""Simulate the servos and check that follow_path moves smoothly along a line (no robot needed)."""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "MyRobot", "Python"))
import MyRobot as robot_module
from MyRobot import MyRobot

DEG_S_PER_UNIT = 0.111 * 6  # AX-12A moving speed: 1 unit = 0.111 rpm


class SimServos:
    """Joints move straight toward their goal at their set speed; time only advances when the code sleeps."""

    def __init__(self, robot, start):
        self.robot, self.t, self.log = robot, 0.0, []
        self.pos, self.goal = np.array(start, float), np.array(start, float)
        self.speed = np.full(4, 0.1 * 1023 * DEG_S_PER_UNIT)
        robot._write2, robot.read_joint_angles = self.write2, lambda: self.pos.copy()
        robot_module.time = SimpleNamespace(sleep=self.sleep, time=lambda: self.t)

    def write2(self, dxl_id, addr, value):
        i = dxl_id - 1
        if addr == MyRobot.ADDR_MX_GOAL_POSITION:
            self.goal[i] = MyRobot.ticks_to_deg(value) - self.robot.joint_offsets[i]
        elif addr == MyRobot.ADDR_MX_MOVING_SPEED:
            self.speed[i] = (value or 1023) * DEG_S_PER_UNIT  # 0 means full speed on these servos

    def sleep(self, dt):
        for _ in range(max(1, int(dt / 0.01))):
            step = np.clip(self.goal - self.pos, -self.speed * 0.01, self.speed * 0.01)
            self.pos += step
            self.t += 0.01
            self.log.append((self.t, self.robot.forward(self.pos)[:3, 3]))


def make_robot():
    robot = MyRobot.__new__(MyRobot)
    robot.motor_ids = [1, 2, 3, 4]
    robot.joint_offsets = np.array([150, 150, 145, 150], dtype=float)
    robot.joint_limits = np.array([[-90, 90], [-180, 180], [-140, 140], [-140, 140]])
    robot.use_smooth_speed_flag = False
    robot.motor_speed = np.full(4, 0.1)
    return robot


if __name__ == "__main__":
    s = np.arange(0, 120, 5.0)  # curved line on the table, points 5 mm apart
    points = np.column_stack([150 + 30 * np.sin(s / 40), s - 60, np.full_like(s, -30.0)])
    robot = make_robot()
    sim = SimServos(robot, robot.inverse(*points[0] + [0, 0, 20], np.pi / 2))
    robot.follow_path(points, hover_mm=20, speed_mm_s=20)

    t = np.array([row[0] for row in sim.log])
    tip = np.array([row[1] for row in sim.log])
    on_line = np.abs(tip[:, 2] - points[0, 2]) < 2  # fingertip down at line height, not approaching or lifting
    t_line, tip_line = t[on_line], tip[on_line]
    speed = np.linalg.norm(np.diff(tip_line, axis=0), axis=1) / np.diff(t_line)
    off_line = max(np.min(np.linalg.norm(points - p, axis=1)) for p in tip_line)
    print(f"line of {len(points)} points, {np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)):.0f} mm, "
          f"traced in {t_line[-1] - t_line[0]:.1f} s (target {np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) / 20:.1f} s)")
    p5, p50, p95 = np.percentile(speed, [5, 50, 95])
    print(f"fingertip speed along the line: 5% {p5:.1f}, median {p50:.1f}, 95% {p95:.1f} mm/s (target 20); "
          f"moments below 2 mm/s: {np.sum(speed < 2)} of {len(speed)}")
    print(f"furthest from a line point: {off_line:.1f} mm (points are 5 mm apart, so up to 2.5 mm is between points)")
    print(f"ends at {np.round(tip[-1], 1)}, last line point + lift {np.round(points[-1] + [0, 0, 20], 1)}")
