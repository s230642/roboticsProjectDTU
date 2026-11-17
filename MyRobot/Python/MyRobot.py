import os
import sys
import inspect
import math
import numpy as np
import time

# --- Dynamixel SDK path bootstrap -----------------------
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sdk_path = os.path.join(parentdir, "DynamixelSDK-3.7.31", "python", "src", "dynamixel_sdk")
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from dynamixel_sdk import PortHandler, PacketHandler


def cosd(x_deg):  # cos with degrees
    return math.cos(math.radians(x_deg))


def sind(x_deg):  # sin with degrees
    return math.sin(math.radians(x_deg))


class MyRobot:
    # ----------------------------- Constants (AX-12A) -------------------------
    ADDR_MX_TORQUE_ENABLE = 24       # 1 byte
    ADDR_MX_CW_COMPLIANCE_SLOPE = 28   # 1 byte
    ADDR_MX_CCW_COMPLIANCE_SLOPE = 29  # 1 byte
    ADDR_MX_GOAL_POSITION = 30       # 2 bytes
    ADDR_MX_MOVING_SPEED = 32        # 2 bytes
    ADDR_MX_TORQUE_LIMIT = 34        # 2 bytes
    ADDR_MX_PRESENT_POSITION = 36    # 2 bytes

    PROTOCOL_VERSION = 1.0
    BAUDRATE = 1000000
    DEVICENAME = "COM10"              # e.g., Windows: 'COM3' | Linux: '/dev/ttyUSB0' | Mac: '/dev/tty.usbserial-*'

    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    COMM_SUCCESS = 0
    DXL_MOVING_STATUS_THRESHOLD = 10  # [ticks], not used directly here
    TICKS_PER_DEG = 1.0 / 0.29        # ≈ 3.448275862 ticks per degree
    DEG_PER_TICK = 0.29
    FINGER_LENGTH = 80.0             # fingertip distance along the last link from the joint 4 axis (mm)
    FINGER_OFFSET = 14.0             # fingertip offset above the last link, in the arm's plane (mm)

    # ------------------------------- Init -------------------------------------
    def __init__(self):
        # Robot + kinematics state
        self.motor_ids = [1, 2, 3, 4]
        self.gripper_motor_id = 5
        self.forward_transform = np.zeros((4, 4))
        self.joint_angles = np.array([0.0, 0.0, 0.0, 0.0])  # deg
        self.joint_pos = np.zeros((4, 4))
        self.draw_robot_flag = False
        self.use_smooth_speed_flag = False
        self.gripper_open_flag = True

        # deg limits: [-130,130], [-180,0], [-140,140], [-140,100]
        self.joint_limits = np.array([[-90, 90], [-180, 180], [-140, 140], [-140, 140]])

        # IK config (simple analytical below; no external toolbox)
        self.ik_weights = np.array([0.25, 0.25, 0.25, 1, 1, 1])

        # Motor offsets (deg)
        self.joint_offsets = np.array([150, 150, 145, 150], dtype=float)

        self.joint_angle_error = np.array([0.0, 0.0, 0.0, 0.0])
        self.init_status = 0

        self.movement_history = []  # list of dicts
        self.motor_speed = np.array([0.1, 0.1, 0.1, 0.1])   # [0..1]
        self.motor_torque = np.array([1.0, 1.0, 1.0, 1.0])  # [0..1]
        self.pitch = 0.0

        # --- SDK setup ---
        try:
            self.port_handler = PortHandler(self.DEVICENAME)
            self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

            if self.port_handler.openPort():
                print("Succeeded to open the port!")
            else:
                print("Failed to open the port! Reconnect Robot.")
                raise RuntimeError("openPort failed")

            if self.port_handler.setBaudRate(self.BAUDRATE):
                print("Succeeded to change the baudrate!")
            else:
                print("Failed to change the baudrate! Reconnect Robot.")
                raise RuntimeError("setBaudRate failed")

            # Initialize defaults
            self.set_speed(self.motor_speed, overwrite_speeds=True)
            self.set_torque_limit(self.motor_torque)
            #self.move_j(0, 0, 0, 0)
            self.enable_motors()
            self.init_status = 1

        except Exception as e:
            print(f"Initialization error: {e}")
            self.init_status = 0

    # ----------------------------- Utilities ----------------------------------
    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @classmethod
    def deg_to_ticks(cls, deg):
        return int(round(deg * cls.TICKS_PER_DEG))

    @classmethod
    def ticks_to_deg(cls, ticks):
        return ticks * cls.DEG_PER_TICK

    def _check_limits_one(self, deg, idx):
        lo, hi = self.joint_limits[idx]
        if not (lo <= deg <= hi):
            if idx == 0:
                raise ValueError("Angle Limits for first Axis Reached, Min/Max: ±130°")
            elif idx == 1:
                raise ValueError("Angle Limits for second Axis Reached, Min/Max: [−180°, 0°]")
            else:
                raise ValueError("Angle Limits Reached, Min/Max: ±100°")
        return deg

    # --------------------------- Motor I/O wrappers ---------------------------
    def _write1(self, dxl_id, addr, value):
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, addr, int(value))
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL write1 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={dxl_error}")

    def _write2(self, dxl_id, addr, value):
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler, dxl_id, addr, int(value))
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL write2 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={dxl_error}")

    def _read2(self, dxl_id, addr):
        value, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(self.port_handler, dxl_id, addr)
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL read2 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={dxl_error}")
        return value

    # ------------------------------- Speeds/Torque ----------------------------
    def smooth_speed(self, joint_angles_delta_deg):
        max_angle = float(np.max(np.abs(joint_angles_delta_deg)))
        if max_angle == 0:
            return
        speed_per_deg = max_angle / 100.0
        if speed_per_deg != 0:
            new_speeds = np.abs(joint_angles_delta_deg / speed_per_deg) * 0.01
            # preserve zeros by falling back to previous speeds
            for i in range(len(self.motor_speed)):
                if new_speeds[i] == 0:
                    new_speeds[i] = self.motor_speed[i]
                else:
                    new_speeds[i] = new_speeds[i] * self.motor_speed[i]
            self.set_speed(new_speeds, overwrite_speeds=False)

    def set_speed(self, speeds, overwrite_speeds=True):
        speeds = np.asarray(speeds, dtype=float)
        if overwrite_speeds:
            self.motor_speed = speeds.copy()

        for i, dxl_id in enumerate(self.motor_ids):
            s = float(speeds[i])
            if 0 < s <= 1:
                raw = int(round(s * 1023))
                self._write2(dxl_id, self.ADDR_MX_MOVING_SPEED, raw)
            else:
                print("Movement speed out of range, enter value between (0, 1].")

    def set_torque_limit(self, torques):
        torques = np.asarray(torques, dtype=float)
        self.motor_torque = torques.copy()
        for i, dxl_id in enumerate(self.motor_ids):
            t = float(torques[i])
            if 0 < t <= 1:
                self._write2(dxl_id, self.ADDR_MX_TORQUE_LIMIT, int(round(t * 1023)))
            else:
                print("Torque limit out of range, enter value between (0, 1].")

    def set_compliance_slope(self, slopes):
        """How gently each servo pushes near its goal (default 32). Lower = stiffer: 2, 4, 8, 16, 32, 64 or 128."""
        for dxl_id, slope in zip(self.motor_ids, slopes):
            self._write1(dxl_id, self.ADDR_MX_CW_COMPLIANCE_SLOPE, slope)
            self._write1(dxl_id, self.ADDR_MX_CCW_COMPLIANCE_SLOPE, slope)

    # --------------------------------- Power ----------------------------------
    def enable_motors(self):
        for dxl_id in self.motor_ids:
            self._write1(dxl_id, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_ENABLE)
        print("Dynamixel torque enabled on all motors.")

    def disable_motors(self):
        for dxl_id in self.motor_ids:
            try:
                self._write1(dxl_id, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_DISABLE)
            except Exception as e:
                print(f"Disable motor {dxl_id} error: {e}")
        try:
            self.port_handler.closePort()
        finally:
            self.init_status = 0
        print("Dynamixels disabled and port closed.")

    # ------------------------------- Telemetry --------------------------------
    def get_position(self, motor_id):
        ticks = self._read2(motor_id, self.ADDR_MX_PRESENT_POSITION)
        return self.ticks_to_deg(ticks)

    def read_joint_angles(self):
        j_a = np.zeros(4, dtype=float)
        for i, dxl_id in enumerate(self.motor_ids):
            ticks = self._read2(dxl_id, self.ADDR_MX_PRESENT_POSITION)
            deg = self.ticks_to_deg(ticks) - self.joint_offsets[i]
            j_a[i] = deg
            self.joint_angle_error[i] = deg - self.joint_angles[i]
        return j_a

    # --------------------------- Kinematics (FK / IK) -------------------------
   # Please input your forward and inverse kinematics here for position and pose conversion to joint angles.
   #  The output should be in the format of an array

    def forward(self, joint_angles_deg):
        #DH parameters
        d__1 , a__1 , alpha__1 = 50, 0 , -np.pi / 2
        d__2 , a__2 , alpha__2 = 0, 93 , 0
        d__3 , a__3 , alpha__3 = 0, 93 , 0
        d__4 , a__4 , alpha__4 = 0, self.FINGER_LENGTH , 0

        #Joint angles and map to radians 
        q1, q2, q3, q4  = joint_angles_deg
        # The axis are flipped on the robot compared to denavit hartenbug
        theta__1, theta__2, theta__3, theta__4 = np.deg2rad([q1, -q2 - 90, -q3, -q4])

        t_01 = np.array([[np.cos(theta__1), -np.sin(theta__1)*np.cos(alpha__1), np.sin(theta__1)*np.sin(alpha__1), a__1*np.cos(theta__1)], [np.sin(theta__1), np.cos(theta__1)*np.cos(alpha__1), -np.cos(theta__1)*np.sin(alpha__1), a__1*np.sin(theta__1)], [0, np.sin(alpha__1), np.cos(alpha__1), d__1], [0, 0, 0, 1]])
        t_12 = np.array([[np.cos(theta__2), -np.sin(theta__2)*np.cos(alpha__2), np.sin(theta__2)*np.sin(alpha__2), a__2*np.cos(theta__2)], [np.sin(theta__2), np.cos(theta__2)*np.cos(alpha__2), -np.cos(theta__2)*np.sin(alpha__2), a__2*np.sin(theta__2)], [0, np.sin(alpha__2), np.cos(alpha__2), d__2], [0, 0, 0, 1]])
        t_23 = np.array([[np.cos(theta__3), -np.sin(theta__3)*np.cos(alpha__3), np.sin(theta__3)*np.sin(alpha__3), a__3*np.cos(theta__3)], [np.sin(theta__3), np.cos(theta__3)*np.cos(alpha__3), -np.cos(theta__3)*np.sin(alpha__3), a__3*np.sin(theta__3)], [0, np.sin(alpha__3), np.cos(alpha__3), d__3], [0, 0, 0, 1]])
        t_34 = np.array([[np.cos(theta__4), -np.sin(theta__4)*np.cos(alpha__4), np.sin(theta__4)*np.sin(alpha__4), a__4*np.cos(theta__4)], [np.sin(theta__4), np.cos(theta__4)*np.cos(alpha__4), -np.cos(theta__4)*np.sin(alpha__4), a__4*np.sin(theta__4)], [0, np.sin(alpha__4), np.cos(alpha__4), d__4], [0, 0, 0, 1]])

        # Fingertip sits FINGER_OFFSET above the link line (DH y4 points below the link)
        t_tip = np.eye(4)
        t_tip[1, 3] = -self.FINGER_OFFSET

        return t_01@t_12@t_23@t_34@t_tip

    def inverse(self, x, y, z, pitch, elbow_up=True):
        """
        Joint angles (deg, servo convention) that put the pointer tip at (x, y, z) mm in the base frame.

        pitch: angle of the pointer below horizontal in radians (pi/2 = pointing straight down).
        Raises ValueError if the point cannot be reached with that pitch.
        """
        d__1, a__2, a__3, a__4, h = 50, 93, 93, self.FINGER_LENGTH, self.FINGER_OFFSET

        theta__1 = np.arctan2(y, x)

        # Wrist position in the arm's vertical plane: u outward from the base axis, w downward from the shoulder
        u = np.hypot(x, y) - a__4 * np.cos(pitch) - h * np.sin(pitch)
        w = (d__1 - z) - a__4 * np.sin(pitch) + h * np.cos(pitch)

        c3 = (u**2 + w**2 - a__2**2 - a__3**2) / (2 * a__2 * a__3)
        if abs(c3) > 1:
            raise ValueError(f"Point ({x:.1f}, {y:.1f}, {z:.1f}) is out of reach with pitch {np.degrees(pitch):.1f} deg")

        # Positive DH angles bend downward, so elbow up means theta3 > 0
        theta__3 = np.arccos(c3) if elbow_up else -np.arccos(c3)
        theta__2 = np.arctan2(w, u) - np.arctan2(a__3 * np.sin(theta__3), a__2 + a__3 * np.cos(theta__3))
        theta__4 = pitch - theta__2 - theta__3

        # Back from DH angles to the robot's servo convention (inverse of the mapping in forward)
        t1, t2, t3, t4 = np.degrees([theta__1, theta__2, theta__3, theta__4])
        q = np.array([t1, -t2 - 90, -t3, -t4])
        return (q + 180) % 360 - 180

    # ------------------------------- Motion -----------------------------------
    def _write_goal(self, joint_deg):
        cmd_deg = joint_deg + self.joint_offsets
        for i, dxl_id in enumerate(self.motor_ids):
            self._write2(dxl_id, self.ADDR_MX_GOAL_POSITION, self.deg_to_ticks(cmd_deg[i]))

    def move_j(self, j1_deg, j2_deg, j3_deg, j4_deg, wait=True, tol_deg=2.0, poll_period=0.05, timeout_s=10.0,
               settle_s=0.3, max_push_deg=5.0):
        # Limits
        desired = np.array([j1_deg, j2_deg, j3_deg, j4_deg], dtype=float)
        for i in range(4):
            desired[i] = self._check_limits_one(desired[i], i)

        # Smooth speed option
        if self.use_smooth_speed_flag:
            self.smooth_speed(desired - self.joint_angles)

        # Update internal & FK
        #self.joint_angles = desired.copy()
        #self.forward(desired)

        goal = desired.copy()
        self._write_goal(goal)

        if not wait:
            return


        # Wait until all joints within tolerance, but give up loudly after timeout_s
        deadline = time.time() + timeout_s
        ja, prev, still_since, last_error = None, None, time.time(), None
        while time.time() < deadline:
            try:
                ja = self.read_joint_angles()
            except Exception as e:
                # If read fails transiently, just try again shortly.
                last_error = e
                time.sleep(poll_period)
                continue
            if np.all(np.abs(ja - desired) < tol_deg):
                return
            # Stopped short of the target (e.g. sagging under gravity): move the goal past it by the remaining error
            if prev is None or np.any(np.abs(ja - prev) > 0.3):
                still_since = time.time()
            elif time.time() - still_since > settle_s:
                goal = np.clip(goal + desired - ja, desired - max_push_deg, desired + max_push_deg)
                self._write_goal(goal)
                still_since = time.time()
            prev = ja
            time.sleep(poll_period)
        if ja is None:
            raise TimeoutError(f"move_j: could not read joint angles for {timeout_s} s (last error: {last_error})")
        raise TimeoutError(f"move_j: joints did not reach target within {timeout_s} s. "
                           f"Target {np.round(desired, 1)}, reached {np.round(ja, 1)}, error {np.round(ja - desired, 1)} deg")

    def move_until_confirmed(self, instruction):
        """Let the user jog the arm by typing joint angles until they press Enter. Returns the joint angles (deg)."""
        print("\n" + instruction)
        while True:
            cmd = input("  Type 4 joint angles to move (e.g. '0 -45 -90 0'), or press Enter when in position: ").strip()
            if not cmd:
                return self.read_joint_angles()
            try:
                self.move_j(*[float(a) for a in cmd.split()])
            except Exception as e:
                print(f"  Could not move: {e}")

    def move_c(self, x, y, z, pitch_deg, **kwargs):
        j_deg = self.inverse(x, y, z, math.radians(pitch_deg))
        self.move_j(j_deg[0], j_deg[1], j_deg[2], j_deg[3], **kwargs)

    def read_ee_position(self):
        j_a = self.read_joint_angles()
        return self.forward(j_a)

    def find_pitch(self, points, pitches_deg=(90, 100, 80, 110, 70, 120, 60)):
        """Return (pitch_deg, joint angles per point) for the first pitch that reaches all points, or None."""
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        for pitch in pitches_deg:
            try:
                qs = np.array([self.inverse(*p, math.radians(pitch)) for p in points])
            except ValueError:
                continue
            if np.all((lo <= qs) & (qs <= hi)):
                return pitch, qs
        return None

    def follow_path(self, points, hover_mm=20):
        """Lower onto the first point, move through all points with one pitch, then lift."""
        lift = np.array([0, 0, hover_mm])
        solution = self.find_pitch(np.vstack([points[0] + lift, points, points[-1] + lift]))
        if solution is None:
            raise ValueError("No single pitch reaches the whole path")
        for q in solution[1]:
            self.move_j(*q)

    # --------------------------- History / Macros ------------------------------
    def record_configuration(self):
        entry = {
            "j1": float(self.joint_angles[0]),
            "j2": float(self.joint_angles[1]),
            "j3": float(self.joint_angles[2]),
            "j4": float(self.joint_angles[3]),
            "speed": float(self.motor_speed[0]),
            "torque": float(self.motor_torque[0]),
            "gripper_open": bool(self.gripper_open_flag),
        }
        self.movement_history.append(entry)
        print(
            f"Recorded Speed: {entry['speed']:.3f}, Torque: {entry['torque']:.3f}\n"
            f"Joint Positions: {entry['j1']:.2f}, {entry['j2']:.2f}, {entry['j3']:.2f}, {entry['j4']:.2f}\n"
            f"Gripper open: {int(entry['gripper_open'])}"
        )

    def delete_last_recorded_configuration(self):
        if not self.movement_history:
            print("No last history position")
            return
        self.movement_history.pop()

    def play_configuration_history(self, dwell_after_move_s=1.0, dwell_after_grip_s=3.0):
        if not self.movement_history:
            print("No recorded configurations.")
            return
        for rec in self.movement_history:
            s = rec["speed"]
            t = rec["torque"]
            self.set_speed([s, s, s, s], overwrite_speeds=True)
            self.set_torque_limit([t, t, t, t])
            self.move_j(rec["j1"], rec["j2"], rec["j3"], rec["j4"])
            time.sleep(dwell_after_move_s)
            if self.gripper_open_flag != rec["gripper_open"]:
                self.actuate_gripper()
                time.sleep(dwell_after_grip_s)


# ------------------------------- Example usage --------------------------------
# Example 1:
# Demonstrates how to move the robot joints into desired joint positions [degrees]
# You might need to adjust the offsets in the code above to make the zero point suit your needs.
# Just comment out the code snippet, input desired joint angles in robot.move_j() and run.

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3)
    robot = MyRobot()
    if robot.init_status:
            #Reading off position for calibration:
            joint_angles = robot.read_joint_angles()
            print("Initial joint angles (deg):", joint_angles)
            print("Initial forward kinematic matrix:\n", robot.forward(joint_angles))
            #robot.move_j(0, 0, 0, 0)
            #robot.move_j(-0.7, -79.5, -37.4, -43.1)       # pointer tip to (x, y, z) mm, pointing straight down
            #q = robot.inverse(147.1, -8.4, -42.2, np.pi/2)   # just get the joint angles (servo convention, deg)
            #print(q)
 