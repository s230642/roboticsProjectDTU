import numpy as np
import sympy as sp

# ==========================================
# EXERCISE 1: Transformation Matrix from DH Parameters
# ==========================================
def get_dh_matrix(theta, d, a, alpha):
    """
    Calculates the Homogeneous Transformation Matrix T for a single joint.
    """
    c_theta = sp.cos(theta)
    s_theta = sp.sin(theta)
    c_alpha = sp.cos(alpha)
    s_alpha = sp.sin(alpha)
    
    T= sp.Matrix([
        [c_theta, -s_theta * c_alpha,  s_theta * s_alpha, a * c_theta],
        [s_theta,  c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
        [0,        s_alpha,            c_alpha,           d          ],
        [0,        0,                  0,                 1          ]
    ])
    return T.applyfunc(lambda expr: expr.replace(lambda x: x.is_Float and abs(x) < 1e-10, lambda x: sp.S.Zero))

def get_all_fk_matrices(dh_table):
    """
    Calculates Forward Kinematics and returns ALL intermediate matrices.
    Takes a DH table as input to remain generic.
    """
    # Store intermediate matrices starting with Base Frame (Identity)
    T_matrices = [sp.eye(4)] 
    T_current = sp.eye(4)
    
    # Iterate through the provided DH table
    for row in dh_table:
        T_step = get_dh_matrix(*row)
        T_current = T_current @ T_step
        T_matrices.append(T_current)
        
    # Apply Tool Offset for Frame 5 (Camera Frame)
    # T_matrices[4] is the End Effector (T0_4)
    T4_5 = sp.Matrix([
        [1, 0, 0, -15],
        [0, 1, 0,  45],
        [0, 0, 1,   0],
        [0, 0, 0,   1]
    ])
    T0_5 = T_matrices[4] @ T4_5 
    T0_5 = T0_5.applyfunc(lambda expr: expr.replace(lambda x: x.is_Float and abs(x) < 1e-10, lambda x: sp.S.Zero))
    
    # Add T0_5 to the end of the list
    T_matrices.append(T0_5)
    
    # Returns: [T0_0, T0_1, T0_2, T0_3, T0_4, T0_5]
    return T_matrices

q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4')
dh_parameters = [
    [q1,    50.0,  0.0,   np.pi/2], # Joint 1
    [q2 + np.pi/2,    0.0,   93.0,  0.0],     # Joint 2
    [q3,    0.0,   93.0,  0.0],     # Joint 3
    [q4,    0.0,   50.0,  0.0]      # Joint 4
]

all_T_matrices = get_all_fk_matrices(dh_parameters)
# Calculate the final transformation matrix from base to end effector
T0_4 = all_T_matrices[4]  # Transformation from base to end-effector
T0_4 = sp.simplify(T0_4)
T0_5 = all_T_matrices[5]  # Transformation from base to camera frame
T0_5 = sp.simplify(T0_5)

if __name__ == "__main__":
    # Print the result
    print("Final Transformation Matrix T for", len(dh_parameters), "joints:")
    sp.pprint(T0_4)

    print("\nFinal Transformation Matrix T0_5 (Base to Camera Frame):")
    sp.pprint(T0_5)

# ==========================================================
# EXERCISE 2: Analytical Inverse Kinematics
# ==========================================================
def inverse_kinematics_analytical(target_pos, elbow_down=True):
    """
    Calculates q = [q1, q2, q3, q4] analytically.
    target_pos: [x, y, z] desired position of the end-effector (o_0^4).
    elbow_down: If True, calculates for the elbow-down configuration; if False, for the elbow-up configuration.
    """
    x, y, z = target_pos
    
    # Dimensions extracted from the T_0^4 matrix
    d1 = 50.0   # Base height
    a2 = 93.0   # Link length 2
    a3 = 93.0   # Link length 3
    a4 = 50.0   # Pencil length / end-effector length

    # Angle q1: Projection in the XY plane
    q1 = np.arctan2(y, x)

    # Decoupling of orientation (Horizontal pencil: q2 + q3 + q4 = 0)
    r = np.sqrt(x**2 + y**2)
    r_prime = r - a4 * 1.0
    s_prime = z - d1

    # Angle q3 (Cosine Rule)
    D = (r_prime**2 + s_prime**2 - a2**2 - a3**2) / (2.0 * a2 * a3)
    if D > 1.000001 or D < -1.000001:
        print("Warning: Unreachable position. The point is outside the workspace of the robot.")
        return None
    D = np.clip(D, -1.0, 1.0)  # Avoid numerical issues with arccos
    if elbow_down:
        s3 = -np.sqrt(1.0 - D**2)  # Configuration elbow down
    else:
        s3 = np.sqrt(1.0 - D**2)   # Configuration elbow up

    q3 = np.arctan2(s3, D)

    # Angle q2
    q2 = np.arctan2(s_prime, r_prime) - np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3))- np.pi/2

    # Angle q4
    q4 = -(q2+ np.pi/2 + q3)

    q_rad = np.array([q1, q2, q3, q4])
    q_deg = np.rad2deg(q_rad)

    return q_rad, q_deg


# ==========================================================
# EXERCISE 3: Loop to Trace the Circle (Results in Degrees)
# ==========================================================
R = 32.0                                # Radius in mm
p0c = np.array([150.0, 0.0, 120.0])     # Center of the circle [x, y, z] in mm

# 37 points (36 equal divisions between 0 and 2*pi)
phi_points = np.linspace(0, 2 * np.pi, 37)

# Store configurations in degrees
q_trajectory_rad = []
q_trajectory_deg = []

if __name__ == "__main__":
    print("=" * 75)
    print(f"{'Point (j)':^10} | {'phi (deg)':^10} | {'q1 (rad / deg)':^11} | {'q2 (rad / deg)':^11} | {'q3 (rad / deg)':^11} | {'q4 (rad / deg)':^11}")
    print("=" * 75)

for j, phi in enumerate(phi_points):
    # Circle equation given in the statement
    target_pos = p0c + R * np.array([0.0, np.cos(phi), np.sin(phi)])
    
    # Calculate q(j) in degrees
    q_rad, q_deg = inverse_kinematics_analytical(target_pos, elbow_down=True)
    if q_rad is None:
        raise ValueError(f"Target position {target_pos} is unreachable. Please check the workspace limits.")
    
    q_trajectory_rad.append(q_rad)
    q_trajectory_deg.append(q_deg)
    
    phi_deg = np.rad2deg(phi)
    q1_str = f"{q_rad[0]:6.3f} / {q_deg[0]:6.1f}"
    q2_str = f"{q_rad[1]:6.3f} / {q_deg[1]:6.1f}"
    q3_str = f"{q_rad[2]:6.3f} / {q_deg[2]:6.1f}"
    q4_str = f"{q_rad[3]:6.3f} / {q_deg[3]:6.1f}"
    print(f"{j:^7} | {phi_deg:^9.1f} | {q1_str:^19} | {q2_str:^19} | {q3_str:^19} | {q4_str:^19}")

q_trajectory_rad = np.array(q_trajectory_rad)
q_trajectory_deg = np.array(q_trajectory_deg)
#print("=" * 75)
#print(f"Sequence generated: array of shape {q_trajectory_deg.shape} with all angles in degrees.")
#print(f"Sequence generated: array of shape {q_trajectory_rad.shape} with all angles in radians.")

# ==========================================================
# EXERCISE 4: Calculate the Geometric Jacobian
# ==========================================================

def calculate_jacobians_sym(T_matrices):
    """Calculates Symbolic Jacobians extracting Z axes and Origins."""
    Z0 = T_matrices[0][0:3, 2];  O0 = T_matrices[0][0:3, 3]
    Z1 = T_matrices[1][0:3, 2];  O1 = T_matrices[1][0:3, 3]
    Z2 = T_matrices[2][0:3, 2];  O2 = T_matrices[2][0:3, 3]
    Z3 = T_matrices[3][0:3, 2];  O3 = T_matrices[3][0:3, 3]
    
    O_ee = T_matrices[4][0:3, 3]
    O_cam  = T_matrices[5][0:3, 3]
    
    # --- Camera Jacobian ---
    Jv1_cam = Z0.cross(O_cam - O0)
    Jv2_cam = Z1.cross(O_cam - O1)
    Jv3_cam = Z2.cross(O_cam - O2)
    Jv4_cam = Z3.cross(O_cam - O3)
    
    J_cam = sp.Matrix.vstack(
        sp.Matrix.hstack(Jv1_cam, Jv2_cam, Jv3_cam, Jv4_cam),
        sp.Matrix.hstack(Z0, Z1, Z2, Z3)
    )
    
    # --- End-Effector Jacobian ---
    Jv1_ee = Z0.cross(O_ee - O0)
    Jv2_ee = Z1.cross(O_ee - O1)
    Jv3_ee = Z2.cross(O_ee - O2)
    Jv4_ee = Z3.cross(O_ee - O3)
    
    J_ee = sp.Matrix.vstack(
        sp.Matrix.hstack(Jv1_ee, Jv2_ee, Jv3_ee, Jv4_ee),
        sp.Matrix.hstack(Z0, Z1, Z2, Z3)
    )
    
    return sp.simplify(J_cam), sp.simplify(J_ee)

J_camera_sym, J_ee_sym = calculate_jacobians_sym(all_T_matrices)

J_camera_sym = sp.simplify(J_camera_sym)
J_ee_sym = sp.simplify(J_ee_sym)

sp.pprint(J_camera_sym)
sp.pprint(J_ee_sym)

J_cam_func = sp.lambdify((q1, q2, q3, q4), J_camera_sym, modules="numpy")
J_ee_func = sp.lambdify((q1, q2, q3, q4), J_ee_sym, modules="numpy")
T0_4_func = sp.lambdify((q1, q2, q3, q4), T0_4, modules="numpy")

jacobians_ee_store = {}
jacobians_cam_store = {}

path_points = {
    "phi = 0":      q_trajectory_rad[0], 
    "phi = pi/2":   q_trajectory_rad[9],
    "phi = pi":     q_trajectory_rad[18],
    "phi = 3pi/2":  q_trajectory_rad[27]
}

for point_name, q_vals in path_points.items():
    
    J_cam_num = J_cam_func(q_vals[0], q_vals[1], q_vals[2], q_vals[3])
    J_ee_num = J_ee_func(q_vals[0], q_vals[1], q_vals[2], q_vals[3])

    jacobians_cam_store[point_name] = J_cam_num
    jacobians_ee_store[point_name] = J_ee_num
    if __name__ == "__main__":
        print(f"\nEvaluating at {point_name} (q = [{q_vals[0]:.2f}, {q_vals[1]:.2f}, {q_vals[2]:.2f}]):")
        
        print("Camera Jacobian (J_cam):")
        print(np.round(J_cam_num, 4))
        
        print("\nEnd-Effector Jacobian (J_ee):")
        print(np.round(J_ee_num, 4))
        print("-" * 40)

# ==========================================================
# EXERCISE 5: Joint Velocities for a Given End-Effector Velocity
# ==========================================================
def compute_joint_velocities(point_name,target_velocity):
    """
    Computes the joint velocities (q_dot) by automatically retrieving 
    the configuration and Jacobian associated with a specific path point.
    """
    # Check if the point exists
    if point_name not in path_points:
        print(f"Error: The point '{point_name}' was not found in path_points.")
        return None
        
    # Extract stored q values and the numerical Jacobian
    q_vals = path_points[point_name]
    J_ee = jacobians_ee_store[point_name]
    
    # Evaluate T0_4 numerically
    T0_4_num = T0_4_func(*q_vals)
    
    # Extract the linear and angular parts of the End-Effector Jacobian
    J_v = J_ee[0:3, :]  # Top 3 rows (Linear velocity)
    J_w = J_ee[3:6, :]  # Bottom 3 rows (Angular velocity)

    x4_x = T0_4_num[0, 0]
    x4_y = T0_4_num[1, 0]
    
    # Angular constraint equation: w_x * x4_y - w_y * x4_x = 0
    J_w_x = J_w[0, :]
    J_w_y = J_w[1, :]
    constraint_row = (x4_y * J_w_x) - (x4_x * J_w_y)
    
    A = np.vstack((J_v, constraint_row))
    b = np.array(target_velocity)
    
    # Linear system A * q_dot = b
    q_dot = np.linalg.pinv(A) @ b
    
    return q_dot

target_v = [0.0, -3.0, 0.0, 0.0]
points_to_calculate = ["phi = pi/2"]

for point in points_to_calculate:
    q_dot_res = compute_joint_velocities(point, target_v)
    
    if q_dot_res is not None:
        if __name__ == "__main__":
            print(f"\nComputed Joint Velocities (rad/s) for {point}:")
            print(np.round(q_dot_res, 4))


# ==========================================================
# EXERCISE 6: Quintic Trajectory Planning
# ==========================================================

# The assignment uses five knot-points and four segments of 2 seconds each.
# All joint variables are kept in radians in this section.
segment_duration = 2.0

knot_indices = [0, 9, 18, 27, 36]
q_knots = q_trajectory_rad[knot_indices]

# Cartesian velocities prescribed by the assignment, augmented with the
# angular constraint used in Exercise 5.  These are velocities at the
# corresponding knot points, not at the start of the following segment:
# q(0) has zero velocity, q(9) has [0, -27, 0], etc.
knot_target_velocities = {
    "phi = 0": [0.0, 0.0, 0.0, 0.0],
    "phi = pi/2": [0.0, -27.0, 0.0, 0.0],
    "phi = pi": [0.0, 0.0, -27.0, 0.0],
    "phi = 3pi/2": [0.0, 27.0, 0.0, 0.0],
    "phi = 2pi": [0.0, 0.0, 0.0, 0.0]
}

q_dot_knots = np.vstack([
    np.zeros(4),
    compute_joint_velocities("phi = pi/2", knot_target_velocities["phi = pi/2"]),
    compute_joint_velocities("phi = pi", knot_target_velocities["phi = pi"]),
    compute_joint_velocities("phi = 3pi/2", knot_target_velocities["phi = 3pi/2"]),
    np.zeros(4)
])
q_ddot_knots = np.zeros_like(q_knots) # knot accelerations are zero for this exercise


def quintic_coefficients(q_start, q_dot_start, q_ddot_start,
                         q_end, q_dot_end, q_ddot_end,
                         t_i, t_f):
    """Solve the Lecture 6 quintic interpolation problem.

    For each joint, the six conditions are q, q_dot and q_ddot at the
    beginning and at the end of the segment. The returned array has shape
    (4, 6), with one row [c0, ..., c5] per joint.
    """
    coefficient_matrix = np.array([
        [1.0, t_i, t_i**2, t_i**3, t_i**4, t_i**5],
        [0.0, 1.0, 2.0*t_i, 3.0*t_i**2,
         4.0*t_i**3, 5.0*t_i**4],
        [0.0, 0.0, 2.0, 6.0*t_i, 12.0*t_i**2,
         20.0*t_i**3],
        [1.0, t_f, t_f**2, t_f**3, t_f**4, t_f**5],
        [0.0, 1.0, 2.0*t_f, 3.0*t_f**2,
         4.0*t_f**3, 5.0*t_f**4],
        [0.0, 0.0, 2.0, 6.0*t_f, 12.0*t_f**2,
         20.0*t_f**3]
    ])

    boundary_conditions = np.column_stack((
        q_start,
        q_dot_start,
        q_ddot_start,
        q_end,
        q_dot_end,
        q_ddot_end
    ))

    return np.linalg.solve(
        coefficient_matrix,
        boundary_conditions.T
    ).T


segment_coefficients = [
    quintic_coefficients(
        q_knots[i], q_dot_knots[i], q_ddot_knots[i],
        q_knots[i + 1], q_dot_knots[i + 1], q_ddot_knots[i + 1],
        t_i=0.0,
        t_f=segment_duration
    )
    for i in range(len(q_knots) - 1)
]


def evaluate_quintic_segment(coefficients, local_time): #calculates q, q_dot and q_ddot for a given segment
    """Evaluate q, q_dot and q_ddot for one segment."""
    t = local_time

    powers = np.array([1.0, t, t**2, t**3, t**4, t**5])
    velocity_powers = np.array([0.0, 1.0, 2.0*t, 3.0*t**2,
                                4.0*t**3, 5.0*t**4])
    acceleration_powers = np.array([0.0, 0.0, 2.0, 6.0*t,
                                    12.0*t**2, 20.0*t**3])

    q = coefficients @ powers
    q_dot = coefficients @ velocity_powers
    q_ddot = coefficients @ acceleration_powers

    return q, q_dot, q_ddot


def validate_piecewise_trajectory(coefficients_list, knot_q, knot_q_dot, #calculates the maximum error at the knot points for q, q_dot and q_ddot
                                  knot_q_ddot, segment_time):
    """Check q, q_dot and q_ddot at every segment boundary."""
    errors = []

    for index, coefficients in enumerate(coefficients_list):
        q_start, q_dot_start, q_ddot_start = \
            evaluate_quintic_segment(coefficients, 0.0)
        q_end, q_dot_end, q_ddot_end = \
            evaluate_quintic_segment(coefficients, segment_time)

        errors.extend([
            np.max(np.abs(q_start - knot_q[index])),
            np.max(np.abs(q_dot_start - knot_q_dot[index])),
            np.max(np.abs(q_ddot_start - knot_q_ddot[index])),
            np.max(np.abs(q_end - knot_q[index + 1])),
            np.max(np.abs(q_dot_end - knot_q_dot[index + 1])),
            np.max(np.abs(q_ddot_end - knot_q_ddot[index + 1]))
        ])

    return max(errors)


print("\nExercise 6: quintic segment coefficients [c0, ..., c5]")
for index, coefficients in enumerate(segment_coefficients):
    print(f"\nSegment {chr(ord('A') + index)}:")
    print(np.round(coefficients, 6))

five_knot_continuity_error = validate_piecewise_trajectory(
    segment_coefficients,
    q_knots,
    q_dot_knots,
    q_ddot_knots,
    segment_duration
)

if __name__ == "__main__":
    print(
        f"\nMaximum 5-knot interpolation condition error: "
        f"{five_knot_continuity_error:.3e}"
    )


# ==========================================================
# EXERCISE 7: Cartesian Path Comparison with 9 Knot-Points
# ==========================================================

total_duration = 8.0
sample_times = np.linspace(0.0, total_duration, 801)


def evaluate_cartesian_trajectory(coefficients_list, segment_time,
                                  global_time):
    """Evaluate one piecewise joint trajectory at a global time."""
    number_of_segments = len(coefficients_list)
    trajectory_duration = number_of_segments * segment_time
    t = float(np.clip(global_time, 0.0, trajectory_duration))

    if t >= trajectory_duration:
        segment_index = number_of_segments - 1
        local_time = segment_time
    else:
        segment_index = int(t // segment_time)
        local_time = t - segment_index * segment_time

    return evaluate_quintic_segment(
        coefficients_list[segment_index],
        local_time
    )


def cartesian_positions_from_trajectory(coefficients_list, segment_time):
    """Convert a piecewise joint trajectory into Cartesian positions."""
    positions = []

    for global_time in sample_times:
        q_sample, _, _ = evaluate_cartesian_trajectory(
            coefficients_list,
            segment_time,
            global_time
        )
        T_sample = np.asarray(T0_4_func(*q_sample), dtype=float)
        positions.append(T_sample[0:3, 3])

    return np.asarray(positions)


def desired_circle_positions():
    """Return the desired circle sampled over the complete motion."""
    positions = []

    for global_time in sample_times:
        phi_sample = 2.0 * np.pi * global_time / total_duration
        positions.append(
            p0c + R * np.array([
                0.0,
                np.cos(phi_sample),
                np.sin(phi_sample)
            ])
        )

    return np.asarray(positions)


# The improved trajectory uses 9 equally spaced knot-points. This gives
# 8 segments, each lasting 1 second over the same total duration of 8 s.
nine_phi_points = np.linspace(0.0, 2.0 * np.pi, 9)
nine_q_knots = np.asarray([
    inverse_kinematics_analytical(
        p0c + R * np.array([0.0, np.cos(phi), np.sin(phi)]),
        elbow_down=True
    )[0]
    for phi in nine_phi_points
])

nine_q_dot_knots = []
circle_angular_speed = 2.0 * np.pi / total_duration

for index, (phi, q_values) in enumerate(zip(nine_phi_points, nine_q_knots)):
    point_name = f"nine-knot {index}"

    # Reuse the existing Jacobian-based velocity function by registering
    # the additional configurations in the same stores used in Exercise 5.
    path_points[point_name] = q_values
    jacobians_ee_store[point_name] = J_ee_func(*q_values)

    if index == 0 or index == len(nine_phi_points) - 1:
        nine_q_dot_knots.append(np.zeros(4))
        continue

    cartesian_velocity = [
        0.0,
        -R * np.sin(phi) * circle_angular_speed,
        R * np.cos(phi) * circle_angular_speed,
        0.0
    ]
    nine_q_dot_knots.append(
        compute_joint_velocities(point_name, cartesian_velocity)
    )

nine_q_dot_knots = np.asarray(nine_q_dot_knots, dtype=float)
nine_q_ddot_knots = np.zeros_like(nine_q_knots)
nine_segment_duration = total_duration / (len(nine_q_knots) - 1)

nine_segment_coefficients = [
    quintic_coefficients(
        nine_q_knots[i], nine_q_dot_knots[i], nine_q_ddot_knots[i],
        nine_q_knots[i + 1], nine_q_dot_knots[i + 1],
        nine_q_ddot_knots[i + 1],
        t_i=0.0,
        t_f=nine_segment_duration
    )
    for i in range(len(nine_q_knots) - 1)
]

nine_knot_continuity_error = validate_piecewise_trajectory(
    nine_segment_coefficients,
    nine_q_knots,
    nine_q_dot_knots,
    nine_q_ddot_knots,
    nine_segment_duration
)

desired_positions = desired_circle_positions()
five_knot_positions = cartesian_positions_from_trajectory(
    segment_coefficients,
    segment_duration
)
nine_knot_positions = cartesian_positions_from_trajectory(
    nine_segment_coefficients,
    nine_segment_duration
)

five_knot_errors = np.linalg.norm(
    five_knot_positions - desired_positions,
    axis=1
)
nine_knot_errors = np.linalg.norm(
    nine_knot_positions - desired_positions,
    axis=1
)


if __name__ == "__main__":
    print("\nExercise 7: comparison of 5 and 9 knot-points")
    print("5 knot-points:")
    print(f"Maximum error: {np.max(five_knot_errors):.4f} mm")
    print(f"Mean error:    {np.mean(five_knot_errors):.4f} mm")
    print(f"RMS error:     {np.sqrt(np.mean(five_knot_errors**2)):.4f} mm")
    print("9 knot-points:")
    print(f"Maximum error: {np.max(nine_knot_errors):.4f} mm")
    print(f"Mean error:    {np.mean(nine_knot_errors):.4f} mm")
    print(f"RMS error:     {np.sqrt(np.mean(nine_knot_errors**2)):.4f} mm")


    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(
        desired_positions[:, 0],
        desired_positions[:, 1],
        desired_positions[:, 2],
        "k--",
        label="Desired circle"
    )
    ax.plot(
        five_knot_positions[:, 0],
        five_knot_positions[:, 1],
        five_knot_positions[:, 2],
        "b",
        label="5 knot-points"
    )
    ax.plot(
        nine_knot_positions[:, 0],
        nine_knot_positions[:, 1],
        nine_knot_positions[:, 2],
        "r",
        label="9 knot-points"
    )
    ax.scatter(
        p0c[0] + 0.0 * nine_phi_points,
        p0c[1] + R * np.cos(nine_phi_points),
        p0c[2] + R * np.sin(nine_phi_points),
        c="k",
        s=18,
        label="9 knot locations"
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("Exercise 7: end-effector path")
    ax.legend()

    error_ax = fig.add_subplot(122)
    error_ax.plot(sample_times, five_knot_errors, label="5 knot-points")
    error_ax.plot(sample_times, nine_knot_errors, label="9 knot-points")
    error_ax.set_xlabel("Time [s]")
    error_ax.set_ylabel("Position error [mm]")
    error_ax.set_title("Cartesian position error")
    error_ax.grid(True)
    error_ax.legend()

    plt.tight_layout()
    fig.savefig("exercise_7_path_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
