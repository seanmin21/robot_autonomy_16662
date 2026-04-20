import numpy as np
import math
import xml.etree.ElementTree as ET


def rpyxyz2H(rpy: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from RPY angles and XYZ position."""
    Ht = [[1, 0, 0, xyz[0]],
          [0, 1, 0, xyz[1]],
          [0, 0, 1, xyz[2]],
          [0, 0, 0, 1]]

    Hx = [[1, 0, 0, 0],
          [0, math.cos(rpy[0]), -math.sin(rpy[0]), 0],
          [0, math.sin(rpy[0]),  math.cos(rpy[0]), 0],
          [0, 0, 0, 1]]

    Hy = [[ math.cos(rpy[1]), 0, math.sin(rpy[1]), 0],
          [0, 1, 0, 0],
          [-math.sin(rpy[1]), 0, math.cos(rpy[1]), 0],
          [0, 0, 0, 1]]

    Hz = [[math.cos(rpy[2]), -math.sin(rpy[2]), 0, 0],
          [math.sin(rpy[2]),  math.cos(rpy[2]), 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]

    return np.matmul(np.matmul(np.matmul(Ht, Hz), Hy), Hx)


def R2axisang(R: np.ndarray) -> (np.ndarray, float):
    """Decompose a rotation matrix into axis-angle form."""
    ang = math.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
    Z = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if Z == 0:
        return [1, 0, 0], 0.
    x = (R[2, 1] - R[1, 2]) / Z
    y = (R[0, 2] - R[2, 0]) / Z
    z = (R[1, 0] - R[0, 1]) / Z
    return np.array([x, y, z]), ang


def MatrixExp(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula returned as a 4x4 homogeneous matrix."""
    S = so3(axis)
    R = np.eye(3) + np.sin(theta) * S + (1 - np.cos(theta)) * np.matmul(S, S)
    last = np.zeros((1, 4))
    last[0, 3] = 1
    return np.vstack((np.hstack((R, np.zeros((3, 1)))), last))


def so3(axis: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for a 3-vector."""
    return np.asarray([
        [0, -axis[2],  axis[1]],
        [axis[2], 0,  -axis[0]],
        [-axis[1], axis[0], 0]
    ])


def RPY_to_rot(rpy):
    """Roll-pitch-yaw to 3x3 rotation matrix (ZYX convention)."""
    roll, pitch, yaw = rpy

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def FindNearest(prevPoints, newPoint):
    D = np.array([np.linalg.norm(np.array(p) - np.array(newPoint)) for p in prevPoints])
    return D.argmin()


def BlockDesc2Points(H, Dim):
	center = H[0:3, 3]
	axes = [H[0:3, 0], H[0:3, 1], H[0:3, 2]]

	corners = [
		center,
		center + (axes[0]*Dim[0]/2.) + (axes[1]*Dim[1]/2.) + (axes[2]*Dim[2]/2.),
		center + (axes[0]*Dim[0]/2.) + (axes[1]*Dim[1]/2.) - (axes[2]*Dim[2]/2.),
		center + (axes[0]*Dim[0]/2.) - (axes[1]*Dim[1]/2.) + (axes[2]*Dim[2]/2.),
		center + (axes[0]*Dim[0]/2.) - (axes[1]*Dim[1]/2.) - (axes[2]*Dim[2]/2.),
		center - (axes[0]*Dim[0]/2.) + (axes[1]*Dim[1]/2.) + (axes[2]*Dim[2]/2.),
		center - (axes[0]*Dim[0]/2.) + (axes[1]*Dim[1]/2.) - (axes[2]*Dim[2]/2.),
		center - (axes[0]*Dim[0]/2.) - (axes[1]*Dim[1]/2.) + (axes[2]*Dim[2]/2.),
		center - (axes[0]*Dim[0]/2.) - (axes[1]*Dim[1]/2.) - (axes[2]*Dim[2]/2.),
	]

	return corners, axes


def CheckPointOverlap(pointsA, pointsB, axis):
	pA = np.matmul(axis, np.transpose(pointsA))
	pB = np.matmul(axis, np.transpose(pointsB))

	maxA, minA = np.max(pA), np.min(pA)
	maxB, minB = np.max(pB), np.min(pB)

	if maxA <= maxB and maxA >= minB:
		return True
	if minA <= maxB and minA >= minB:
		return True
	if maxB <= maxA and maxB >= minA:
		return True
	if minB <= maxA and minB >= minA:
		return True

	return False


def CheckBoxBoxCollision(pointsA, axesA, pointsB, axesB):
	# quick sphere check first
	if np.linalg.norm(pointsA[0] - pointsB[0]) > (np.linalg.norm(pointsA[0] - pointsA[1]) + np.linalg.norm(pointsB[0] - pointsB[1])):
		return False

	for i in range(3):
		if not CheckPointOverlap(pointsA, pointsB, axesA[i]):
			return False

	for j in range(3):
		if not CheckPointOverlap(pointsA, pointsB, axesB[j]):
			return False

	for i in range(3):
		for j in range(3):
			if not CheckPointOverlap(pointsA, pointsB, np.cross(axesA[i], axesB[j])):
				return False

	return True


def axis_angle_between(v1, v2):
    """Axis and angle needed to rotate v1 onto v2."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    dot   = np.clip(np.dot(v1, v2), -1.0, 1.0)
    cross = np.cross(v1, v2)
    angle = np.arccos(dot)

    if np.isclose(angle, 0.0):
        return np.array([1, 0, 0]), 0.0
    elif np.isclose(angle, np.pi):
        orth = np.array([1, 0, 0])
        if np.allclose(v1, orth) or np.allclose(v1, -orth):
            orth = np.array([0, 1, 0])
        axis = np.cross(v1, orth)
        axis /= np.linalg.norm(axis)
        return axis, np.pi
    else:
        return cross / np.linalg.norm(cross), angle


def interp_min_jerk(q_start, q_goal, t, T):
    tau = np.clip(t / max(T, 1e-6), 0.0, 1.0)
    s   = 10*tau**3 - 15*tau**4 + 6*tau**5
    ds  = (30*tau**2 - 60*tau**3 + 30*tau**4) / max(T, 1e-6)
    return q_start + s * (q_goal - q_start), ds * (q_goal - q_start)


PANDA_EE_OFFSET = np.array([0.0, 0.0, 0.08])


def _make_rot(z_axis, y_axis):
    """Build a right-handed rotation matrix from desired z and y directions."""
    z = np.asarray(z_axis, dtype=float)
    z /= np.linalg.norm(z)
    y = np.asarray(y_axis, dtype=float)
    y -= np.dot(y, z) * z
    ny = np.linalg.norm(y)
    if ny < 1e-8:
        raise ValueError("y_axis must not be parallel to z_axis")
    y /= ny
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def top_down_grasp_pose(block_pos, pregrasp_height=0.1, finger_axis=None):
    """Top-down grasp: approach from above."""
    if finger_axis is None:
        finger_axis = np.array([1.0, 0.0, 0.0])
    rot = _make_rot(np.array([0.0, 0.0, -1.0]), finger_axis)
    grasp_pos    = np.array(block_pos, dtype=float)
    pregrasp_pos = grasp_pos + np.array([0.0, 0.0, pregrasp_height])
    return grasp_pos, rot, pregrasp_pos


def side_grasp_pose(block_pos, approach_dir, pregrasp_dist=0.1, finger_axis=None):
    """Side grasp: move horizontally into block from approach_dir."""
    if finger_axis is None:
        finger_axis = np.array([0.0, 0.0, 1.0])
    d = np.asarray(approach_dir, dtype=float)
    d /= np.linalg.norm(d)
    rot          = _make_rot(d, finger_axis)
    grasp_pos    = np.array(block_pos, dtype=float)
    pregrasp_pos = grasp_pos - d * pregrasp_dist
    return grasp_pos, rot, pregrasp_pos


def place_pose_above(target_pos, approach_height=0.1):
    """Top-down place: returns place position and approach waypoint above it."""
    place_pos    = np.array(target_pos, dtype=float)
    preplace_pos = place_pos + np.array([0.0, 0.0, approach_height])
    return place_pos, preplace_pos


def compute_ik(model, data, target_pos, target_rot, arm_idx, q_init=None,
               body_name="hand",
               max_iter=200, pos_tol=1e-2, rot_tol=1e-2,
               step_size=0.5, damping=1e-2):
    """Damped-least-squares IK. Returns (q, converged)."""
    import mujoco as mj
    ee_off = np.asarray(PANDA_EE_OFFSET, dtype=float)

    if q_init is not None:
        data.qpos[arm_idx] = np.asarray(q_init, dtype=float)

    nv      = model.nv
    arm_idx = list(arm_idx)
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    lo = model.jnt_range[arm_idx, 0]
    hi = model.jnt_range[arm_idx, 1]

    dp = np.ones(3)
    dR = np.ones(3)

    for _ in range(max_iter):
        mj.mj_forward(model, data)
        xpos = np.array(data.xpos[body_id])
        xmat = np.array(data.xmat[body_id]).reshape(3, 3)

        ee_pos = xpos + xmat @ ee_off
        dp     = target_pos - ee_pos
        axis, ang = R2axisang(target_rot @ xmat.T)
        dR = axis * ang

        if np.linalg.norm(dp) < pos_tol and np.linalg.norm(dR) < rot_tol:
            return data.qpos[arm_idx].copy(), True

        jacp_flat = np.zeros((3, nv))
        jacr_flat = np.zeros((3, nv))
        mj.mj_jacBody(model, data, jacp_flat, jacr_flat, body_id)
        Jp    = jacp_flat[:, arm_idx]
        Jr    = jacr_flat[:, arm_idx]
        Jp_ee = Jp - so3(xmat @ ee_off) @ Jr

        J   = np.vstack([Jp_ee, Jr])
        err = np.concatenate([dp, dR])
        A   = J @ J.T + damping * np.eye(6)
        dq  = step_size * (J.T @ np.linalg.solve(A, err))

        data.qpos[arm_idx] = np.clip(data.qpos[arm_idx] + dq, lo, hi)

    print(f"[IK] did not converge — pos: {np.linalg.norm(dp):.4f} m, rot: {np.linalg.norm(dR):.4f} rad")
    return data.qpos[arm_idx].copy(), False


def add_free_block_to_model(tree, name, pos, density, size, rgba, free):
    worldbody = tree.getroot().find("worldbody")
    body = ET.SubElement(worldbody, "body", {"name": {name}, "pos": f"{pos[0]} {pos[1]} {pos[2]}"})
    ET.SubElement(body, "geom", {"type": "box", "density": f"{density}", "size": f"{size[0]} {size[1]} {size[2]}", "rgba": f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"})
    if free is True:
        ET.SubElement(body, "freejoint")
