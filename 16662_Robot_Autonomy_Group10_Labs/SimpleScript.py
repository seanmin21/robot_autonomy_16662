import os
os.environ['GLFW_PLATFORM'] = 'x11'

import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET

import RobotUtil as rt
from Franka import FrankaArm

ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
MODEL_XML      = "franka_emika_panda/panda_torque_table_shelves.xml"

WAYPOINTS = np.array([
    [0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04],
    [0.0,  0.65, 0.0, -2.0,  0.0,  2.65, 0.8, 0.04],
    [0.0,  0.65, 0.0, -2.0,  0.0,  2.65, 0.8, 0.0125],
    [1.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.0125],
    [1.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04],
    [0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04],
], dtype=float)

OPEN  = 0.04
CLOSE = 0.015
FINGER_QPOS_IDX = [7, 8]
ENDOFTABLE = 0.55 + 0.135 + 0.05

LEFT_SHELF_LEVEL_1_SURFACE_Z  = 0.1375 + 0.005
RIGHT_SHELF_LEVEL_1_SURFACE_Z = 0.1375 + 0.005
RIGHT_SHELF_LEVEL_2_SURFACE_Z = 0.1375 + 0.005 + 0.2
RIGHT_SHELF_LEVEL_3_SURFACE_Z = 0.1375 + 0.005 + 0.4

LEFT_SHELF_PLACE_XY = np.array(
    [ENDOFTABLE - 0.275 - 0.135 + 0.0225,  0.504 - 0.09 - 0.135 / 2.0],
    dtype=float,
)
RIGHT_SHELF_PLACE_XY = np.array(
    [ENDOFTABLE - 0.275 - 0.135 + 0.0225, -0.504 + 0.09 + 0.135 / 2.0],
    dtype=float,
)
LEFT_SHELF_FRONT_GRASP_Y = 0.32


def build_task_space_waypoints(
    block_pos: np.ndarray,
    block_size: np.ndarray,
    right_shelf_surface_z: float = RIGHT_SHELF_LEVEL_1_SURFACE_Z,
) -> np.ndarray:
    home = np.array([0.386, 0.0, 0.8, 3.14, 0.0, 0.0, OPEN], dtype=float)

    top_contact = block_pos.copy()
    top_contact[0] += block_size[0] / 2.
    top_contact[2] -= block_size[2] / 2. + 0.01
    top_pregrasp = top_contact.copy()
    top_pregrasp[2] += 0.12
    top_lift = top_pregrasp.copy()

    left_place_center = np.array(
        [LEFT_SHELF_PLACE_XY[0], LEFT_SHELF_FRONT_GRASP_Y, LEFT_SHELF_LEVEL_1_SURFACE_Z + block_size[2]],
        dtype=float,
    )
    left_place_approach = left_place_center + np.array([0.0, 0.0, 0.0625])

    left_side_contact = left_place_center.copy()
    left_side_contact[2] += 0.0075
    left_side_pregrasp = left_side_contact + np.array([0.0, 0.0, 0.07])
    left_side_lift = left_side_pregrasp.copy()
    left_flip_clear = np.array(
        [left_place_center[0], left_place_center[1] - 0.08, left_side_pregrasp[2] + 0.03],
        dtype=float,
    )

    right_place_center = np.array(
        [*RIGHT_SHELF_PLACE_XY, right_shelf_surface_z + block_size[2]],
        dtype=float,
    )
    clearance = 0.04 if right_shelf_surface_z >= RIGHT_SHELF_LEVEL_3_SURFACE_Z else 0.02
    right_place_release  = right_place_center  + np.array([0.0, 0.0, clearance])
    right_place_retreat  = right_place_release + np.array([0.0, 0.05, 0.0])
    right_place_approach = right_place_retreat + np.array([0.0, 0.0, 0.02])
    transfer_mid = np.array([
        0.42,
        0.5 * (left_side_lift[1] + right_place_approach[1]),
        max(left_side_lift[2], right_place_approach[2]) + 0.12,
    ], dtype=float)

    return np.array([
        # pick from table
        home,
        [*top_pregrasp, 3.14, 0.0, 0.0, OPEN],
        [*top_contact,  3.14, 0.0, 0.0, OPEN],
        [*top_contact,  3.14, 0.0, 0.0, CLOSE],
        [*top_contact,  3.14, 0.0, 0.0, CLOSE],
        [*top_lift,     3.14, 0.0, 0.0, CLOSE],

        # place on left shelf
        [*left_place_approach, 3.14, 0.0, 1.57, CLOSE],
        [*left_place_center,   3.14, 0.0, 1.57, CLOSE],
        [*left_place_center,   3.14, 0.0, 1.57, OPEN],
        [*left_place_center,   3.14, 0.0, 1.57, OPEN],
        [*left_place_approach, 3.14, 0.0, 1.57, OPEN],
        [*left_flip_clear,     3.14, 0.0, 1.57, OPEN],
        [*left_flip_clear,    -1.57, 1.57, 0.0, OPEN],

        # side-grasp from left shelf
        [*left_side_pregrasp, -1.57, 1.57, 0.0, OPEN],
        [*left_side_contact,  -1.57, 1.57, 0.0, OPEN],
        [*left_side_contact,  -1.57, 1.57, 0.0, CLOSE],
        [*left_side_contact,  -1.57, 1.57, 0.0, CLOSE],
        [*left_side_lift,     -1.57, 1.57, 0.0, CLOSE],

        # transfer and place on right shelf
        [*transfer_mid,         1.57, 1.57, 0.0, CLOSE],
        [*right_place_approach, 1.57, 1.57, 0.0, CLOSE],
        [*right_place_release,  1.57, 1.57, 0.0, CLOSE],
        [*right_place_release,  1.57, 1.57, 0.0, OPEN],
        [*right_place_retreat,  1.57, 1.57, 0.0, OPEN],
        [*right_place_approach, 1.57, 1.57, 0.0, OPEN],

        # return home
        home,
    ], dtype=float)


SEGMENT_DURATION = 2.0
HOLD_DURATION    = 1.0

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float) * 2


def run_simulation(right_shelf_surface_z: float = RIGHT_SHELF_LEVEL_1_SURFACE_Z) -> None:
    np.random.seed(13)

    model = mj.MjModel.from_xml_path(MODEL_XML)
    data  = mj.MjData(model)
    arm   = FrankaArm(model, data)
    arm_idx     = [0, 1, 2, 3, 4, 5, 6]
    gripper_idx = 7

    q_home = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
    data.qpos[arm_idx]         = q_home
    data.qpos[FINGER_QPOS_IDX] = OPEN
    data.ctrl[gripper_idx]     = OPEN
    arm.ForwardKin()

    block_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "Block")
    block_geom_id = model.body_geomadr[block_body_id]
    block_pos  = data.xpos[block_body_id].copy()
    block_size = model.geom_size[block_geom_id].copy()
    print("Block pos:", np.round(block_pos, 4))
    print("Block half-size:", np.round(block_size, 4))

    ts_waypoints = build_task_space_waypoints(block_pos, block_size, right_shelf_surface_z=right_shelf_surface_z)

    # convert task-space waypoints to joint space
    js_waypoints = []
    for idx, wp in enumerate(ts_waypoints):
        pos, rpy, gripper_width = wp[:3], wp[3:6], wp[6]
        rot = rt.RPY_to_rot(rpy)

        data.qpos[arm_idx] = q_home
        arm.ForwardKin()
        arm.IterInvKin(pos, rot)

        q_sol = data.qpos[arm_idx].copy()
        js_waypoints.append([*q_sol, gripper_width])
        print(f"Waypoint {idx} -> joints: {np.round(q_sol, 3)}")

    js_waypoints = np.array(js_waypoints)

    data.qpos[arm_idx]         = js_waypoints[0][:7]
    data.qpos[FINGER_QPOS_IDX] = OPEN
    data.qvel[arm_idx]         = 0.0
    data.ctrl[gripper_idx]     = OPEN
    mj.mj_forward(model, data)

    dt = model.opt.timestep

    print("\n=== Planning RRT path ===")
    rrt_path = arm.plan_waypoints(js_waypoints.tolist())

    if rrt_path is None:
        print("Planning failed.")
    else:
        print(f"Path has {len(rrt_path)} steps")

        v = viewer.launch_passive(model, data)
        v.cam.distance = 3.0
        v.cam.azimuth += 90

        seg_steps = int(SEGMENT_DURATION / dt)

        try:
            for i in range(len(rrt_path) - 1):
                if not v.is_running():
                    break

                q0 = np.array(rrt_path[i][:7])
                q1 = np.array(rrt_path[i + 1][:7])
                gripper_cmd = rrt_path[i + 1][-1] if len(rrt_path[i + 1]) > 7 else OPEN

                t = 0.0
                for _ in range(seg_steps):
                    if not v.is_running():
                        break

                    q_des, qd_des = rt.interp_min_jerk(q0, q1, t, SEGMENT_DURATION)
                    q  = data.qpos[arm_idx].copy()
                    qd = data.qvel[arm_idx].copy()
                    tau = KP * (q_des - q) + KD * (qd_des - qd)

                    data.ctrl[gripper_idx] = gripper_cmd
                    data.ctrl[arm_idx]     = tau + data.qfrc_bias[:7]

                    mj.mj_step(model, data)
                    v.sync()
                    t += dt
        finally:
            v.close()


if __name__ == "__main__":
    run_simulation()
