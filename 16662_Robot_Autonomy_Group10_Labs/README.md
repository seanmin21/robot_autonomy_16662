# Robot Autonomy - Lab 2

**16-662 Robot Autonomy** | Prof. Oliver Kroemer | Due: March 30

---

## Overview

This lab implements a full multi-stage manipulation pipeline for a 7-DoF Franka Emika Panda arm in MuJoCo simulation. The robot performs a pick-and-place task involving a top-down grasp, a side regrasp, and placement on a shelf — planned with RRT and executed with a joint-space PD controller.

## Task Sequence

1. **Top-down grasp** — pick up a block from the table
2. **Place on left shelf** — place the block near the shelf edge from above
3. **Side regrasp** — regrasp the block horizontally from the side
4. **Place on right shelf** — move and place the block on one of the three right shelves
5. **Return home** — release the block and return the arm to its home configuration


## Running

```bash
cd 16_662_LAB2
python SimpleScript.py
```


## File Structure

```
16_662_LAB2/
├── SimpleScript.py              # Main entry point — IK, RRT planning, simulation execution
├── Franka.py                    # FrankaArm class: FK, IK, collision detection, RRT planner
├── RobotUtil.py                 # Utilities: RPY↔rotation, grasp pose helpers, min-jerk interpolation
├── visualize_waypoints.py       # Visualize task-space waypoints in MuJoCo viewer
├── test/ik_test.py              # IK unit tests
└── franka_emika_panda/
    ├── panda_torque_table_shelves.xml   # Scene with shelves and block (used for execution)
    └── panda_torque_table.xml           # Base table scene
```

## Implementation Details

### Inverse Kinematics (`Franka.py`)
Weighted damped least-squares numerical IK (`IterInvKin`):
- Iteratively minimizes position + orientation error via the 6×7 Jacobian
- Task-space weights: 10⁶ on position, 10³ on orientation
- Joint-space weights penalize wrist joints to prefer natural configurations

### Motion Planning (`Franka.py`)
RRT with the following features:
- **Adaptive goal bias**: linearly increases from 20% to 50% over iterations
- **Extend-to-sample**: grows the tree toward sampled configs in steps of 0.05 rad
- **Path shortening**: 80 random shortcut attempts after RRT succeeds
- **Collision checking**: MuJoCo contact detection (robot vs. environment only)
- **Dense edge checking**: 5 intermediate samples per edge via `DetectCollisionEdge`
- **Interpolation**: linear joint-space interpolation at 0.05 rad resolution

### Execution (`SimpleScript.py`)
- Task-space waypoints (xyz + RPY + gripper) are solved to joint space via IK
- `plan_waypoints` runs RRT between consecutive joint-space waypoints
- Gripper state is carried per segment through the full path
- **Controller**: joint-space PD with gravity compensation
  ```
  τ = Kp·(q_des − q) + Kd·(qd_des − qd) + qfrc_bias
  ```
- Trajectory: minimum-jerk interpolation between RRT path steps

### Grasp Pose Helpers (`RobotUtil.py`)
- `top_down_grasp_pose` — EE z-axis pointing down onto block
- `side_grasp_pose` — EE approaching horizontally from a specified direction
- `place_pose_above` — top-down placement with configurable approach height
