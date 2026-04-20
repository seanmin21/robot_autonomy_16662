import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET

import RobotUtil as rt

# ── Model paths
BASE_XML   = "franka_emika_panda/panda_torque_table.xml"
SCENE_XML  = "franka_emika_panda/panda_lab3.xml"

# ── Block / table geometry 
HALF_SIDE    = 0.02
TABLE_Z      = 0.02
CENTER_Z     = TABLE_Z + HALF_SIDE

# ── 3x3 grid layout 
#   row 0 = far from robot (+x direction)
#   col 0 = robot's left  (+y direction)
CELL_GAP  = 0.060
GRID_X    = 0.46
GRID_Y    = 0.00

def cell_world_pos(row, col):
    x = GRID_X + (1 - row) * CELL_GAP
    y = GRID_Y + (1 - col) * CELL_GAP
    return np.array([x, y, CENTER_Z])

GRID_POSITIONS = [[cell_world_pos(r, c) for c in range(3)] for r in range(3)]

# ── Staging areas (unplaced blocks)
STAGING_GAP    = 0.08
_STAGE_X_VALS  = [GRID_X + (i - 1.5) * STAGING_GAP for i in range(4)]
STAGE_Y_RED    =  0.24
STAGE_Y_BLUE   = -0.24
RED_STAGING    = [np.array([x, STAGE_Y_RED,  CENTER_Z]) for x in _STAGE_X_VALS]
BLUE_STAGING   = [np.array([x, STAGE_Y_BLUE, CENTER_Z]) for x in _STAGE_X_VALS]

# ── Target patterns 
#   None = empty, 'R' = red, 'B' = blue
#   row 0 = top of figure, col 0 = left
PATTERNS = [
    [['B', 'B', 'B'], [None, 'R', None], [None, 'R', None]],  # pickaxe
    [['B', 'B', None], ['B', 'R', None], [None, 'R', None]],  # axe
    [['B', 'R', 'B'], ['R', None, 'R'], ['B', 'R', 'B']],     # box
    [['B', 'R', None], ['B', None, 'R'], ['B', 'R', None]],   # bow
]
PATTERN_LABELS = ["pickaxe", "axe", "box", "bow"]

# ── Controller gains 
JOINT_KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float) * 0.22
JOINT_KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float) * 2.6

# ── Timing  
SEGMENT_TIME = 1.8   # was 2.5 — increase for smoother/safer, decrease for faster
DWELL_TIME   = 0.20  # was 0.30

# ── Gripper widths 
GRIP_OPEN   = 0.025
GRIP_CLOSED = 0.0125

ARM_JOINTS     = list(range(7))
GRIPPER_JOINT  = 7

HOME_CONFIG  = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
APPROACH_H   = 0.1   # metres above block for pre-grasp


# ── Scene construction 

def build_scene():
    """Parse the base XML and inject the table surface plus all staging blocks."""
    tree = ET.parse(BASE_XML)

    rt.add_free_block_to_model(
        tree, "TableTop",
        pos=[GRID_X, 0.0, TABLE_Z / 2],
        density=2000,
        size=[0.22, 0.36, TABLE_Z / 2],
        rgba=[0.75, 0.65, 0.50, 1],
        free=False,
    )

    for i, pos in enumerate(RED_STAGING):
        rt.add_free_block_to_model(
            tree, f"RedBlock{i}", pos=pos.tolist(),
            density=100, size=[HALF_SIDE] * 3,
            rgba=[0.85, 0.12, 0.10, 1], free=True,
        )

    for i, pos in enumerate(BLUE_STAGING):
        rt.add_free_block_to_model(
            tree, f"BlueBlock{i}", pos=pos.tolist(),
            density=100, size=[HALF_SIDE] * 3,
            rgba=[0.10, 0.28, 0.90, 1], free=True,
        )

    tree.write(SCENE_XML, encoding="utf-8", xml_declaration=True)
    print(f"[setup] Scene written to {SCENE_XML}")


# ── World-state tracker 

class GridState:
    """Symbolic model of which blocks are on the grid and which are in staging."""

    def __init__(self):
        self.cells      = [[None] * 3 for _ in range(3)]
        self.red_pool   = list(range(4))
        self.blue_pool  = list(range(4))

    def clone(self):
        s = GridState()
        s.cells     = [row[:] for row in self.cells]
        s.red_pool  = self.red_pool[:]
        s.blue_pool = self.blue_pool[:]
        return s

    @staticmethod
    def color_of(block_name):
        return 'R' if 'Red' in block_name else 'B'

    def pop_from_staging(self, color):
        pool = self.red_pool if color == 'R' else self.blue_pool
        if not pool:
            return None, None
        idx  = pool.pop(0)
        pos  = (RED_STAGING if color == 'R' else BLUE_STAGING)[idx].copy()
        name = f"{'Red' if color == 'R' else 'Blue'}Block{idx}"
        return name, pos

    def push_to_staging(self, block_name):
        idx  = int(block_name.replace("RedBlock", "").replace("BlueBlock", ""))
        pool = self.red_pool if 'Red' in block_name else self.blue_pool
        if idx not in pool:
            pool.append(idx)
            pool.sort()

    def staging_position(self, block_name):
        idx = int(block_name.replace("RedBlock", "").replace("BlueBlock", ""))
        return (RED_STAGING if 'Red' in block_name else BLUE_STAGING)[idx].copy()


# ── Finger-axis selection 

def _check_axis_clearance(r, c, cells):
    """Return (x_clear, y_clear) — True when the axis has at most one neighbour."""
    def occupied(rr, cc):
        return 0 <= rr < 3 and 0 <= cc < 3 and cells[rr][cc] is not None

    x_clear = not (occupied(r - 1, c) and occupied(r + 1, c))
    y_clear = not (occupied(r, c - 1) and occupied(r, c + 1))
    return x_clear, y_clear


def _clearance_score(r, c, cells):
    x, y = _check_axis_clearance(r, c, cells)
    return int(x) + int(y)


def select_finger_axis(r, c, cells):
    """Pick a gripper finger axis that won't clip neighbouring blocks."""
    x_ok, y_ok = _check_axis_clearance(r, c, cells)
    if x_ok:
        return np.array([1.0, 0.0, 0.0])
    if y_ok:
        return np.array([0.0, 1.0, 0.0])

    # Both axes blocked — fall back to whichever side has fewer neighbours
    def occupied(rr, cc):
        return 0 <= rr < 3 and 0 <= cc < 3 and cells[rr][cc] is not None

    x_count = int(occupied(r - 1, c)) + int(occupied(r + 1, c))
    y_count = int(occupied(r, c - 1)) + int(occupied(r, c + 1))
    print(f"  [warn] cell ({r},{c}) fully surrounded — using least-blocked axis")
    return np.array([1.0, 0.0, 0.0]) if x_count <= y_count else np.array([0.0, 1.0, 0.0])


_AXIS_CANDIDATES = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]


def _axis_priority(rc, cells):
    """Return axes in order of preference for a given cell (or default for staging)."""
    if rc is None:
        return [np.array([0.0, -1.0, 0.0])]
    preferred = select_finger_axis(*rc, cells)
    fallbacks = [a for a in _AXIS_CANDIDATES if not np.allclose(a, preferred)]
    return [preferred] + fallbacks


# ── Low-level execution 

def execute_waypoints(model, data, viewer_handle, waypoints):
    """Step the simulation through a joint-space waypoint sequence."""
    dt          = model.opt.timestep
    move_steps  = max(1, int(SEGMENT_TIME / dt))
    hold_steps  = max(1, int(DWELL_TIME   / dt))

    for i in range(len(waypoints) - 1):
        q0   = np.array(waypoints[i][:7])
        q1   = np.array(waypoints[i + 1][:7])
        grip = waypoints[i + 1][-1]
        t    = 0.0

        for _ in range(move_steps + hold_steps):
            q_des, qd_des = rt.interp_min_jerk(q0, q1, t, SEGMENT_TIME)
            q   = data.qpos[ARM_JOINTS].copy()
            qd  = data.qvel[ARM_JOINTS].copy()
            tau = JOINT_KP * (q_des - q) + JOINT_KD * (qd_des - qd)

            data.ctrl[ARM_JOINTS]    = tau + data.qfrc_bias[:7]
            data.ctrl[GRIPPER_JOINT] = grip
            mj.mj_step(model, data)
            viewer_handle.sync()
            t += dt


# ── Pick-and-place primitive 

def move_block(model, data, viewer_handle,
               pick_pos, place_pos,
               cells_at_pick, cells_at_place,
               pick_cell=None, place_cell=None):
    """
    Attempt a pick-and-place operation, trying both finger axes for pick and
    place independently.  Returns True on success, False if IK fails.
    """
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()

    def restore_state():
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel

    # Solve IK for the pick phase
    pick_ik = None
    for axis in _axis_priority(pick_cell, cells_at_pick):
        at_block, rot, above_block = rt.top_down_grasp_pose(pick_pos, APPROACH_H, axis)
        restore_state()
        q_above, ok1 = rt.compute_ik(model, data, above_block, rot, ARM_JOINTS, q_init=HOME_CONFIG)
        restore_state()
        q_at,    ok2 = rt.compute_ik(model, data, at_block,    rot, ARM_JOINTS, q_init=q_above)
        restore_state()
        q_lift,  ok3 = rt.compute_ik(model, data, above_block, rot, ARM_JOINTS, q_init=q_at)
        if ok1 and ok2 and ok3:
            pick_ik = (q_above, q_at, q_lift)
            break

    # Solve IK for the place phase
    place_ik = None
    for axis in _axis_priority(place_cell, cells_at_place):
        at_place, rot_p, above_place = rt.top_down_grasp_pose(place_pos, APPROACH_H, axis)
        restore_state()
        q_preplace, ok4 = rt.compute_ik(model, data, above_place, rot_p, ARM_JOINTS, q_init=HOME_CONFIG)
        restore_state()
        q_set_down, ok5 = rt.compute_ik(model, data, at_place,    rot_p, ARM_JOINTS, q_init=q_preplace)
        if ok4 and ok5:
            place_ik = (q_preplace, q_set_down)
            break

    restore_state()
    mj.mj_forward(model, data)

    if pick_ik is None or place_ik is None:
        failures = (["pick"]  if pick_ik  is None else []) + \
                   (["place"] if place_ik is None else [])
        print(f"  [IK failed] could not solve: {', '.join(failures)} — skipping this move")
        return False

    q_above, q_at, q_lift       = pick_ik
    q_preplace, q_set_down      = place_ik

    sequence = [
        [*HOME_CONFIG,   GRIP_OPEN  ],  # 0: home position
        [*q_above,       GRIP_OPEN  ],  # 1: hover above block
        [*q_at,          GRIP_OPEN  ],  # 2: descend to block
        [*q_at,          GRIP_CLOSED],  # 3: grasp
        [*q_lift,        GRIP_CLOSED],  # 4: lift clear
        [*q_preplace,    GRIP_CLOSED],  # 5: move to target hover
        [*q_set_down,    GRIP_CLOSED],  # 6: descend to target
        [*q_set_down,    GRIP_OPEN  ],  # 7: release
        [*q_preplace,    GRIP_OPEN  ],  # 8: retreat upward
        [*HOME_CONFIG,   GRIP_OPEN  ],  # 9: return home
    ]
    execute_waypoints(model, data, viewer_handle, sequence)
    return True


# ── Task planner 

def compute_ops(current_state: GridState, target):
    """
    Produce an ordered list of remove/place operations that transitions
    the current grid state to the target pattern.
    """
    ops = []
    sim = current_state.clone()

    # Identify cells whose current block has the wrong color
    to_clear = [
        (r, c, sim.cells[r][c])
        for r in range(3) for c in range(3)
        if sim.cells[r][c] is not None
        and target[r][c] != GridState.color_of(sim.cells[r][c])
    ]

    # Remove wrong-color blocks, most-accessible first
    while to_clear:
        to_clear.sort(
            key=lambda x: (
                _clearance_score(x[0], x[1], sim.cells),
                -sum(
                    1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= x[0] + dr < 3
                    and 0 <= x[1] + dc < 3
                    and sim.cells[x[0] + dr][x[1] + dc] is not None
                ),
            ),
            reverse=True,
        )
        r, c, name = to_clear.pop(0)
        ops.append({'type': 'remove', 'r': r, 'c': c, 'name': name})
        sim.cells[r][c] = None
        sim.push_to_staging(name)

    # Identify empty cells that need a block placed
    to_fill = [
        (r, c, target[r][c])
        for r in range(3) for c in range(3)
        if target[r][c] is not None
        and (sim.cells[r][c] is None
             or GridState.color_of(sim.cells[r][c]) != target[r][c])
    ]

    # Place blocks, most-accessible and most-central first
    while to_fill:
        to_fill.sort(
            key=lambda x: (
                _clearance_score(x[0], x[1], sim.cells),
                abs(x[0] - 1) + abs(x[1] - 1),
            ),
            reverse=True,
        )
        r, c, color = to_fill.pop(0)
        name, _ = sim.pop_from_staging(color)
        if name is None:
            print(f"  [planner] no {color} block available for cell ({r},{c}) — skipping")
            continue
        ops.append({'type': 'place', 'r': r, 'c': c, 'color': color, 'name': name})
        sim.cells[r][c] = name

    return ops


# ── Main

if __name__ == "__main__":
    np.random.seed(42)

    build_scene()
    model = mj.MjModel.from_xml_path(SCENE_XML)
    data  = mj.MjData(model)

    data.qpos[ARM_JOINTS] = HOME_CONFIG
    mj.mj_forward(model, data)

    v = viewer.launch_passive(model, data)
    v.cam.distance  = 2.5
    v.cam.azimuth  += 90
    v.cam.elevation = -25

    world = GridState()

    for idx, (label, target) in enumerate(zip(PATTERN_LABELS, PATTERNS)):
        print(f"\n{'─' * 55}")
        print(f"  Pattern {idx + 1} of {len(PATTERNS)}: {label.upper()}")
        print(f"{'─' * 55}")

        ops = compute_ops(world, target)
        if not ops:
            print("  Grid already matches target — nothing to do.")
            continue

        success = True

        for op in ops:
            r, c = op['r'], op['c']

            if op['type'] == 'remove':
                name      = op['name']
                pick_pos  = GRID_POSITIONS[r][c].copy()
                place_pos = world.staging_position(name)
                snap      = [row[:] for row in world.cells]

                print(f"  removing {name} from ({r},{c}) → staging")
                ok = move_block(
                    model, data, v,
                    pick_pos, place_pos,
                    cells_at_pick=snap, cells_at_place=None,
                    pick_cell=(r, c), place_cell=None,
                )
                if ok:
                    world.cells[r][c] = None
                    world.push_to_staging(name)
                else:
                    print(f"  [error] failed to remove {name} — aborting pattern")
                    success = False
                    break

            elif op['type'] == 'place':
                color          = op['color']
                name, pick_pos = world.pop_from_staging(color)
                if name is None:
                    print(f"  [error] no {color} block in staging — skipping cell ({r},{c})")
                    continue

                place_pos  = GRID_POSITIONS[r][c].copy()
                snap       = [row[:] for row in world.cells]
                snap[r][c] = None

                print(f"  placing {name} ({color}) → cell ({r},{c})")
                ok = move_block(
                    model, data, v,
                    pick_pos, place_pos,
                    cells_at_pick=None, cells_at_place=snap,
                    pick_cell=None, place_cell=(r, c),
                )
                if ok:
                    world.cells[r][c] = name
                else:
                    world.push_to_staging(name)
                    print(f"  [error] failed to place {name} at ({r},{c}) — aborting pattern")
                    success = False
                    break

        status = "done" if success else "incomplete"
        print(f"\n  Pattern '{label}': {status}")

        # Brief settle pause between patterns
        for _ in range(int(1.0 / model.opt.timestep)):
            mj.mj_step(model, data)
            v.sync()

    print("\nAll patterns attempted. Close the viewer to exit.")
    while v.is_running():
        mj.mj_step(model, data)
        v.sync()

    v.close()