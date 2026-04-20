import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET

import RobotUtil as rt

ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
MODEL_XML      = "franka_emika_panda/panda_lab3.xml"

BLOCK_HALF   = 0.02
TABLE_TOP_Z  = 0.02
BLOCK_Z      = TABLE_TOP_Z + BLOCK_HALF

# 3x3 grid: row 0 = far from robot (+x), col 0 = robot's left (+y)
GRID_SPACING = 0.060  # 4 cm block + 1cm gap each side
GRID_CX      = 0.46   # center of grid, matches the table center in XMl file
GRID_CY      = 0.00   # grid is centered on robot midline

# mapss row,col to world coord
def grid_pos(row, col):
    x = GRID_CX + (1 - row) * GRID_SPACING
    y = GRID_CY + (1 - col) * GRID_SPACING
    return np.array([x, y, BLOCK_Z])

GRID = [[grid_pos(r, c) for c in range(3)] for r in range(3)]

STAGE_SPACING = 0.08  # wider gap between staging blocks (4 cm block + 2 cm gap each side)
_STAGE_X     = [GRID_CX + (i - 1.5) * STAGE_SPACING for i in range(4)]
STAGE_Y_RED  =  0.24
STAGE_Y_BLUE = -0.24
STAGING_RED  = [np.array([x, STAGE_Y_RED,  BLOCK_Z]) for x in _STAGE_X]
STAGING_BLUE = [np.array([x, STAGE_Y_BLUE, BLOCK_Z]) for x in _STAGE_X]

# None = empty, 'R' = red, 'B' = blue; row 0 = top of figure, col 0 = left
PATTERNS = [
    [['B', 'B', 'B'], [None, 'R', None], [None, 'R', None]],  # Pickaxe
    [['B', 'B', None], ['B', 'R', None], [None, 'R', None]],  # Axe
    [['B', 'R', 'B'], ['R', None, 'R'], ['B', 'R', 'B']],     # Box
    [['B', 'R', None], ['B', None, 'R'], ['B', 'R', None]],   # Bow
]
PATTERN_NAMES = ["pickaxe", "axe", "box", "bow"]

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float) * 0.2
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float) * 2.5
MOVE_DUR  = 2.5
HOLD_DUR  = 0.30

OPEN      = 0.025
CLOSED    = 0.0125

ARM_IDX     = list(range(7))
GRIPPER_IDX = 7

HOME_Q     = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
PREGRASP_H = 0.1


def build_scene():
    # this adds table and staged blocks to the scene
    tree = ET.parse(ROOT_MODEL_XML)
    rt.add_free_block_to_model(tree, "TableTop",
                               pos=[GRID_CX, 0.0, TABLE_TOP_Z / 2],
                               density=2000,
                               size=[0.22, 0.36, TABLE_TOP_Z / 2],
                               rgba=[0.75, 0.65, 0.50, 1],
                               free=False)
    for i, pos in enumerate(STAGING_RED):
        rt.add_free_block_to_model(tree, f"RedBlock{i}", pos=pos.tolist(),
                                   density=100, size=[BLOCK_HALF]*3,
                                   rgba=[0.85, 0.12, 0.10, 1], free=True)
    for i, pos in enumerate(STAGING_BLUE):
        rt.add_free_block_to_model(tree, f"BlueBlock{i}", pos=pos.tolist(),
                                   density=100, size=[BLOCK_HALF]*3,
                                   rgba=[0.10, 0.28, 0.90, 1], free=True)
    tree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)
    print(f"[Scene] Wrote {MODEL_XML}")


class BlockState:
    # a class for defining the abstract world state
    def __init__(self):
        self.grid       = [[None] * 3 for _ in range(3)]
        self.red_avail  = list(range(4))
        self.blue_avail = list(range(4))

    def copy(self):
        s = BlockState()
        s.grid       = [row[:] for row in self.grid]
        s.red_avail  = self.red_avail[:]
        s.blue_avail = self.blue_avail[:]
        return s

    @staticmethod
    def block_color(name):
        return 'R' if 'Red' in name else 'B'

    def take_from_staging(self, color):
        if color == 'R' and self.red_avail:
            idx = self.red_avail.pop(0)
            return f"RedBlock{idx}", STAGING_RED[idx].copy()
        if color == 'B' and self.blue_avail:
            idx = self.blue_avail.pop(0)
            return f"BlueBlock{idx}", STAGING_BLUE[idx].copy()
        return None, None

    def return_to_staging(self, name):
        idx = int(name.replace("RedBlock", "").replace("BlueBlock", ""))
        avail = self.red_avail if 'Red' in name else self.blue_avail
        if idx not in avail:
            avail.append(idx)
            avail.sort()

    def staging_pos(self, name):
        idx = int(name.replace("RedBlock", "").replace("BlueBlock", ""))
        return (STAGING_RED if 'Red' in name else STAGING_BLUE)[idx].copy()

def _axes_clear(r, c, grid):
    # Axis is only blocked when BOTH in-grid neighbours on that side are occupied.
    def occ(rr, cc):
        return 0 <= rr < 3 and 0 <= cc < 3 and grid[rr][cc] is not None
    x_clear = not (occ(r - 1, c) and occ(r + 1, c))
    y_clear = not (occ(r, c - 1) and occ(r, c + 1))
    return x_clear, y_clear


def choose_finger_axis(r, c, grid):
    x_clear, y_clear = _axes_clear(r, c, grid)
    if x_clear:
        return np.array([1.0, 0.0, 0.0])
    if y_clear:
        return np.array([0.0, 1.0, 0.0])
    # fallback: fewer occupied neighbours wins
    def occ(rr, cc):
        return 0 <= rr < 3 and 0 <= cc < 3 and grid[rr][cc] is not None
    x_occ = int(occ(r-1, c)) + int(occ(r+1, c))
    y_occ = int(occ(r, c-1)) + int(occ(r, c+1))
    print(f"  [Warn] grid[{r}][{c}]: all neighbours occupied; using best-effort axis.")
    return np.array([1.0, 0.0, 0.0]) if x_occ <= y_occ else np.array([0.0, 1.0, 0.0])


def _clearance_score(r, c, grid):
    x, y = _axes_clear(r, c, grid)
    return int(x) + int(y)

# this runs the trajectory using mujoco
def run_waypoints(model, data, v, waypoints):
    dt         = model.opt.timestep
    seg_steps  = max(1, int(MOVE_DUR / dt))
    hold_steps = max(1, int(HOLD_DUR / dt))

    for i in range(len(waypoints) - 1):
        q_start = np.array(waypoints[i][:7])
        q_goal  = np.array(waypoints[i + 1][:7])
        grip    = waypoints[i + 1][-1]
        t = 0.0
        for _ in range(seg_steps + hold_steps):
            q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, MOVE_DUR)
            q  = data.qpos[ARM_IDX].copy()
            qd = data.qvel[ARM_IDX].copy()
            tau = KP * (q_des - q) + KD * (qd_des - qd)
            data.ctrl[ARM_IDX]     = tau + data.qfrc_bias[:7]
            data.ctrl[GRIPPER_IDX] = grip
            mj.mj_step(model, data)
            v.sync()
            t += dt

_CANDIDATE_AXES = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]

def _ordered_axes(rc, grid):
    if rc is None:
        return [np.array([0., -1., 0.])]
    pref = choose_finger_axis(*rc, grid)
    others = [a for a in _CANDIDATE_AXES if not np.allclose(a, pref)]
    return [pref] + others

# this generates the waypoints for mujoco to execute
def pick_and_place(model, data, v,
                   pick_pos, place_pos,
                   grid_at_pick, grid_at_place,
                   pick_rc=None, place_rc=None):
    """
    Tries each candidate finger axis for pick and place independently,
    using the grid-aware preferred axis first.  Falls back to the other
    axis automatically so IK failures don't abort the whole operation.
    """
    qpos_saved = data.qpos.copy()
    qvel_saved = data.qvel.copy()

    def restore():
        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved

    # --- find a working pick IK ---
    pick_solution = None
    for f_pick in _ordered_axes(pick_rc, grid_at_pick):
        gp, gr, gpre = rt.top_down_grasp_pose(pick_pos, PREGRASP_H, f_pick)
        restore(); q_prepick, ok1 = rt.compute_ik(model, data, gpre, gr, ARM_IDX, q_init=HOME_Q)
        restore(); q_pick_ik, ok2 = rt.compute_ik(model, data, gp,   gr, ARM_IDX, q_init=q_prepick)
        restore(); q_lift,    ok3 = rt.compute_ik(model, data, gpre, gr, ARM_IDX, q_init=q_pick_ik)
        if ok1 and ok2 and ok3:
            pick_solution = (q_prepick, q_pick_ik, q_lift)
            break

    # --- find a working place IK ---
    place_solution = None
    for f_place in _ordered_axes(place_rc, grid_at_place):
        pp, pr, ppre = rt.top_down_grasp_pose(place_pos, PREGRASP_H, f_place)
        restore(); q_preplace, ok4 = rt.compute_ik(model, data, ppre, pr, ARM_IDX, q_init=HOME_Q)
        restore(); q_place_ik, ok5 = rt.compute_ik(model, data, pp,   pr, ARM_IDX, q_init=q_preplace)
        if ok4 and ok5:
            place_solution = (q_preplace, q_place_ik)
            break

    restore()
    mj.mj_forward(model, data)

    if pick_solution is None or place_solution is None:
        failed = ([" pick"] if pick_solution  is None else []) + \
                 (["place"] if place_solution is None else [])
        print(f"  [IK fail] No valid orientation for: {', '.join(failed)} – skipping.")
        return False

    q_prepick, q_pick_ik, q_lift   = pick_solution
    q_preplace, q_place_ik          = place_solution

    wp = [
        [*HOME_Q,      OPEN  ],   # 0: home
        [*q_prepick,   OPEN  ],   # 1: pre-grasp
        [*q_pick_ik,   OPEN  ],   # 2: at block
        [*q_pick_ik,   CLOSED],   # 3: close gripper
        [*q_lift,      CLOSED],   # 4: lift/retreat
        [*q_preplace,  CLOSED],   # 5: pre-place
        [*q_place_ik,  CLOSED],   # 6: at target location
        [*q_place_ik,  OPEN  ],   # 7: release
        [*q_preplace,  OPEN  ],   # 8: retreat
        [*HOME_Q,      OPEN  ],   # 9: home
    ]
    run_waypoints(model, data, v, wp)
    return True


def plan_transition(state: BlockState, target_pattern):
    # remove wrong blocks then place new blocks
    ops = []
    sim = state.copy()

    to_remove = [
        (r, c, sim.grid[r][c])
        for r in range(3) for c in range(3)
        if sim.grid[r][c] is not None
        and target_pattern[r][c] != sim.block_color(sim.grid[r][c])
    ]
    while to_remove:
        to_remove.sort(
            key=lambda x: (_clearance_score(x[0], x[1], sim.grid),
                           -sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                if 0<=x[0]+dr<3 and 0<=x[1]+dc<3
                                and sim.grid[x[0]+dr][x[1]+dc] is not None)),
            reverse=True)
        r, c, name = to_remove.pop(0)
        ops.append({'type': 'remove', 'r': r, 'c': c, 'name': name})
        sim.grid[r][c] = None
        sim.return_to_staging(name)

    needs_place = [
        (r, c, target_pattern[r][c])
        for r in range(3) for c in range(3)
        if target_pattern[r][c] is not None
        and (sim.grid[r][c] is None
             or sim.block_color(sim.grid[r][c]) != target_pattern[r][c])
    ]
    while needs_place:
        needs_place.sort(
            key=lambda x: (_clearance_score(x[0], x[1], sim.grid),
                           abs(x[0] - 1) + abs(x[1] - 1)),
            reverse=True)
        r, c, color = needs_place.pop(0)
        name, _ = sim.take_from_staging(color)
        if name is None:
            print(f"  [Plan] No {color} block available for grid[{r}][{c}] – skipping.")
            continue
        ops.append({'type': 'place', 'r': r, 'c': c, 'color': color, 'name': name})
        sim.grid[r][c] = name

    return ops


if __name__ == "__main__":
    np.random.seed(42)

    build_scene()
    model = mj.MjModel.from_xml_path(MODEL_XML)
    data  = mj.MjData(model)

    data.qpos[ARM_IDX] = HOME_Q
    mj.mj_forward(model, data)

    v = viewer.launch_passive(model, data)
    v.cam.distance  = 2.5
    v.cam.azimuth  += 90
    v.cam.elevation = -25

    state = BlockState()

    for pat_idx, (pat_name, target) in enumerate(zip(PATTERN_NAMES, PATTERNS)):
        print(f"\n{'='*60}")
        print(f"  Pattern {pat_idx + 1}: {pat_name.upper()}")
        print(f"{'='*60}")

        ops = plan_transition(state, target)
        if not ops:
            print("Already in target configuration.")
            continue

        pattern_ok = True

        for op in ops:
            r, c = op['r'], op['c']

            if op['type'] == 'remove':
                name      = op['name']
                pick_pos  = GRID[r][c].copy()
                place_pos = state.staging_pos(name)
                grid_pick = [row[:] for row in state.grid]

                print(f"Remove {name} from grid[{r}][{c}] → staging")
                ok = pick_and_place(model, data, v,
                                    pick_pos, place_pos,
                                    grid_at_pick=grid_pick, grid_at_place=None,
                                    pick_rc=(r, c), place_rc=None)
                if ok:
                    state.grid[r][c] = None
                    state.return_to_staging(name)
                else:
                    print(f"[WARNNN] Could not remove {name}! aborting pattern.")
                    pattern_ok = False
                    break

            elif op['type'] == 'place':
                color = op['color']
                name, pick_pos = state.take_from_staging(color)
                if name is None:
                    print(f"[ERROR] No {color} block available! skipping cell.")
                    continue

                place_pos  = GRID[r][c].copy()
                grid_place = [row[:] for row in state.grid]
                grid_place[r][c] = None

                print(f"Place {name} ({color}) staging → grid[{r}][{c}]")
                ok = pick_and_place(model, data, v,
                                    pick_pos, place_pos,
                                    grid_at_pick=None, grid_at_place=grid_place,
                                    pick_rc=None, place_rc=(r, c))
                if ok:
                    state.grid[r][c] = name
                else:
                    state.return_to_staging(name)
                    print(f"[WARNNN] Could not place {name} at grid[{r}][{c}]!aborting pattern.")
                    pattern_ok = False
                    break

        print(f"Pattern '{pat_name}' {'COMPLETE' if pattern_ok else 'SKIPPED'}.")

        for _ in range(int(1.0 / model.opt.timestep)):
            mj.mj_step(model, data)
            v.sync()

    print("\nAll patterns done. Close the viewer to exit.")
    while v.is_running():
        mj.mj_step(model, data)
        v.sync()

    v.close()