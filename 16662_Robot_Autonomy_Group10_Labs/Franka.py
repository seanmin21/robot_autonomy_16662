import random
import time
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import Optional

import mujoco as mj
import RobotUtil as rt

@dataclass
class FrankaArmConfig:
    ik_max_iter: int = 100
    ik_pos_tol: float = 1e-2
    ik_ang_tol: float = np.radians(2)
    ik_step_size: float = np.radians(5)
    ik_task_weight: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [1e6, 1e6, 1e6, 1e3, 1e3, 1e3], dtype=np.float64
        )
    )
    ik_joint_weight: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [1, 1, 100, 100, 1, 1, 100], dtype=np.float64
        )
    )


class FrankaArm:

    def __init__(
            self,
            model: mj.MjModel,
            data: mj.MjData,
            config: FrankaArmConfig | None = None,
            arm_joint_names: str | None = None,
            ee_site_name: str = "grasp_center",
        ):
        self.model = model
        self.data = data
        self.cfg = config if config is not None else FrankaArmConfig()

        if arm_joint_names is None:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]

        self.joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in arm_joint_names]
        self.body_ids = [model.jnt_bodyid[id] for id in self.joint_ids]

        # include end-effector and base bodies for collision checking
        for extra_name in ["link0", "hand", "left_finger", "right_finger"]:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, extra_name)
            if bid >= 0 and bid not in self.body_ids:
                self.body_ids.append(bid)

        self.ee_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, ee_site_name)

        self.qmin = model.jnt_range[self.joint_ids, 0]
        self.qmax = model.jnt_range[self.joint_ids, 1]
        self.axes = model.jnt_axis[self.joint_ids]

        self.ik_W = np.diag(self.cfg.ik_joint_weight)
        self.ik_C = np.diag(self.cfg.ik_task_weight)

        self._tmp_data = mj.MjData(model)

        # skip manipulated objects during collision checks
        self.ignored_body_ids = set()
        for name in ["Block"]:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.ignored_body_ids.add(bid)

        self.ForwardKin()

    def ForwardKin(self, ang: npt.NDArray[np.float64] | None = None):
        if ang is not None:
            self.data.qpos[self.joint_ids] = ang
        mj.mj_forward(self.model, self.data)

        self.ee_pos  = self.data.site_xpos[self.ee_site_id].copy()
        self.ee_rmat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        self.J = np.vstack([jacp[:, self.joint_ids], jacr[:, self.joint_ids]])

    def IterInvKin(
            self,
            target_pos: npt.NDArray[np.float64],
            target_rmat: npt.NDArray[np.float64],
    ):
        itr = 1
        converged = False
        c_inv = np.linalg.inv(self.ik_C)
        w_inv = np.linalg.inv(self.ik_W)
        dp = 0
        ang_err = 0

        while itr <= self.cfg.ik_max_iter:
            dp = target_pos - self.ee_pos
            rot_diff = target_rmat @ self.ee_rmat.T
            axis, ang_err = rt.R2axisang(rot_diff)
            rvec = np.array(axis, dtype=np.float64) * ang_err

            if np.linalg.norm(dp) <= self.cfg.ik_pos_tol and ang_err <= self.cfg.ik_ang_tol:
                converged = True
                break

            err = np.concatenate([dp, rvec])
            J_pinv = w_inv @ self.J.T @ np.linalg.inv(self.J @ w_inv @ self.J.T + c_inv)
            dq = np.clip(J_pinv @ err, -self.cfg.ik_step_size, self.cfg.ik_step_size)

            q = self.data.qpos[self.joint_ids] + dq
            self.data.qpos[self.joint_ids] = np.clip(q, self.qmin, self.qmax)
            self.ForwardKin()
            itr += 1

        if converged:
            print(f"IK solved at iteration {itr}")
        else:
            print(f"IK did not converge: pos_err={dp}, ang_err={ang_err}")

    def DetectCollision(self, q):
        mj.mj_resetData(self.model, self._tmp_data)
        self._tmp_data.qpos[self.joint_ids] = q
        mj.mj_forward(self.model, self._tmp_data)

        for i in range(self._tmp_data.ncon):
            contact = self._tmp_data.contact[i]
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]

            if b1 in self.ignored_body_ids or b2 in self.ignored_body_ids:
                continue

            if (b1 in self.body_ids) != (b2 in self.body_ids):
                return True
        return False

    def DetectCollisionEdge(self, q1, q2, resolution=0.01):
        q1 = np.array(q1)
        q2 = np.array(q2)
        n_steps = max(10, int(np.ceil(np.max(np.abs(q2 - q1)) / resolution)))
        for s in np.linspace(0, 1, n_steps):
            if self.DetectCollision(q1 + s * (q2 - q1)):
                return True
        return False

    def SampleRobotConfig(self):
        return [self.qmin[i] + (self.qmax[i] - self.qmin[i]) * random.random() for i in range(7)]

    def FindNearest(self, tree, q):
        dists = np.array([np.linalg.norm(np.array(node) - np.array(q)) for node in tree])
        return dists.argmin()

    def naive_interpolation(self, plan):
        res = 0.2
        np_plan = np.array(plan)
        out = [np_plan[0]]

        for i in range(np_plan.shape[0] - 1):
            n_steps = int(np.ceil(np.max(np.abs(np_plan[i + 1] - np_plan[i])) / res))
            if n_steps == 0:
                continue
            inc = (np_plan[i + 1] - np_plan[i]) / n_steps
            for j in range(1, n_steps + 1):
                out.append(np_plan[i] + j * inc)

        out = np.array(out)
        print("Plan interpolated.")
        print(len(plan), len(out))
        return out

    def rrt_plan_bidirectional(self, q_start: list[float], q_goal: list[float],
                               max_iter: int = 5000, goal_bias: float = 0.3,
                               step_size: float = 0.03, goal_thresh: float = 0.05
                               ) -> Optional[list[list[float]]]:
        tree_a = [list(q_start)]
        par_a  = [-1]
        tree_b = [list(q_goal)]
        par_b  = [-1]

        if not self.DetectCollisionEdge(q_start, q_goal):
            print("[RRT] Direct connection found.")
            return [q_start, q_goal]

        found = False
        for k in range(max_iter):
            if k % 2 == 0:
                t_from, p_from = tree_a, par_a
                t_to,   p_to   = tree_b, par_b
            else:
                t_from, p_from = tree_b, par_b
                t_to,   p_to   = tree_a, par_a

            bias = min(goal_bias + k / max_iter * 0.2, 0.5)
            if random.random() < bias:
                q_rand = np.array(t_to[0]) + np.random.uniform(-0.05, 0.05, size=7)
            else:
                q_rand = self.SampleRobotConfig()

            nn_idx = self.FindNearest(t_from, q_rand)
            q_near = np.array(t_from[nn_idx])

            diff = q_rand - q_near
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue
            direction = diff / dist

            q_cur  = q_near.copy()
            cur_idx = nn_idx
            while True:
                rem = np.linalg.norm(q_rand - q_cur)
                if rem < 1e-6:
                    break
                q_new = q_cur + direction * min(step_size, rem)

                if self.DetectCollisionEdge(q_cur, q_new):
                    break

                t_from.append(list(q_new))
                p_from.append(cur_idx)
                cur_idx = len(t_from) - 1
                q_cur = q_new

                nn_to = self.FindNearest(t_to, q_cur)
                if not self.DetectCollisionEdge(q_cur, np.array(t_to[nn_to])):
                    if k % 2 == 0:
                        seg_a = self._extract_path(t_from, p_from, len(t_from) - 1)
                        seg_b = self._extract_path(t_to,   p_to,   nn_to)
                        seg_b.reverse()
                    else:
                        seg_a = self._extract_path(t_to,   p_to,   nn_to)
                        seg_b = self._extract_path(t_from, p_from, len(t_from) - 1)
                        seg_b.reverse()
                    plan = seg_a + seg_b
                    found = True
                    print(f"[RRT] Connected at iter {k}, nodes: {len(tree_a) + len(tree_b)}")
                    break
            if found:
                break

        if not found:
            print("[RRT] No path found.")
            return None

        for attempt in range(80):
            n = len(plan)
            if n < 3:
                break
            a = np.random.randint(0, n - 2)
            b = np.random.randint(a + 2, n)
            if not self.DetectCollisionEdge(np.array(plan[a]), np.array(plan[b])):
                plan = plan[:a + 1] + plan[b:]
                print(f"{attempt}, [RRT] Shortcut: removed {a+1}..{b-1}, len={len(plan)}")

        return plan

    def _extract_path(self, tree, parent, idx):
        path = []
        c = idx
        while c != -1:
            path.append(tree[c])
            c = parent[c]
        path.reverse()
        return path

    def plan_waypoints(self, waypoints: list[list[float]]) -> Optional[list[list[float]]]:
        """
        Plan a collision-free joint-space path through the given waypoints.
        Waypoints may include a gripper state as the 8th value; it's passed
        through but not used in collision planning.
        """
        full_path = []
        for i in range(len(waypoints) - 1):
            print(f"\nSegment {i} -> {i+1}")
            q0 = np.array(waypoints[i][:7])
            q1 = np.array(waypoints[i + 1][:7])
            print("  start collision:", self.DetectCollision(q0))
            print("  goal  collision:", self.DetectCollision(q1))

            if np.allclose(q0, q1, atol=1e-6):
                seg = [q0.copy(), q1.copy()]
            else:
                seg = self.rrt_plan_bidirectional(q0, q1, step_size=0.05)
                if seg is None:
                    print(f"  Failed segment {i} -> {i+1}")
                    return None

            if len(waypoints[i + 1]) > 7:
                g = waypoints[i + 1][-1]
                seg = [np.append(q, g) for q in seg]

            full_path.extend(seg[1:] if full_path else seg)
        return full_path

    def __del__(self):
        if hasattr(self, "_tmp_data"):
            del self._tmp_data
