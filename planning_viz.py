#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
import heapq
from scipy.linalg import solve_continuous_are

# ============================================================
# Helpers and base model
# ============================================================

def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class KinematicBicycle:
    L: float
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0

    def step(self, v: float, delta: float, dt: float):
        x_dot = v * np.cos(self.psi)
        y_dot = v * np.sin(self.psi)
        psi_dot = v / self.L * np.tan(delta)

        self.x += x_dot * dt
        self.y += y_dot * dt
        self.psi = wrap_angle(self.psi + psi_dot * dt)
        return self.x, self.y, self.psi


# ============================================================
# Grid world + obstacles
# ============================================================

class GridWorld:
    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.5):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.res = resolution

        self.nx = int((x_max - x_min) / resolution) + 1
        self.ny = int((y_max - y_min) / resolution) + 1

        self.occ = np.zeros((self.nx, self.ny), dtype=bool)
        self.obstacles = []

    def add_circular_obstacle(self, xc, yc, r):
        self.obstacles.append((xc, yc, r))
        for ix in range(self.nx):
            for iy in range(self.ny):
                x, y = self.index_to_world(ix, iy)
                if np.hypot(x - xc, y - yc) <= r:
                    self.occ[ix, iy] = True

    def in_bounds(self, x, y):
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

    def world_to_index(self, x, y):
        ix = int(round((x - self.x_min) / self.res))
        iy = int(round((y - self.y_min) / self.res))
        return ix, iy

    def index_to_world(self, ix, iy):
        x = self.x_min + ix * self.res
        y = self.y_min + iy * self.res
        return x, y

    def collision_free_segment(self, xs, ys, robot_radius=0.7):
        for x, y in zip(xs, ys):
            if not self.in_bounds(x, y):
                return False
            ix, iy = self.world_to_index(x, y)
            if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
                return False
            if self.occ[ix, iy]:
                return False
            for xc, yc, r in self.obstacles:
                if np.hypot(x - xc, y - yc) <= r + robot_radius:
                    return False
        return True


# ============================================================
# Hybrid A*-like search
# ============================================================

@dataclass(order=True)
class PQNode:
    f: float
    idx: int


@dataclass
class Node:
    x: float
    y: float
    psi: float
    g: float
    h: float
    parent: int


def heuristic(x, y, goal):
    return np.hypot(x - goal[0], y - goal[1])


def hybrid_astar(
    grid: GridWorld,
    start_state,
    goal_xy,
    L=2.5,
    v_step=2.0,
    dt=0.1,
    T_step=1.0,
    delta_set=None,
    yaw_bins=24,
    robot_radius=0.7,
    max_iter=14000,
    goal_tolerance=1.5,
):
    if delta_set is None:
        delta_set = [-0.5, 0.0, 0.5]

    car_model = KinematicBicycle(L=L)

    def yaw_to_bin(psi):
        psi_wrap = wrap_angle(psi)
        bin_width = 2 * np.pi / yaw_bins
        return int(np.floor((psi_wrap + np.pi) / bin_width))

    closed = np.full((grid.nx, grid.ny, yaw_bins), False, dtype=bool)

    nodes = []
    x0, y0, psi0 = start_state
    h0 = heuristic(x0, y0, goal_xy)
    nodes.append(Node(x0, y0, psi0, g=0.0, h=h0, parent=-1))

    pq = []
    heapq.heappush(pq, PQNode(f=h0, idx=0))

    step_n = max(2, int(T_step / dt))

    for it in range(max_iter):
        if not pq:
            return None

        cur = heapq.heappop(pq)
        n_idx = cur.idx
        node = nodes[n_idx]

        x, y, psi = node.x, node.y, node.psi

        if np.hypot(x - goal_xy[0], y - goal_xy[1]) <= goal_tolerance:
            path = []
            idx = n_idx
            while idx != -1:
                n = nodes[idx]
                path.append((n.x, n.y, n.psi))
                idx = n.parent
            path.reverse()
            print(f"Goal reached in {it} expansions, nodes={len(path)}, cost={node.g:.2f}")
            return path

        ix, iy = grid.world_to_index(x, y)
        if ix < 0 or ix >= grid.nx or iy < 0 or iy >= grid.ny:
            continue
        yaw_bin = yaw_to_bin(psi)
        if closed[ix, iy, yaw_bin]:
            continue
        closed[ix, iy, yaw_bin] = True

        for delta in delta_set:
            xs = [x]
            ys = [y]
            psi_ = psi

            for _ in range(step_n):
                car_model.x, car_model.y, car_model.psi = xs[-1], ys[-1], psi_
                x_, y_, psi_ = car_model.step(v_step, delta, dt)
                xs.append(x_)
                ys.append(y_)

            if not grid.collision_free_segment(xs, ys, robot_radius):
                continue

            x_new, y_new, psi_new = xs[-1], ys[-1], psi_

            g_new = node.g + np.hypot(x_new - x, y_new - y) + 0.05 * abs(delta)
            h_new = heuristic(x_new, y_new, goal_xy)
            f_new = g_new + h_new

            nodes.append(Node(x_new, y_new, psi_new, g=g_new, h=h_new, parent=n_idx))
            heapq.heappush(pq, PQNode(f=f_new, idx=len(nodes) - 1))

    return None


# ============================================================
# Controllers
# ============================================================

class PIDSteering:
    """PID on lateral error + heading error around delta_ff."""
    def __init__(self, Kp_y=2.0, Ki_y=0.1, Kd_y=0.6, Kp_psi=4.0, delta_max=np.deg2rad(35)):
        self.Kp_y = Kp_y
        self.Ki_y = Ki_y
        self.Kd_y = Kd_y
        self.Kp_psi = Kp_psi
        self.delta_max = delta_max
        self.reset()

    def reset(self):
        self.e_y_int = 0.0
        self.e_y_prev = 0.0
        self.first = True

    def control(self, e_y, e_psi, delta_ff, dt):
        self.e_y_int += e_y * dt
        if self.first:
            de_y = 0.0
            self.first = False
        else:
            de_y = (e_y - self.e_y_prev) / dt
        self.e_y_prev = e_y

        delta_fb = (self.Kp_y * e_y + self.Ki_y * self.e_y_int + self.Kd_y * de_y + self.Kp_psi * e_psi)
        delta = delta_ff - delta_fb
        return float(np.clip(delta, -self.delta_max, self.delta_max))


class LQRSteering:
    """Continuous-time LQR on [e_y, e_psi], delta = delta_ff + u."""
    def __init__(self, v, L, Q=None, R=None, delta_max=np.deg2rad(35)):
        self.v = v
        self.L = L
        self.delta_max = delta_max

        A = np.array([[0.0, v],
                      [0.0, 0.0]])
        B = np.array([[0.0],
                      [v / L]])

        if Q is None:
            Q = np.diag([20.0, 8.0])
        if R is None:
            R = np.array([[0.4]])
        elif np.isscalar(R):
            R = np.array([[R]])

        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P  # 1x2

    def control(self, e_y, e_psi, delta_ff):
        x = np.array([[e_y], [e_psi]])
        u = -self.K @ x
        delta = delta_ff + float(u.item())
        return float(np.clip(delta, -self.delta_max, self.delta_max))


# ============================================================
# Path utilities (curvature + lookahead target)
# ============================================================

def compute_path_geometry(xs, ys):
    N = len(xs)
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    psi = np.arctan2(dy, dx)

    ds = np.hypot(dx, dy) + 1e-9
    dpsi = np.gradient(np.unwrap(psi))
    kappa = dpsi / ds

    s = np.zeros(N)
    for i in range(1, N):
        s[i] = s[i - 1] + np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])

    return psi, kappa, s


def advance_progress_index(xs, ys, x, y, idx_prev, window=30):
    N = len(xs)
    i0 = max(0, idx_prev)
    i1 = min(N, idx_prev + window)
    if i0 >= i1:
        return N - 1

    d2 = (xs[i0:i1] - x) ** 2 + (ys[i0:i1] - y) ** 2
    return i0 + int(np.argmin(d2))


def lookahead_index(s_ref, idx_closest, Ld):
    s_target = s_ref[idx_closest] + Ld
    idx = int(np.searchsorted(s_ref, s_target))
    return min(idx, len(s_ref) - 1)


def lateral_heading_error(x, y, psi, x_ref, y_ref, psi_ref):
    dx = x - x_ref
    dy = y - y_ref
    e_y = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy
    e_psi = wrap_angle(psi - psi_ref)
    return float(e_y), float(e_psi)


# ============================================================
# Cute car geometry
# ============================================================

def car_outline(x, y, yaw, L, W):
    L_car = 1.6 * L
    half_w = W / 2.0
    corners_body = np.array([[0.0, -half_w],
                             [0.0,  half_w],
                             [L_car, half_w],
                             [L_car, -half_w]])
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])
    corners_world = (R @ corners_body.T).T
    corners_world[:, 0] += x
    corners_world[:, 1] += y
    return corners_world


def wheel_centers_world(x, y, yaw, L, W):
    half_w = W / 2.0
    centers_body = np.array([[0.0, +half_w],
                             [0.0, -half_w],
                             [L,   +half_w],
                             [L,   -half_w]])
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])
    centers_world = (R @ centers_body.T).T
    centers_world[:, 0] += x
    centers_world[:, 1] += y
    return centers_world


def wheel_polygon(center_world, theta, length, width):
    half_l = length / 2.0
    half_w = width / 2.0
    corners_local = np.array([[-half_l, -half_w],
                              [-half_l,  half_w],
                              [ half_l,  half_w],
                              [ half_l, -half_w]])
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    corners_world = (R @ corners_local.T).T
    corners_world[:, 0] += center_world[0]
    corners_world[:, 1] += center_world[1]
    return corners_world


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # --- Planner world ---
    grid = GridWorld(x_min=0, x_max=40, y_min=-10, y_max=10, resolution=0.5)
    grid.add_circular_obstacle(15, 0, 3)
    grid.add_circular_obstacle(25, 4, 2)
    grid.add_circular_obstacle(30, -3, 2)

    start = (2.0, -5.0, 0.0)
    goal_xy = (35.0, 5.0)

    path = hybrid_astar(
        grid,
        start_state=start,
        goal_xy=goal_xy,
        L=2.5,
        v_step=3.0,
        dt=0.1,
        T_step=1.0,
        delta_set=[-0.5, -0.25, 0.0, 0.25, 0.5],
        yaw_bins=32,
        robot_radius=0.8,
        goal_tolerance=1.5,
    )

    if path is None or len(path) < 5:
        print("No path found (or too short).")
        raise SystemExit

    xs_ref = np.array([p[0] for p in path], dtype=float)
    ys_ref = np.array([p[1] for p in path], dtype=float)
    psi_ref, kappa_ref, s_ref = compute_path_geometry(xs_ref, ys_ref)

    # --- Tracking setup ---
    L = 2.5
    W = 1.5
    v = 4.0
    dt = 0.05
    T_sim = 18.0
    N = int(T_sim / dt)

    Ld = 2.5  # lookahead distance

    car_pid = KinematicBicycle(L=L, x=start[0], y=start[1], psi=start[2])
    car_lqr = KinematicBicycle(L=L, x=start[0], y=start[1], psi=start[2])

    pid = PIDSteering()
    lqr = LQRSteering(v=v, L=L)

    # progress indices (MUTABLE so update() can modify without nonlocal)
    idx_pid = [0]
    idx_lqr = [0]

    # logs
    xs_pid = np.zeros(N)
    ys_pid = np.zeros(N)
    xs_lqr = np.zeros(N)
    ys_lqr = np.zeros(N)

    # --- Figure: two separate graphs ---
    fig, (ax_pid, ax_lqr) = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax in (ax_pid, ax_lqr):
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle="--", alpha=0.3)
        for xc, yc, r in grid.obstacles:
            ax.add_patch(plt.Circle((xc, yc), r, color="k", alpha=0.25))
        ax.plot(xs_ref, ys_ref, "--", color="#cccccc", linewidth=2.0, label="Planned path")
        ax.scatter(start[0], start[1], c="g", s=40, label="Start")
        ax.scatter(goal_xy[0], goal_xy[1], c="r", s=60, marker="x", label="Goal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    ax_pid.set_title("PID tracking (lookahead target)")
    ax_lqr.set_title("LQR tracking (lookahead target)")

    margin = 3.0
    ax_pid.set_xlim(xs_ref.min() - margin, xs_ref.max() + margin)
    ax_pid.set_ylim(ys_ref.min() - margin, ys_ref.max() + margin)

    # Trajectory lines
    line_pid, = ax_pid.plot([], [], color="#1f77b4", linewidth=2.0, label="PID traj")
    line_lqr, = ax_lqr.plot([], [], color="#d62728", linewidth=2.0, label="LQR traj")

    ax_pid.legend(loc="upper right", fontsize=8)
    ax_lqr.legend(loc="upper right", fontsize=8)

    # Car patches (PID)
    wheel_len = 0.6
    wheel_wid = 0.2
    body_pid = plt.Polygon(
        car_outline(car_pid.x, car_pid.y, car_pid.psi, L, W),
        closed=True, facecolor="#1f77b4", edgecolor="black", alpha=0.7
    )
    ax_pid.add_patch(body_pid)
    wcp = wheel_centers_world(car_pid.x, car_pid.y, car_pid.psi, L, W)
    wheels_pid = []
    for c in wcp:
        p = plt.Polygon(wheel_polygon(c, car_pid.psi, wheel_len, wheel_wid),
                        closed=True, facecolor="black")
        ax_pid.add_patch(p)
        wheels_pid.append(p)

    # Car patches (LQR)
    body_lqr = plt.Polygon(
        car_outline(car_lqr.x, car_lqr.y, car_lqr.psi, L, W),
        closed=True, facecolor="#d62728", edgecolor="black", alpha=0.7
    )
    ax_lqr.add_patch(body_lqr)
    wcl = wheel_centers_world(car_lqr.x, car_lqr.y, car_lqr.psi, L, W)
    wheels_lqr = []
    for c in wcl:
        p = plt.Polygon(wheel_polygon(c, car_lqr.psi, wheel_len, wheel_wid),
                        closed=True, facecolor="black")
        ax_lqr.add_patch(p)
        wheels_lqr.append(p)

    # --- Animation update ---
    def update(i):
        # ========== PID ==========
        idx_pid[0] = advance_progress_index(xs_ref, ys_ref, car_pid.x, car_pid.y, idx_pid[0], window=40)
        tgt_pid = lookahead_index(s_ref, idx_pid[0], Ld)

        xR, yR, psiR, kap = xs_ref[tgt_pid], ys_ref[tgt_pid], psi_ref[tgt_pid], kappa_ref[tgt_pid]
        delta_ff = np.arctan(L * kap)

        e_y, e_psi = lateral_heading_error(car_pid.x, car_pid.y, car_pid.psi, xR, yR, psiR)
        delta = pid.control(e_y, e_psi, delta_ff, dt)

        car_pid.step(v, delta, dt)

        xs_pid[i] = car_pid.x
        ys_pid[i] = car_pid.y
        line_pid.set_data(xs_pid[:i + 1], ys_pid[:i + 1])

        body_pid.set_xy(car_outline(car_pid.x, car_pid.y, car_pid.psi, L, W))
        centers = wheel_centers_world(car_pid.x, car_pid.y, car_pid.psi, L, W)
        for wp, c in zip(wheels_pid, centers):
            wp.set_xy(wheel_polygon(c, car_pid.psi, wheel_len, wheel_wid))

        # ========== LQR ==========
        idx_lqr[0] = advance_progress_index(xs_ref, ys_ref, car_lqr.x, car_lqr.y, idx_lqr[0], window=40)
        tgt_lqr = lookahead_index(s_ref, idx_lqr[0], Ld)

        xR, yR, psiR, kap = xs_ref[tgt_lqr], ys_ref[tgt_lqr], psi_ref[tgt_lqr], kappa_ref[tgt_lqr]
        delta_ff = np.arctan(L * kap)

        e_y, e_psi = lateral_heading_error(car_lqr.x, car_lqr.y, car_lqr.psi, xR, yR, psiR)
        delta = lqr.control(e_y, e_psi, delta_ff)

        car_lqr.step(v, delta, dt)

        xs_lqr[i] = car_lqr.x
        ys_lqr[i] = car_lqr.y
        line_lqr.set_data(xs_lqr[:i + 1], ys_lqr[:i + 1])

        body_lqr.set_xy(car_outline(car_lqr.x, car_lqr.y, car_lqr.psi, L, W))
        centers = wheel_centers_world(car_lqr.x, car_lqr.y, car_lqr.psi, L, W)
        for wp, c in zip(wheels_lqr, centers):
            wp.set_xy(wheel_polygon(c, car_lqr.psi, wheel_len, wheel_wid))

        return [line_pid, body_pid, *wheels_pid, line_lqr, body_lqr, *wheels_lqr]

    ani = FuncAnimation(fig, update, frames=N, interval=dt * 1000, blit=True)
    fig.suptitle("Search-based (Hybrid A*) plan tracking — Separate PID vs LQR plots", fontsize=14)
    plt.tight_layout()
    plt.show()
