#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from scipy.linalg import solve_continuous_are


# ============================
# Helpers and model
# ============================

def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class KinematicBicycle:
    """Kinematic bicycle model (Ackermann)."""
    L: float
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0  # heading

    def step(self, v: float, delta: float, dt: float):
        x_dot = v * np.cos(self.psi)
        y_dot = v * np.sin(self.psi)
        psi_dot = v / self.L * np.tan(delta)

        self.x += x_dot * dt
        self.y += y_dot * dt
        self.psi = wrap_angle(self.psi + psi_dot * dt)

        return self.x, self.y, self.psi


# ============================
# Reference paths: x_r, y_r, psi_r, kappa_r
# ============================

def ref_lane_change(t: float, v: float, Dy: float = 3.0, T_c: float = 5.0):
    """Lane change with curvature (half-cosine)."""
    x = v * t
    if t <= T_c:
        y = 0.5 * Dy * (1 - np.cos(np.pi * t / T_c))
        y_dot = 0.5 * Dy * (np.pi / T_c) * np.sin(np.pi * t / T_c)
        y_ddot = 0.5 * Dy * (np.pi / T_c) ** 2 * np.cos(np.pi * t / T_c)
    else:
        y = Dy
        y_dot = 0.0
        y_ddot = 0.0

    x_dot = v
    x_ddot = 0.0

    psi = np.arctan2(y_dot, x_dot)

    denom = (x_dot**2 + y_dot**2) ** 1.5
    if denom < 1e-6:
        kappa = 0.0
    else:
        kappa = (x_dot * y_ddot - y_dot * x_ddot) / denom

    return x, y, psi, kappa


def ref_circle(t: float, v: float, R: float = 10.0):
    """Constant radius circle with curvature 1/R."""
    w = v / R
    th = w * t
    x = R * np.sin(th)
    y = R * (1 - np.cos(th))

    x_dot = v * np.cos(th)
    y_dot = v * np.sin(th)
    psi = np.arctan2(y_dot, x_dot)

    kappa = 1.0 / R
    return x, y, psi, kappa


def ref_wave(t: float, v: float, A: float = 2.0, T_s: float = 10.0):
    """Simple sine wave road with curvature."""
    x = v * t
    w = 2 * np.pi / T_s
    y = A * np.sin(w * t)

    x_dot = v
    y_dot = A * w * np.cos(w * t)

    x_ddot = 0.0
    y_ddot = -A * w**2 * np.sin(w * t)

    psi = np.arctan2(y_dot, x_dot)

    denom = (x_dot**2 + y_dot**2) ** 1.5
    if denom < 1e-6:
        kappa = 0.0
    else:
        kappa = (x_dot * y_ddot - y_dot * x_ddot) / denom

    return x, y, psi, kappa


def ref_complicated(t: float, v: float, A: float = 2.0, T_s: float = 12.0):
    """Wiggly road (sum of sines) with curvature."""
    x = v * t
    w = 2 * np.pi / T_s
    y = A * np.sin(w * t) + 0.5 * A * np.sin(3 * w * t)

    x_dot = v
    y_dot = A * w * np.cos(w * t) + 0.5 * A * 3 * w * np.cos(3 * w * t)

    x_ddot = 0.0
    y_ddot = -A * w**2 * np.sin(w * t) - 0.5 * A * 9 * w**2 * np.sin(3 * w * t)

    psi = np.arctan2(y_dot, x_dot)

    denom = (x_dot**2 + y_dot**2) ** 1.5
    if denom < 1e-6:
        kappa = 0.0
    else:
        kappa = (x_dot * y_ddot - y_dot * x_ddot) / denom

    return x, y, psi, kappa


# ============================
# LQR Steering Controller
# ============================

class LQRSteering:
    """
    LQR on linearized lateral error dynamics:

      state x = [e_y, e_psi]^T
      x_dot = A x + B u,  u = delta_tilde

      A = [[0, v],
           [0, 0]]
      B = [[0],
           [v / L]]

    delta = delta_ff + delta_tilde, where delta_ff from curvature.
    """

    def __init__(
        self,
        v: float,
        L: float,
        Q: np.ndarray | None = None,
        R: float | np.ndarray | None = None,
        delta_max: float = np.deg2rad(35.0),
    ):
        self.v = v
        self.L = L
        self.delta_max = delta_max

        A = np.array([[0.0, v],
                      [0.0, 0.0]])
        B = np.array([[0.0],
                      [v / L]])

        if Q is None:
            Q = np.diag([10.0, 5.0])   # lateral & heading error weights
        if R is None:
            R = np.array([[1.0]])
        elif np.isscalar(R):
            R = np.array([[R]])

        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P   # 1x2 gain

    def control(self, e_y: float, e_psi: float, delta_ff: float) -> float:
        x = np.array([[e_y],
                      [e_psi]])
        u = -self.K @ x     # delta_tilde
        delta = delta_ff + u.item()
        return np.clip(delta, -self.delta_max, self.delta_max)


# ============================
# Car geometry for viz
# ============================

def car_outline(x: float, y: float, yaw: float, L: float, W: float):
    L_car = 1.6 * L
    half_w = W / 2.0
    corners_body = np.array(
        [
            [0.0, -half_w],
            [0.0, +half_w],
            [L_car, +half_w],
            [L_car, -half_w],
        ]
    )
    c = np.cos(yaw); s = np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners_world = (R @ corners_body.T).T
    corners_world[:, 0] += x
    corners_world[:, 1] += y
    return corners_world


def wheel_centers_world(x: float, y: float, yaw: float, L: float, W: float):
    half_w = W / 2.0
    centers_body = np.array(
        [
            [0.0, +half_w],  # RL
            [0.0, -half_w],  # RR
            [L, +half_w],    # FL
            [L, -half_w],    # FR
        ]
    )
    c = np.cos(yaw); s = np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    centers_world = (R @ centers_body.T).T
    centers_world[:, 0] += x
    centers_world[:, 1] += y
    return centers_world


def wheel_polygon(center_world, theta, length, width):
    half_l = length / 2.0
    half_w = width / 2.0
    corners_local = np.array(
        [
            [-half_l, -half_w],
            [-half_l, +half_w],
            [+half_l, +half_w],
            [+half_l, -half_w],
        ]
    )
    c = np.cos(theta); s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    corners_world = (R @ corners_local.T).T
    corners_world[:, 0] += center_world[0]
    corners_world[:, 1] += center_world[1]
    return corners_world


# ============================
# Simulation with LQR
# ============================

def simulate_path_lqr(ref_fun, name: str, L=2.5, v=5.0, T=10.0, n_frames=300):
    dt = T / n_frames
    car = KinematicBicycle(L=L)
    lqr = LQRSteering(v=v, L=L)

    xs = np.zeros(n_frames)
    ys = np.zeros(n_frames)
    psis = np.zeros(n_frames)
    xr = np.zeros(n_frames)
    yr = np.zeros(n_frames)

    for k in range(n_frames):
        t = k * dt

        # reference pose + curvature
        x_ref, y_ref, psi_ref, kappa_ref = ref_fun(t, v)
        xr[k], yr[k] = x_ref, y_ref

        # errors in path frame
        dx = car.x - x_ref
        dy = car.y - y_ref
        e_y = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy
        e_psi = wrap_angle(car.psi - psi_ref)

        # curvature feedforward steering
        delta_ff = np.arctan(L * kappa_ref)

        # LQR correction
        delta = lqr.control(e_y, e_psi, delta_ff)

        # propagate model
        x, y, psi = car.step(v, delta, dt)

        xs[k], ys[k], psis[k] = x, y, psi

    return {
        "name": name,
        "xs": xs,
        "ys": ys,
        "psis": psis,
        "xr": xr,
        "yr": yr,
        "dt": dt,
        "L": L,
        "W": 1.5,
    }


# ============================
# Main: 4-path LQR viz
# ============================

if __name__ == "__main__":
    L = 2.5
    W = 1.5
    v = 5.0
    T = 10.0
    n_frames = 300

    paths = [
        ("Lane change",  ref_lane_change),
        ("Circle",       ref_circle),
        ("Wave",         ref_wave),
        ("Complicated",  ref_complicated),
    ]

    # simulate all
    sim_results = [
        simulate_path_lqr(ref_fun, name, L=L, v=v, T=T, n_frames=n_frames)
        for name, ref_fun in paths
    ]

    # ---- colors per path ----
    path_colors = {
        "Lane change": ("#1f77b4", "#3399ff"),     # body, traj
        "Circle":      ("#d62728", "#ff6347"),
        "Wave":        ("#2ca02c", "#32cd32"),
        "Complicated": ("#9467bd", "#ba55d3"),
    }

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs = axs.ravel()

    cars = []
    wheel_len = 0.6
    wheel_wid = 0.2

    for ax, res in zip(axs, sim_results):
        name = res["name"]
        xs, ys, psis = res["xs"], res["ys"], res["psis"]
        xr, yr = res["xr"], res["yr"]

        body_color, traj_color = path_colors[name]

        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(f"{name} – LQR", fontsize=10)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)

        # reference path (light gray)
        ax.plot(xr, yr, color="#cccccc", linestyle="--", linewidth=1.0, label="ref")
        ax.legend(fontsize=7, loc="best")

        margin = 2.0
        ax.set_xlim(min(xs.min(), xr.min()) - margin,
                    max(xs.max(), xr.max()) + margin)
        ax.set_ylim(min(ys.min(), yr.min()) - margin,
                    max(ys.max(), yr.max()) + margin)

        # moving trajectory line (colored)
        path_line, = ax.plot([], [], linewidth=2.2, color=traj_color)

        # car body (colored)
        body_poly = plt.Polygon(
            car_outline(xs[0], ys[0], psis[0], L, W),
            closed=True,
            facecolor=body_color,
            edgecolor="black",
            alpha=0.7,
            linewidth=1.2,
        )
        ax.add_patch(body_poly)

        # wheels (black)
        centers0 = wheel_centers_world(xs[0], ys[0], psis[0], L, W)
        rl = plt.Polygon(
            wheel_polygon(centers0[0], psis[0], wheel_len, wheel_wid),
            closed=True, facecolor="black"
        )
        rr = plt.Polygon(
            wheel_polygon(centers0[1], psis[0], wheel_len, wheel_wid),
            closed=True, facecolor="black"
        )
        fl = plt.Polygon(
            wheel_polygon(centers0[2], psis[0], wheel_len, wheel_wid),
            closed=True, facecolor="black"
        )
        fr = plt.Polygon(
            wheel_polygon(centers0[3], psis[0], wheel_len, wheel_wid),
            closed=True, facecolor="black"
        )
        for p in (rl, rr, fl, fr):
            ax.add_patch(p)

        cars.append((path_line, body_poly, (rl, rr, fl, fr), xs, ys, psis))

    # ---- animation ----
    def update(frame_idx):
        for path_line, body_poly, (rl, rr, fl, fr), xs, ys, psis in cars:
            path_line.set_data(xs[:frame_idx + 1], ys[:frame_idx + 1])

            body_poly.set_xy(
                car_outline(xs[frame_idx], ys[frame_idx], psis[frame_idx], L, W)
            )

            centers = wheel_centers_world(
                xs[frame_idx], ys[frame_idx], psis[frame_idx], L, W
            )
            rl.set_xy(
                wheel_polygon(centers[0], psis[frame_idx], wheel_len, wheel_wid)
            )
            rr.set_xy(
                wheel_polygon(centers[1], psis[frame_idx], wheel_len, wheel_wid)
            )
            fl.set_xy(
                wheel_polygon(centers[2], psis[frame_idx], wheel_len, wheel_wid)
            )
            fr.set_xy(
                wheel_polygon(centers[3], psis[frame_idx], wheel_len, wheel_wid)
            )
        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=(T / n_frames) * 1000,
        blit=False,
    )

    fig.suptitle("Ackermann car – LQR tracking on 4 paths (color coded)", fontsize=12)
    plt.tight_layout()
    plt.show()
