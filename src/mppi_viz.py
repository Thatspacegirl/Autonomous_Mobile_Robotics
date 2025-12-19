#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

# ============================================================
# Basic helpers and model
# ============================================================

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


# ============================================================
# Reference paths
# All return: x_ref, y_ref, psi_ref, kappa_ref
# ============================================================

def ref_circle(t: float, v: float, R: float = 10.0):
    """Constant-speed circle of radius R."""
    w = v / R
    th = w * t
    x = R * np.sin(th)
    y = R * (1 - np.cos(th))

    x_dot = v * np.cos(th)
    y_dot = v * np.sin(th)
    psi = np.arctan2(y_dot, x_dot)

    kappa = 1.0 / R
    return x, y, psi, kappa


def mountain_profile(t, A1=4.0, A2=3.0, t1=3.0, s1=1.0, t2=7.0, s2=1.5):
    """
    Double-Gaussian 'mountain' profile in y(t).
    y(t) = A1*exp(-((t-t1)/s1)^2) + A2*exp(-((t-t2)/s2)^2)
    Returns y, y_dot, y_ddot.
    """
    # First peak
    z1 = (t - t1) / s1
    e1 = np.exp(-z1**2)
    y1 = A1 * e1
    y1_dot = y1 * (-2.0 * z1 / s1)
    y1_ddot = y1 * (4.0 * z1**2 - 2.0) / (s1**2)

    # Second peak
    z2 = (t - t2) / s2
    e2 = np.exp(-z2**2)
    y2 = A2 * e2
    y2_dot = y2 * (-2.0 * z2 / s2)
    y2_ddot = y2 * (4.0 * z2**2 - 2.0) / (s2**2)

    y = y1 + y2
    y_dot = y1_dot + y2_dot
    y_ddot = y1_ddot + y2_ddot
    return y, y_dot, y_ddot


def ref_mountain(t: float, v: float):
    """
    'Mountain road': x = v t, y from a double-Gaussian profile.
    We compute psi and curvature from derivatives.
    """
    x = v * t
    y, y_dot, y_ddot = mountain_profile(t)

    x_dot = v
    x_ddot = 0.0

    psi = np.arctan2(y_dot, x_dot)

    denom = (x_dot**2 + y_dot**2)**1.5
    if denom < 1e-8:
        kappa = 0.0
    else:
        kappa = (x_dot * y_ddot - y_dot * x_ddot) / denom

    return x, y, psi, kappa


def generate_mackey_glass(T_total, dt,
                          beta=0.2, gamma=0.1, n=10, tau=17.0,
                          x0=1.2):
    """
    Generate a Mackey-Glass time series x(t) on [0, T_total].
    Simple Euler discretization of the DDE with constant history x0.
    """
    t_start = -tau
    N_total = int((T_total - t_start) / dt) + 1
    times = t_start + np.arange(N_total) * dt

    x = np.zeros(N_total)
    x[:] = x0

    delay_steps = int(round(tau / dt))

    for i in range(delay_steps, N_total - 1):
        x_tau = x[i - delay_steps]
        dxdt = beta * x_tau / (1.0 + x_tau**n) - gamma * x[i]
        x[i + 1] = x[i] + dxdt * dt

    mask = times >= 0.0
    return times[mask], x[mask]


def make_mackey_ref(T, dt, v):
    """
    Precompute Mackey-Glass-based reference y(t) and build
    ref_mg(t, v) that returns x_ref, y_ref, psi_ref, kappa_ref.
    """
    times, y_all = generate_mackey_glass(T, dt)
    N = len(times)

    y_dot = np.zeros(N)
    y_ddot = np.zeros(N)

    for i in range(N):
        if 0 < i < N - 1:
            y_dot[i] = (y_all[i + 1] - y_all[i - 1]) / (2 * dt)
            y_ddot[i] = (y_all[i + 1] - 2 * y_all[i] + y_all[i - 1]) / (dt**2)
        elif i == 0:
            y_dot[i] = (y_all[i + 1] - y_all[i]) / dt
            y_ddot[i] = (y_all[i + 2] - 2 * y_all[i + 1] + y_all[i]) / (dt**2)
        else:
            y_dot[i] = (y_all[i] - y_all[i - 1]) / dt
            y_ddot[i] = (y_all[i] - 2 * y_all[i - 1] + y_all[i - 2]) / (dt**2)

    def ref_mg(t: float, v_local: float):
        idx = int(round(t / dt))
        idx = max(0, min(N - 1, idx))

        x_ref = v_local * t
        y_ref = y_all[idx]

        x_dot = v_local
        x_ddot = 0.0

        psi_ref = np.arctan2(y_dot[idx], x_dot)

        denom = (x_dot**2 + y_dot[idx]**2)**1.5
        if denom < 1e-8:
            kappa_ref = 0.0
        else:
            kappa_ref = (x_dot * y_ddot[idx] - y_dot[idx] * x_ddot) / denom

        return x_ref, y_ref, psi_ref, kappa_ref

    return ref_mg, times, y_all


def triangle_profile(t, A=3.0, T_p=6.0):
    """
    Symmetric triangle wave in time:
    - period T_p
    - amplitude +/- A
    Returns y, y_dot, y_ddot (y_ddot ~ 0 except at corners).
    """
    phase = (t % T_p) / T_p
    if phase < 0.25:
        y = 4 * A * phase
        y_dot = 4 * A / T_p
    elif phase < 0.75:
        y = 2 * A - 4 * A * phase
        y_dot = -4 * A / T_p
    else:
        y = -4 * A + 4 * A * phase
        y_dot = 4 * A / T_p

    y_ddot = 0.0  # ignore dirac-like corners for curvature
    return y, y_dot, y_ddot


def ref_triangle(t: float, v: float):
    """
    Triangle-wave road: x = v t, y from triangle_profile(t).
    """
    x = v * t
    y, y_dot, y_ddot = triangle_profile(t)

    x_dot = v
    x_ddot = 0.0

    psi = np.arctan2(y_dot, x_dot)

    denom = (x_dot**2 + y_dot**2)**1.5
    if denom < 1e-8:
        kappa = 0.0
    else:
        kappa = (x_dot * y_ddot - y_dot * x_ddot) / denom

    return x, y, psi, kappa


# ============================================================
# Geometry for visualization (cute car)
# ============================================================

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
    c = np.cos(yaw)
    s = np.sin(yaw)
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
    c = np.cos(yaw)
    s = np.sin(yaw)
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
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    corners_world = (R @ corners_local.T).T
    corners_world[:, 0] += center_world[0]
    corners_world[:, 1] += center_world[1]
    return corners_world


# ============================================================
# MPPI steering controller
# ============================================================

class MPPISteering:
    """
    MPPI controller for steering:
    - state: [x, y, psi]
    - control: delta (steering angle)
    - v is constant forward speed
    """

    def __init__(
        self,
        L: float,
        v: float,
        ref_fun,
        horizon_H: int = 15,
        num_samples: int = 80,
        dt: float = 0.05,
        lambda_: float = 1.0,
        sigma: float = np.deg2rad(8.0),
        delta_max: float = np.deg2rad(35.0),
        q_y: float = 8.0,
        q_psi: float = 2.0,
        r_delta: float = 0.1,
    ):
        self.L = L
        self.v = v
        self.ref_fun = ref_fun
        self.H = horizon_H
        self.K = num_samples
        self.dt = dt
        self.lambda_ = lambda_
        self.sigma = sigma
        self.delta_max = delta_max

        self.q_y = q_y
        self.q_psi = q_psi
        self.r_delta = r_delta

        # initial nominal control sequence (all zeros)
        self.u_seq = np.zeros(self.H)

    def _rollout_cost(self, x0, y0, psi0, t0, u_seq_pert):
        x = x0
        y = y0
        psi = psi0
        cost = 0.0

        for h in range(self.H):
            t_h = t0 + h * self.dt
            x_ref, y_ref, psi_ref, kappa_ref = self.ref_fun(t_h, self.v)

            dx = x - x_ref
            dy = y - y_ref
            e_y = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy
            e_psi = wrap_angle(psi - psi_ref)

            delta = np.clip(u_seq_pert[h], -self.delta_max, self.delta_max)

            cost += (
                self.q_y * e_y**2
                + self.q_psi * e_psi**2
                + self.r_delta * delta**2
            )

            x_dot = self.v * np.cos(psi)
            y_dot = self.v * np.sin(psi)
            psi_dot = self.v / self.L * np.tan(delta)

            x += x_dot * self.dt
            y += y_dot * self.dt
            psi = wrap_angle(psi + psi_dot * self.dt)

        return cost

    def control(self, x0: float, y0: float, psi0: float, t0: float) -> float:
        noise = self.sigma * np.random.randn(self.K, self.H)
        costs = np.zeros(self.K)

        for k in range(self.K):
            u_pert = self.u_seq + noise[k, :]
            costs[k] = self._rollout_cost(x0, y0, psi0, t0, u_pert)

        S_min = np.min(costs)
        weights = np.exp(-(costs - S_min) / self.lambda_)
        weights_sum = np.sum(weights) + 1e-8

        du = np.sum(weights[:, None] * noise, axis=0) / weights_sum
        self.u_seq += du

        delta = np.clip(self.u_seq[0], -self.delta_max, self.delta_max)

        self.u_seq[:-1] = self.u_seq[1:]
        self.u_seq[-1] = 0.0

        return delta


# ============================================================
# Simulation + visualization (4 paths in one window)
# ============================================================

if __name__ == "__main__":
    # Car and control parameters
    L = 2.5          # wheelbase
    W = 1.5          # track width (for viz only)
    v = 5.0          # constant speed [m/s]

    # Simulation horizon
    T = 12.0         # seconds
    dt = 0.05
    N = int(T / dt)
    ts = np.linspace(0.0, T, N)

    # Mackey–Glass ref (precomputed)
    ref_mg, mg_times, mg_y = make_mackey_ref(T, dt, v)

    # Define the four scenarios
    scenarios_def = [
        ("Circle",        ref_circle),
        ("Mountain",      ref_mountain),
        ("Mackey-Glass",  ref_mg),
        ("Triangle",      ref_triangle),
    ]

    # Colors for each scenario (body, traj)
    scenario_colors = {
        "Circle":       ("#1f77b4", "#3399ff"),
        "Mountain":     ("#d62728", "#ff6347"),
        "Mackey-Glass": ("#2ca02c", "#32cd32"),
        "Triangle":     ("#9467bd", "#ba55d3"),
    }

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs = axs.ravel()

    # Store per-scenario state + artists
    scenarios = []
    wheel_len = 0.6
    wheel_wid = 0.2

    for ax, (name, ref_fun) in zip(axs, scenarios_def):
        # precompute reference (xr, yr) for plotting
        xr = np.zeros(N)
        yr = np.zeros(N)
        for i, t in enumerate(ts):
            x_ref, y_ref, _, _ = ref_fun(t, v)
            xr[i] = x_ref
            yr[i] = y_ref

        body_color, traj_color = scenario_colors[name]

        # create car + MPPI controller
        car = KinematicBicycle(L=L)
        controller = MPPISteering(
            L=L, v=v, ref_fun=ref_fun,
            horizon_H=15, num_samples=80,
            dt=dt, lambda_=3.0,
            sigma=np.deg2rad(6.0),
            delta_max=np.deg2rad(35.0),
            q_y=10.0, q_psi=3.0, r_delta=0.05,
        )

        xs = np.zeros(N)
        ys = np.zeros(N)
        psis = np.zeros(N)

        # subplot setup
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(f"{name} – MPPI", fontsize=10)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)

        # reference path
        ax.plot(xr, yr, color="#cccccc", linestyle="--", linewidth=1.3, label="ref")
        ax.legend(fontsize=7, loc="best")

        margin = 5.0
        ax.set_xlim(xr.min() - margin, xr.max() + margin)
        ax.set_ylim(yr.min() - margin, yr.max() + margin)

        # moving trajectory line
        traj_line, = ax.plot([], [], linewidth=2.0, color=traj_color)

        # car body
        body_poly = plt.Polygon(
            car_outline(car.x, car.y, car.psi, L, W),
            closed=True,
            facecolor=body_color,
            edgecolor="black",
            alpha=0.7,
            linewidth=1.2,
        )
        ax.add_patch(body_poly)

        # wheels (black)
        centers0 = wheel_centers_world(car.x, car.y, car.psi, L, W)
        rl = plt.Polygon(wheel_polygon(centers0[0], car.psi, wheel_len, wheel_wid),
                         closed=True, facecolor="black")
        rr = plt.Polygon(wheel_polygon(centers0[1], car.psi, wheel_len, wheel_wid),
                         closed=True, facecolor="black")
        fl = plt.Polygon(wheel_polygon(centers0[2], car.psi, wheel_len, wheel_wid),
                         closed=True, facecolor="black")
        fr = plt.Polygon(wheel_polygon(centers0[3], car.psi, wheel_len, wheel_wid),
                         closed=True, facecolor="black")
        for p in (rl, rr, fl, fr):
            ax.add_patch(p)

        scenarios.append(
            {
                "name": name,
                "ax": ax,
                "car": car,
                "controller": controller,
                "xs": xs,
                "ys": ys,
                "psis": psis,
                "traj_line": traj_line,
                "body_poly": body_poly,
                "wheels": (rl, rr, fl, fr),
            }
        )

    # Animation update: step each scenario
    def update(frame_idx):
        t = frame_idx * dt
        artists = []

        for s in scenarios:
            car = s["car"]
            controller = s["controller"]
            xs = s["xs"]
            ys = s["ys"]
            psis = s["psis"]
            traj_line = s["traj_line"]
            body_poly = s["body_poly"]
            rl, rr, fl, fr = s["wheels"]

            delta = controller.control(car.x, car.y, car.psi, t)
            x, y, psi = car.step(v, delta, dt)
            xs[frame_idx] = x
            ys[frame_idx] = y
            psis[frame_idx] = psi

            traj_line.set_data(xs[:frame_idx + 1], ys[:frame_idx + 1])
            body_poly.set_xy(car_outline(x, y, psi, L, W))

            centers = wheel_centers_world(x, y, psi, L, W)
            rl.set_xy(wheel_polygon(centers[0], psi, wheel_len, wheel_wid))
            rr.set_xy(wheel_polygon(centers[1], psi, wheel_len, wheel_wid))
            fl.set_xy(wheel_polygon(centers[2], psi, wheel_len, wheel_wid))
            fr.set_xy(wheel_polygon(centers[3], psi, wheel_len, wheel_wid))

            artists.extend([traj_line, body_poly, rl, rr, fl, fr])

        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=N,
        interval=dt * 1000,
        blit=True,
    )

    fig.suptitle("Ackermann car – MPPI tracking on 4 paths (circle, mountain, Mackey-Glass, triangle)", fontsize=12)
    plt.tight_layout()
    plt.show()
