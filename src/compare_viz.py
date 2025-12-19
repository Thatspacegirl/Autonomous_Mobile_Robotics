#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from scipy.linalg import solve_continuous_are

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
# Reference paths: return (x_ref, y_ref, psi_ref, kappa_ref)
# ============================================================

def ref_sine(t: float, v: float, A: float = 3.0, T_s: float = 8.0):
    """Sine wave road."""
    x = v * t
    w = 2 * np.pi / T_s
    y = A * np.sin(w * t)

    x_dot = v
    y_dot = A * w * np.cos(w * t)

    x_ddot = 0.0
    y_ddot = -A * w**2 * np.sin(w * t)

    psi = np.arctan2(y_dot, x_dot)

    denom = (x_dot**2 + y_dot**2) ** 1.5
    if denom < 1e-8:
        kappa = 0.0
    else:
        kappa = (x_dot * y_ddot - y_dot * x_ddot) / denom

    return x, y, psi, kappa


def make_random_fourier_ref(T: float, K: int = 3, A_scale: float = 3.0, seed: int = 0):
    """
    Random Fourier road:
        y(t) = Σ_k [a_k sin(k ω t) + b_k cos(k ω t)],
        ω = 2π / T
    """
    rng = np.random.default_rng(seed)
    w = 2 * np.pi / T

    # random coefficients
    a = A_scale * rng.normal(size=K) / np.arange(1, K + 1)
    b = A_scale * rng.normal(size=K) / np.arange(1, K + 1)

    def ref_random(t: float, v: float):
        x = v * t

        ks = np.arange(1, K + 1, dtype=float)
        kwt = np.outer(ks, t * w)  # shape (K,) effectively, but keeps formula explicit

        sin_kwt = np.sin(ks * w * t)
        cos_kwt = np.cos(ks * w * t)

        y = np.sum(a * sin_kwt + b * cos_kwt)

        # first derivative
        y_dot = np.sum(a * ks * w * cos_kwt - b * ks * w * sin_kwt)

        # second derivative
        y_ddot = np.sum(-a * (ks * w) ** 2 * sin_kwt - b * (ks * w) ** 2 * cos_kwt)

        x_dot = v
        x_ddot = 0.0

        psi = np.arctan2(y_dot, x_dot)

        denom = (x_dot**2 + y_dot**2) ** 1.5
        if denom < 1e-8:
            kappa = 0.0
        else:
            kappa = (x_dot * y_ddot - y_dot * x_ddot) / denom

        return x, y, psi, kappa

    return ref_random


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
    """Mountain road: x = v t, y from double-Gaussian profile."""
    x = v * t
    y, y_dot, y_ddot = mountain_profile(t)

    x_dot = v
    x_ddot = 0.0

    psi = np.arctan2(y_dot, x_dot)

    denom = (x_dot**2 + y_dot**2) ** 1.5
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
# Controllers
# ============================================================

class PIDSteering:
    """
    Aggressive PID on lateral error and heading error,
    with curvature feedforward (delta_ff).
    """

    def __init__(
        self,
        Kp_y: float = 3.0,
        Ki_y: float = 0.4,
        Kd_y: float = 0.8,
        Kp_psi: float = 5.0,
        delta_max: float = np.deg2rad(35.0),
    ):
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

    def control(self, e_y: float, e_psi: float, delta_ff: float, dt: float) -> float:
        self.e_y_int += e_y * dt

        if self.first:
            de_y = 0.0
            self.first = False
        else:
            de_y = (e_y - self.e_y_prev) / dt
        self.e_y_prev = e_y

        # Aggressive tracking around curvature feedforward
        delta_fb = (
            self.Kp_y * e_y
            + self.Ki_y * self.e_y_int
            + self.Kd_y * de_y
            + self.Kp_psi * e_psi
        )
        delta = delta_ff - delta_fb
        return np.clip(delta, -self.delta_max, self.delta_max)


class LQRSteering:
    """
    LQR on linearized lateral dynamics:

      x = [e_y, e_psi]^T
      x_dot = A x + B u,  u = delta_tilde

      A = [[0,  v],
           [0,  0]]
      B = [[0],
           [v / L]]

    delta = delta_ff + delta_tilde; delta_ff from curvature.
    """

    def __init__(
        self,
        v: float,
        L: float,
        Q=None,
        R=None,
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
            # aggressive tracking weights
            Q = np.diag([30.0, 10.0])
        if R is None:
            R = np.array([[0.2]])
        elif np.isscalar(R):
            R = np.array([[R]])

        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P  # 1x2

    def control(self, e_y: float, e_psi: float, delta_ff: float) -> float:
        x = np.array([[e_y],
                      [e_psi]])
        u = -self.K @ x     # delta_tilde
        delta = delta_ff + u.item()
        return np.clip(delta, -self.delta_max, self.delta_max)


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
        horizon_H: int = 12,
        num_samples: int = 60,
        dt: float = 0.05,
        lambda_: float = 2.0,
        sigma: float = np.deg2rad(8.0),
        delta_max: float = np.deg2rad(35.0),
        q_y: float = 10.0,
        q_psi: float = 3.0,
        r_delta: float = 0.05,
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
# Simulation + visualization (4 paths, 3 controllers each)
# ============================================================

if __name__ == "__main__":
    # Car and control parameters
    L = 2.5          # wheelbase
    W = 1.5          # track width (for viz only)
    v = 5.0          # constant speed [m/s]

    # Simulation horizon
    T = 10.0         # seconds
    dt = 0.05
    N = int(T / dt)
    ts = np.linspace(0.0, T, N)

    # Random Fourier reference (R2)
    ref_random = make_random_fourier_ref(T, K=3, A_scale=3.0, seed=42)

    # Define the four scenarios
    scenarios_def = [
        ("Sine",          ref_sine),
        ("Random Fourier", ref_random),
        ("Circle",        ref_circle),
        ("Mountain",      ref_mountain),
    ]

    # Colors per controller (same across subplots)
    ctrl_colors = {
        "PID":  "#1f77b4",  # blue
        "LQR":  "#d62728",  # red
        "MPPI": "#2ca02c",  # green
    }

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    # Each scenario will hold controllers, cars, etc.
    scenarios = []
    wheel_len = 0.6
    wheel_wid = 0.2

    for ax, (name, ref_fun) in zip(axs, scenarios_def):
        # Precompute reference (xr, yr) for plotting
        xr = np.zeros(N)
        yr = np.zeros(N)
        for i, t in enumerate(ts):
            x_ref, y_ref, _, _ = ref_fun(t, v)
            xr[i] = x_ref
            yr[i] = y_ref

        # Subplot setup
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(f"{name}", fontsize=10)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)

        # reference path
        ax.plot(xr, yr, color="#cccccc", linestyle="--", linewidth=1.3, label="ref")
        ax.legend(fontsize=7, loc="best")

        margin = 5.0
        ax.set_xlim(xr.min() - margin, xr.max() + margin)
        ax.set_ylim(yr.min() - margin, yr.max() + margin)

        # For this scenario, create 3 controllers: PID, LQR, MPPI
        controllers_dict = {}
        for ctrl_name in ["PID", "LQR", "MPPI"]:
            color = ctrl_colors[ctrl_name]

            car = KinematicBicycle(L=L)

            if ctrl_name == "PID":
                ctrl = PIDSteering(
                    Kp_y=3.0, Ki_y=0.4, Kd_y=0.8,
                    Kp_psi=5.0,
                    delta_max=np.deg2rad(35.0),
                )
            elif ctrl_name == "LQR":
                ctrl = LQRSteering(
                    v=v, L=L,
                    Q=np.diag([30.0, 10.0]),
                    R=0.2,
                    delta_max=np.deg2rad(35.0),
                )
            else:  # MPPI
                ctrl = MPPISteering(
                    L=L, v=v, ref_fun=ref_fun,
                    horizon_H=12, num_samples=60,
                    dt=dt, lambda_=2.0,
                    sigma=np.deg2rad(8.0),
                    delta_max=np.deg2rad(35.0),
                    q_y=10.0, q_psi=3.0, r_delta=0.05,
                )

            xs = np.zeros(N)
            ys = np.zeros(N)
            psis = np.zeros(N)

            # trajectory line
            traj_line, = ax.plot([], [], linewidth=1.8, color=color, label=ctrl_name)

            # car body
            body_poly = plt.Polygon(
                car_outline(car.x, car.y, car.psi, L, W),
                closed=True,
                facecolor=color,
                edgecolor="black",
                alpha=0.7,
                linewidth=1.2,
            )
            ax.add_patch(body_poly)

            # wheels (black)
            centers0 = wheel_centers_world(car.x, car.y, car.psi, L, W)
            rl = plt.Polygon(
                wheel_polygon(centers0[0], car.psi, wheel_len, wheel_wid),
                closed=True, facecolor="black"
            )
            rr = plt.Polygon(
                wheel_polygon(centers0[1], car.psi, wheel_len, wheel_wid),
                closed=True, facecolor="black"
            )
            fl = plt.Polygon(
                wheel_polygon(centers0[2], car.psi, wheel_len, wheel_wid),
                closed=True, facecolor="black"
            )
            fr = plt.Polygon(
                wheel_polygon(centers0[3], car.psi, wheel_len, wheel_wid),
                closed=True, facecolor="black"
            )
            for p in (rl, rr, fl, fr):
                ax.add_patch(p)

            controllers_dict[ctrl_name] = {
                "car": car,
                "ctrl": ctrl,
                "xs": xs,
                "ys": ys,
                "psis": psis,
                "traj_line": traj_line,
                "body_poly": body_poly,
                "wheels": (rl, rr, fl, fr),
                "color": color,
            }

        # add a small legend for controllers (once per subplot)
        # (just reuse one line's label)
        ax.legend(fontsize=7, loc="upper right")

        scenarios.append(
            {
                "name": name,
                "ax": ax,
                "ref_fun": ref_fun,
                "xr": xr,
                "yr": yr,
                "controllers": controllers_dict,
            }
        )

    # Animation update: step each scenario & controller
    def update(frame_idx):
        t = frame_idx * dt
        artists = []

        for s in scenarios:
            ref_fun = s["ref_fun"]
            for ctrl_name, cdata in s["controllers"].items():
                car = cdata["car"]
                ctrl = cdata["ctrl"]
                xs = cdata["xs"]
                ys = cdata["ys"]
                psis = cdata["psis"]
                traj_line = cdata["traj_line"]
                body_poly = cdata["body_poly"]
                rl, rr, fl, fr = cdata["wheels"]

                if ctrl_name in ["PID", "LQR"]:
                    # need e_y, e_psi, delta_ff
                    x_ref, y_ref, psi_ref, kappa_ref = ref_fun(t, v)
                    dx = car.x - x_ref
                    dy = car.y - y_ref
                    e_y = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy
                    e_psi = wrap_angle(car.psi - psi_ref)
                    delta_ff = np.arctan(L * kappa_ref)

                    if ctrl_name == "PID":
                        delta = ctrl.control(e_y, e_psi, delta_ff, dt)
                    else:
                        delta = ctrl.control(e_y, e_psi, delta_ff)
                else:
                    # MPPI uses full state
                    delta = ctrl.control(car.x, car.y, car.psi, t)

                x, y, psi = car.step(v, delta, dt)
                xs[frame_idx] = x
                ys[frame_idx] = y
                psis[frame_idx] = psi

                # update trajectory
                traj_line.set_data(xs[:frame_idx + 1], ys[:frame_idx + 1])

                # update car geometry
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

    fig.suptitle("Ackermann – aggressive PID vs LQR vs MPPI on 4 paths",
                 fontsize=14)
    plt.tight_layout()
    plt.show()
