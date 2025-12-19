#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass

# ============================
# Helper functions / classes
# ============================

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


# ============================
# Reference paths
# ============================

def ref_lane_change(t: float, v: float, Dy: float = 3.0, T_c: float = 5.0):
    x = v * t
    if t <= T_c:
        y = 0.5 * Dy * (1 - np.cos(np.pi * t / T_c))
        y_dot = 0.5 * Dy * (np.pi / T_c) * np.sin(np.pi * t / T_c)
    else:
        y = Dy
        y_dot = 0.0
    x_dot = v
    psi = np.arctan2(y_dot, x_dot)
    return x, y, psi


def ref_circle(t: float, v: float, R: float = 10.0):
    w = v / R
    th = w * t
    x = R * np.sin(th)
    y = R * (1 - np.cos(th))
    x_dot = v * np.cos(th)
    y_dot = v * np.sin(th)
    psi = np.arctan2(y_dot, x_dot)
    return x, y, psi


def ref_wave(t: float, v: float, A: float = 2.0, T_s: float = 10.0):
    x = v * t
    w = 2 * np.pi / T_s
    y = A * np.sin(w * t)
    x_dot = v
    y_dot = A * w * np.cos(w * t)
    psi = np.arctan2(y_dot, x_dot)
    return x, y, psi


def ref_complicated(t: float, v: float, A: float = 2.0, T_s: float = 12.0):
    x = v * t
    w = 2 * np.pi / T_s
    y = A * np.sin(w * t) + 0.5 * A * np.sin(3 * w * t)
    x_dot = v
    y_dot = A * w * np.cos(w * t) + 0.5 * A * 3 * w * np.cos(3 * w * t)
    psi = np.arctan2(y_dot, x_dot)
    return x, y, psi


# ============================
# PID steering controller
# ============================

class PIDSteering:
    def __init__(
        self,
        Kp_y: float = 1.0,
        Ki_y: float = 0.2,
        Kd_y: float = 0.1,
        Kp_psi: float = 1.0,
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

    def control(self, e_y: float, e_psi: float, dt: float) -> float:
        self.e_y_int += e_y * dt
        if self.first:
            de_y = 0.0
            self.first = False
        else:
            de_y = (e_y - self.e_y_prev) / dt
        self.e_y_prev = e_y

        delta = -(
            self.Kp_y * e_y
            + self.Ki_y * self.e_y_int
            + self.Kd_y * de_y
            + self.Kp_psi * e_psi
        )
        return np.clip(delta, -self.delta_max, self.delta_max)


# ============================
# Car geometry
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


# ============================
# Simulate a path
# ============================

def simulate_path(ref_fun, name: str, L=2.5, v=5.0, T=10.0, n_frames=200):
    """
    Simulate with fixed number of frames (n_frames) → lighter GIF.
    """
    dt = T / n_frames
    car = KinematicBicycle(L=L)
    pid = PIDSteering()
    pid.reset()

    xs = np.zeros(n_frames)
    ys = np.zeros(n_frames)
    psis = np.zeros(n_frames)
    xr = np.zeros(n_frames)
    yr = np.zeros(n_frames)

    for k in range(n_frames):
        t = k * dt
        x_ref, y_ref, psi_ref = ref_fun(t, v)
        xr[k], yr[k] = x_ref, y_ref

        dx = car.x - x_ref
        dy = car.y - y_ref
        e_y = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy
        e_psi = wrap_angle(car.psi - psi_ref)

        delta = pid.control(e_y, e_psi, dt)
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
# Main: build animation + save GIF
# ============================

if __name__ == "__main__":
    L = 2.5
    W = 1.5
    v = 5.0
    T = 10.0         # shorter sim
    n_frames = 200   # fewer frames → lighter GIF

    path_defs = [
        ("Lane change",  ref_lane_change),
        ("Circle",       ref_circle),
        ("Wave",         ref_wave),
        ("Complicated",  ref_complicated),
    ]

    sim_results = [
        simulate_path(ref_fun, name, L=L, v=v, T=T, n_frames=n_frames)
        for name, ref_fun in path_defs
    ]

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # smaller figure
    axs = axs.ravel()

    cars = []
    wheel_len = 0.5
    wheel_wid = 0.18

    for ax, res in zip(axs, sim_results):
        name = res["name"]
        xs, ys, psis = res["xs"], res["ys"], res["psis"]
        xr, yr = res["xr"], res["yr"]

        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)

        ax.plot(xr, yr, "k--", linewidth=1.0, label="ref")
        ax.legend(fontsize=7, loc="best")

        margin = 2.0
        ax.set_xlim(min(xs.min(), xr.min()) - margin,
                    max(xs.max(), xr.max()) + margin)
        ax.set_ylim(min(ys.min(), yr.min()) - margin,
                    max(ys.max(), yr.max()) + margin)

        path_line, = ax.plot([], [], linewidth=1.5)

        body_poly = plt.Polygon(
            car_outline(xs[0], ys[0], psis[0], L, W),
            closed=True,
            alpha=0.4,
        )
        ax.add_patch(body_poly)

        centers0 = wheel_centers_world(xs[0], ys[0], psis[0], L, W)
        rl = plt.Polygon(
            wheel_polygon(centers0[0], psis[0], wheel_len, wheel_wid),
            closed=True,
            alpha=0.9,
        )
        rr = plt.Polygon(
            wheel_polygon(centers0[1], psis[0], wheel_len, wheel_wid),
            closed=True,
            alpha=0.9,
        )
        fl = plt.Polygon(
            wheel_polygon(centers0[2], psis[0], wheel_len, wheel_wid),
            closed=True,
            alpha=0.9,
        )
        fr = plt.Polygon(
            wheel_polygon(centers0[3], psis[0], wheel_len, wheel_wid),
            closed=True,
            alpha=0.9,
        )
        for p in (rl, rr, fl, fr):
            ax.add_patch(p)

        cars.append((path_line, body_poly, (rl, rr, fl, fr), xs, ys, psis))

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
        # blit=False → we don't need to return artists, but we can:
        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=50,     # ms
        blit=False,      # IMPORTANT: easier for Pillow
    )

    fig.suptitle("Ackermann car – PID tracking on 4 paths", fontsize=11)
    plt.tight_layout()

    # Save as GIF with Pillow
    writer = PillowWriter(fps=20)
    ani.save("ackermann_4paths.gif", writer=writer, dpi=80)  # smaller dpi → less memory

    # Also show interactively
    plt.show()
