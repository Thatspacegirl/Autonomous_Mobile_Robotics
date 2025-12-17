#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =========================================================
# Helpers
# =========================================================

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# =========================================================
# Kinematic bicycle (Ackermann)
# =========================================================

@dataclass
class KinematicBicycle:
    L: float              # wheelbase
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0      # heading

    def step(self, v, delta, dt):
        x_dot = v * np.cos(self.psi)
        y_dot = v * np.sin(self.psi)
        psi_dot = v / self.L * np.tan(delta)

        self.x  += x_dot * dt
        self.y  += y_dot * dt
        self.psi = wrap_angle(self.psi + psi_dot * dt)
        return self.x, self.y, self.psi


# =========================================================
# Reference paths: (x_r, y_r, psi_r, kappa_r)
# =========================================================

def ref_straight(t, v):
    x = v * t
    y = 0.0
    psi = 0.0
    kappa = 0.0
    return x, y, psi, kappa


def ref_circle(t, v, R=10.0):
    w = v / R
    th = w * t
    x = R * np.sin(th)
    y = R * (1 - np.cos(th))
    psi = th
    kappa = 1.0 / R
    return x, y, psi, kappa


def ref_s_curve(t, v, A=2.0, T_s=8.0):
    x = v * t
    w = 2 * np.pi / T_s
    y = A * np.sin(w * t)

    x_dot = v
    y_dot = A * w * np.cos(w * t)
    psi = np.arctan2(y_dot, x_dot)

    x_ddot = 0.0
    y_ddot = -A * w**2 * np.sin(w * t)
    denom = (x_dot**2 + y_dot**2)**1.5
    kappa = 0.0 if denom < 1e-6 else (x_dot * y_ddot - y_dot * x_ddot) / denom
    return x, y, psi, kappa


def ref_lane_change(t, v, Dy=3.0, T_c=5.0):
    x = v * t
    if t <= T_c:
        y = 0.5 * Dy * (1 - np.cos(np.pi * t / T_c))
        y_dot = 0.5 * Dy * (np.pi / T_c) * np.sin(np.pi * t / T_c)
        y_ddot = 0.5 * Dy * (np.pi / T_c)**2 * np.cos(np.pi * t / T_c)
    else:
        y = Dy
        y_dot = 0.0
        y_ddot = 0.0

    x_dot = v
    x_ddot = 0.0
    psi = np.arctan2(y_dot, x_dot)
    denom = (x_dot**2 + y_dot**2)**1.5
    kappa = 0.0 if denom < 1e-6 else (x_dot * y_ddot - y_dot * x_ddot) / denom
    return x, y, psi, kappa


# =========================================================
# PID steering controller (only)
# =========================================================

class PIDSteering:
    """
    Steering PID on lateral error e_y plus P on heading error e_psi.

    delta = - (Kp_y * e_y + Ki_y * int(e_y) + Kd_y * de_y/dt)
            - Kp_psi * e_psi
    """
    def __init__(self,
                 Kp_y=0.8, Ki_y=0.2, Kd_y=0.05,
                 Kp_psi=1.0,
                 delta_max=np.deg2rad(35.0)):
        self.Kp_y = Kp_y
        self.Ki_y = Ki_y
        self.Kd_y = Kd_y
        self.Kp_psi = Kp_psi
        self.delta_max = delta_max

        self.e_y_int = 0.0
        self.e_y_prev = 0.0
        self.first_call = True

    def reset(self):
        self.e_y_int = 0.0
        self.e_y_prev = 0.0
        self.first_call = True

    def control(self, e_y, e_psi, dt):
        # integral on lateral error
        self.e_y_int += e_y * dt

        # derivative on lateral error
        if self.first_call:
            de_y = 0.0
            self.first_call = False
        else:
            de_y = (e_y - self.e_y_prev) / dt
        self.e_y_prev = e_y

        delta = -(self.Kp_y * e_y +
                  self.Ki_y * self.e_y_int +
                  self.Kd_y * de_y +
                  self.Kp_psi * e_psi)

        return np.clip(delta, -self.delta_max, self.delta_max)


# =========================================================
# Simulation with PID only
# =========================================================

def simulate_path(name, ref_fun, controller,
                  v=5.0, L=2.5, T=12.0, dt=0.01):
    car = KinematicBicycle(L=L)
    controller.reset()

    N = int(T / dt)
    t_hist  = np.zeros(N)
    ey_hist = np.zeros(N)
    x_hist  = np.zeros(N)
    y_hist  = np.zeros(N)
    xr_hist = np.zeros(N)
    yr_hist = np.zeros(N)

    for k in range(N):
        t = k * dt
        t_hist[k] = t

        # reference
        x_r, y_r, psi_r, _ = ref_fun(t, v)
        xr_hist[k], yr_hist[k] = x_r, y_r

        # errors in PATH frame (important)
        dx = car.x - x_r
        dy = car.y - y_r
        e_y = -np.sin(psi_r) * dx + np.cos(psi_r) * dy
        e_psi = wrap_angle(car.psi - psi_r)

        ey_hist[k] = e_y

        # PID steering (no feedforward)
        delta = controller.control(e_y, e_psi, dt)

        # kinematic update
        x, y, psi = car.step(v, delta, dt)
        x_hist[k], y_hist[k] = x, y

    return {
        "name": name,
        "t": t_hist,
        "e_y": ey_hist,
        "x": x_hist, "y": y_hist,
        "x_r": xr_hist, "y_r": yr_hist,
    }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    L = 2.5
    v = 5.0
    T = 12.0
    dt = 0.01

    pid = PIDSteering(
        Kp_y=0.8,
        Ki_y=0.2,
        Kd_y=0.05,
        Kp_psi=1.0
    )

    paths = [
        ("Straight",    lambda t, v: ref_straight(t, v)),
        ("Circle",      lambda t, v: ref_circle(t, v, R=10.0)),
        ("S-curve",     lambda t, v: ref_s_curve(t, v, A=2.0, T_s=8.0)),
        ("Lane change", lambda t, v: ref_lane_change(t, v, Dy=3.0, T_c=5.0)),
    ]

    fig_traj, axs_traj = plt.subplots(2, 2, figsize=(12, 8))
    fig_err,  axs_err  = plt.subplots(2, 2, figsize=(12, 8))
    axs_traj = axs_traj.ravel()
    axs_err  = axs_err.ravel()

    for i, (name, ref_fun) in enumerate(paths):
        res = simulate_path(name, ref_fun, pid, v=v, L=L, T=T, dt=dt)

        # trajectories
        ax_t = axs_traj[i]
        ax_t.plot(res["x_r"], res["y_r"], "k--", label="reference")
        ax_t.plot(res["x"], res["y"], label="PID")
        ax_t.set_title(f"{name} – trajectories")
        ax_t.set_aspect("equal", "box")
        ax_t.grid(True)
        ax_t.legend()

        # lateral error
        ax_e = axs_err[i]
        ax_e.plot(res["t"], res["e_y"])
        ax_e.set_title(f"{name} – lateral error")
        ax_e.set_xlabel("time [s]")
        ax_e.set_ylabel("e_y [m]")
        ax_e.grid(True)

        rms = np.sqrt(np.mean(res["e_y"]**2))
        print(f"{name}: RMS lateral error = {rms:.3f} m")

    fig_traj.suptitle("Path tracking with PID steering only")
    fig_err.suptitle("Lateral error with PID steering only")
    plt.tight_layout()
    plt.show()
