#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass


# ================================
# Ackermann Kinematic Model
# ================================

@dataclass
class AckermannKinematics:
    wheelbase: float       # L [m]
    track_width: float     # W [m]
    x: float = 0.0         # rear axle center x
    y: float = 0.0         # rear axle center y
    yaw: float = 0.0       # heading [rad]

    def ackermann_wheel_angles(self, delta_center: float):
        """
        Given a virtual steering angle (bicycle model),
        compute inner and outer front wheel angles.
        """
        if abs(delta_center) < 1e-6:
            return 0.0, 0.0

        L = self.wheelbase
        W = self.track_width

        # Turning radius of rear axle center
        R = L / np.tan(delta_center)

        if delta_center > 0:  # left turn
            R_in = R - W / 2.0
            R_out = R + W / 2.0
        else:                 # right turn
            R_in = R + W / 2.0
            R_out = R - W / 2.0

        delta_in = np.arctan(L / R_in)
        delta_out = np.arctan(L / R_out)
        return delta_in, delta_out

    def step(self, v: float, delta_center: float, dt: float):
        """
        One kinematic integration step.
        """
        x_dot = v * np.cos(self.yaw)
        y_dot = v * np.sin(self.yaw)
        yaw_dot = v / self.wheelbase * np.tan(delta_center)

        self.x += x_dot * dt
        self.y += y_dot * dt
        self.yaw += yaw_dot * dt

        # wrap yaw to [-pi, pi]
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi

        delta_in, delta_out = self.ackermann_wheel_angles(delta_center)
        return self.x, self.y, self.yaw, delta_in, delta_out


# ================================
# Geometry helpers
# ================================

def car_outline(x, y, yaw, L, W):
    """
    Return polygon for car body.
    State (x, y, yaw) is at the rear axle center.
    """
    L_car = 1.6 * L      # a bit longer than wheelbase
    half_w = W / 2.0

    # rectangle in body frame (rear axle at origin)
    corners_body = np.array([
        [0.0,    -half_w],
        [0.0,     half_w],
        [L_car,   half_w],
        [L_car,  -half_w],
    ])

    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])

    corners_world = (R @ corners_body.T).T
    corners_world[:, 0] += x
    corners_world[:, 1] += y
    return corners_world


def wheel_polygon(center_world, theta, length, width):
    """
    Small rectangle representing a wheel, centered at center_world,
    oriented with angle theta.
    """
    half_l = length / 2.0
    half_w = width / 2.0

    # wheel rect in its own local frame
    corners_local = np.array([
        [-half_l, -half_w],
        [-half_l,  half_w],
        [ half_l,  half_w],
        [ half_l, -half_w],
    ])

    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])

    corners_world = (R @ corners_local.T).T
    corners_world[:, 0] += center_world[0]
    corners_world[:, 1] += center_world[1]
    return corners_world


def wheel_centers_world(x, y, yaw, L, W):
    """
    Centers of the four wheels in world frame.
    Order: rear-left, rear-right, front-left, front-right.
    """
    half_w = W / 2.0

    # positions in body frame
    centers_body = np.array([
        [0.0, +half_w],   # rear-left
        [0.0, -half_w],   # rear-right
        [L,   +half_w],   # front-left
        [L,   -half_w],   # front-right
    ])

    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])

    centers_world = (R @ centers_body.T).T
    centers_world[:, 0] += x
    centers_world[:, 1] += y
    return centers_world


# ================================
# Simulation (nice S-curve path)
# ================================

def simulate_trajectory():
    """
    Smooth S-curve: steering angle is sinusoidal.
    """
    L = 2.5
    W = 1.5
    kin = AckermannKinematics(wheelbase=L, track_width=W)

    dt = 0.02
    T = 20.0
    steps = int(T / dt)

    xs = np.zeros(steps)
    ys = np.zeros(steps)
    yaws = np.zeros(steps)
    delta_fl = np.zeros(steps)
    delta_fr = np.zeros(steps)

    v = 2.0                         # constant speed [m/s]
    delta_max = np.deg2rad(25.0)    # max steering
    T_turn = 10.0                   # sinusoid period

    for k in range(steps):
        t = k * dt
        # smooth left-right S-turn
        delta_center = delta_max * np.sin(2.0 * np.pi * t / T_turn)

        x, y, yaw, d_in, d_out = kin.step(v, delta_center, dt)
        xs[k], ys[k], yaws[k] = x, y, yaw

        # Assign inner / outer to left/right wheels
        if abs(delta_center) < 1e-6:
            delta_fl[k] = 0.0
            delta_fr[k] = 0.0
        elif delta_center > 0:  # left turn
            delta_fl[k] = d_in
            delta_fr[k] = d_out
        else:                   # right turn
            delta_fl[k] = d_out
            delta_fr[k] = d_in

    return xs, ys, yaws, delta_fl, delta_fr, L, W, dt


# ================================
# Animation
# ================================

def animate_trajectory(xs, ys, yaws, delta_fl, delta_fr, L, W, dt):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal', 'box')

    margin = 3.0
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Ackermann Car  (S-curve path)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Full trajectory
    ax.plot(xs, ys, linestyle="--", alpha=0.4, label="Planned path")
    ax.legend()

    # Moving path
    path_line, = ax.plot([], [], linewidth=2.0)

    # Car body polygon
    body_poly = plt.Polygon(
        car_outline(xs[0], ys[0], yaws[0], L, W),
        closed=True, alpha=0.5
    )
    ax.add_patch(body_poly)

    #
    wheel_len = 0.7
    wheel_wid = 0.25

    centers0 = wheel_centers_world(xs[0], ys[0], yaws[0], L, W)
    rear_left_poly = plt.Polygon(
        wheel_polygon(centers0[0], yaws[0], wheel_len, wheel_wid),
        closed=True, alpha=0.9
    )
    rear_right_poly = plt.Polygon(
        wheel_polygon(centers0[1], yaws[0], wheel_len, wheel_wid),
        closed=True, alpha=0.9
    )
    front_left_poly = plt.Polygon(
        wheel_polygon(centers0[2], yaws[0] + delta_fl[0], wheel_len, wheel_wid),
        closed=True, alpha=0.9
    )
    front_right_poly = plt.Polygon(
        wheel_polygon(centers0[3], yaws[0] + delta_fr[0], wheel_len, wheel_wid),
        closed=True, alpha=0.9
    )

    for poly in [rear_left_poly, rear_right_poly, front_left_poly, front_right_poly]:
        ax.add_patch(poly)

    def init():
        path_line.set_data([], [])
        body_poly.set_xy(car_outline(xs[0], ys[0], yaws[0], L, W))

        centers = wheel_centers_world(xs[0], ys[0], yaws[0], L, W)
        rear_left_poly.set_xy(
            wheel_polygon(centers[0], yaws[0], wheel_len, wheel_wid)
        )
        rear_right_poly.set_xy(
            wheel_polygon(centers[1], yaws[0], wheel_len, wheel_wid)
        )
        front_left_poly.set_xy(
            wheel_polygon(centers[2], yaws[0] + delta_fl[0], wheel_len, wheel_wid)
        )
        front_right_poly.set_xy(
            wheel_polygon(centers[3], yaws[0] + delta_fr[0], wheel_len, wheel_wid)
        )
        return (path_line, body_poly,
                rear_left_poly, rear_right_poly,
                front_left_poly, front_right_poly)

    def update(i):
        # path so far
        path_line.set_data(xs[:i+1], ys[:i+1])

        # body
        body_poly.set_xy(car_outline(xs[i], ys[i], yaws[i], L, W))

        # wheels
        centers = wheel_centers_world(xs[i], ys[i], yaws[i], L, W)
        rear_left_poly.set_xy(
            wheel_polygon(centers[0], yaws[i], wheel_len, wheel_wid)
        )
        rear_right_poly.set_xy(
            wheel_polygon(centers[1], yaws[i], wheel_len, wheel_wid)
        )
        front_left_poly.set_xy(
            wheel_polygon(centers[2], yaws[i] + delta_fl[i], wheel_len, wheel_wid)
        )
        front_right_poly.set_xy(
            wheel_polygon(centers[3], yaws[i] + delta_fr[i], wheel_len, wheel_wid)
        )

        return (path_line, body_poly,
                rear_left_poly, rear_right_poly,
                front_left_poly, front_right_poly)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(xs),
        init_func=init,
        interval=dt * 1000.0,
        blit=True
    )

    plt.show()


if __name__ == "__main__":
    xs, ys, yaws, delta_fl, delta_fr, L, W, dt = simulate_trajectory()
    animate_trajectory(xs, ys, yaws, delta_fl, delta_fr, L, W, dt)
