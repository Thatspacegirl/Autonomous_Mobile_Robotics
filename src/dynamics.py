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
    """
    Kinematic bicycle model + Ackermann wheel-angle geometry.
    State is at rear axle center: (x, y, yaw).
    """
    wheelbase: float       # L [m]
    track_width: float     # W [m]

    # Vehicle state in world frame
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0       # [rad]

    def ackermann_wheel_angles(self, delta_center: float):
        """
        Given a 'virtual' steering angle delta_center (bicycle model),
        compute inner and outer front wheel steering angles.

        Positive delta_center -> left turn.
        """
        if np.abs(delta_center) < 1e-6:
            # Straight line: inner = outer = 0
            return 0.0, 0.0

        L = self.wheelbase
        W = self.track_width

        # Turning radius of rear axle center
        R = L / np.tan(delta_center)

        if delta_center > 0:  # left turn
            R_in = R - W / 2.0
            R_out = R + W / 2.0
        else:  # right turn
            R_in = R + W / 2.0
            R_out = R - W / 2.0

        delta_in = np.arctan(L / R_in)
        delta_out = np.arctan(L / R_out)
        return delta_in, delta_out

    def step(self, v: float, delta_center: float, dt: float):
        """
        Integrate kinematic bicycle model one step.

        v: longitudinal speed [m/s]
        delta_center: 'virtual' steering angle [rad]
        dt: timestep [s]
        """
        # Derivatives
        x_dot = v * np.cos(self.yaw)
        y_dot = v * np.sin(self.yaw)
        yaw_dot = v / self.wheelbase * np.tan(delta_center)

        # Euler integration
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.yaw += yaw_dot * dt

        # Normalize yaw to [-pi, pi]
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi

        # Also return real wheel angles if needed
        delta_in, delta_out = self.ackermann_wheel_angles(delta_center)
        return self.x, self.y, self.yaw, delta_in, delta_out


# ================================
# Visualization helpers
# ================================

def car_outline(x, y, yaw, L, W):
    """
    Compute the 2D polygon for a simple car body.
    State (x, y, yaw) is at the REAR axle center.
    We draw a rectangle from rear axle to some length forward.
    """
    # Choose a car length slightly larger than wheelbase
    L_car = 1.5 * L
    half_width = W / 2.0

    # Car corners in body frame (rear axle at (0, 0))
    # Order: rear-right, rear-left, front-left, front-right
    corners_body = np.array([
        [0.0,        -half_width],
        [0.0,         half_width],
        [L_car,       half_width],
        [L_car,      -half_width],
    ])

    # Rotation matrix
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])

    # Rotate + translate
    corners_world = (R @ corners_body.T).T
    corners_world[:, 0] += x
    corners_world[:, 1] += y

    return corners_world


# ================================
# Simulation + animation
# ================================

def simulate_trajectory():
    """
    Run a simple simulation:
    - accelerate straight
    - then turn with constant steering
    - then straighten again
    Returns arrays for x, y, yaw over time.
    """
    # Vehicle params
    L = 2.5   # wheelbase [m]
    W = 1.5   # track width [m]
    kin = AckermannKinematics(wheelbase=L, track_width=W)

    dt = 0.02
    T = 12.0
    steps = int(T / dt)

    xs = np.zeros(steps)
    ys = np.zeros(steps)
    yaws = np.zeros(steps)
    deltas = np.zeros(steps)

    v = 0.0
    for k in range(steps):
        t = k * dt

        # Simple speed profile: accelerate then hold
        if t < 3.0:
            v = 0.5 * t  # ramp up
        else:
            v = 1.5      # constant

        # Simple steering profile:
        # 0-3s straight, 3-8s turning, 8-12s straight
        if 3.0 <= t < 8.0:
            delta = np.deg2rad(20.0)  # 20 deg left
        else:
            delta = 0.0

        xs[k], ys[k], yaws[k], _, _ = kin.step(v, delta, dt)
        deltas[k] = delta

    return xs, ys, yaws, deltas, L, W, dt


def animate_trajectory(xs, ys, yaws, L, W, dt):
    """
    Create an animation of the car moving along the trajectory.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal', 'box')

    # Pad around trajectory for nicer view
    margin = 2.0
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Ackermann Kinematic Model Trajectory")

    # Plot full path
    ax.plot(xs, ys, '--', alpha=0.5, label="Path")
    ax.legend()

    # Moving objects: car + traced path up to current time
    path_line, = ax.plot([], [], lw=2)
    car_patch = plt.Polygon(car_outline(xs[0], ys[0], yaws[0], L, W),
                            closed=True, alpha=0.4)

    ax.add_patch(car_patch)

    def init():
        path_line.set_data([], [])
        car_patch.set_xy(car_outline(xs[0], ys[0], yaws[0], L, W))
        return path_line, car_patch

    def update(i):
        # Update trajectory up to i
        path_line.set_data(xs[:i+1], ys[:i+1])

        # Update car polygon
        poly = car_outline(xs[i], ys[i], yaws[i], L, W)
        car_patch.set_xy(poly)

        return path_line, car_patch

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
    xs, ys, yaws, deltas, L, W, dt = simulate_trajectory()
    animate_trajectory(xs, ys, yaws, L, W, dt)
