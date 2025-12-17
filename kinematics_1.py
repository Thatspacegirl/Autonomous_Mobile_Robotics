#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def unit_from_angle(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def angle_of(v):
    return np.arctan2(v[1], v[0])

# ------------------------------------------------------------
# Ackermann geometry (robot frame: x forward, y left)
# Origin at rear-axle midpoint
# ------------------------------------------------------------
def ackermann_front_wheel_angles(delta_center, L, W, eps=1e-9):
    """
    delta_center: "virtual" center steering angle (bicycle model) [rad]
    returns: (delta_FL, delta_FR) [rad]
    """
    if abs(delta_center) < eps:
        return 0.0, 0.0

    R = L / np.tan(delta_center)      # signed
    Rabs = abs(R)

    denom_inner = max(Rabs - W / 2.0, 1e-6)
    denom_outer = max(Rabs + W / 2.0, 1e-6)

    delta_inner = np.arctan(L / denom_inner)
    delta_outer = np.arctan(L / denom_outer)

    if R > 0:      # left
        return delta_inner, delta_outer
    else:          # right
        return -delta_outer, -delta_inner

def icr_location(delta_center, L, eps=1e-9):
    if abs(delta_center) < eps:
        return None
    R = L / np.tan(delta_center)
    return np.array([0.0, R])   # ICR at (0, R) in this frame

def velocity_direction_about_icr(p, icr, omega=1.0):
    r = p - icr
    v = omega * np.array([-r[1], r[0]])  # zhat x r
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0])

# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------
def main():
    # Vehicle geometry
    L = 0.50      # wheelbase [m]
    W = 0.30      # track width [m]

    wheel_len = 0.12
    arrow_len = 0.18

    # Wheel positions (origin rear axle midpoint)
    p_RL = np.array([0.0,  +W/2])
    p_RR = np.array([0.0,  -W/2])
    p_FL = np.array([L,    +W/2])
    p_FR = np.array([L,    -W/2])

    wheel_pts = {"RL": p_RL, "RR": p_RR, "FL": p_FL, "FR": p_FR}

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (forward) [m]")
    ax.set_ylabel("y (left) [m]")
    ax.grid(True)
    ax.set_title("Ackermann Kinematics (Animated)\nGreen=wheel heading, Blue=velocity dir, Red=ICR")

    # Chassis outline
    chassis = np.array([
        [0.0,   +W/2],
        [L,     +W/2],
        [L,     -W/2],
        [0.0,   -W/2],
        [0.0,   +W/2],
    ])
    ax.plot(chassis[:, 0], chassis[:, 1], linewidth=2)

    # Wheel labels
    for k, p in wheel_pts.items():
        ax.plot(p[0], p[1], "ko")
        ax.text(p[0] + 0.01, p[1] + 0.01, k, fontsize=10)

    # Artists
    wheel_lines = {}
    head_arrows = {}
    vel_arrows = {}

    for k, p in wheel_pts.items():
        ln, = ax.plot([], [], "k-", linewidth=4)   # wheel segment
        wheel_lines[k] = ln

        qh = ax.quiver(p[0], p[1], 1, 0, angles="xy", scale_units="xy", scale=1, width=0.007)
        qh.set_color("g")
        head_arrows[k] = qh

        qv = ax.quiver(p[0], p[1], 1, 0, angles="xy", scale_units="xy", scale=1, width=0.007)
        qv.set_color("b")
        vel_arrows[k] = qv

    icr_dot, = ax.plot([], [], "rx", markersize=10, mew=2)
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=11)
    err_text  = ax.text(0.02, 0.78, "", transform=ax.transAxes, va="top",
                        fontsize=11, family="monospace")

    ax.set_xlim(-0.25, L + 0.35)
    ax.set_ylim(-0.85, 0.85)

    # Steering animation settings
    amp_deg = 30.0      # max steering (center) [deg]
    period_s = 6.0      # seconds per full cycle
    fps = 30
    dt = 1.0 / fps

    def frame_to_delta(frame_idx):
        t = frame_idx * dt
        delta_deg = amp_deg * np.sin(2 * np.pi * t / period_s)
        return np.deg2rad(delta_deg), delta_deg

    def update(frame_idx):
        delta, delta_deg = frame_to_delta(frame_idx)

        dFL, dFR = ackermann_front_wheel_angles(delta, L, W)
        headings = {"RL": 0.0, "RR": 0.0, "FL": dFL, "FR": dFR}

        icr = icr_location(delta, L)

        if icr is None:
            icr_dot.set_data([], [])
            icr_str = "ICR: straight (infinity)"
            vel_dirs = {k: np.array([1.0, 0.0]) for k in wheel_pts.keys()}
        else:
            icr_dot.set_data([icr[0]], [icr[1]])
            icr_str = f"ICR: (x={icr[0]:.3f}, y={icr[1]:.3f}) m"
            vel_dirs = {k: velocity_direction_about_icr(p, icr) for k, p in wheel_pts.items()}

        # Update visuals + slip error
        err_lines = []
        for k, p in wheel_pts.items():
            th = headings[k]
            hdir = unit_from_angle(th)
            vdir = vel_dirs[k]

            slip = wrap_to_pi(angle_of(vdir) - angle_of(hdir))
            slip_deg = np.rad2deg(slip)
            err_lines.append(f"{k}: slip={slip_deg:+6.2f} deg")

            # wheel segment
            a = p - 0.5 * wheel_len * hdir
            b = p + 0.5 * wheel_len * hdir
            wheel_lines[k].set_data([a[0], b[0]], [a[1], b[1]])

            # arrows
            head_arrows[k].set_offsets([p])
            head_arrows[k].set_UVC(hdir[0] * arrow_len, hdir[1] * arrow_len)

            vel_arrows[k].set_offsets([p])
            vel_arrows[k].set_UVC(vdir[0] * arrow_len, vdir[1] * arrow_len)

        info_text.set_text(f"delta_center = {delta_deg:+.1f} deg\n{icr_str}")
        err_text.set_text("Slip angle (heading vs velocity dir):\n" + "\n".join(err_lines))

        # Return artists for blitting
        artists = [icr_dot, info_text, err_text]
        artists += list(wheel_lines.values())
        artists += list(head_arrows.values())
        artists += list(vel_arrows.values())
        return artists

    ani = FuncAnimation(fig, update, frames=int(period_s * fps), interval=1000/fps, blit=True, repeat=True)
    plt.show()

if __name__ == "__main__":
    main()
