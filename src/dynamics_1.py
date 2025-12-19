#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def t_hat(th):
    return np.array([np.cos(th), np.sin(th)])

def n_hat(th):
    return np.array([-np.sin(th), np.cos(th)])

def rk4_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + 0.5*dt*k1)
    k3 = f(y + 0.5*dt*k2)
    k4 = f(y + dt*k3)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def main():
    # --- parameters ---
    m = 12.0     # kg
    I = 1.0      # kg*m^2
    L_draw = 0.6 # m (just for drawing rectangle)
    W_draw = 0.35

    # --- sim ---
    dt = 1/60
    T  = 18.0
    N  = int(T/dt)
    t  = np.arange(N)*dt

    # state y = [x, y, theta, xdot, ydot, thetadot]
    y = np.array([0.0, 0.0, 0.0,
                  1.2, 0.0, 0.0], dtype=float)

    X = np.zeros((N, 6))
    Lam = np.zeros(N)

    # input profiles (edit freely)
    def inputs(tt):
        # longitudinal force (N)
        f1 = 10.0 + 4.0*np.sin(2*np.pi*tt/5.0)
        # yaw torque (N*m)
        f2 = 0.8*np.sin(2*np.pi*tt/3.5)
        return f1, f2

    def dynamics(y, tt):
        x, ypos, th, xd, yd, rd = y
        f1, f2 = inputs(tt)

        # longitudinal speed v = t_hat^T [xd, yd]
        v = t_hat(th).dot(np.array([xd, yd]))

        # Lagrange multiplier that enforces no lateral slip
        lam = m * v * rd
        # translational acceleration
        acc = (f1 * t_hat(th) + lam * n_hat(th)) / m

        xdd, ydd = acc[0], acc[1]
        rdd = f2 / I

        return np.array([xd, yd, rd, xdd, ydd, rdd], dtype=float), lam

    # integrate with RK4 + projection to enforce constraint
    for k in range(N):
        X[k] = y

        # store lambda
        _, lam = dynamics(y, t[k])
        Lam[k] = lam

        def f(yy):
            dy, _ = dynamics(yy, t[k])
            return dy

        y_next = rk4_step(f, y, dt)

        # project velocity to satisfy constraint: n_hat^T v = 0
        thn = y_next[2]
        v_long = t_hat(thn).dot(y_next[3:5])
        y_next[3:5] = v_long * t_hat(thn)

        # keep theta bounded
        y_next[2] = wrap_pi(y_next[2])

        # safety
        if not np.all(np.isfinite(y_next)):
            print("NaN encountered; stopping.")
            X = X[:k+1]
            Lam = Lam[:k+1]
            t = t[:k+1]
            N = k+1
            break

        y = y_next

    # --- animation ---
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Constrained Dynamic Model (no lateral slip)")

    ax.plot(X[:,0], X[:,1], linewidth=1)

    body_line, = ax.plot([], [], linewidth=2)
    vel_q = ax.quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=1, width=0.007)
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    # body rectangle in body frame (centered)
    rect_b = np.array([
        [ +L_draw/2, +W_draw/2],
        [ +L_draw/2, -W_draw/2],
        [ -L_draw/2, -W_draw/2],
        [ -L_draw/2, +W_draw/2],
        [ +L_draw/2, +W_draw/2],
    ])

    def set_view(i):
        cx, cy = X[i,0], X[i,1]
        ax.set_xlim(cx - 3.0, cx + 3.0)
        ax.set_ylim(cy - 3.0, cy + 3.0)

    def update(i):
        x, ypos, th, xd, yd, rd = X[i]
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
        rect_w = (R @ rect_b.T).T + np.array([x, ypos])
        body_line.set_data(rect_w[:,0], rect_w[:,1])

        vvec = np.array([xd, yd])
        vel_q.set_offsets([[x, ypos]])
        vel_q.set_UVC(vvec[0], vvec[1])

        # constraint check (should be ~0)
        lat = n_hat(th).dot(vvec)
        f1, f2 = inputs(t[i])

        info.set_text(
            f"t={t[i]:.2f}s\n"
            f"theta={np.rad2deg(th):+.1f} deg, r={np.rad2deg(rd):+.1f} deg/s\n"
            f"v=({xd:.2f},{yd:.2f}) m/s\n"
            f"n^T v (lateral vel) = {lat:+.3e}\n"
            f"f1={f1:.2f} N, f2={f2:.2f} N·m\n"
            f"lambda={Lam[i]:+.2f} N"
        )

        set_view(i)
        return body_line, vel_q, info

    ani = FuncAnimation(fig, update, frames=N, interval=1000*dt, blit=False, repeat=True)
    plt.show()

if __name__ == "__main__":
    main()
