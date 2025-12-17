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
    # -------------------------
    # parameters
    # -------------------------
    m = 12.0     # kg
    I = 1.0      # kg*m^2
    L_draw = 0.6 # for drawing
    W_draw = 0.35

    dt = 1/60
    T  = 18.0
    N  = int(T/dt)
    t  = np.arange(N)*dt

    # state y = [x, y, theta, xdot, ydot, r]
    y = np.array([0.0, 0.0, 0.0,
                  1.2, 0.0, 0.0], dtype=float)

    X = np.zeros((N, 6))
    Lam = np.zeros(N)
    U = np.zeros((N, 2))      # f1, f2
    resid = np.zeros(N)       # n^T v

    # -------------------------
    # inputs (edit freely)
    # -------------------------
    def inputs(tt):
        f1 = 10.0 + 4.0*np.sin(2*np.pi*tt/5.0)    # N
        f2 = 0.8*np.sin(2*np.pi*tt/3.5)           # N*m
        return f1, f2

    # -------------------------
    # constrained dynamics
    # -------------------------
    def dynamics(y, tt):
        x, ypos, th, xd, yd, r = y
        f1, f2 = inputs(tt)

        v = t_hat(th).dot(np.array([xd, yd]))     # longitudinal speed
        lam = m * v * r                            # Lagrange multiplier

        acc = (f1 * t_hat(th) + lam * n_hat(th)) / m
        xdd, ydd = acc[0], acc[1]
        rdd = f2 / I

        dy = np.array([xd, yd, r, xdd, ydd, rdd], dtype=float)
        return dy, lam, f1, f2

    # integrate with RK4 + projection
    for k in range(N):
        X[k] = y
        dy, lam, f1, f2 = dynamics(y, t[k])
        Lam[k] = lam
        U[k] = [f1, f2]
        resid[k] = n_hat(y[2]).dot(y[3:5])

        def f(yy):
            dyy, *_ = dynamics(yy, t[k])
            return dyy

        y_next = rk4_step(f, y, dt)

        # project velocity to satisfy n^T v = 0
        thn = y_next[2]
        v_long = t_hat(thn).dot(y_next[3:5])
        y_next[3:5] = v_long * t_hat(thn)

        y_next[2] = wrap_pi(y_next[2])

        if not np.all(np.isfinite(y_next)):
            print("NaN encountered; stopping.")
            X = X[:k+1]
            Lam = Lam[:k+1]
            U = U[:k+1]
            resid = resid[:k+1]
            t = t[:k+1]
            N = k+1
            break

        y = y_next

    # =========================================================
    # PLOTS (time series)
    # =========================================================
    x = X[:,0]; yv = X[:,1]; th = X[:,2]
    xd = X[:,3]; yd = X[:,4]; r = X[:,5]
    speed = np.sqrt(xd**2 + yd**2)

    figp, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axs[0].plot(t, x, label="x")
    axs[0].plot(t, yv, label="y")
    axs[0].set_ylabel("position [m]")
    axs[0].grid(True); axs[0].legend()

    axs[1].plot(t, np.rad2deg(th), label="theta [deg]")
    axs[1].set_ylabel("heading [deg]")
    axs[1].grid(True); axs[1].legend()

    axs[2].plot(t, xd, label="xdot")
    axs[2].plot(t, yd, label="ydot")
    axs[2].plot(t, speed, label="|v|")
    axs[2].plot(t, np.rad2deg(r), label="r [deg/s]")
    axs[2].set_ylabel("vel / yaw rate")
    axs[2].grid(True); axs[2].legend(ncols=4)

    axs[3].plot(t, U[:,0], label="f1 [N]")
    axs[3].plot(t, U[:,1], label="f2 [N·m]")
    axs[3].plot(t, Lam, label="lambda [N]")
    axs[3].plot(t, resid, label="n^T v (residual)")
    axs[3].set_ylabel("inputs / constraint")
    axs[3].set_xlabel("time [s]")
    axs[3].grid(True); axs[3].legend(ncols=4)

    figp.suptitle("Constrained Dynamic Model — Time Plots", y=0.995)
    figp.tight_layout()

    # =========================================================
    # ANIMATION (separate window)
    # =========================================================
    figa, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Constrained Dynamic Model (no lateral slip) — Animation")

    ax.plot(x, yv, linewidth=1)

    body_line, = ax.plot([], [], linewidth=2)
    vel_q = ax.quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=1, width=0.007)
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    rect_b = np.array([
        [ +L_draw/2, +W_draw/2],
        [ +L_draw/2, -W_draw/2],
        [ -L_draw/2, -W_draw/2],
        [ -L_draw/2, +W_draw/2],
        [ +L_draw/2, +W_draw/2],
    ])

    def set_view(i):
        cx, cy = x[i], yv[i]
        ax.set_xlim(cx - 3.0, cx + 3.0)
        ax.set_ylim(cy - 3.0, cy + 3.0)

    def update(i):
        th_i = th[i]
        Rm = np.array([[np.cos(th_i), -np.sin(th_i)],
                       [np.sin(th_i),  np.cos(th_i)]])
        rect_w = (Rm @ rect_b.T).T + np.array([x[i], yv[i]])
        body_line.set_data(rect_w[:,0], rect_w[:,1])

        vvec = np.array([xd[i], yd[i]])
        vel_q.set_offsets([[x[i], yv[i]]])
        vel_q.set_UVC(vvec[0], vvec[1])

        info.set_text(
            f"t={t[i]:.2f}s\n"
            f"theta={np.rad2deg(th_i):+.1f} deg, r={np.rad2deg(r[i]):+.1f} deg/s\n"
            f"v=({xd[i]:.2f},{yd[i]:.2f}) m/s, |v|={speed[i]:.2f}\n"
            f"residual n^T v = {resid[i]:+.3e}\n"
            f"f1={U[i,0]:.2f} N, f2={U[i,1]:.2f} N·m\n"
            f"lambda={Lam[i]:+.2f} N"
        )

        set_view(i)
        return body_line, vel_q, info

    ani = FuncAnimation(figa, update, frames=N, interval=1000*dt, blit=False, repeat=True)
    plt.show()

if __name__ == "__main__":
    main()
