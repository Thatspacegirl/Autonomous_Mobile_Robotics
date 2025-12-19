
import numpy as np
import heapq
from dataclasses import dataclass



@dataclass
class KinematicBicycle:
    L: float  # wheelbase

    def step(self, x, y, psi, v, delta, dt):
        """One integration step."""
        x_dot = v * np.cos(psi)
        y_dot = v * np.sin(psi)
        psi_dot = v / self.L * np.tan(delta)
        x_new = x + x_dot * dt
        y_new = y + y_dot * dt
        psi_new = (psi + psi_dot * dt + np.pi) % (2 * np.pi) - np.pi
        return x_new, y_new, psi_new



class GridWorld:
    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.5):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.res = resolution

        self.nx = int((x_max - x_min) / resolution) + 1
        self.ny = int((y_max - y_min) / resolution) + 1

        # obstacle grid (bool)
        self.occ = np.zeros((self.nx, self.ny), dtype=bool)
        self.obstacles = []

    def add_circular_obstacle(self, xc, yc, r):
        self.obstacles.append((xc, yc, r))
        for ix in range(self.nx):
            for iy in range(self.ny):
                x, y = self.index_to_world(ix, iy)
                if np.hypot(x - xc, y - yc) <= r:
                    self.occ[ix, iy] = True

    def in_bounds(self, x, y):
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

    def world_to_index(self, x, y):
        ix = int(round((x - self.x_min) / self.res))
        iy = int(round((y - self.y_min) / self.res))
        return ix, iy

    def index_to_world(self, ix, iy):
        x = self.x_min + ix * self.res
        y = self.y_min + iy * self.res
        return x, y

    def collision_free_segment(self, xs, ys, robot_radius=0.7):
        """Check if a segment (xs, ys arrays) is free."""
        for x, y in zip(xs, ys):
            if not self.in_bounds(x, y):
                return False
            ix, iy = self.world_to_index(x, y)
            if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
                return False
            if self.occ[ix, iy]:
                return False
            # inflate by robot radius w.r.t. continuous obstacles
            for xc, yc, r in self.obstacles:
                if np.hypot(x - xc, y - yc) <= r + robot_radius:
                    return False
        return True


# ==============================
# Hybrid-A*-like search
# ==============================

@dataclass(order=True)
class PQNode:
    f: float
    idx: int  # index in node list


@dataclass
class Node:
    x: float
    y: float
    psi: float
    g: float
    h: float
    parent: int


def heuristic(x, y, goal):
    """Straight-line heuristic."""
    return np.hypot(x - goal[0], y - goal[1])


def hybrid_astar(
    grid: GridWorld,
    start_state,
    goal_xy,
    L=2.5,
    v_step=2.0,
    dt=0.1,
    T_step=1.0,
    delta_set=None,
    yaw_bins=16,
    robot_radius=0.7,
    max_iter=10000,
    goal_tolerance=1.0,
):
    if delta_set is None:
        delta_set = [-0.5, 0.0, 0.5]  # radians

    car = KinematicBicycle(L)

    # discretization in yaw
    def yaw_to_bin(psi):
        psi_wrap = (psi + np.pi) % (2 * np.pi) - np.pi
        bin_width = 2 * np.pi / yaw_bins
        return int(np.floor((psi_wrap + np.pi) / bin_width))

    # closed set: 3D occupancy (x_idx, y_idx, yaw_bin)
    closed = np.full((grid.nx, grid.ny, yaw_bins), False, dtype=bool)

    nodes = []
    x0, y0, psi0 = start_state
    h0 = heuristic(x0, y0, goal_xy)
    nodes.append(Node(x0, y0, psi0, g=0.0, h=h0, parent=-1))

    pq = []
    heapq.heappush(pq, PQNode(f=h0, idx=0))

    step_n = int(T_step / dt)

    for it in range(max_iter):
        if not pq:
            print("Search failed: open list empty.")
            return None

        cur = heapq.heappop(pq)
        n_idx = cur.idx
        node = nodes[n_idx]

        x, y, psi = node.x, node.y, node.psi

        # goal check (position-only)
        if np.hypot(x - goal_xy[0], y - goal_xy[1]) <= goal_tolerance:
            print(f"Goal reached in {it} iterations, cost={node.g:.2f}")
            # reconstruct path
            path = []
            idx = n_idx
            while idx != -1:
                n = nodes[idx]
                path.append((n.x, n.y, n.psi))
                idx = n.parent
            path.reverse()
            return path

        ix, iy = grid.world_to_index(x, y)
        if ix < 0 or ix >= grid.nx or iy < 0 or iy >= grid.ny:
            continue
        yaw_bin = yaw_to_bin(psi)
        if closed[ix, iy, yaw_bin]:
            continue
        closed[ix, iy, yaw_bin] = True

        # expand successors
        for delta in delta_set:
            xs = [x]
            ys = [y]
            psi_ = psi

            # simulate short motion
            for _ in range(step_n):
                x_, y_, psi_ = car.step(xs[-1], ys[-1], psi_, v_step, delta, dt)
                xs.append(x_)
                ys.append(y_)

            if not grid.collision_free_segment(xs, ys, robot_radius):
                continue

            x_new, y_new, psi_new = xs[-1], ys[-1], psi_

            g_new = node.g + np.hypot(x_new - x, y_new - y)
            h_new = heuristic(x_new, y_new, goal_xy)
            f_new = g_new + h_new

            new_node = Node(x_new, y_new, psi_new, g=g_new, h=h_new, parent=n_idx)
            nodes.append(new_node)
            new_idx = len(nodes) - 1

            heapq.heappush(pq, PQNode(f=f_new, idx=new_idx))

    print("Search terminated: max iterations reached.")
    return None


# ==============================
# Example usage
# ==============================

if __name__ == "__main__":
    # Define grid + obstacles
    grid = GridWorld(x_min=0, x_max=40, y_min=-10, y_max=10, resolution=0.5)
    grid.add_circular_obstacle(15, 0, 3)
    grid.add_circular_obstacle(25, 4, 2)
    grid.add_circular_obstacle(30, -3, 2)

    start = (2.0, -5.0, 0.0)    # (x, y, psi)
    goal_xy = (35.0, 5.0)

    path = hybrid_astar(
        grid,
        start_state=start,
        goal_xy=goal_xy,
        L=2.5,
        v_step=3.0,
        dt=0.1,
        T_step=1.0,
        delta_set=[-0.5, 0.0, 0.5],
        yaw_bins=24,
        robot_radius=0.7,
    )

    if path is None:
        print("No path found.")
    else:
        print(f"Path length (nodes): {len(path)}")

        # OPTIONAL: quick visualization
        import matplotlib.pyplot as plt

        # plot obstacles
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_aspect("equal", "box")
        for xc, yc, r in grid.obstacles:
            circle = plt.Circle((xc, yc), r, color="k", alpha=0.3)
            ax.add_patch(circle)

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, "-o", color="tab:blue", label="Hybrid A* path", linewidth=2)

        ax.plot(start[0], start[1], "go", label="Start")
        ax.plot(goal_xy[0], goal_xy[1], "rx", label="Goal", markersize=10)

        ax.set_xlim(grid.x_min - 1, grid.x_max + 1)
        ax.set_ylim(grid.y_min - 1, grid.y_max + 1)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        ax.set_title("Search-based Ackermann Planning (Hybrid-A*-like)")
        plt.show()

