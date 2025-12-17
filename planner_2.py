#!/usr/bin/env python3
"""
ENHANCED: Hybrid A* with Multiple Obstacle Avoidance + Visualization

Features:
- Multiple complex obstacle scenarios
- Rich visualization showing obstacle avoidance
- Animated path following
- Detailed analysis dashboard

Usage: python hybrid_astar_multi_obstacle_viz.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
import heapq
from dataclasses import dataclass
import time
from typing import List, Tuple, Dict, Optional


# ============================================================================
# PLANNER COMPONENTS
# ============================================================================

@dataclass
class KinematicBicycle:
    L: float
    
    def step(self, x, y, psi, v, delta, dt):
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
        
        self.occ = np.zeros((self.nx, self.ny), dtype=bool)
        self.obstacles = []
        self.rect_obstacles = []
    
    def add_circular_obstacle(self, xc, yc, r):
        self.obstacles.append((xc, yc, r))
        for ix in range(self.nx):
            for iy in range(self.ny):
                x, y = self.index_to_world(ix, iy)
                if np.hypot(x - xc, y - yc) <= r:
                    self.occ[ix, iy] = True
    
    def add_rectangular_obstacle(self, x_min, x_max, y_min, y_max):
        self.rect_obstacles.append((x_min, x_max, y_min, y_max))
        for ix in range(self.nx):
            for iy in range(self.ny):
                x, y = self.index_to_world(ix, iy)
                if x_min <= x <= x_max and y_min <= y <= y_max:
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
        for x, y in zip(xs, ys):
            if not self.in_bounds(x, y):
                return False
            ix, iy = self.world_to_index(x, y)
            if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
                return False
            if self.occ[ix, iy]:
                return False
            for xc, yc, r in self.obstacles:
                if np.hypot(x - xc, y - yc) <= r + robot_radius:
                    return False
        return True


@dataclass(order=True)
class PQNode:
    f: float
    idx: int


@dataclass
class Node:
    x: float
    y: float
    psi: float
    g: float
    h: float
    parent: int


def heuristic(x, y, goal):
    return np.hypot(x - goal[0], y - goal[1])


def hybrid_astar_instrumented(
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
    """Hybrid A* with statistics."""
    
    if delta_set is None:
        delta_set = [-0.5, 0.0, 0.5]
    
    car = KinematicBicycle(L)
    
    def yaw_to_bin(psi):
        psi_wrap = (psi + np.pi) % (2 * np.pi) - np.pi
        bin_width = 2 * np.pi / yaw_bins
        return int(np.floor((psi_wrap + np.pi) / bin_width))
    
    closed = np.full((grid.nx, grid.ny, yaw_bins), False, dtype=bool)
    
    nodes = []
    x0, y0, psi0 = start_state
    h0 = heuristic(x0, y0, goal_xy)
    nodes.append(Node(x0, y0, psi0, g=0.0, h=h0, parent=-1))
    
    pq = []
    heapq.heappush(pq, PQNode(f=h0, idx=0))
    
    step_n = int(T_step / dt)
    nodes_expanded = 0
    
    for it in range(max_iter):
        if not pq:
            return None, {'iterations': it, 'nodes_expanded': nodes_expanded, 
                         'nodes_generated': len(nodes), 'success': False}
        
        cur = heapq.heappop(pq)
        n_idx = cur.idx
        node = nodes[n_idx]
        
        x, y, psi = node.x, node.y, node.psi
        
        if np.hypot(x - goal_xy[0], y - goal_xy[1]) <= goal_tolerance:
            path = []
            idx = n_idx
            while idx != -1:
                n = nodes[idx]
                path.append((n.x, n.y, n.psi))
                idx = n.parent
            path.reverse()
            
            stats = {
                'iterations': it,
                'nodes_expanded': nodes_expanded,
                'nodes_generated': len(nodes),
                'success': True
            }
            return path, stats
        
        ix, iy = grid.world_to_index(x, y)
        if ix < 0 or ix >= grid.nx or iy < 0 or iy >= grid.ny:
            continue
        yaw_bin = yaw_to_bin(psi)
        if closed[ix, iy, yaw_bin]:
            continue
        closed[ix, iy, yaw_bin] = True
        nodes_expanded += 1
        
        for delta in delta_set:
            xs = [x]
            ys = [y]
            psi_ = psi
            
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
    
    stats = {
        'iterations': max_iter,
        'nodes_expanded': nodes_expanded,
        'nodes_generated': len(nodes),
        'success': False
    }
    return None, stats


# ============================================================================
# MULTI-OBSTACLE SCENARIOS
# ============================================================================

class ScenarioGenerator:
    """Generate challenging multi-obstacle scenarios."""
    
    @staticmethod
    def obstacle_field():
        """Multiple scattered obstacles - like a parking lot."""
        grid = GridWorld(0, 50, -15, 15, resolution=0.5)
        
        # Create obstacle field
        obstacles = [
            (10, 8, 2.5),
            (10, -8, 2.5),
            (20, 5, 2.0),
            (20, -5, 2.0),
            (30, 10, 2.5),
            (30, 0, 2.0),
            (30, -10, 2.5),
            (40, 7, 2.0),
            (40, -7, 2.0),
        ]
        
        for xc, yc, r in obstacles:
            grid.add_circular_obstacle(xc, yc, r)
        
        start = (2.0, 0.0, 0.0)
        goal = (48.0, 0.0)
        return {
            'name': 'Obstacle Field',
            'grid': grid,
            'start': start,
            'goal': goal,
            'description': '9 obstacles - parking lot style'
        }
    
    @staticmethod
    def corridor_with_obstacles():
        """Narrow corridor with obstacles."""
        grid = GridWorld(0, 50, -8, 8, resolution=0.5)
        
        # Corridor walls
        grid.add_rectangular_obstacle(0, 50, 6, 8)
        grid.add_rectangular_obstacle(0, 50, -8, -6)
        
        # Obstacles in corridor
        grid.add_circular_obstacle(12, 3, 1.5)
        grid.add_circular_obstacle(12, -3, 1.5)
        grid.add_circular_obstacle(25, 0, 2.0)
        grid.add_circular_obstacle(35, 4, 1.5)
        grid.add_circular_obstacle(35, -4, 1.5)
        
        start = (2.0, 0.0, 0.0)
        goal = (48.0, 0.0)
        return {
            'name': 'Corridor Challenge',
            'grid': grid,
            'start': start,
            'goal': goal,
            'description': 'Narrow corridor + 5 obstacles'
        }
    
    @staticmethod
    def slalom_course():
        """Slalom-style obstacle course."""
        grid = GridWorld(0, 60, -12, 12, resolution=0.5)
        
        # Alternating obstacles (slalom pattern)
        y_positions = [8, -8, 8, -8, 8, -8, 8]
        x_positions = np.linspace(10, 50, 7)
        
        for x, y in zip(x_positions, y_positions):
            grid.add_circular_obstacle(x, y, 2.5)
        
        start = (2.0, 0.0, 0.0)
        goal = (58.0, 0.0)
        return {
            'name': 'Slalom Course',
            'grid': grid,
            'start': start,
            'goal': goal,
            'description': '7 obstacles - slalom pattern'
        }
    
    @staticmethod
    def dense_forest():
        """Dense obstacle environment."""
        grid = GridWorld(0, 50, -15, 15, resolution=0.5)
        
        # Random-ish dense obstacles
        np.random.seed(42)
        n_obstacles = 15
        
        for i in range(n_obstacles):
            x = np.random.uniform(8, 45)
            y = np.random.uniform(-12, 12)
            r = np.random.uniform(1.5, 2.5)
            
            # Don't block start/goal
            if np.hypot(x - 2, y) > 5 and np.hypot(x - 48, y) > 5:
                grid.add_circular_obstacle(x, y, r)
        
        start = (2.0, 0.0, 0.0)
        goal = (48.0, 0.0)
        return {
            'name': 'Dense Forest',
            'grid': grid,
            'start': start,
            'goal': goal,
            'description': '15 obstacles - dense environment'
        }


# ============================================================================
# ENHANCED VISUALIZATION
# ============================================================================

def create_obstacle_avoidance_viz(scenario, path, stats):
    """Create detailed visualization showing obstacle avoidance."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.3)
    
    name = scenario['name']
    desc = scenario['description']
    grid = scenario['grid']
    start = scenario['start']
    goal = scenario['goal']
    
    fig.suptitle(f'Multi-Obstacle Avoidance: {name}', fontsize=16, fontweight='bold')
    
    # 1. MAIN PATH VIEW (Large)
    ax1 = fig.add_subplot(gs[:, :2])
    
    # Plot obstacles
    for xc, yc, r in grid.obstacles:
        circle = Circle((xc, yc), r, color='red', alpha=0.3, 
                       edgecolor='darkred', linewidth=2)
        ax1.add_patch(circle)
        # Add danger zone
        danger = Circle((xc, yc), r + 0.7, color='red', alpha=0.1, 
                       linestyle='--', fill=False, linewidth=1)
        ax1.add_patch(danger)
    
    # Plot rectangular obstacles
    for x_min, x_max, y_min, y_max in grid.rect_obstacles:
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                        color='gray', alpha=0.4, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
    
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        
        # Plot path with gradient color
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', linewidth=3)
        lc.set_array(np.linspace(0, 1, len(xs)))
        ax1.add_collection(lc)
        
        # Start and goal
        ax1.plot(start[0], start[1], 'go', markersize=15, 
                label='Start', zorder=10, markeredgecolor='darkgreen', markeredgewidth=2)
        ax1.plot(goal[0], goal[1], 'r*', markersize=20, 
                label='Goal', zorder=10, markeredgewidth=2)
        
        # Add direction arrows along path
        arrow_step = max(1, len(path) // 10)
        for i in range(0, len(path) - 1, arrow_step):
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            ax1.arrow(xs[i], ys[i], dx*0.8, dy*0.8,
                     head_width=0.8, head_length=0.5,
                     fc='blue', ec='blue', alpha=0.5, zorder=5)
    
    ax1.set_xlabel('x [m]', fontsize=12)
    ax1.set_ylabel('y [m]', fontsize=12)
    ax1.set_title(f'{desc}', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.axis('equal')
    ax1.set_xlim(grid.x_min - 2, grid.x_max + 2)
    ax1.set_ylim(grid.y_min - 2, grid.y_max + 2)
    
    # Add text annotations
    ax1.text(0.02, 0.98, f'Obstacles: {len(grid.obstacles)}',
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. STATISTICS
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    stats_text = "PLANNING STATS\n" + "="*25 + "\n\n"
    stats_text += f"Status: {'✓ SUCCESS' if path else '✗ FAILED'}\n\n"
    
    if path:
        path_length = sum(np.hypot(path[i+1][0] - path[i][0], 
                                   path[i+1][1] - path[i][1])
                         for i in range(len(path) - 1))
        optimal = np.hypot(goal[0] - start[0], goal[1] - start[1])
        
        stats_text += f"Iterations: {stats['iterations']}\n"
        stats_text += f"Nodes: {stats['nodes_generated']}\n\n"
        stats_text += f"Path Length: {path_length:.1f}m\n"
        stats_text += f"Straight Line: {optimal:.1f}m\n"
        stats_text += f"Sub-optimal: {path_length/optimal:.3f}\n\n"
        stats_text += f"Waypoints: {len(path)}\n"
    else:
        stats_text += f"Iterations: {stats['iterations']}\n"
        stats_text += f"Nodes: {stats['nodes_generated']}\n"
        stats_text += f"\nNo path found"
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. CLEARANCE ANALYSIS
    ax3 = fig.add_subplot(gs[1, 2])
    
    if path:
        # Compute minimum clearance to obstacles
        clearances = []
        for x, y, _ in path:
            min_dist = float('inf')
            for xc, yc, r in grid.obstacles:
                dist = np.hypot(x - xc, y - yc) - r
                min_dist = min(min_dist, dist)
            clearances.append(max(0, min_dist))
        
        ax3.plot(clearances, 'b-', linewidth=2)
        ax3.axhline(0.7, color='orange', linestyle='--', 
                   label='Robot radius', linewidth=2)
        ax3.fill_between(range(len(clearances)), 0, 0.7, 
                        alpha=0.2, color='red', label='Danger zone')
        ax3.set_xlabel('Waypoint', fontsize=10)
        ax3.set_ylabel('Clearance [m]', fontsize=10)
        ax3.set_title('Obstacle Clearance Along Path', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        min_clear = min(clearances)
        avg_clear = np.mean(clearances)
        ax3.text(0.98, 0.98, f'Min: {min_clear:.2f}m\nAvg: {avg_clear:.2f}m',
                transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def create_animated_path(scenario, path):
    """Create animation of robot following path."""
    
    if not path:
        print("No path to animate")
        return None
    
    grid = scenario['grid']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot obstacles
    for xc, yc, r in grid.obstacles:
        circle = Circle((xc, yc), r, color='red', alpha=0.3, 
                       edgecolor='darkred', linewidth=2)
        ax.add_patch(circle)
    
    for x_min, x_max, y_min, y_max in grid.rect_obstacles:
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                        color='gray', alpha=0.4, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
    
    # Plot full path (light)
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, 'b--', alpha=0.3, linewidth=1, label='Planned path')
    
    # Start and goal
    ax.plot(scenario['start'][0], scenario['start'][1], 'go', 
           markersize=12, label='Start', zorder=10)
    ax.plot(scenario['goal'][0], scenario['goal'][1], 'r*', 
           markersize=15, label='Goal', zorder=10)
    
    # Robot (will be animated)
    robot = Circle((xs[0], ys[0]), 0.7, color='blue', alpha=0.6, zorder=15)
    ax.add_patch(robot)
    
    # Direction arrow
    arrow = ax.arrow(xs[0], ys[0], 1, 0, head_width=0.5, head_length=0.3,
                    fc='darkblue', ec='darkblue', zorder=16)
    
    # Trail
    trail_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.8, label='Traveled')
    
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title(f'{scenario["name"]} - Robot Navigation', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.axis('equal')
    ax.set_xlim(grid.x_min - 2, grid.x_max + 2)
    ax.set_ylim(grid.y_min - 2, grid.y_max + 2)
    
    # Animation
    def animate(frame):
        if frame < len(path):
            x, y, psi = path[frame]
            robot.center = (x, y)
            
            # Update arrow
            dx = 1.5 * np.cos(psi)
            dy = 1.5 * np.sin(psi)
            arrow.set_data(x=x, y=y, dx=dx, dy=dy)
            
            # Update trail
            trail_line.set_data(xs[:frame+1], ys[:frame+1])
        
        return robot, arrow, trail_line
    
    ani = FuncAnimation(fig, animate, frames=len(path), 
                       interval=50, blit=True, repeat=True)
    
    return fig, ani


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run multi-obstacle avoidance demo."""
    
    print("\n" + "="*70)
    print(" HYBRID A* - MULTI-OBSTACLE AVOIDANCE DEMO")
    print("="*70)
    print("\nShowing 4 challenging scenarios with multiple obstacles")
    print("="*70 + "\n")
    
    # Create scenarios
    scenarios = [
        ScenarioGenerator.obstacle_field(),
        ScenarioGenerator.corridor_with_obstacles(),
        ScenarioGenerator.slalom_course(),
        ScenarioGenerator.dense_forest(),
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/4] Planning: {scenario['name']}")
        print(f"      {scenario['description']}")
        
        start_time = time.time()
        path, stats = hybrid_astar_instrumented(
            scenario['grid'],
            scenario['start'],
            scenario['goal'],
            L=2.5,
            v_step=3.0,
            yaw_bins=16,
            max_iter=8000
        )
        elapsed = time.time() - start_time
        
        status = "✓ SUCCESS" if path else "✗ FAILED"
        print(f"      Result: {status} ({elapsed:.2f}s, {stats['iterations']} iters)")
        
        if path:
            path_length = sum(np.hypot(path[i+1][0] - path[i][0], 
                                       path[i+1][1] - path[i][1])
                             for i in range(len(path) - 1))
            optimal = np.hypot(scenario['goal'][0] - scenario['start'][0],
                              scenario['goal'][1] - scenario['start'][1])
            print(f"      Path: {path_length:.1f}m (sub-opt: {path_length/optimal:.3f})")
        print()
        
        results.append((scenario, path, stats))
    
    # Generate visualizations
    print("="*70)
    print(" GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    for i, (scenario, path, stats) in enumerate(results, 1):
        print(f"[{i}/4] Creating visualization: {scenario['name']}")
        
        fig = create_obstacle_avoidance_viz(scenario, path, stats)
        filename = f"obstacle_avoidance_{i}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"      Saved: {filename}")
    
    print("\n" + "="*70)
    print(" COMPLETE!")
    print("="*70)
    print("\n📁 Output Files:")
    for i in range(1, 5):
        print(f"  - obstacle_avoidance_{i}.png")
    
    print("\n💡 Tip: Run with animation to see robot navigate!")
    print("   (Uncomment animation code in main())")
    
    # OPTIONAL: Create animation for first successful path
    print("\n" + "="*70)
    print("Creating animation for first scenario...")
    
    for scenario, path, stats in results:
        if path:
            print(f"Animating: {scenario['name']}")
            fig_anim, ani = create_animated_path(scenario, path)
            
            # Save animation
            print("Saving animation (this may take a moment)...")
            try:
                ani.save('robot_navigation.gif', writer='pillow', fps=20)
                print("✓ Saved: robot_navigation.gif")
            except:
                print("(Could not save animation - showing instead)")
            
            plt.show()
            break
    
    print("\n✅ All visualizations complete!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user\n")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()