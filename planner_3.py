#!/usr/bin/env python3
"""
DYNAMIC OBSTACLE AVOIDANCE - Hybrid A* with Moving Obstacles

Features:
- Moving obstacles with various patterns (linear, circular, bouncing)
- Space-time planning (predicts future collisions)
- Real-time visualization showing dynamic avoidance
- Interactive animation showing robot navigating moving obstacles

Usage: python dynamic_obstacle_avoidance.py
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for animations
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrow
from matplotlib.animation import FuncAnimation
import heapq
from dataclasses import dataclass
import time
from typing import List, Tuple, Dict, Optional

# ============================================================================
# ANIMATION SETTINGS (customize as needed)
# ============================================================================
ANIMATION_FPS = 20      # Frames per second
ANIMATION_SPEED = 100   # milliseconds per frame (lower = faster)


# ============================================================================
# MOVING OBSTACLE CLASS
# ============================================================================

class MovingObstacle:
    """Represents a moving obstacle with position and velocity."""
    
    def __init__(self, x0, y0, vx, vy, r, pattern='linear', 
                 x_bounds=None, y_bounds=None, omega=0.0, orbit_center=None):
        self.x0 = x0  # Initial position
        self.y0 = y0
        self.vx = vx  # Velocity
        self.vy = vy
        self.r = r    # Radius
        self.pattern = pattern  # 'linear', 'circular', 'bounce'
        self.x_bounds = x_bounds if x_bounds else (0, 50)
        self.y_bounds = y_bounds if y_bounds else (-15, 15)
        self.omega = omega  # Angular velocity for circular motion
        self.orbit_center = orbit_center if orbit_center else (x0, y0)
        
    def get_position(self, t):
        """Get obstacle position at time t."""
        
        if self.pattern == 'linear':
            # Simple linear motion
            x = self.x0 + self.vx * t
            y = self.y0 + self.vy * t
            
        elif self.pattern == 'circular':
            # Circular orbit
            cx, cy = self.orbit_center
            orbit_r = np.hypot(self.x0 - cx, self.y0 - cy)
            theta0 = np.arctan2(self.y0 - cy, self.x0 - cx)
            theta = theta0 + self.omega * t
            x = cx + orbit_r * np.cos(theta)
            y = cy + orbit_r * np.sin(theta)
            
        elif self.pattern == 'bounce':
            # Bouncing motion (reflects at boundaries)
            x = self.x0 + self.vx * t
            y = self.y0 + self.vy * t
            
            # Bounce off boundaries
            x_range = self.x_bounds[1] - self.x_bounds[0]
            y_range = self.y_bounds[1] - self.y_bounds[0]
            
            # Handle bouncing
            x_normalized = (x - self.x_bounds[0]) / x_range
            x_bounces = int(x_normalized)
            x_frac = x_normalized - x_bounces
            if x_bounces % 2 == 1:
                x_frac = 1 - x_frac
            x = self.x_bounds[0] + x_frac * x_range
            
            y_normalized = (y - self.y_bounds[0]) / y_range
            y_bounces = int(y_normalized)
            y_frac = y_normalized - y_bounces
            if y_bounces % 2 == 1:
                y_frac = 1 - y_frac
            y = self.y_bounds[0] + y_frac * y_range
            
        else:
            x, y = self.x0, self.y0
            
        return x, y
    
    def get_velocity(self, t):
        """Get obstacle velocity at time t."""
        if self.pattern == 'circular':
            # Velocity is perpendicular to radius
            x, y = self.get_position(t)
            cx, cy = self.orbit_center
            orbit_r = np.hypot(x - cx, y - cy)
            vx = -self.omega * (y - cy)
            vy = self.omega * (x - cx)
            return vx, vy
        else:
            return self.vx, self.vy


# ============================================================================
# KINEMATIC BICYCLE
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


# ============================================================================
# DYNAMIC GRID (considers time)
# ============================================================================

class DynamicGridWorld:
    """Grid world with moving obstacles."""
    
    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.5):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.res = resolution
        
        self.nx = int((x_max - x_min) / resolution) + 1
        self.ny = int((y_max - y_min) / resolution) + 1
        
        self.static_obstacles = []  # (x, y, r)
        self.dynamic_obstacles = []  # MovingObstacle objects
        
    def add_static_obstacle(self, xc, yc, r):
        self.static_obstacles.append((xc, yc, r))
        
    def add_dynamic_obstacle(self, obstacle: MovingObstacle):
        self.dynamic_obstacles.append(obstacle)
        
    def in_bounds(self, x, y):
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)
    
    def world_to_index(self, x, y):
        ix = int(round((x - self.x_min) / self.res))
        iy = int(round((y - self.y_min) / self.res))
        return ix, iy
    
    def check_collision(self, x, y, t, robot_radius=0.7):
        """Check collision at position (x,y) at time t."""
        
        # Check bounds
        if not self.in_bounds(x, y):
            return True
            
        # Check static obstacles
        for xc, yc, r in self.static_obstacles:
            if np.hypot(x - xc, y - yc) <= r + robot_radius:
                return True
        
        # Check dynamic obstacles at time t
        for obs in self.dynamic_obstacles:
            obs_x, obs_y = obs.get_position(t)
            if np.hypot(x - obs_x, y - obs_y) <= obs.r + robot_radius:
                return True
                
        return False
    
    def collision_free_segment(self, xs, ys, ts, robot_radius=0.7):
        """Check if trajectory segment is collision-free."""
        for x, y, t in zip(xs, ys, ts):
            if self.check_collision(x, y, t, robot_radius):
                return False
        return True


# ============================================================================
# SPACE-TIME A* PLANNER
# ============================================================================

@dataclass(order=True)
class PQNode:
    f: float
    idx: int


@dataclass
class Node:
    x: float
    y: float
    psi: float
    t: float  # Time dimension!
    g: float
    h: float
    parent: int


def heuristic(x, y, goal):
    return np.hypot(x - goal[0], y - goal[1])


def dynamic_hybrid_astar(
    grid: DynamicGridWorld,
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
    """Space-time Hybrid A* for dynamic obstacle avoidance."""
    
    if delta_set is None:
        delta_set = [-0.5, 0.0, 0.5]
    
    car = KinematicBicycle(L)
    
    def yaw_to_bin(psi):
        psi_wrap = (psi + np.pi) % (2 * np.pi) - np.pi
        bin_width = 2 * np.pi / yaw_bins
        return int(np.floor((psi_wrap + np.pi) / bin_width))
    
    # State space: (x, y, yaw, time_bin)
    time_resolution = 0.5  # seconds per time bin
    max_time_bins = 100
    
    closed = {}  # Use dict instead of array for sparse time dimension
    
    nodes = []
    x0, y0, psi0 = start_state
    t0 = 0.0
    h0 = heuristic(x0, y0, goal_xy)
    nodes.append(Node(x0, y0, psi0, t=t0, g=0.0, h=h0, parent=-1))
    
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
        
        x, y, psi, t = node.x, node.y, node.psi, node.t
        
        # Goal check
        if np.hypot(x - goal_xy[0], y - goal_xy[1]) <= goal_tolerance:
            # Reconstruct path
            path = []
            idx = n_idx
            while idx != -1:
                n = nodes[idx]
                path.append((n.x, n.y, n.psi, n.t))
                idx = n.parent
            path.reverse()
            
            stats = {
                'iterations': it,
                'nodes_expanded': nodes_expanded,
                'nodes_generated': len(nodes),
                'success': True,
                'total_time': t
            }
            return path, stats
        
        # Create state key
        ix, iy = grid.world_to_index(x, y)
        yaw_bin = yaw_to_bin(psi)
        time_bin = int(t / time_resolution)
        
        state_key = (ix, iy, yaw_bin, time_bin)
        
        if state_key in closed:
            continue
        closed[state_key] = True
        nodes_expanded += 1
        
        # Expand successors
        for delta in delta_set:
            xs = [x]
            ys = [y]
            ts = [t]
            psi_ = psi
            t_ = t
            
            # Simulate motion
            for _ in range(step_n):
                x_, y_, psi_ = car.step(xs[-1], ys[-1], psi_, v_step, delta, dt)
                t_ += dt
                xs.append(x_)
                ys.append(y_)
                ts.append(t_)
            
            # Check collision along trajectory (space-time check!)
            if not grid.collision_free_segment(xs, ys, ts, robot_radius):
                continue
            
            x_new, y_new, psi_new, t_new = xs[-1], ys[-1], psi_, ts[-1]
            
            # Time-dependent cost (encourage faster paths)
            g_new = node.g + np.hypot(x_new - x, y_new - y) + 0.1 * T_step
            h_new = heuristic(x_new, y_new, goal_xy)
            f_new = g_new + h_new
            
            new_node = Node(x_new, y_new, psi_new, t=t_new, 
                          g=g_new, h=h_new, parent=n_idx)
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
# DYNAMIC SCENARIOS
# ============================================================================

class DynamicScenarios:
    """Create scenarios with moving obstacles."""
    
    @staticmethod
    def crossing_obstacles():
        """Obstacles crossing the path."""
        grid = DynamicGridWorld(0, 50, -15, 15, resolution=0.5)
        
        # Static obstacles
        grid.add_static_obstacle(25, 10, 2.0)
        grid.add_static_obstacle(25, -10, 2.0)
        
        # Moving obstacles crossing path
        grid.add_dynamic_obstacle(
            MovingObstacle(15, -12, 0, 2.0, r=1.5, pattern='linear')
        )
        grid.add_dynamic_obstacle(
            MovingObstacle(30, 12, 0, -1.5, r=1.5, pattern='linear')
        )
        grid.add_dynamic_obstacle(
            MovingObstacle(40, -12, 0, 2.5, r=1.5, pattern='linear')
        )
        
        start = (2.0, 0.0, 0.0)
        goal = (48.0, 0.0)
        
        return {
            'name': 'Crossing Obstacles',
            'grid': grid,
            'start': start,
            'goal': goal,
            'description': '3 moving obstacles crossing path',
            'duration': 25.0
        }
    
    @staticmethod
    def orbiting_obstacles():
        """Obstacles in circular orbits."""
        grid = DynamicGridWorld(0, 50, -15, 15, resolution=0.5)
        
        # Obstacles orbiting in circles
        grid.add_dynamic_obstacle(
            MovingObstacle(20, 5, 0, 0, r=2.0, pattern='circular',
                         omega=0.3, orbit_center=(20, 0))
        )
        grid.add_dynamic_obstacle(
            MovingObstacle(35, -5, 0, 0, r=2.0, pattern='circular',
                         omega=-0.4, orbit_center=(35, 0))
        )
        
        # Static obstacles
        grid.add_static_obstacle(20, 0, 1.0)
        grid.add_static_obstacle(35, 0, 1.0)
        
        start = (2.0, 0.0, 0.0)
        goal = (48.0, 0.0)
        
        return {
            'name': 'Orbiting Obstacles',
            'grid': grid,
            'start': start,
            'goal': goal,
            'description': '2 obstacles in circular orbits',
            'duration': 25.0
        }
    
    @staticmethod
    def complex_dynamic():
        """Complex scenario with multiple motion patterns."""
        grid = DynamicGridWorld(0, 60, -15, 15, resolution=0.5)
        
        # Mix of motion patterns
        # Linear movers
        grid.add_dynamic_obstacle(
            MovingObstacle(15, 8, 1.5, 0, r=2.0, pattern='linear')
        )
        grid.add_dynamic_obstacle(
            MovingObstacle(25, -8, 0, 1.5, r=2.0, pattern='linear')
        )
        
        # Orbiting
        grid.add_dynamic_obstacle(
            MovingObstacle(40, 5, 0, 0, r=1.8, pattern='circular',
                         omega=0.5, orbit_center=(40, 0))
        )
        
        # Bouncing
        grid.add_dynamic_obstacle(
            MovingObstacle(50, 0, 0, 2.0, r=1.5, pattern='bounce',
                         y_bounds=(-10, 10))
        )
        
        # Static obstacles
        grid.add_static_obstacle(10, 0, 2.5)
        grid.add_static_obstacle(30, 10, 2.0)
        
        start = (2.0, 0.0, 0.0)
        goal = (58.0, 0.0)
        
        return {
            'name': 'Complex Dynamic',
            'grid': grid,
            'start': start,
            'goal': goal,
            'description': '4 dynamic + 2 static obstacles',
            'duration': 30.0
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_dynamic_visualization(scenario, path, stats):
    """Create snapshot visualization showing obstacle motion."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    name = scenario['name']
    grid = scenario['grid']
    start = scenario['start']
    goal = scenario['goal']
    duration = scenario['duration']
    
    fig.suptitle(f'Dynamic Obstacle Avoidance: {name}', 
                fontsize=16, fontweight='bold')
    
    # 1. PATH WITH OBSTACLE POSITIONS AT DIFFERENT TIMES
    ax1 = fig.add_subplot(gs[:, 0])
    
    # Show obstacles at multiple time snapshots
    time_snapshots = [0, duration/3, 2*duration/3, duration]
    colors = ['red', 'orange', 'yellow', 'pink']
    alphas = [0.3, 0.25, 0.2, 0.15]
    
    for t_snap, color, alpha in zip(time_snapshots, colors, alphas):
        for obs in grid.dynamic_obstacles:
            x, y = obs.get_position(t_snap)
            circle = Circle((x, y), obs.r, facecolor=color, alpha=alpha,
                          edgecolor='darkred', linewidth=1)
            ax1.add_patch(circle)
            
            # Add velocity arrow
            vx, vy = obs.get_velocity(t_snap)
            if np.hypot(vx, vy) > 0.1:
                ax1.arrow(x, y, vx*0.5, vy*0.5, head_width=0.5,
                         head_length=0.3, fc=color, ec='darkred', alpha=alpha*2)
    
    # Static obstacles
    for xc, yc, r in grid.static_obstacles:
        circle = Circle((xc, yc), r, facecolor='gray', alpha=0.5,
                       edgecolor='black', linewidth=2)
        ax1.add_patch(circle)
    
    # Plot path
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        
        from matplotlib.collections import LineCollection
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap='viridis', linewidth=3.5)
        lc.set_array(np.linspace(0, 1, len(xs)))
        ax1.add_collection(lc)
        
        ax1.plot(start[0], start[1], 'go', markersize=12, 
                label='Start', zorder=10)
        ax1.plot(goal[0], goal[1], 'r*', markersize=15, 
                label='Goal', zorder=10)
    
    ax1.set_xlabel('x [m]', fontsize=12)
    ax1.set_ylabel('y [m]', fontsize=12)
    ax1.set_title(f'{scenario["description"]}', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.axis('equal')
    ax1.set_xlim(grid.x_min - 2, grid.x_max + 2)
    ax1.set_ylim(grid.y_min - 2, grid.y_max + 2)
    
    # Legend for obstacle times
    legend_text = "Obstacle positions at:\n"
    for t, c in zip(time_snapshots, colors):
        legend_text += f"  {c}: t={t:.1f}s\n"
    ax1.text(0.02, 0.02, legend_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. SPACE-TIME DIAGRAM
    ax2 = fig.add_subplot(gs[0, 1])
    
    if path:
        # Plot robot trajectory in space-time
        xs = [p[0] for p in path]
        ts = [p[3] for p in path]
        ax2.plot(xs, ts, 'b-', linewidth=2.5, label='Robot')
        
        # Plot obstacle trajectories
        t_range = np.linspace(0, duration, 100)
        for i, obs in enumerate(grid.dynamic_obstacles):
            obs_xs = [obs.get_position(t)[0] for t in t_range]
            ax2.plot(obs_xs, t_range, '--', alpha=0.6, 
                    label=f'Obstacle {i+1}')
        
        ax2.set_xlabel('x position [m]', fontsize=10)
        ax2.set_ylabel('Time [s]', fontsize=10)
        ax2.set_title('Space-Time Diagram (x vs t)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
    
    # 3. STATISTICS
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    stats_text = "PLANNING RESULTS\n" + "="*28 + "\n\n"
    stats_text += f"Status: {'✓ SUCCESS' if path else '✗ FAILED'}\n\n"
    
    if path:
        path_length = sum(np.hypot(path[i+1][0] - path[i][0],
                                   path[i+1][1] - path[i][1])
                         for i in range(len(path) - 1))
        optimal = np.hypot(goal[0] - start[0], goal[1] - start[1])
        total_time = path[-1][3]
        
        stats_text += f"Planning:\n"
        stats_text += f"  Iterations: {stats['iterations']}\n"
        stats_text += f"  Nodes: {stats['nodes_generated']}\n\n"
        
        stats_text += f"Path Quality:\n"
        stats_text += f"  Length: {path_length:.1f}m\n"
        stats_text += f"  Optimal: {optimal:.1f}m\n"
        stats_text += f"  Sub-opt: {path_length/optimal:.3f}\n\n"
        
        stats_text += f"Timing:\n"
        stats_text += f"  Total time: {total_time:.2f}s\n"
        stats_text += f"  Avg speed: {path_length/total_time:.2f}m/s\n\n"
        
        stats_text += f"Dynamic:\n"
        stats_text += f"  Moving obs: {len(grid.dynamic_obstacles)}\n"
        stats_text += f"  Static obs: {len(grid.static_obstacles)}\n"
    else:
        stats_text += f"Iterations: {stats['iterations']}\n"
        stats_text += f"Nodes: {stats['nodes_generated']}\n"
        stats_text += f"\nCould not find safe path"
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig


def create_animated_dynamic_avoidance(scenario, path):
    """Create animation showing dynamic obstacle avoidance."""
    
    if not path:
        print("No path to animate")
        return None
    
    grid = scenario['grid']
    duration = scenario['duration']
    
    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Calculate actual simulation time from path
    max_time = path[-1][3]
    
    # LEFT: Main view
    ax1.set_xlim(grid.x_min - 2, grid.x_max + 2)
    ax1.set_ylim(grid.y_min - 2, grid.y_max + 2)
    ax1.set_xlabel('x [m]', fontsize=12)
    ax1.set_ylabel('y [m]', fontsize=12)
    ax1.set_title(f'{scenario["name"]} - Dynamic Navigation', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # RIGHT: Time evolution
    ax2.set_xlim(-1, max_time + 1)
    ax2.set_ylim(0, len(grid.dynamic_obstacles) + len(grid.static_obstacles) + 1)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Obstacle ID', fontsize=12)
    ax2.set_title('Collision Risk Over Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Static obstacles (don't move)
    static_circles = []
    for xc, yc, r in grid.static_obstacles:
        circle = Circle((xc, yc), r, facecolor='gray', alpha=0.5,
                       edgecolor='black', linewidth=2)
        ax1.add_patch(circle)
        static_circles.append(circle)
    
    # Dynamic obstacles (will be updated)
    dynamic_circles = []
    velocity_arrows = []
    for obs in grid.dynamic_obstacles:
        x, y = obs.get_position(0)
        circle = Circle((x, y), obs.r, facecolor='red', alpha=0.4,
                       edgecolor='darkred', linewidth=2)
        ax1.add_patch(circle)
        dynamic_circles.append(circle)
        
        arrow = ax1.arrow(x, y, 0, 0, head_width=0.5, head_length=0.3,
                         fc='red', ec='darkred', alpha=0.6)
        velocity_arrows.append(arrow)
    
    # Robot
    robot = Circle((path[0][0], path[0][1]), 0.7, facecolor='blue', 
                  alpha=0.7, edgecolor='darkblue', linewidth=2, zorder=15)
    ax1.add_patch(robot)
    
    # Robot direction arrow
    psi0 = path[0][2]
    dx0 = 1.5 * np.cos(psi0)
    dy0 = 1.5 * np.sin(psi0)
    robot_arrow = ax1.arrow(path[0][0], path[0][1], dx0, dy0,
                           head_width=0.5, head_length=0.3,
                           fc='darkblue', ec='darkblue', zorder=16)
    
    # Trail
    trail_line, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.6, 
                          label='Robot path')
    
    # Start/Goal
    ax1.plot(scenario['start'][0], scenario['start'][1], 'go',
            markersize=10, label='Start', zorder=10)
    ax1.plot(scenario['goal'][0], scenario['goal'][1], 'r*',
            markersize=12, label='Goal', zorder=10)
    ax1.legend(loc='upper left', fontsize=9)
    
    # Collision risk plot elements
    risk_lines = []
    for i in range(len(grid.dynamic_obstacles)):
        line, = ax2.plot([], [], 'o-', label=f'Obs {i+1}')
        risk_lines.append(line)
    
    robot_risk, = ax2.plot([], [], 'b-', linewidth=2, label='Robot')
    ax2.legend(fontsize=8)
    ax2.set_yticks(range(len(grid.dynamic_obstacles) + 1))
    ax2.set_yticklabels(['Robot'] + [f'Obs{i+1}' for i in range(len(grid.dynamic_obstacles))])
    
    # Time text
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Animation data
    xs_path = [p[0] for p in path]
    ys_path = [p[1] for p in path]
    ts_path = [p[3] for p in path]
    psis_path = [p[2] for p in path]
    
    # Precompute collision risks
    risk_data = {i: {'times': [], 'risks': []} for i in range(len(grid.dynamic_obstacles))}
    robot_risks = {'times': [], 'risks': []}
    
    def animate(frame):
        nonlocal robot_arrow  # Allow modifying outer scope variable
        
        # Determine current time
        if frame >= len(path):
            return []
        
        t_current = ts_path[frame]
        
        # Update dynamic obstacles
        for i, obs in enumerate(grid.dynamic_obstacles):
            x, y = obs.get_position(t_current)
            dynamic_circles[i].center = (x, y)
            
            # Update velocity arrow
            vx, vy = obs.get_velocity(t_current)
            if velocity_arrows[i] in ax1.patches:
                velocity_arrows[i].remove()
            velocity_arrows[i] = ax1.arrow(x, y, vx*0.5, vy*0.5,
                                          head_width=0.5, head_length=0.3,
                                          fc='red', ec='darkred', alpha=0.6)
        
        # Update robot
        robot.center = (xs_path[frame], ys_path[frame])
        
        # Update robot arrow
        if robot_arrow in ax1.patches:
            robot_arrow.remove()
        dx = 1.5 * np.cos(psis_path[frame])
        dy = 1.5 * np.sin(psis_path[frame])
        robot_arrow = ax1.arrow(xs_path[frame], ys_path[frame], dx, dy,
                               head_width=0.5, head_length=0.3,
                               fc='darkblue', ec='darkblue', zorder=16)
        
        # Update trail
        trail_line.set_data(xs_path[:frame+1], ys_path[:frame+1])
        
        # Update time text
        time_text.set_text(f'Time: {t_current:.2f}s')
        
        # Update risk plot
        robot_x, robot_y = xs_path[frame], ys_path[frame]
        
        for i, obs in enumerate(grid.dynamic_obstacles):
            obs_x, obs_y = obs.get_position(t_current)
            distance = np.hypot(robot_x - obs_x, robot_y - obs_y)
            risk = max(0, 5 - distance)  # Risk increases as distance decreases
            
            risk_data[i]['times'].append(t_current)
            risk_data[i]['risks'].append(risk)
            
            risk_lines[i].set_data(risk_data[i]['times'], risk_data[i]['risks'])
        
        # Robot overall risk
        max_risk = max([max(0, 5 - np.hypot(robot_x - obs.get_position(t_current)[0],
                                             robot_y - obs.get_position(t_current)[1]))
                       for obs in grid.dynamic_obstacles]) if grid.dynamic_obstacles else 0
        
        robot_risks['times'].append(t_current)
        robot_risks['risks'].append(max_risk)
        robot_risk.set_data(robot_risks['times'], robot_risks['risks'])
        
        return [robot, robot_arrow, trail_line, time_text] + dynamic_circles + velocity_arrows
    
    ani = FuncAnimation(fig, animate, frames=len(path),
                       interval=ANIMATION_SPEED, blit=False, repeat=True)
    
    return fig, ani


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run dynamic obstacle avoidance demo."""
    
    print("\n" + "="*70)
    print(" DYNAMIC OBSTACLE AVOIDANCE - Space-Time Planning")
    print("="*70)
    print("\nPlanning paths that avoid MOVING obstacles!")
    print("="*70 + "\n")
    
    # Create scenarios
    scenarios = [
        DynamicScenarios.crossing_obstacles(),
        DynamicScenarios.orbiting_obstacles(),
        DynamicScenarios.complex_dynamic(),
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/3] Planning: {scenario['name']}")
        print(f"      {scenario['description']}")
        
        start_time = time.time()
        path, stats = dynamic_hybrid_astar(
            scenario['grid'],
            scenario['start'],
            scenario['goal'],
            L=2.5,
            v_step=3.0,
            yaw_bins=12,
            max_iter=10000
        )
        elapsed = time.time() - start_time
        
        status = "✓ SUCCESS" if path else "✗ FAILED"
        print(f"      Result: {status} ({elapsed:.2f}s, {stats['iterations']} iters)")
        
        if path:
            path_length = sum(np.hypot(path[i+1][0] - path[i][0],
                                       path[i+1][1] - path[i][1])
                             for i in range(len(path) - 1))
            print(f"      Path: {path_length:.1f}m in {path[-1][3]:.2f}s")
        print()
        
        results.append((scenario, path, stats))
    
    # Generate visualizations
    print("="*70)
    print(" GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    for i, (scenario, path, stats) in enumerate(results, 1):
        print(f"[{i}/3] Snapshot: {scenario['name']}")
        fig = create_dynamic_visualization(scenario, path, stats)
        filename = f"dynamic_avoidance_{i}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"      Saved: {filename}")
    
    # Create and play animation
    print("\n" + "="*70)
    print(" PLAYING ANIMATION")
    print("="*70 + "\n")
    
    for scenario, path, stats in results:
        if path:
            print(f"Animating: {scenario['name']}")
            print("  → Creating animation...")
            
            try:
                fig_anim, ani = create_animated_dynamic_avoidance(scenario, path)
                
                print("\n  ✓ Animation created!")
                print("  → Opening animation window...")
                print("     - Animation will play automatically")
                print("     - Use toolbar to pause/play/zoom")
                print("     - Close window when done\n")
                
                # Make sure window stays open
                plt.show(block=True)
                
                print("\n  ✓ Animation complete!")
                
            except Exception as e:
                print(f"\n  ✗ Animation failed: {e}")
                print("  → Creating static snapshots instead...")
                
                # Fallback: Show multiple frames as static images
                create_frame_snapshots(scenario, path)
                
            break
    
    print("\n" + "="*70)
    print(" ✅ COMPLETE!")
    print("="*70)
    print("\n📁 Output Files:")
    for i in range(1, 4):
        print(f"  - dynamic_avoidance_{i}.png (Snapshot)")
    
    print("\n🎬 Animation:")
    print("  - Played in interactive window")
    print("  - Close window to exit")
    
    print("\n✅ Dynamic obstacle avoidance complete!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted\n")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()