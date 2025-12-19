

import numpy as np
import time
from typing import List, Tuple, Dict
import heapq
from dataclasses import dataclass

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
    
    def add_circular_obstacle(self, xc, yc, r):
        self.obstacles.append((xc, yc, r))
        for ix in range(self.nx):
            for iy in range(self.ny):
                x, y = self.index_to_world(ix, iy)
                if np.hypot(x - xc, y - yc) <= r:
                    self.occ[ix, iy] = True
    
    def add_rectangular_obstacle(self, x_min, x_max, y_min, y_max):
        """Add rectangular obstacle."""
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


# Simplified planner for testing
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
    """Instrumented version that returns additional statistics."""
    
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
    
    # Statistics
    nodes_expanded = 0
    
    for it in range(max_iter):
        if not pq:
            return None, {'iterations': it, 'nodes_expanded': nodes_expanded, 'nodes_generated': len(nodes)}
        
        cur = heapq.heappop(pq)
        n_idx = cur.idx
        node = nodes[n_idx]
        
        x, y, psi = node.x, node.y, node.psi
        
        if np.hypot(x - goal_xy[0], y - goal_xy[1]) <= goal_tolerance:
            # Reconstruct path
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
        
        # Expand
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


# =====================
# Test Scenarios
# =====================

class ScenarioGenerator:
    """Generate test scenarios of varying difficulty."""
    
    @staticmethod
    def empty_space():
        """Simple empty space - should find near-optimal path."""
        grid = GridWorld(0, 40, -10, 10, resolution=0.5)
        start = (2.0, 0.0, 0.0)
        goal = (35.0, 0.0)
        return {
            'name': 'Empty Space',
            'grid': grid,
            'start': start,
            'goal': goal,
            'difficulty': 'easy',
            'expected_success': True
        }
    
    @staticmethod
    def single_obstacle():
        """Single obstacle requiring deviation."""
        grid = GridWorld(0, 40, -10, 10, resolution=0.5)
        grid.add_circular_obstacle(20, 0, 3)
        start = (2.0, 0.0, 0.0)
        goal = (35.0, 0.0)
        return {
            'name': 'Single Obstacle',
            'grid': grid,
            'start': start,
            'goal': goal,
            'difficulty': 'easy',
            'expected_success': True
        }
    
    @staticmethod
    def cluttered():
        """Multiple obstacles - moderate difficulty."""
        grid = GridWorld(0, 40, -10, 10, resolution=0.5)
        grid.add_circular_obstacle(15, 0, 3)
        grid.add_circular_obstacle(25, 4, 2)
        grid.add_circular_obstacle(30, -3, 2)
        grid.add_circular_obstacle(20, -5, 1.5)
        start = (2.0, -5.0, 0.0)
        goal = (35.0, 5.0)
        return {
            'name': 'Cluttered',
            'grid': grid,
            'start': start,
            'goal': goal,
            'difficulty': 'medium',
            'expected_success': True
        }
    
    @staticmethod
    def narrow_passage():
        """Narrow passage requiring precise navigation."""
        grid = GridWorld(0, 40, -10, 10, resolution=0.5)
        # Create walls with narrow gap
        grid.add_rectangular_obstacle(18, 20, -10, -2)
        grid.add_rectangular_obstacle(18, 20, 3, 10)
        start = (2.0, 0.0, 0.0)
        goal = (35.0, 0.0)
        return {
            'name': 'Narrow Passage',
            'grid': grid,
            'start': start,
            'goal': goal,
            'difficulty': 'hard',
            'expected_success': True
        }
    
    @staticmethod
    def u_shaped_obstacle():
        """U-shaped obstacle - requires going around."""
        grid = GridWorld(0, 40, -10, 10, resolution=0.5)
        grid.add_rectangular_obstacle(15, 17, -8, 3)
        grid.add_rectangular_obstacle(15, 28, 3, 5)
        grid.add_rectangular_obstacle(26, 28, -8, 5)
        start = (2.0, 0.0, 0.0)
        goal = (22.0, -2.0)
        return {
            'name': 'U-Shaped Obstacle',
            'grid': grid,
            'start': start,
            'goal': goal,
            'difficulty': 'hard',
            'expected_success': True
        }
    
    @staticmethod
    def tight_turns():
        """Requires multiple tight turns."""
        grid = GridWorld(0, 40, -10, 10, resolution=0.5)
        grid.add_rectangular_obstacle(10, 30, 2, 10)
        grid.add_rectangular_obstacle(10, 30, -10, -3)
        start = (2.0, 0.0, 0.0)
        goal = (35.0, 0.0)
        return {
            'name': 'Tight Turns',
            'grid': grid,
            'start': start,
            'goal': goal,
            'difficulty': 'hard',
            'expected_success': True
        }
    
    @staticmethod
    def impossible():
        """Blocked goal - should fail gracefully."""
        grid = GridWorld(0, 40, -10, 10, resolution=0.5)
        grid.add_circular_obstacle(35, 0, 8)
        start = (2.0, 0.0, 0.0)
        goal = (35.0, 0.0)
        return {
            'name': 'Impossible (Blocked Goal)',
            'grid': grid,
            'start': start,
            'goal': goal,
            'difficulty': 'impossible',
            'expected_success': False
        }


# =====================
# Testing Framework
# =====================

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.details = ""
        
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.name}\n  {self.details}"


class HybridAStarTester:
    """Comprehensive testing suite."""
    
    def __init__(self):
        self.results = []
    
    def test_basic_scenarios(self):
        """Test 1: Basic scenario coverage."""
        result = TestResult("Basic Scenario Coverage")
        
        scenarios = [
            ScenarioGenerator.empty_space(),
            ScenarioGenerator.single_obstacle(),
            ScenarioGenerator.cluttered(),
        ]
        
        all_passed = True
        details = []
        
        for scenario in scenarios:
            start_time = time.time()
            path, stats = hybrid_astar_instrumented(
                scenario['grid'],
                scenario['start'],
                scenario['goal'],
                L=2.5,
                v_step=3.0,
                max_iter=5000
            )
            elapsed = time.time() - start_time
            
            success = path is not None
            expected = scenario['expected_success']
            passed = success == expected
            
            all_passed = all_passed and passed
            
            status = "✓" if passed else "✗"
            details.append(
                f"  {status} {scenario['name']:20s}: "
                f"{'Success' if success else 'Failed':8s} "
                f"({elapsed:.2f}s, {stats['iterations']} iters)"
            )
        
        result.passed = all_passed
        result.details = "\n".join(details)
        self.results.append(result)
        return result
    
    def test_difficult_scenarios(self):
        """Test 2: Challenging scenarios."""
        result = TestResult("Difficult Scenarios")
        
        scenarios = [
            ScenarioGenerator.narrow_passage(),
            ScenarioGenerator.u_shaped_obstacle(),
            ScenarioGenerator.tight_turns(),
        ]
        
        all_passed = True
        details = []
        
        for scenario in scenarios:
            start_time = time.time()
            path, stats = hybrid_astar_instrumented(
                scenario['grid'],
                scenario['start'],
                scenario['goal'],
                L=2.5,
                v_step=2.0,
                max_iter=10000,
                yaw_bins=24  # Finer resolution
            )
            elapsed = time.time() - start_time
            
            success = path is not None
            passed = success == scenario['expected_success']
            
            all_passed = all_passed and passed
            
            status = "✓" if passed else "✗"
            details.append(
                f"  {status} {scenario['name']:20s}: "
                f"{'Success' if success else 'Failed':8s} "
                f"({elapsed:.2f}s)"
            )
        
        result.passed = all_passed
        result.details = "\n".join(details)
        self.results.append(result)
        return result
    
    def test_impossible_scenario(self):
        """Test 3: Graceful failure on impossible scenarios."""
        result = TestResult("Impossible Scenario Handling")
        
        scenario = ScenarioGenerator.impossible()
        
        start_time = time.time()
        path, stats = hybrid_astar_instrumented(
            scenario['grid'],
            scenario['start'],
            scenario['goal'],
            max_iter=2000  # Limited iterations
        )
        elapsed = time.time() - start_time
        
        # Should fail (path == None)
        success = path is None
        
        result.passed = success
        result.details = (
            f"  Correctly identified impossible scenario: "
            f"{'✓ Yes' if success else '✗ No'}\n"
            f"  Terminated in {elapsed:.2f}s after {stats['iterations']} iterations"
        )
        
        self.results.append(result)
        return result
    
    def test_parameter_ranges(self):
        """Test 4: Robustness to parameter variations."""
        result = TestResult("Parameter Robustness")
        
        scenario = ScenarioGenerator.cluttered()
        
        # Test different parameter sets
        param_sets = [
            {'name': 'Coarse res', 'resolution': 1.0, 'yaw_bins': 12},
            {'name': 'Fine res', 'resolution': 0.3, 'yaw_bins': 36},
            {'name': 'Few actions', 'delta_set': [-0.5, 0.0, 0.5]},
            {'name': 'Many actions', 'delta_set': [-0.6, -0.3, 0.0, 0.3, 0.6]},
        ]
        
        all_passed = True
        details = []
        
        for params in param_sets:
            # Apply parameters to grid
            grid = GridWorld(0, 40, -10, 10, resolution=params.get('resolution', 0.5))
            grid.add_circular_obstacle(15, 0, 3)
            grid.add_circular_obstacle(25, 4, 2)
            grid.add_circular_obstacle(30, -3, 2)
            grid.add_circular_obstacle(20, -5, 1.5)
            
            path, stats = hybrid_astar_instrumented(
                grid,
                scenario['start'],
                scenario['goal'],
                yaw_bins=params.get('yaw_bins', 16),
                delta_set=params.get('delta_set', None),
                max_iter=5000
            )
            
            success = path is not None
            all_passed = all_passed and success
            
            status = "✓" if success else "✗"
            details.append(f"  {status} {params['name']:15s}: {'Success' if success else 'Failed'}")
        
        result.passed = all_passed
        result.details = "\n".join(details)
        self.results.append(result)
        return result
    
    def test_computational_efficiency(self):
        """Test 5: Computational efficiency benchmarks."""
        result = TestResult("Computational Efficiency")
        
        scenario = ScenarioGenerator.cluttered()
        
        # Run multiple times for statistics
        times = []
        iterations_list = []
        nodes_list = []
        
        n_runs = 5
        for _ in range(n_runs):
            start_time = time.time()
            path, stats = hybrid_astar_instrumented(
                scenario['grid'],
                scenario['start'],
                scenario['goal'],
                L=2.5,
                v_step=3.0,
                max_iter=5000
            )
            elapsed = time.time() - start_time
            
            if path:
                times.append(elapsed)
                iterations_list.append(stats['iterations'])
                nodes_list.append(stats['nodes_generated'])
        
        if times:
            avg_time = np.mean(times)
            avg_iters = np.mean(iterations_list)
            avg_nodes = np.mean(nodes_list)
            
            # Performance targets
            time_ok = avg_time < 2.0  # Should solve in <2s
            efficiency_ok = avg_nodes < 5000  # Reasonable node count
            
            result.passed = time_ok and efficiency_ok
            result.details = (
                f"  Avg Time: {avg_time:.3f}s "
                f"({'✓' if time_ok else '✗'} < 2s)\n"
                f"  Avg Iterations: {avg_iters:.0f}\n"
                f"  Avg Nodes Generated: {avg_nodes:.0f} "
                f"({'✓' if efficiency_ok else '✗'} < 5000)"
            )
        else:
            result.passed = False
            result.details = "  ✗ Failed to find path in any run"
        
        self.results.append(result)
        return result
    
    def test_path_quality(self):
        """Test 6: Path quality metrics."""
        result = TestResult("Path Quality")
        
        scenario = ScenarioGenerator.single_obstacle()
        
        path, stats = hybrid_astar_instrumented(
            scenario['grid'],
            scenario['start'],
            scenario['goal'],
            L=2.5,
            v_step=3.0
        )
        
        if path:
            from planner_analysis import PathAnalyzer
            
            # Compute metrics
            path_length = PathAnalyzer.compute_path_length(path)
            optimal = PathAnalyzer.compute_straightline_distance(
                scenario['start'], scenario['goal']
            )
            subopt = path_length / optimal
            
            smoothness = PathAnalyzer.compute_smoothness(path)
            feasibility = PathAnalyzer.compute_kinematic_feasibility(path, L=2.5)
            
            # Quality checks
            subopt_ok = subopt < 1.5  # Not too suboptimal
            smooth_ok = smoothness['smoothness_score'] > 0.5
            feasible_ok = feasibility['feasible']
            
            result.passed = subopt_ok and smooth_ok and feasible_ok
            result.details = (
                f"  Sub-optimality: {subopt:.3f} "
                f"({'✓' if subopt_ok else '✗'} < 1.5)\n"
                f"  Smoothness: {smoothness['smoothness_score']:.3f} "
                f"({'✓' if smooth_ok else '✗'} > 0.5)\n"
                f"  Kinematically Feasible: "
                f"{'✓ Yes' if feasible_ok else '✗ No'}"
            )
        else:
            result.passed = False
            result.details = "  ✗ Failed to find path"
        
        self.results.append(result)
        return result
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "="*70)
        print("  COMPREHENSIVE HYBRID A* PLANNER VALIDATION SUITE")
        print("="*70)
        print("\nTesting motion planning algorithm...\n")
        
        start_time = time.time()
        
        tests = [
            ("Basic Scenarios", self.test_basic_scenarios),
            ("Difficult Scenarios", self.test_difficult_scenarios),
            ("Impossible Scenario", self.test_impossible_scenario),
            ("Parameter Robustness", self.test_parameter_ranges),
            ("Computational Efficiency", self.test_computational_efficiency),
            ("Path Quality", self.test_path_quality),
        ]
        
        for i, (name, test_func) in enumerate(tests, 1):
            print(f"\n[Test {i}/{len(tests)}] {name}...")
            print("-" * 70)
            result = test_func()
            print(result)
        
        elapsed = time.time() - start_time
        
        # Summary
        print("\n" + "="*70)
        print("  TEST SUMMARY")
        print("="*70)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"\nTests passed: {passed}/{total} ({passed/total*100:.0f}%)")
        print(f"Execution time: {elapsed:.1f}s\n")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - Planner is validated! ✓")
        elif passed >= total * 0.8:
            print("⚠️  MOST TESTS PASSED - Minor issues detected")
        else:
            print("❌ MULTIPLE FAILURES - Planner needs tuning")
        
        print("\n" + "="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    tester = HybridAStarTester()
    results = tester.run_all_tests()
    

    print("✅ Validation complete!")
