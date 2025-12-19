

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class PathAnalyzer:
    """Analyze path quality and characteristics."""
    
    @staticmethod
    def compute_path_length(path: List[Tuple[float, float, float]]) -> float:
        """Compute total path length."""
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.hypot(dx, dy)
        
        return length
    
    @staticmethod
    def compute_straightline_distance(start, goal) -> float:
        """Compute straight-line distance from start to goal."""
        return np.hypot(goal[0] - start[0], goal[1] - start[1])
    
    @staticmethod
    def compute_suboptimality(path_length: float, optimal_length: float) -> float:
        """
        Compute sub-optimality ratio.
        
        Sub-optimality = path_length / optimal_length
        
        Perfect: 1.0
        Typical for A*: 1.0-1.2
        Acceptable: < 1.5
        """
        if optimal_length == 0:
            return float('inf')
        return path_length / optimal_length
    
    @staticmethod
    def compute_smoothness(path: List[Tuple[float, float, float]]) -> Dict:
        """
        Compute path smoothness metrics.
        
        Returns:
        - heading_changes: List of heading changes between segments
        - curvature: Approximate curvature at each point
        - max_curvature: Maximum curvature
        - avg_abs_curvature: Average absolute curvature
        """
        if len(path) < 3:
            return {
                'heading_changes': [],
                'curvature': [],
                'max_curvature': 0.0,
                'avg_abs_curvature': 0.0,
                'smoothness_score': 1.0
            }
        
        # Heading changes
        heading_changes = []
        for i in range(len(path) - 1):
            psi_diff = abs(path[i+1][2] - path[i][2])
            # Wrap to [-pi, pi]
            if psi_diff > np.pi:
                psi_diff = 2*np.pi - psi_diff
            heading_changes.append(psi_diff)
        
        # Approximate curvature (change in heading / distance)
        curvatures = []
        for i in range(len(path) - 1):
            ds = np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if ds > 1e-6:
                kappa = heading_changes[i] / ds
                curvatures.append(kappa)
        
        max_curv = max(curvatures) if curvatures else 0.0
        avg_curv = np.mean(np.abs(curvatures)) if curvatures else 0.0
        
        # Smoothness score (lower curvature = smoother)
        smoothness_score = 1.0 / (1.0 + avg_curv)
        
        return {
            'heading_changes': heading_changes,
            'curvature': curvatures,
            'max_curvature': max_curv,
            'avg_abs_curvature': avg_curv,
            'smoothness_score': smoothness_score
        }
    
    @staticmethod
    def compute_kinematic_feasibility(path: List[Tuple[float, float, float]], 
                                     L: float, 
                                     max_steering: float = 0.6) -> Dict:
        """
        Check if path respects kinematic constraints.
        
        For Ackermann steering:
        κ = tan(δ) / L
        
        Max curvature: κ_max = tan(δ_max) / L
        """
        max_kappa_allowed = abs(np.tan(max_steering) / L)
        
        smoothness = PathAnalyzer.compute_smoothness(path)
        curvatures = smoothness['curvature']
        
        violations = [k for k in curvatures if k > max_kappa_allowed]
        
        return {
            'max_curvature_allowed': max_kappa_allowed,
            'max_curvature_actual': max(curvatures) if curvatures else 0.0,
            'num_violations': len(violations),
            'feasible': len(violations) == 0,
            'violation_percentage': len(violations) / len(curvatures) * 100 if curvatures else 0.0
        }


# =====================
# Planner Performance Analyzer
# =====================

class HybridAStarAnalyzer:
    """Comprehensive analysis of Hybrid A* planner performance."""
    
    def __init__(self):
        self.results = []
        
    def analyze_planning_result(self, 
                                path: Optional[List],
                                start: Tuple,
                                goal: Tuple,
                                planning_time: float,
                                iterations: int,
                                nodes_expanded: int) -> Dict:
        """
        Analyze a single planning result.
        
        Returns comprehensive metrics about the planning attempt.
        """
        result = {
            'success': path is not None,
            'planning_time': planning_time,
            'iterations': iterations,
            'nodes_expanded': nodes_expanded,
        }
        
        if path is None:
            result.update({
                'path_length': float('inf'),
                'path_nodes': 0,
                'suboptimality': float('inf'),
                'smoothness_score': 0.0,
                'max_curvature': float('inf'),
                'feasible': False
            })
        else:
            # Path quality
            path_length = PathAnalyzer.compute_path_length(path)
            optimal_length = PathAnalyzer.compute_straightline_distance(start, goal[:2])
            subopt = PathAnalyzer.compute_suboptimality(path_length, optimal_length)
            
            smoothness = PathAnalyzer.compute_smoothness(path)
            feasibility = PathAnalyzer.compute_kinematic_feasibility(path, L=2.5)
            
            result.update({
                'path_length': path_length,
                'path_nodes': len(path),
                'optimal_length': optimal_length,
                'suboptimality': subopt,
                'smoothness_score': smoothness['smoothness_score'],
                'max_curvature': smoothness['max_curvature'],
                'avg_curvature': smoothness['avg_abs_curvature'],
                'kinematic_feasible': feasibility['feasible'],
                'curvature_violations': feasibility['num_violations']
            })
        
        return result
    
    def parameter_sensitivity_analysis(self, 
                                      planner_func,
                                      grid,
                                      start,
                                      goal,
                                      base_params: Dict) -> Dict:
        """
        Analyze sensitivity to planner parameters.
        
        Tests:
        - Resolution sensitivity
        - Delta_set sensitivity (steering discretization)
        - Yaw_bins sensitivity (heading discretization)
        - T_step sensitivity (motion primitive length)
        """
        results = {}
        
        # 1. Resolution sensitivity
        print("  → Testing resolution sensitivity...")
        res_values = [0.3, 0.5, 0.7, 1.0]
        results['resolution'] = self._test_parameter(
            planner_func, grid, start, goal, base_params,
            'resolution', res_values
        )
        
        # 2. Steering discretization
        print("  → Testing steering discretization...")
        delta_sets = [
            [-0.5, 0.0, 0.5],
            [-0.6, -0.3, 0.0, 0.3, 0.6],
            [-0.7, -0.4, -0.2, 0.0, 0.2, 0.4, 0.7],
        ]
        results['delta_set'] = self._test_delta_sets(
            planner_func, grid, start, goal, base_params, delta_sets
        )
        
        # 3. Yaw discretization
        print("  → Testing yaw discretization...")
        yaw_bins_values = [8, 16, 24, 36]
        results['yaw_bins'] = self._test_parameter(
            planner_func, grid, start, goal, base_params,
            'yaw_bins', yaw_bins_values
        )
        
        # 4. Motion primitive length
        print("  → Testing motion primitive length...")
        t_step_values = [0.5, 1.0, 1.5, 2.0]
        results['t_step'] = self._test_parameter(
            planner_func, grid, start, goal, base_params,
            'T_step', t_step_values
        )
        
        return results
    
    def _test_parameter(self, planner_func, grid, start, goal, 
                       base_params, param_name, values):
        """Test single parameter across multiple values."""
        results_list = []
        
        for val in values:
            params = base_params.copy()
            params[param_name] = val
            
            start_time = time.time()
            try:
                path = planner_func(grid, start, goal, **params)
                elapsed = time.time() - start_time
                
                if path:
                    result = {
                        'value': val,
                        'success': True,
                        'time': elapsed,
                        'path_length': PathAnalyzer.compute_path_length(path),
                        'path_nodes': len(path)
                    }
                else:
                    result = {
                        'value': val,
                        'success': False,
                        'time': elapsed,
                        'path_length': float('inf'),
                        'path_nodes': 0
                    }
            except Exception as e:
                result = {
                    'value': val,
                    'success': False,
                    'time': 0.0,
                    'error': str(e)
                }
            
            results_list.append(result)
        
        return results_list
    
    def _test_delta_sets(self, planner_func, grid, start, goal, 
                        base_params, delta_sets):
        """Test different steering angle sets."""
        results_list = []
        
        for delta_set in delta_sets:
            params = base_params.copy()
            params['delta_set'] = delta_set
            
            start_time = time.time()
            try:
                path = planner_func(grid, start, goal, **params)
                elapsed = time.time() - start_time
                
                if path:
                    smoothness = PathAnalyzer.compute_smoothness(path)
                    result = {
                        'n_actions': len(delta_set),
                        'success': True,
                        'time': elapsed,
                        'path_length': PathAnalyzer.compute_path_length(path),
                        'smoothness': smoothness['smoothness_score']
                    }
                else:
                    result = {
                        'n_actions': len(delta_set),
                        'success': False,
                        'time': elapsed
                    }
            except Exception as e:
                result = {
                    'n_actions': len(delta_set),
                    'success': False,
                    'error': str(e)
                }
            
            results_list.append(result)
        
        return results_list
    
    def scenario_testing(self, planner_func, scenarios: List[Dict]) -> Dict:
        """
        Test planner across multiple scenarios.
        
        Each scenario dict contains:
        - grid: GridWorld
        - start: (x, y, psi)
        - goal: (x, y)
        - name: str
        - params: Dict
        """
        results = []
        
        for scenario in scenarios:
            print(f"\n  Testing: {scenario['name']}")
            
            start_time = time.time()
            path = planner_func(
                scenario['grid'],
                scenario['start'],
                scenario['goal'],
                **scenario.get('params', {})
            )
            elapsed = time.time() - start_time
            
            result = {
                'name': scenario['name'],
                'success': path is not None,
                'time': elapsed
            }
            
            if path:
                path_length = PathAnalyzer.compute_path_length(path)
                optimal = PathAnalyzer.compute_straightline_distance(
                    scenario['start'], scenario['goal']
                )
                smoothness = PathAnalyzer.compute_smoothness(path)
                
                result.update({
                    'path_length': path_length,
                    'optimal_length': optimal,
                    'suboptimality': path_length / optimal if optimal > 0 else float('inf'),
                    'path_nodes': len(path),
                    'smoothness': smoothness['smoothness_score'],
                    'max_curvature': smoothness['max_curvature']
                })
            
            results.append(result)
        
        # Summary statistics
        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_time = np.mean([r['time'] for r in results])
        
        successful = [r for r in results if r['success']]
        if successful:
            avg_subopt = np.mean([r['suboptimality'] for r in successful])
            avg_smoothness = np.mean([r['smoothness'] for r in successful])
        else:
            avg_subopt = float('inf')
            avg_smoothness = 0.0
        
        summary = {
            'scenarios': results,
            'success_rate': success_rate,
            'avg_time': avg_time,
            'avg_suboptimality': avg_subopt,
            'avg_smoothness': avg_smoothness
        }
        
        return summary


# =====================
# Heuristic Analysis
# =====================

class HeuristicAnalyzer:
    """Analyze heuristic function effectiveness."""
    
    @staticmethod
    def compute_heuristic_accuracy(path: List[Tuple], 
                                   goal: Tuple,
                                   heuristic_func) -> Dict:
        """
        Analyze how well the heuristic estimates remaining cost.
        
        For each point on the path, compare:
        - h(point): Heuristic estimate
        - actual_cost: Actual cost-to-go from that point
        """
        if not path or len(path) < 2:
            return {
                'admissible': True,
                'consistency': True,
                'accuracy': 1.0
            }
        
        # Compute actual costs from each point
        actual_costs = [0.0]
        for i in range(len(path) - 1, 0, -1):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            actual_costs.insert(0, actual_costs[0] + np.hypot(dx, dy))
        
        # Compute heuristic values
        heuristic_values = [
            heuristic_func(p[0], p[1], goal)
            for p in path
        ]
        
        # Check admissibility: h(n) <= h*(n) for all n
        admissible = all(h <= actual + 1e-6 
                        for h, actual in zip(heuristic_values, actual_costs))
        
        # Check consistency: h(n) <= c(n,n') + h(n') for all n,n'
        consistent = True
        for i in range(len(path) - 1):
            c_nn = np.hypot(path[i+1][0] - path[i][0], 
                           path[i+1][1] - path[i][1])
            if heuristic_values[i] > c_nn + heuristic_values[i+1] + 1e-6:
                consistent = False
                break
        
        # Accuracy: how close is h to h* on average
        errors = [abs(h - actual) for h, actual in zip(heuristic_values, actual_costs)]
        avg_error = np.mean(errors)
        accuracy = 1.0 / (1.0 + avg_error)
        
        return {
            'admissible': admissible,
            'consistent': consistent,
            'accuracy': accuracy,
            'avg_error': avg_error,
            'max_error': max(errors) if errors else 0.0
        }


# =====================
# Visualization
# =====================

class PlannerVisualizer:
    """Visualization tools for planner analysis."""
    
    @staticmethod
    def plot_parameter_sensitivity(sensitivity_results: Dict, param_name: str):
        """Plot how performance varies with parameter."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        results = sensitivity_results[param_name]
        values = [r['value'] for r in results]
        
        # Success rate
        ax = axes[0, 0]
        success = [1 if r['success'] else 0 for r in results]
        ax.plot(values, success, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Success (1) / Failure (0)')
        ax.set_title(f'Success vs {param_name}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])
        
        # Planning time
        ax = axes[0, 1]
        times = [r.get('time', 0) for r in results if r['success']]
        values_success = [r['value'] for r in results if r['success']]
        if times:
            ax.plot(values_success, times, 'o-', linewidth=2, markersize=8, color='orange')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Planning Time [s]')
            ax.set_title(f'Time vs {param_name}')
            ax.grid(True, alpha=0.3)
        
        # Path length
        ax = axes[1, 0]
        lengths = [r.get('path_length', 0) for r in results if r['success']]
        if lengths:
            ax.plot(values_success, lengths, 'o-', linewidth=2, markersize=8, color='green')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Path Length [m]')
            ax.set_title(f'Path Length vs {param_name}')
            ax.grid(True, alpha=0.3)
        
        # Path nodes
        ax = axes[1, 1]
        nodes = [r.get('path_nodes', 0) for r in results if r['success']]
        if nodes:
            ax.plot(values_success, nodes, 'o-', linewidth=2, markersize=8, color='purple')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Path Nodes')
            ax.set_title(f'Path Complexity vs {param_name}')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Parameter Sensitivity: {param_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_scenario_comparison(scenario_results: Dict):
        """Compare planner performance across scenarios."""
        scenarios = scenario_results['scenarios']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        names = [s['name'] for s in scenarios]
        y_pos = np.arange(len(names))
        
        # Success/Failure
        ax = axes[0, 0]
        success = [1 if s['success'] else 0 for s in scenarios]
        colors = ['green' if s else 'red' for s in success]
        ax.barh(y_pos, success, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim([0, 1.2])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Fail', 'Success'])
        ax.set_title('Success Rate by Scenario')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Planning time
        ax = axes[0, 1]
        times = [s.get('time', 0) for s in scenarios]
        ax.barh(y_pos, times, alpha=0.7, color='orange')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Time [s]')
        ax.set_title('Planning Time')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Sub-optimality
        ax = axes[1, 0]
        successful = [s for s in scenarios if s['success']]
        if successful:
            subopt = [s.get('suboptimality', 0) for s in successful]
            names_success = [s['name'] for s in successful]
            y_pos_success = np.arange(len(names_success))
            ax.barh(y_pos_success, subopt, alpha=0.7, color='purple')
            ax.set_yticks(y_pos_success)
            ax.set_yticklabels(names_success, fontsize=8)
            ax.set_xlabel('Sub-optimality Ratio')
            ax.set_title('Path Quality (1.0 = optimal)')
            ax.axvline(1.0, color='g', linestyle='--', linewidth=2, alpha=0.5)
            ax.grid(True, axis='x', alpha=0.3)
        
        # Smoothness
        ax = axes[1, 1]
        if successful:
            smoothness = [s.get('smoothness', 0) for s in successful]
            ax.barh(y_pos_success, smoothness, alpha=0.7, color='blue')
            ax.set_yticks(y_pos_success)
            ax.set_yticklabels(names_success, fontsize=8)
            ax.set_xlabel('Smoothness Score')
            ax.set_title('Path Smoothness (1.0 = smoothest)')
            ax.grid(True, axis='x', alpha=0.3)
        
        fig.suptitle('Scenario Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_path_analysis(path: List[Tuple], start: Tuple, goal: Tuple):
        """Detailed analysis of a single path."""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        psis = [p[2] for p in path]
        
        # 1. Path in XY plane
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(xs, ys, 'b-o', linewidth=2, markersize=4, label='Path')
        ax1.plot(start[0], start[1], 'go', markersize=10, label='Start')
        ax1.plot(goal[0], goal[1], 'rx', markersize=12, label='Goal')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.set_title('Path in XY Plane')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # 2. Curvature profile
        smoothness = PathAnalyzer.compute_smoothness(path)
        curvatures = smoothness['curvature']
        
        ax2 = fig.add_subplot(gs[0, 2])
        if curvatures:
            ax2.plot(curvatures, 'r-', linewidth=2)
            ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax2.set_xlabel('Segment')
            ax2.set_ylabel('Curvature [1/m]')
            ax2.set_title('Curvature Profile')
            ax2.grid(True, alpha=0.3)
        
        # 3. Heading profile
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(np.rad2deg(psis), 'b-', linewidth=2)
        ax3.set_xlabel('Node')
        ax3.set_ylabel('Heading [deg]')
        ax3.set_title('Heading Evolution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Heading changes
        ax4 = fig.add_subplot(gs[1, 1])
        heading_changes = smoothness['heading_changes']
        if heading_changes:
            ax4.plot(np.rad2deg(heading_changes), 'g-', linewidth=2)
            ax4.set_xlabel('Segment')
            ax4.set_ylabel('Heading Change [deg]')
            ax4.set_title('Heading Changes per Segment')
            ax4.grid(True, alpha=0.3)
        
        # 5. Speed profile (constant in this case)
        ax5 = fig.add_subplot(gs[1, 2])
        s_values = np.cumsum([0] + [np.hypot(xs[i+1]-xs[i], ys[i+1]-ys[i]) 
                                     for i in range(len(xs)-1)])
        ax5.plot(s_values, 'purple', linewidth=2)
        ax5.set_xlabel('Node')
        ax5.set_ylabel('Arc Length [m]')
        ax5.set_title('Cumulative Distance')
        ax5.grid(True, alpha=0.3)
        
        # 6. Metrics summary
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        path_length = PathAnalyzer.compute_path_length(path)
        optimal = PathAnalyzer.compute_straightline_distance(start, goal[:2])
        subopt = path_length / optimal if optimal > 0 else float('inf')
        
        metrics_text = "PATH METRICS\n" + "="*60 + "\n\n"
        metrics_text += f"Path Length: {path_length:.2f} m\n"
        metrics_text += f"Straight-Line Distance: {optimal:.2f} m\n"
        metrics_text += f"Sub-optimality: {subopt:.3f} "
        metrics_text += f"({'✓ Good' if subopt < 1.3 else '⚠ High'})\n\n"
        
        metrics_text += f"Smoothness Score: {smoothness['smoothness_score']:.3f}\n"
        metrics_text += f"Max Curvature: {smoothness['max_curvature']:.4f} 1/m\n"
        metrics_text += f"Avg Curvature: {smoothness['avg_abs_curvature']:.4f} 1/m\n\n"
        
        metrics_text += f"Path Nodes: {len(path)}\n"
        metrics_text += f"Avg Segment Length: {path_length/(len(path)-1):.2f} m\n"
        
        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.suptitle('Detailed Path Analysis', fontsize=14, fontweight='bold')
        return fig


if __name__ == "__main__":
    print("Hybrid A* Motion Planner - Analysis Tools")
    print("Import this module to use analysis functions")
    print("\nKey classes:")
    print("  - PathAnalyzer: Analyze path quality")
    print("  - HybridAStarAnalyzer: Comprehensive planner analysis")
    print("  - HeuristicAnalyzer: Heuristic effectiveness")

    print("  - PlannerVisualizer: Visualization tools")
