#!/usr/bin/env python3
"""
Comprehensive Stability Analysis for LQR Steering Controller

This module analyzes the stability and performance of the LQR lateral controller through:
1. Closed-loop eigenvalue analysis
2. Frequency domain analysis (Bode plots, margins)
3. Time domain performance metrics
4. Q/R weight sensitivity analysis
5. Robustness testing across operating points
6. Comparison with different weight matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg
from scipy.linalg import solve_continuous_are
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ========================
# LQR System Model
# ========================

class LinearizedLateralDynamics:
    """
    Linearized lateral error dynamics for Ackermann vehicle.
    
    State: x = [e_y, e_psi]^T (lateral error, heading error)
    Input: u = delta_tilde (corrective steering)
    
    Dynamics:
    ẋ = A*x + B*u
    
    where:
    A = [[0,  v  ],
         [0,  0  ]]
    
    B = [[0    ],
         [v/L  ]]
    """
    
    def __init__(self, v, L):
        self.v = v
        self.L = L
        self.A = np.array([[0.0, v],
                          [0.0, 0.0]])
        self.B = np.array([[0.0],
                          [v / L]])
        
    def get_state_space(self):
        """Return A, B matrices"""
        return self.A, self.B
    
    def is_controllable(self):
        """Check controllability"""
        controllability_matrix = np.hstack([self.B, self.A @ self.B])
        rank = np.linalg.matrix_rank(controllability_matrix)
        return rank == self.A.shape[0]


class LQRController:
    """
    LQR controller for lateral tracking.
    
    Solves: min J = ∫(x'Qx + u'Ru)dt
    Subject to: ẋ = Ax + Bu
    
    Solution: u = -Kx, where K = R^(-1)B'P
    P solves: A'P + PA - PBR^(-1)B'P + Q = 0 (Algebraic Riccati Equation)
    """
    
    def __init__(self, v, L, Q=None, R=None):
        self.v = v
        self.L = L
        
        # Default weight matrices
        if Q is None:
            Q = np.diag([10.0, 5.0])  # [lateral error, heading error]
        if R is None:
            R = np.array([[1.0]])
        elif np.isscalar(R):
            R = np.array([[R]])
            
        self.Q = Q
        self.R = R
        
        # System matrices
        self.A = np.array([[0.0, v],
                          [0.0, 0.0]])
        self.B = np.array([[0.0],
                          [v / L]])
        
        # Solve Riccati equation
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        
        # Compute gain
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P  # shape (1, 2)
        
        # Closed-loop system
        self.A_cl = self.A - self.B @ self.K
        
    def get_closed_loop_system(self):
        """Return closed-loop A matrix"""
        return self.A_cl
    
    def get_gain(self):
        """Return feedback gain K"""
        return self.K
    
    def get_cost_matrix(self):
        """Return solution to ARE"""
        return self.P


# ========================
# Stability Analysis
# ========================

class LQRStabilityAnalyzer:
    """Comprehensive stability analysis for LQR controller."""
    
    def __init__(self, v=5.0, L=2.5):
        self.v = v
        self.L = L
        
    def analyze_eigenvalues(self, Q, R):
        """
        Analyze closed-loop eigenvalues.
        
        LQR guarantees stability if:
        1. (A, B) is controllable
        2. Q ≥ 0, R > 0
        
        Returns eigenvalues and stability metrics.
        """
        lqr = LQRController(self.v, self.L, Q, R)
        A_cl = lqr.get_closed_loop_system()
        K = lqr.get_gain()
        
        # Eigenvalues of closed-loop system
        eigs = linalg.eigvals(A_cl)
        
        # Stability check
        stable = np.all(eigs.real < 0)
        
        # Stability margins
        max_real_part = np.max(eigs.real)
        stability_margin = -max_real_part
        
        # Damping and natural frequency for complex pairs
        damping_ratios = []
        natural_freqs = []
        
        for eig in eigs:
            if np.abs(eig.imag) > 1e-10:  # Complex eigenvalue
                wn = np.abs(eig)  # Natural frequency
                zeta = -eig.real / wn  # Damping ratio
                natural_freqs.append(wn)
                damping_ratios.append(zeta)
        
        return {
            'eigenvalues': eigs,
            'stable': stable,
            'max_real_part': max_real_part,
            'stability_margin': stability_margin,
            'gain': K,
            'damping_ratios': damping_ratios,
            'natural_frequencies': natural_freqs,
            'Q': Q,
            'R': R
        }
    
    def frequency_domain_analysis(self, Q, R):
        """
        Frequency domain analysis: Bode plots and stability margins.
        """
        lqr = LQRController(self.v, self.L, Q, R)
        A_cl = lqr.get_closed_loop_system()
        K = lqr.get_gain()
        
        # Open-loop transfer function (loop transfer function)
        # L(s) = K(sI - A)^(-1)B
        A = lqr.A
        B = lqr.B
        C = K  # Output is control input
        D = np.array([[0.0]])
        
        sys_ol = signal.StateSpace(A, B, C, D)
        
        # Frequency range
        w = np.logspace(-2, 2, 500)
        
        # Bode plot
        w, mag, phase = signal.bode(sys_ol, w)
        
        # Gain and phase margins
        try:
            gm, pm, wgc, wpc = signal.margin(sys_ol)
            gm_db = 20 * np.log10(gm) if gm > 0 else -np.inf
        except:
            gm_db = np.nan
            pm = np.nan
            wgc = np.nan
            wpc = np.nan
        
        return {
            'frequencies': w,
            'magnitude_db': mag,
            'phase_deg': phase,
            'gain_margin_db': gm_db,
            'phase_margin_deg': pm,
            'gain_crossover_freq': wgc,
            'phase_crossover_freq': wpc
        }
    
    def step_response_metrics(self, Q, R, T_sim=10.0):
        """
        Time domain step response analysis.
        """
        lqr = LQRController(self.v, self.L, Q, R)
        A_cl = lqr.get_closed_loop_system()
        
        # Closed-loop transfer function (from disturbance to e_y)
        C = np.array([[1.0, 0.0]])  # Measure lateral error
        D = np.array([[0.0]])
        
        # Use B as disturbance input
        sys_cl = signal.StateSpace(A_cl, lqr.B, C, D)
        
        # Step response
        dt = 0.01
        t = np.arange(0, T_sim, dt)
        t_step, y_step = signal.step(sys_cl, T=t)
        
        # Calculate metrics
        final_value = y_step[-1]
        
        # Rise time (10% to 90%)
        if len(y_step) > 10:
            idx_10 = np.where(y_step >= 0.1 * final_value)[0]
            idx_90 = np.where(y_step >= 0.9 * final_value)[0]
            rise_time = (t_step[idx_90[0]] - t_step[idx_10[0]]) if len(idx_10) > 0 and len(idx_90) > 0 else np.nan
        else:
            rise_time = np.nan
        
        # Settling time (within 2%)
        settling_band = 0.02 * abs(final_value)
        settled = np.abs(y_step - final_value) <= settling_band
        if np.any(settled):
            settling_idx = np.where(settled)[0][0]
            if np.all(settled[settling_idx:]):
                settling_time = t_step[settling_idx]
            else:
                settling_time = np.nan
        else:
            settling_time = np.nan
        
        # Overshoot
        max_value = np.max(y_step)
        overshoot_pct = 100 * (max_value - final_value) / final_value if final_value != 0 else 0
        
        return {
            'time': t_step,
            'response': y_step,
            'rise_time': rise_time,
            'settling_time': settling_time,
            'overshoot_percent': overshoot_pct,
            'final_value': final_value
        }
    
    def velocity_stability_analysis(self, Q, R, v_range=None):
        """
        Analyze stability across different velocities.
        Shows how eigenvalues move with speed.
        """
        if v_range is None:
            v_range = np.linspace(1, 20, 50)
        
        eigenvalues_list = []
        stable_list = []
        stability_margins = []
        
        for v_test in v_range:
            analyzer_temp = LQRStabilityAnalyzer(v_test, self.L)
            result = analyzer_temp.analyze_eigenvalues(Q, R)
            eigenvalues_list.append(result['eigenvalues'])
            stable_list.append(result['stable'])
            stability_margins.append(result['stability_margin'])
        
        return {
            'velocities': v_range,
            'eigenvalues': eigenvalues_list,
            'stable': np.array(stable_list),
            'stability_margins': np.array(stability_margins)
        }
    
    def q_r_weight_analysis(self, q_lateral_range, q_heading_range, R_value=1.0):
        """
        Analyze effect of Q weights on closed-loop eigenvalues.
        
        Q = [[q_lateral, 0],
             [0, q_heading]]
        """
        q_lat_vals = np.logspace(q_lateral_range[0], q_lateral_range[1], 30)
        q_head_vals = np.logspace(q_heading_range[0], q_heading_range[1], 30)
        
        # Store dominant eigenvalue real part
        real_parts = np.zeros((len(q_head_vals), len(q_lat_vals)))
        
        for i, q_head in enumerate(q_head_vals):
            for j, q_lat in enumerate(q_lat_vals):
                Q = np.diag([q_lat, q_head])
                R = np.array([[R_value]])
                
                result = self.analyze_eigenvalues(Q, R)
                real_parts[i, j] = result['max_real_part']
        
        return {
            'q_lateral_values': q_lat_vals,
            'q_heading_values': q_head_vals,
            'real_parts': real_parts
        }
    
    def compare_weight_matrices(self, weight_configs):
        """
        Compare different Q, R weight configurations.
        
        weight_configs: List of (name, Q, R) tuples
        """
        results = []
        
        for name, Q, R in weight_configs:
            eig_result = self.analyze_eigenvalues(Q, R)
            freq_result = self.frequency_domain_analysis(Q, R)
            step_result = self.step_response_metrics(Q, R)
            
            results.append({
                'name': name,
                'Q': Q,
                'R': R,
                'eigenvalues': eig_result,
                'frequency': freq_result,
                'step': step_result
            })
        
        return results


# ========================
# Robustness Analysis
# ========================

class LQRRobustnessAnalyzer:
    """Analyze LQR robustness to parameter variations."""
    
    @staticmethod
    def parameter_sensitivity(v, L, Q, R, param_variations, n_samples=100):
        """
        Monte Carlo analysis with parameter uncertainty.
        
        Tests robustness to variations in:
        - Wheelbase (L)
        - Velocity (v)
        """
        results = {
            'stable_count': 0,
            'unstable_count': 0,
            'stability_margins': [],
            'max_real_parts': []
        }
        
        for _ in range(n_samples):
            # Add random variations
            L_test = L * (1 + np.random.uniform(-param_variations['L'], param_variations['L']))
            v_test = v * (1 + np.random.uniform(-param_variations['v'], param_variations['v']))
            
            analyzer = LQRStabilityAnalyzer(v_test, L_test)
            result = analyzer.analyze_eigenvalues(Q, R)
            
            if result['stable']:
                results['stable_count'] += 1
            else:
                results['unstable_count'] += 1
            
            results['stability_margins'].append(result['stability_margin'])
            results['max_real_parts'].append(result['max_real_part'])
        
        results['stability_rate'] = results['stable_count'] / n_samples
        results['stability_margins'] = np.array(results['stability_margins'])
        results['max_real_parts'] = np.array(results['max_real_parts'])
        
        return results


# ========================
# Visualization
# ========================

class LQRVisualizer:
    """Visualization tools for LQR analysis."""
    
    @staticmethod
    def plot_eigenvalue_analysis(result, title="LQR Eigenvalue Analysis"):
        """Plot eigenvalues in complex plane."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        eigs = result['eigenvalues']
        ax.scatter(eigs.real, eigs.imag, s=150, c='blue', marker='x', 
                  linewidths=3, label='Closed-loop poles', zorder=3)
        
        # Stability boundary
        ax.axvline(0, color='r', linestyle='--', alpha=0.5, linewidth=2, 
                  label='Stability boundary')
        
        # Add damping ratio lines
        for zeta in [0.3, 0.5, 0.7, 0.9]:
            theta = np.arccos(zeta)
            r = np.linspace(0, max(abs(eigs)) * 1.2, 100)
            x = -r * np.cos(theta)
            y1 = r * np.sin(theta)
            y2 = -r * np.sin(theta)
            ax.plot(x, y1, 'k--', alpha=0.2, linewidth=0.8)
            ax.plot(x, y2, 'k--', alpha=0.2, linewidth=0.8)
            if y1[-1] > 0:
                ax.text(x[-1], y1[-1], f'ζ={zeta}', fontsize=8, alpha=0.5)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Real Part', fontsize=12)
        ax.set_ylabel('Imaginary Part', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.axis('equal')
        
        # Status text
        stable_text = "✓ STABLE" if result['stable'] else "✗ UNSTABLE"
        color = 'green' if result['stable'] else 'red'
        
        info_text = f"{stable_text}\n\n"
        info_text += f"K = {result['gain'].flatten()}\n\n"
        
        if len(result['damping_ratios']) > 0:
            info_text += f"ζ = {result['damping_ratios'][0]:.3f}\n"
            info_text += f"ωₙ = {result['natural_frequencies'][0]:.3f} rad/s"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor=color, linewidth=2))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_frequency_response(result):
        """Plot Bode diagrams."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        w = result['frequencies']
        mag = result['magnitude_db']
        phase = result['phase_deg']
        
        # Magnitude plot
        ax1.semilogx(w, mag, 'b-', linewidth=2)
        ax1.grid(True, which='both', alpha=0.3)
        ax1.set_ylabel('Magnitude [dB]', fontsize=11)
        ax1.set_title('LQR Loop Transfer Function - Bode Diagram', fontsize=12, fontweight='bold')
        
        if not np.isnan(result['gain_margin_db']):
            ax1.axhline(result['gain_margin_db'], color='r', linestyle='--',
                       linewidth=1.5, label=f"GM = {result['gain_margin_db']:.2f} dB")
            ax1.legend(fontsize=9)
        
        # Phase plot
        ax2.semilogx(w, phase, 'b-', linewidth=2)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_xlabel('Frequency [rad/s]', fontsize=11)
        ax2.set_ylabel('Phase [deg]', fontsize=11)
        
        if not np.isnan(result['phase_margin_deg']):
            ax2.axhline(-180 + result['phase_margin_deg'], color='r', linestyle='--',
                       linewidth=1.5, label=f"PM = {result['phase_margin_deg']:.2f}°")
            ax2.legend(fontsize=9)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_velocity_stability(result):
        """Plot stability vs velocity."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        v = result['velocities']
        stable = result['stable']
        margins = result['stability_margins']
        eigenvalues = result['eigenvalues']
        
        # Stability regions
        ax1.fill_between(v, 0, 1, where=stable, alpha=0.3, color='g', 
                        label='Stable region')
        ax1.fill_between(v, 0, 1, where=~stable, alpha=0.3, color='r', 
                        label='Unstable region')
        ax1.set_ylabel('Stability', fontsize=11)
        ax1.set_ylim([0, 1.1])
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Unstable', 'Stable'])
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.legend(fontsize=9)
        ax1.set_title('LQR Stability vs Velocity', fontsize=12, fontweight='bold')
        
        # Eigenvalue real parts
        for i, eig_list in enumerate(eigenvalues):
            for eig in eig_list:
                ax2.plot(v[i], eig.real, 'b.', markersize=4, alpha=0.6)
        
        ax2.axhline(0, color='r', linestyle='--', linewidth=2, label='Stability boundary')
        ax2.fill_between(v, -10, 0, alpha=0.2, color='g')
        ax2.set_xlabel('Velocity [m/s]', fontsize=11)
        ax2.set_ylabel('Real(λ)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.set_ylim([-max(abs(margins))*1.5, 1])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_q_weight_heatmap(result):
        """Plot Q weight sensitivity as heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        q_lat = result['q_lateral_values']
        q_head = result['q_heading_values']
        real_parts = result['real_parts']
        
        # Create meshgrid
        Q_lat, Q_head = np.meshgrid(q_lat, q_head)
        
        # Plot heatmap
        levels = np.linspace(real_parts.min(), 0, 20)
        contourf = ax.contourf(Q_lat, Q_head, real_parts, levels=levels, 
                              cmap='RdYlGn_r', alpha=0.8)
        contour = ax.contour(Q_lat, Q_head, real_parts, levels=[-5, -3, -1], 
                            colors='black', linewidths=1, alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        
        # Mark stability boundary
        ax.contour(Q_lat, Q_head, real_parts, levels=[0], 
                  colors='red', linewidths=3)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Q_lateral (lateral error weight)', fontsize=11)
        ax.set_ylabel('Q_heading (heading error weight)', fontsize=11)
        ax.set_title('LQR Closed-Loop Eigenvalue (Max Real Part) vs Q Weights', 
                    fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Max Re(λ)', rotation=270, labelpad=20, fontsize=10)
        
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_weight_comparison(results):
        """Compare different weight configurations."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = [r['name'] for r in results]
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        # 1. Eigenvalues
        ax = axes[0, 0]
        for i, r in enumerate(results):
            eigs = r['eigenvalues']['eigenvalues']
            ax.scatter(eigs.real, eigs.imag, s=100, marker='x', 
                      linewidths=2, label=names[i], color=colors[i])
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title('Eigenvalue Comparison')
        ax.legend(fontsize=8)
        ax.axis('equal')
        
        # 2. Step responses
        ax = axes[0, 1]
        for i, r in enumerate(results):
            t = r['step']['time']
            y = r['step']['response']
            ax.plot(t, y, linewidth=2, label=names[i], color=colors[i])
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Response')
        ax.set_title('Step Response Comparison')
        ax.legend(fontsize=8)
        
        # 3. Performance metrics
        ax = axes[1, 0]
        metrics = ['overshoot_percent', 'settling_time']
        metric_names = ['Overshoot [%]', 'Settling [s]']
        
        x = np.arange(len(names))
        width = 0.35
        
        overshoots = [r['step']['overshoot_percent'] for r in results]
        settling_times = [r['step']['settling_time'] for r in results]
        
        ax.bar(x - width/2, overshoots, width, label='Overshoot %', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, settling_times, width, label='Settling Time', 
               alpha=0.8, color='orange')
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('Overshoot [%]', color='C0')
        ax2.set_ylabel('Settling Time [s]', color='orange')
        ax.set_title('Performance Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Stability margins
        ax = axes[1, 1]
        margins_gm = [r['frequency']['gain_margin_db'] for r in results]
        margins_pm = [r['frequency']['phase_margin_deg'] for r in results]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, margins_gm, width, label='Gain Margin [dB]', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, margins_pm, width, label='Phase Margin [°]', 
               alpha=0.8, color='green')
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('Gain Margin [dB]', color='C0')
        ax2.set_ylabel('Phase Margin [°]', color='green')
        ax.set_title('Stability Margins')
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('LQR Weight Configuration Comparison', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


# ========================
# Main Analysis
# ========================

def run_complete_lqr_analysis():
    """Run comprehensive LQR analysis."""
    
    print("="*70)
    print(" COMPREHENSIVE LQR CONTROLLER STABILITY ANALYSIS")
    print("="*70)
    print()
    
    # System parameters
    v = 5.0
    L = 2.5
    
    # Default weights
    Q = np.diag([10.0, 5.0])
    R = np.array([[1.0]])
    
    analyzer = LQRStabilityAnalyzer(v, L)
    
    # 1. Eigenvalue Analysis
    print("1. EIGENVALUE ANALYSIS")
    print("-" * 70)
    eig_result = analyzer.analyze_eigenvalues(Q, R)
    
    print(f"Closed-Loop Eigenvalues:")
    for i, eig in enumerate(eig_result['eigenvalues']):
        print(f"  λ{i+1} = {eig:.4f}")
    print(f"\nFeedback Gain K = {eig_result['gain'].flatten()}")
    print(f"System: {'✓ STABLE' if eig_result['stable'] else '✗ UNSTABLE'}")
    
    if len(eig_result['damping_ratios']) > 0:
        print(f"Damping Ratio: {eig_result['damping_ratios'][0]:.3f}")
        print(f"Natural Frequency: {eig_result['natural_frequencies'][0]:.3f} rad/s")
    print()
    
    # 2. Frequency Domain
    print("2. FREQUENCY DOMAIN ANALYSIS")
    print("-" * 70)
    freq_result = analyzer.frequency_domain_analysis(Q, R)
    
    print(f"Gain Margin: {freq_result['gain_margin_db']:.2f} dB")
    print(f"Phase Margin: {freq_result['phase_margin_deg']:.2f}°")
    print()
    
    # 3. Generate plots
    print("3. GENERATING PLOTS")
    print("-" * 70)
    
    viz = LQRVisualizer()
    
    fig1 = viz.plot_eigenvalue_analysis(eig_result)
    fig2 = viz.plot_frequency_response(freq_result)
    
    vel_result = analyzer.velocity_stability_analysis(Q, R)
    fig3 = viz.plot_velocity_stability(vel_result)
    
    q_result = analyzer.q_r_weight_analysis([0, 2], [0, 2], R_value=1.0)
    fig4 = viz.plot_q_weight_heatmap(q_result)
    
    print("✓ All plots generated")
    print()
    
    plt.show()
    
    return eig_result


if __name__ == "__main__":
    run_complete_lqr_analysis()
