#!/usr/bin/env python3
"""
Comprehensive Stability Analysis for Ackermann Steering PID Controller

This module analyzes the stability of the PID lateral controller through:
1. Linearization around operating points
2. Eigenvalue analysis
3. Frequency domain analysis (Bode plots, gain/phase margins)
4. Time domain performance metrics
5. Parameter sensitivity analysis
6. Monte Carlo robustness testing
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ========================
# System Linearization
# ========================

class LinearizedAckermann:
    """
    Linearized Ackermann vehicle model around equilibrium point.
    
    State: x = [e_y, e_psi, e_y_dot, e_psi_dot, e_y_int]
    Input: u = [delta]
    
    The lateral error dynamics are linearized assuming small errors.
    """
    
    def __init__(self, L, v, Kp_y, Ki_y, Kd_y, Kp_psi):
        self.L = L
        self.v = v
        self.Kp_y = Kp_y
        self.Ki_y = Ki_y
        self.Kd_y = Kd_y
        self.Kp_psi = Kp_psi
        
    def get_state_space_continuous(self):
        """
        Continuous-time state space model: dx/dt = Ax + Bu
        
        For lateral tracking with PID control:
        State: [e_y, e_psi, e_y_int]
        
        Simplified dynamics (assuming small angles):
        e_y_dot ≈ v * e_psi
        e_psi_dot ≈ v/L * delta
        e_y_int_dot = e_y
        
        With PID control:
        delta = -(Kp_y*e_y + Ki_y*e_y_int + Kd_y*e_y_dot + Kp_psi*e_psi)
        delta = -(Kp_y*e_y + Ki_y*e_y_int + Kd_y*v*e_psi + Kp_psi*e_psi)
        """
        
        v = self.v
        L = self.L
        Kp = self.Kp_y
        Ki = self.Ki_y
        Kd = self.Kd_y
        Kh = self.Kp_psi
        
        # State matrix (closed-loop with PID feedback)
        A = np.array([
            [0,                v,              0],           # e_y_dot
            [-v*Kp/L,         -v*(Kd*v+Kh)/L,  -v*Ki/L],    # e_psi_dot
            [1,                0,              0]            # e_y_int_dot
        ])
        
        # Input matrix (for analyzing disturbances or reference changes)
        B = np.array([[0], [v/L], [0]])
        
        # Output matrix (we observe lateral error)
        C = np.array([[1, 0, 0]])
        
        # Feedthrough
        D = np.array([[0]])
        
        return A, B, C, D
    
    def get_state_space_discrete(self, dt):
        """Discretize the continuous system using zero-order hold."""
        Ac, Bc, Cc, Dc = self.get_state_space_continuous()
        sys_c = signal.StateSpace(Ac, Bc, Cc, Dc)
        sys_d = sys_c.to_discrete(dt)
        return sys_d.A, sys_d.B, sys_d.C, sys_d.D


# ========================
# Stability Analysis
# ========================

class StabilityAnalyzer:
    """Comprehensive stability analysis suite."""
    
    def __init__(self, L=2.5, v=5.0, dt=0.02):
        self.L = L
        self.v = v
        self.dt = dt
        
    def analyze_eigenvalues(self, Kp_y, Ki_y, Kd_y, Kp_psi):
        """
        Analyze system eigenvalues for continuous and discrete systems.
        
        Stability criteria:
        - Continuous: All eigenvalues must have negative real parts
        - Discrete: All eigenvalues must be inside unit circle (|λ| < 1)
        """
        sys = LinearizedAckermann(self.L, self.v, Kp_y, Ki_y, Kd_y, Kp_psi)
        
        # Continuous system eigenvalues
        Ac, Bc, Cc, Dc = sys.get_state_space_continuous()
        eigs_c = linalg.eigvals(Ac)
        
        # Discrete system eigenvalues
        Ad, Bd, Cd, Dd = sys.get_state_space_discrete(self.dt)
        eigs_d = linalg.eigvals(Ad)
        
        # Stability checks
        continuous_stable = np.all(eigs_c.real < 0)
        discrete_stable = np.all(np.abs(eigs_d) < 1)
        
        # Stability margins
        max_real_part = np.max(eigs_c.real)
        max_discrete_mag = np.max(np.abs(eigs_d))
        
        return {
            'continuous_eigenvalues': eigs_c,
            'discrete_eigenvalues': eigs_d,
            'continuous_stable': continuous_stable,
            'discrete_stable': discrete_stable,
            'max_real_part': max_real_part,
            'max_discrete_magnitude': max_discrete_mag,
            'stability_margin_continuous': -max_real_part,
            'stability_margin_discrete': 1 - max_discrete_mag
        }
    
    def frequency_domain_analysis(self, Kp_y, Ki_y, Kd_y, Kp_psi):
        """
        Bode plot analysis and gain/phase margins.
        
        Gain Margin (GM): How much gain can be increased before instability
        Phase Margin (PM): How much phase lag can be added before instability
        
        Good margins: GM > 6 dB, PM > 30°
        """
        sys = LinearizedAckermann(self.L, self.v, Kp_y, Ki_y, Kd_y, Kp_psi)
        Ac, Bc, Cc, Dc = sys.get_state_space_continuous()
        
        # Create transfer function
        sys_tf = signal.StateSpace(Ac, Bc, Cc, Dc)
        
        # Frequency range
        w = np.logspace(-2, 2, 500)
        
        # Bode plot data
        w, mag, phase = signal.bode(sys_tf, w)
        
        # Gain and phase margins
        try:
            gm, pm, wgc, wpc = signal.margin(sys_tf)
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
    
    def step_response_metrics(self, Kp_y, Ki_y, Kd_y, Kp_psi, T_sim=10.0):
        """
        Time domain step response analysis.
        
        Metrics:
        - Rise time: Time to reach 90% of final value
        - Settling time: Time to stay within 2% of final value
        - Overshoot: Maximum overshoot percentage
        - Steady-state error: Final tracking error
        """
        sys = LinearizedAckermann(self.L, self.v, Kp_y, Ki_y, Kd_y, Kp_psi)
        Ac, Bc, Cc, Dc = sys.get_state_space_continuous()
        
        sys_tf = signal.StateSpace(Ac, Bc, Cc, Dc)
        
        # Step response
        t = np.arange(0, T_sim, self.dt)
        t_step, y_step = signal.step(sys_tf, T=t)
        
        # Calculate metrics
        final_value = y_step[-1]
        
        # Rise time (10% to 90%)
        idx_10 = np.where(y_step >= 0.1 * final_value)[0]
        idx_90 = np.where(y_step >= 0.9 * final_value)[0]
        rise_time = (t_step[idx_90[0]] - t_step[idx_10[0]]) if len(idx_10) > 0 and len(idx_90) > 0 else np.nan
        
        # Settling time (within 2%)
        settling_band = 0.02 * abs(final_value)
        settled = np.abs(y_step - final_value) <= settling_band
        if np.any(settled):
            settling_idx = np.where(settled)[0][0]
            # Check if it stays settled
            if np.all(settled[settling_idx:]):
                settling_time = t_step[settling_idx]
            else:
                settling_time = np.nan
        else:
            settling_time = np.nan
        
        # Overshoot
        max_value = np.max(y_step)
        overshoot_pct = 100 * (max_value - final_value) / final_value if final_value != 0 else 0
        
        # Steady-state error (should be near zero for type-1 system with integral control)
        ss_error = abs(1.0 - final_value)
        
        return {
            'time': t_step,
            'response': y_step,
            'rise_time': rise_time,
            'settling_time': settling_time,
            'overshoot_percent': overshoot_pct,
            'steady_state_error': ss_error,
            'final_value': final_value
        }
    
    def velocity_stability_map(self, Kp_y, Ki_y, Kd_y, Kp_psi, v_range=None):
        """
        Analyze stability across different vehicle velocities.
        Critical for understanding speed-dependent behavior.
        """
        if v_range is None:
            v_range = np.linspace(1, 20, 50)
        
        stable_continuous = []
        stable_discrete = []
        max_real_parts = []
        
        for v_test in v_range:
            analyzer_temp = StabilityAnalyzer(self.L, v_test, self.dt)
            result = analyzer_temp.analyze_eigenvalues(Kp_y, Ki_y, Kd_y, Kp_psi)
            stable_continuous.append(result['continuous_stable'])
            stable_discrete.append(result['discrete_stable'])
            max_real_parts.append(result['max_real_part'])
        
        return {
            'velocities': v_range,
            'continuous_stable': np.array(stable_continuous),
            'discrete_stable': np.array(stable_discrete),
            'max_real_parts': np.array(max_real_parts)
        }
    
    def gain_stability_region(self, Kp_range, Kd_range, Ki=0.1, Kp_psi=1.0):
        """
        2D stability region in Kp-Kd parameter space.
        Shows which gain combinations lead to stable behavior.
        """
        Kp_vals = np.linspace(Kp_range[0], Kp_range[1], 50)
        Kd_vals = np.linspace(Kd_range[0], Kd_range[1], 50)
        
        stability_map = np.zeros((len(Kd_vals), len(Kp_vals)))
        
        for i, Kd in enumerate(Kd_vals):
            for j, Kp in enumerate(Kp_vals):
                result = self.analyze_eigenvalues(Kp, Ki, Kd, Kp_psi)
                stability_map[i, j] = result['continuous_stable'] and result['discrete_stable']
        
        return {
            'Kp_values': Kp_vals,
            'Kd_values': Kd_vals,
            'stability_map': stability_map
        }


# ========================
# Performance Validation
# ========================

class PerformanceValidator:
    """Validate controller performance through simulation."""
    
    @staticmethod
    def compute_tracking_metrics(xs, ys, psis, xr, yr, psi_r):
        """
        Compute tracking error metrics.
        
        Returns:
        - RMS lateral error
        - Max lateral error
        - RMS heading error
        - Max heading error
        - Mean absolute errors
        """
        # Lateral errors
        lateral_errors = np.sqrt((xs - xr)**2 + (ys - yr)**2)
        
        # Heading errors
        heading_errors = np.abs(np.arctan2(np.sin(psis - psi_r), np.cos(psis - psi_r)))
        
        metrics = {
            'lateral_error_rms': np.sqrt(np.mean(lateral_errors**2)),
            'lateral_error_max': np.max(lateral_errors),
            'lateral_error_mean': np.mean(lateral_errors),
            'lateral_error_std': np.std(lateral_errors),
            'heading_error_rms': np.sqrt(np.mean(heading_errors**2)),
            'heading_error_max': np.max(heading_errors),
            'heading_error_mean': np.mean(heading_errors),
            'heading_error_std': np.std(heading_errors),
        }
        
        return metrics
    
    @staticmethod
    def analyze_control_effort(deltas, dt):
        """
        Analyze steering command characteristics.
        
        Returns:
        - Max steering angle
        - Mean absolute steering
        - Steering rate statistics
        """
        steering_rates = np.diff(deltas) / dt
        
        metrics = {
            'steering_max': np.max(np.abs(deltas)),
            'steering_mean_abs': np.mean(np.abs(deltas)),
            'steering_std': np.std(deltas),
            'steering_rate_max': np.max(np.abs(steering_rates)),
            'steering_rate_mean': np.mean(np.abs(steering_rates)),
        }
        
        return metrics


# ========================
# Monte Carlo Robustness
# ========================

class RobustnessAnalyzer:
    """Monte Carlo analysis for parameter uncertainty."""
    
    @staticmethod
    def parameter_sensitivity(base_params, param_variations, n_samples=100):
        """
        Perform Monte Carlo analysis varying system parameters.
        
        Tests robustness to:
        - Wheelbase uncertainty
        - Velocity variations
        - Sampling time jitter
        """
        L_base = base_params['L']
        v_base = base_params['v']
        dt_base = base_params['dt']
        gains = base_params['gains']
        
        results = {
            'stable_count': 0,
            'unstable_count': 0,
            'stability_margins': [],
            'max_real_parts': []
        }
        
        for _ in range(n_samples):
            # Add random variations
            L_test = L_base * (1 + np.random.uniform(-param_variations['L'], param_variations['L']))
            v_test = v_base * (1 + np.random.uniform(-param_variations['v'], param_variations['v']))
            dt_test = dt_base * (1 + np.random.uniform(-param_variations['dt'], param_variations['dt']))
            
            analyzer = StabilityAnalyzer(L_test, v_test, dt_test)
            result = analyzer.analyze_eigenvalues(*gains)
            
            if result['continuous_stable'] and result['discrete_stable']:
                results['stable_count'] += 1
            else:
                results['unstable_count'] += 1
            
            results['stability_margins'].append(result['stability_margin_continuous'])
            results['max_real_parts'].append(result['max_real_part'])
        
        results['stability_rate'] = results['stable_count'] / n_samples
        results['stability_margins'] = np.array(results['stability_margins'])
        results['max_real_parts'] = np.array(results['max_real_parts'])
        
        return results


# ========================
# Visualization
# ========================

class StabilityVisualizer:
    """Comprehensive visualization of stability analysis results."""
    
    @staticmethod
    def plot_eigenvalue_analysis(result, title="Eigenvalue Analysis"):
        """Plot eigenvalues in complex plane."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Continuous eigenvalues
        eigs_c = result['continuous_eigenvalues']
        ax1.scatter(eigs_c.real, eigs_c.imag, s=100, c='blue', marker='x', linewidths=2)
        ax1.axvline(0, color='r', linestyle='--', alpha=0.5, label='Stability boundary')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Real Part')
        ax1.set_ylabel('Imaginary Part')
        ax1.set_title('Continuous System Eigenvalues')
        ax1.legend()
        
        stable_text = "STABLE" if result['continuous_stable'] else "UNSTABLE"
        color = 'green' if result['continuous_stable'] else 'red'
        ax1.text(0.05, 0.95, stable_text, transform=ax1.transAxes,
                fontsize=12, fontweight='bold', color=color,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Discrete eigenvalues
        eigs_d = result['discrete_eigenvalues']
        ax2.scatter(eigs_d.real, eigs_d.imag, s=100, c='red', marker='o', linewidths=2)
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax2.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.5, label='Unit circle')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Real Part')
        ax2.set_ylabel('Imaginary Part')
        ax2.set_title('Discrete System Eigenvalues')
        ax2.legend()
        ax2.axis('equal')
        
        stable_text = "STABLE" if result['discrete_stable'] else "UNSTABLE"
        color = 'green' if result['discrete_stable'] else 'red'
        ax2.text(0.05, 0.95, stable_text, transform=ax2.transAxes,
                fontsize=12, fontweight='bold', color=color,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
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
        ax1.set_ylabel('Magnitude [dB]')
        ax1.set_title('Bode Diagram')
        
        # Add gain margin line
        if not np.isnan(result['gain_margin_db']):
            ax1.axhline(result['gain_margin_db'], color='r', linestyle='--',
                       label=f"GM = {result['gain_margin_db']:.2f} dB")
            ax1.legend()
        
        # Phase plot
        ax2.semilogx(w, phase, 'b-', linewidth=2)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.set_ylabel('Phase [deg]')
        
        # Add phase margin line
        if not np.isnan(result['phase_margin_deg']):
            ax2.axhline(-180 + result['phase_margin_deg'], color='r', linestyle='--',
                       label=f"PM = {result['phase_margin_deg']:.2f}°")
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_step_response(result):
        """Plot step response with metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        t = result['time']
        y = result['response']
        
        ax.plot(t, y, 'b-', linewidth=2, label='Step response')
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Reference')
        ax.axhline(1.02, color='g', linestyle=':', alpha=0.3)
        ax.axhline(0.98, color='g', linestyle=':', alpha=0.3, label='2% band')
        
        # Mark settling time
        if not np.isnan(result['settling_time']):
            ax.axvline(result['settling_time'], color='orange', linestyle='--',
                      label=f"Settling time = {result['settling_time']:.2f} s")
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Response')
        ax.set_title('Step Response Analysis')
        ax.legend()
        
        # Add metrics text box
        metrics_text = (f"Rise time: {result['rise_time']:.3f} s\n"
                       f"Settling time: {result['settling_time']:.3f} s\n"
                       f"Overshoot: {result['overshoot_percent']:.2f}%\n"
                       f"SS error: {result['steady_state_error']:.4f}")
        
        ax.text(0.65, 0.4, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_velocity_stability(result):
        """Plot stability vs velocity."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        v = result['velocities']
        stable_c = result['continuous_stable']
        stable_d = result['discrete_stable']
        max_real = result['max_real_parts']
        
        # Stability regions
        ax1.fill_between(v, 0, 1, where=stable_c, alpha=0.3, color='g', label='Continuous stable')
        ax1.fill_between(v, 0, 1, where=~stable_c, alpha=0.3, color='r', label='Continuous unstable')
        ax1.set_ylabel('Stability')
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('Stability vs Velocity')
        
        # Max real part
        ax2.plot(v, max_real, 'b-', linewidth=2)
        ax2.axhline(0, color='r', linestyle='--', label='Stability boundary')
        ax2.fill_between(v, max_real, 0, where=(max_real<0), alpha=0.3, color='g')
        ax2.fill_between(v, max_real, 0, where=(max_real>=0), alpha=0.3, color='r')
        ax2.set_xlabel('Velocity [m/s]')
        ax2.set_ylabel('Max Real Part of Eigenvalues')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_gain_stability_region(result):
        """Plot 2D stability region in gain space."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        Kp = result['Kp_values']
        Kd = result['Kd_values']
        stability = result['stability_map']
        
        # Create meshgrid for contour
        Kp_grid, Kd_grid = np.meshgrid(Kp, Kd)
        
        # Plot stability region
        contour = ax.contourf(Kp_grid, Kd_grid, stability, levels=[0, 0.5, 1],
                              colors=['red', 'lightgreen'], alpha=0.6)
        ax.contour(Kp_grid, Kd_grid, stability, levels=[0.5], colors='black', linewidths=2)
        
        ax.set_xlabel('Kp (Proportional Gain)', fontsize=12)
        ax.set_ylabel('Kd (Derivative Gain)', fontsize=12)
        ax.set_title('Stability Region in Kp-Kd Parameter Space', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, ticks=[0, 1])
        cbar.set_label('Stability', rotation=270, labelpad=20)
        cbar.ax.set_yticklabels(['Unstable', 'Stable'])
        
        plt.tight_layout()
        return fig


# ========================
# Main Analysis Script
# ========================

def run_complete_stability_analysis():
    """Run comprehensive stability analysis suite."""
    
    print("="*70)
    print(" COMPREHENSIVE PID CONTROLLER STABILITY ANALYSIS")
    print("="*70)
    print()
    
    # System parameters
    L = 2.5
    v = 5.0
    dt = 0.02
    
    # Controller gains (from original code)
    Kp_y = 1.0
    Ki_y = 0.2
    Kd_y = 0.1
    Kp_psi = 1.0
    
    analyzer = StabilityAnalyzer(L, v, dt)
    
    # ==================
    # 1. Eigenvalue Analysis
    # ==================
    print("1. EIGENVALUE ANALYSIS")
    print("-" * 70)
    eig_result = analyzer.analyze_eigenvalues(Kp_y, Ki_y, Kd_y, Kp_psi)
    
    print(f"Continuous System Eigenvalues:")
    for i, eig in enumerate(eig_result['continuous_eigenvalues']):
        print(f"  λ{i+1} = {eig:.4f}")
    print(f"Continuous System: {'STABLE ✓' if eig_result['continuous_stable'] else 'UNSTABLE ✗'}")
    print(f"Stability Margin: {eig_result['stability_margin_continuous']:.4f}")
    print()
    
    print(f"Discrete System Eigenvalues:")
    for i, eig in enumerate(eig_result['discrete_eigenvalues']):
        print(f"  λ{i+1} = {eig:.4f} (|λ| = {abs(eig):.4f})")
    print(f"Discrete System: {'STABLE ✓' if eig_result['discrete_stable'] else 'UNSTABLE ✗'}")
    print(f"Max |λ|: {eig_result['max_discrete_magnitude']:.4f}")
    print()
    
    # ==================
    # 2. Frequency Domain Analysis
    # ==================
    print("2. FREQUENCY DOMAIN ANALYSIS")
    print("-" * 70)
    freq_result = analyzer.frequency_domain_analysis(Kp_y, Ki_y, Kd_y, Kp_psi)
    
    print(f"Gain Margin: {freq_result['gain_margin_db']:.2f} dB")
    print(f"  → Gain can increase by factor of {10**(freq_result['gain_margin_db']/20):.2f} before instability")
    print(f"Phase Margin: {freq_result['phase_margin_deg']:.2f}°")
    print(f"  → {'Good margin (>30°)' if freq_result['phase_margin_deg'] > 30 else 'Poor margin (<30°)'}")
    print(f"Gain Crossover Frequency: {freq_result['gain_crossover_freq']:.3f} rad/s")
    print(f"Phase Crossover Frequency: {freq_result['phase_crossover_freq']:.3f} rad/s")
    print()
    
    # ==================
    # 3. Step Response
    # ==================
    print("3. TIME DOMAIN RESPONSE")
    print("-" * 70)
    step_result = analyzer.step_response_metrics(Kp_y, Ki_y, Kd_y, Kp_psi)
    
    print(f"Rise Time: {step_result['rise_time']:.3f} s")
    print(f"Settling Time: {step_result['settling_time']:.3f} s")
    print(f"Overshoot: {step_result['overshoot_percent']:.2f}%")
    print(f"Steady-State Error: {step_result['steady_state_error']:.6f}")
    print()
    
    # ==================
    # 4. Velocity Stability
    # ==================
    print("4. VELOCITY STABILITY ANALYSIS")
    print("-" * 70)
    vel_result = analyzer.velocity_stability_map(Kp_y, Ki_y, Kd_y, Kp_psi)
    
    stable_range = vel_result['velocities'][vel_result['continuous_stable']]
    if len(stable_range) > 0:
        print(f"Stable Velocity Range: {stable_range[0]:.2f} - {stable_range[-1]:.2f} m/s")
    else:
        print("No stable velocity range found!")
    print()
    
    # ==================
    # 5. Robustness Analysis
    # ==================
    print("5. ROBUSTNESS ANALYSIS (Monte Carlo)")
    print("-" * 70)
    
    base_params = {
        'L': L,
        'v': v,
        'dt': dt,
        'gains': (Kp_y, Ki_y, Kd_y, Kp_psi)
    }
    
    param_variations = {
        'L': 0.10,    # ±10% wheelbase variation
        'v': 0.20,    # ±20% velocity variation
        'dt': 0.10    # ±10% sampling time jitter
    }
    
    robust_result = RobustnessAnalyzer.parameter_sensitivity(base_params, param_variations, n_samples=500)
    
    print(f"Stability Rate: {robust_result['stability_rate']*100:.1f}%")
    print(f"  Stable: {robust_result['stable_count']}/500 samples")
    print(f"  Unstable: {robust_result['unstable_count']}/500 samples")
    print(f"Mean Stability Margin: {np.mean(robust_result['stability_margins']):.4f}")
    print(f"Min Stability Margin: {np.min(robust_result['stability_margins']):.4f}")
    print()
    
    # ==================
    # 6. Gain Stability Region
    # ==================
    print("6. GAIN STABILITY REGION")
    print("-" * 70)
    gain_result = analyzer.gain_stability_region([0.1, 3.0], [0.0, 1.0], Ki_y, Kp_psi)
    stable_fraction = np.sum(gain_result['stability_map']) / gain_result['stability_map'].size
    print(f"Stable Region: {stable_fraction*100:.1f}% of parameter space")
    print()
    
    # ==================
    # Generate Plots
    # ==================
    print("7. GENERATING VISUALIZATION PLOTS")
    print("-" * 70)
    
    viz = StabilityVisualizer()
    
    # Eigenvalue plots
    fig1 = viz.plot_eigenvalue_analysis(eig_result, 
                                        f"Eigenvalue Analysis (v={v} m/s, dt={dt} s)")
    
    # Bode plots
    fig2 = viz.plot_frequency_response(freq_result)
    
    # Step response
    fig3 = viz.plot_step_response(step_result)
    
    # Velocity stability
    fig4 = viz.plot_velocity_stability(vel_result)
    
    # Gain stability region
    fig5 = viz.plot_gain_stability_region(gain_result)
    
    print("All plots generated successfully!")
    print()
    
    # ==================
    # Summary & Recommendations
    # ==================
    print("="*70)
    print(" SUMMARY & RECOMMENDATIONS")
    print("="*70)
    print()
    
    if eig_result['continuous_stable'] and eig_result['discrete_stable']:
        print("✓ System is STABLE for nominal operating conditions")
    else:
        print("✗ System is UNSTABLE - gains need adjustment!")
    
    if freq_result['gain_margin_db'] > 6 and freq_result['phase_margin_deg'] > 30:
        print("✓ Good stability margins (GM > 6 dB, PM > 30°)")
    else:
        print("⚠ Stability margins are tight - consider more conservative gains")
    
    if robust_result['stability_rate'] > 0.95:
        print("✓ Robust to parameter variations")
    elif robust_result['stability_rate'] > 0.80:
        print("⚠ Moderately robust - some sensitivity to parameter changes")
    else:
        print("✗ Poor robustness - highly sensitive to parameter variations")
    
    print()
    print("Recommendations:")
    print("  1. Test gains across expected velocity range")
    print("  2. Verify discrete-time stability at actual control frequency")
    print("  3. Add gain scheduling for different velocities if needed")
    print("  4. Consider LQR for optimal tuning across operating conditions")
    print()
    
    plt.show()
    
    return {
        'eigenvalues': eig_result,
        'frequency': freq_result,
        'step': step_result,
        'velocity': vel_result,
        'gains': gain_result,
        'robustness': robust_result
    }


if __name__ == "__main__":
    results = run_complete_stability_analysis()
