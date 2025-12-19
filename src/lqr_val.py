#!/usr/bin/env python3
"""
Comprehensive Testing and Validation Suite for LQR Steering Controller

This script performs extensive testing including:
- Tracking performance validation across different paths
- Robustness to parameter variations
- Q/R weight sensitivity analysis
- Comparative analysis with different weight matrices
- Performance across velocity range
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from dataclasses import dataclass
import time


# =====================
# Vehicle Model
# =====================

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


@dataclass
class KinematicBicycle:
    L: float
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0

    def step(self, v, delta, dt):
        x_dot = v * np.cos(self.psi)
        y_dot = v * np.sin(self.psi)
        psi_dot = v / self.L * np.tan(delta)

        self.x += x_dot * dt
        self.y += y_dot * dt
        self.psi = wrap_angle(self.psi + psi_dot * dt)
        return self.x, self.y, self.psi


class LQRSteering:
    """LQR lateral controller with feedforward."""
    
    def __init__(self, v, L, Q=None, R=None, delta_max=np.deg2rad(35.0)):
        self.v = v
        self.L = L
        self.delta_max = delta_max
        
        A = np.array([[0.0, v],
                     [0.0, 0.0]])
        B = np.array([[0.0],
                     [v / L]])
        
        if Q is None:
            Q = np.diag([10.0, 5.0])
        if R is None:
            R = np.array([[1.0]])
        elif np.isscalar(R):
            R = np.array([[R]])
        
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P
    
    def control(self, e_y, e_psi, delta_ff):
        x = np.array([[e_y], [e_psi]])
        u = -self.K @ x
        delta = delta_ff + u.item()
        return np.clip(delta, -self.delta_max, self.delta_max)


# =====================
# Reference Paths with Curvature
# =====================

def ref_lane_change(t, v, Dy=3.0, T_c=5.0):
    x = v * t
    if t <= T_c:
        y = 0.5 * Dy * (1 - np.cos(np.pi * t / T_c))
        y_dot = 0.5 * Dy * (np.pi / T_c) * np.sin(np.pi * t / T_c)
        y_ddot = 0.5 * Dy * (np.pi / T_c)**2 * np.cos(np.pi * t / T_c)
    else:
        y = Dy
        y_dot = 0.0
        y_ddot = 0.0
    
    x_dot = v
    x_ddot = 0.0
    psi = np.arctan2(y_dot, x_dot)
    
    denom = (x_dot**2 + y_dot**2)**1.5
    kappa = (x_dot * y_ddot - y_dot * x_ddot) / denom if denom > 1e-6 else 0.0
    
    return x, y, psi, kappa


def ref_circle(t, v, R=10.0):
    w = v / R
    th = w * t
    x = R * np.sin(th)
    y = R * (1 - np.cos(th))
    
    x_dot = v * np.cos(th)
    y_dot = v * np.sin(th)
    psi = np.arctan2(y_dot, x_dot)
    kappa = 1.0 / R
    
    return x, y, psi, kappa


def ref_wave(t, v, A=2.0, T_s=10.0):
    x = v * t
    w = 2 * np.pi / T_s
    y = A * np.sin(w * t)
    
    x_dot = v
    y_dot = A * w * np.cos(w * t)
    x_ddot = 0.0
    y_ddot = -A * w**2 * np.sin(w * t)
    
    psi = np.arctan2(y_dot, x_dot)
    denom = (x_dot**2 + y_dot**2)**1.5
    kappa = (x_dot * y_ddot - y_dot * x_ddot) / denom if denom > 1e-6 else 0.0
    
    return x, y, psi, kappa


def ref_step(t, v, y_step=2.0):
    """Step input for disturbance testing."""
    x = v * t
    y = y_step if t > 0 else 0.0
    psi = 0.0
    kappa = 0.0
    return x, y, psi, kappa


# =====================
# Testing Framework
# =====================

class TestResult:
    """Store test results."""
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.metrics = {}
        self.details = ""
        
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.name}\n  {self.details}"


class LQRPerformanceTester:
    """Comprehensive testing suite for LQR controller."""
    
    def __init__(self, L=2.5, v=5.0, dt=0.02):
        self.L = L
        self.v = v
        self.dt = dt
        self.results = []
    
    def simulate_tracking(self, ref_fun, Q, R, T=12.0, initial_offset=(0, 0, 0)):
        """
        Simulate path tracking with LQR.
        
        Args:
            ref_fun: Reference path function (returns x, y, psi, kappa)
            Q: State weight matrix
            R: Control weight
            T: Simulation time
            initial_offset: Initial position error
        """
        car = KinematicBicycle(L=self.L)
        car.x, car.y, car.psi = initial_offset
        
        lqr = LQRSteering(self.v, self.L, Q, R)
        
        N = int(T / self.dt)
        
        # Storage
        t_hist = np.zeros(N)
        xs = np.zeros(N)
        ys = np.zeros(N)
        psis = np.zeros(N)
        xr = np.zeros(N)
        yr = np.zeros(N)
        psi_r = np.zeros(N)
        deltas = np.zeros(N)
        e_lateral = np.zeros(N)
        e_heading = np.zeros(N)
        
        for k in range(N):
            t = k * self.dt
            t_hist[k] = t
            
            # Reference with curvature
            x_ref, y_ref, psi_ref, kappa_ref = ref_fun(t, self.v)
            xr[k], yr[k], psi_r[k] = x_ref, y_ref, psi_ref
            
            # Errors
            dx = car.x - x_ref
            dy = car.y - y_ref
            e_y = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy
            e_psi = wrap_angle(car.psi - psi_ref)
            
            e_lateral[k] = e_y
            e_heading[k] = e_psi
            
            # Feedforward from curvature
            delta_ff = np.arctan(self.L * kappa_ref)
            
            # Control
            delta = lqr.control(e_y, e_psi, delta_ff)
            deltas[k] = delta
            
            # Step
            car.step(self.v, delta, self.dt)
            xs[k], ys[k], psis[k] = car.x, car.y, car.psi
        
        # Compute metrics
        metrics = self._compute_metrics(
            t_hist, xs, ys, psis, xr, yr, psi_r, 
            e_lateral, e_heading, deltas
        )
        
        return {
            'time': t_hist,
            'x': xs, 'y': ys, 'psi': psis,
            'x_ref': xr, 'y_ref': yr, 'psi_ref': psi_r,
            'delta': deltas,
            'e_lateral': e_lateral,
            'e_heading': e_heading,
            'metrics': metrics
        }
    
    def _compute_metrics(self, t, xs, ys, psis, xr, yr, psi_r, 
                        e_lateral, e_heading, deltas):
        """Compute performance metrics."""
        
        position_errors = np.sqrt((xs - xr)**2 + (ys - yr)**2)
        
        metrics = {
            'lateral_rms': np.sqrt(np.mean(e_lateral**2)),
            'lateral_max': np.max(np.abs(e_lateral)),
            'lateral_mean': np.mean(np.abs(e_lateral)),
            'lateral_std': np.std(e_lateral),
            
            'position_rms': np.sqrt(np.mean(position_errors**2)),
            'position_max': np.max(position_errors),
            
            'heading_rms': np.sqrt(np.mean(e_heading**2)),
            'heading_max': np.max(np.abs(e_heading)),
            'heading_mean': np.mean(np.abs(e_heading)),
            
            'steering_max': np.max(np.abs(deltas)),
            'steering_mean': np.mean(np.abs(deltas)),
            'steering_std': np.std(deltas),
            
            'settling_time': self._compute_settling_time(e_lateral, t),
            'overshoot': self._compute_overshoot(e_lateral),
        }
        
        dt = t[1] - t[0]
        steering_rates = np.diff(deltas) / dt
        metrics['steering_rate_max'] = np.max(np.abs(steering_rates))
        metrics['steering_rate_mean'] = np.mean(np.abs(steering_rates))
        
        return metrics
    
    def _compute_settling_time(self, errors, time, threshold=0.02):
        """Find settling time."""
        max_error = np.max(np.abs(errors))
        band = threshold * max_error if max_error > 0 else 0.02
        
        exceeds = np.abs(errors) > band
        if not np.any(exceeds):
            return time[0]
        
        last_exceed_idx = np.where(exceeds)[0][-1]
        return time[last_exceed_idx] if last_exceed_idx < len(time) - 1 else time[-1]
    
    def _compute_overshoot(self, errors):
        """Compute overshoot percentage."""
        if len(errors) < 10:
            return 0.0
        
        mid_idx = len(errors) // 2
        first_half = errors[:mid_idx]
        
        if len(first_half) == 0:
            return 0.0
        
        max_error = np.max(np.abs(first_half))
        final_error = np.mean(np.abs(errors[-100:]))
        
        if final_error < 1e-6:
            return 0.0
        
        overshoot = (max_error - final_error) / final_error * 100
        return max(0, overshoot)
    
    # =====================
    # Test Suite
    # =====================
    
    def test_basic_tracking(self, Q, R):
        """Test 1: Basic tracking performance."""
        result = TestResult("Basic Path Tracking (LQR)")
        
        paths = [
            ("Lane Change", lambda t, v: ref_lane_change(t, v)),
            ("Circle", lambda t, v: ref_circle(t, v)),
            ("Wave", lambda t, v: ref_wave(t, v)),
        ]
        
        all_passed = True
        details = []
        
        for name, ref_fun in paths:
            data = self.simulate_tracking(ref_fun, Q, R, T=12.0)
            m = data['metrics']
            
            # Performance thresholds (LQR should perform well)
            lat_ok = m['lateral_rms'] < 0.3  # Stricter than PID
            head_ok = m['heading_rms'] < np.deg2rad(8)
            
            passed = lat_ok and head_ok
            all_passed = all_passed and passed
            
            status = "✓" if passed else "✗"
            details.append(
                f"  {status} {name}: "
                f"Lat RMS={m['lateral_rms']:.3f}m, "
                f"Head RMS={np.rad2deg(m['heading_rms']):.1f}°"
            )
        
        result.passed = all_passed
        result.details = "\n".join(details)
        self.results.append(result)
        return result
    
    def test_disturbance_rejection(self, Q, R):
        """Test 2: Disturbance rejection."""
        result = TestResult("Disturbance Rejection (LQR)")
        
        offset = (0, 1.0, 0)  # 1m lateral offset
        data = self.simulate_tracking(
            lambda t, v: ref_lane_change(t, v),
            Q, R, T=10.0, initial_offset=offset
        )
        
        m = data['metrics']
        
        settling_ok = m['settling_time'] < 4.0  # LQR should be faster
        final_error = np.mean(np.abs(data['e_lateral'][-100:]))
        converged_ok = final_error < 0.15
        
        result.passed = settling_ok and converged_ok
        result.details = (
            f"  Initial offset: 1.0m\n"
            f"  Settling time: {m['settling_time']:.2f}s "
            f"({'✓' if settling_ok else '✗'} < 4s)\n"
            f"  Final error: {final_error:.3f}m "
            f"({'✓' if converged_ok else '✗'} < 0.15m)"
        )
        
        self.results.append(result)
        return result
    
    def test_velocity_range(self, Q, R, v_range=[2, 5, 10, 15]):
        """Test 3: Performance across velocities."""
        result = TestResult("Velocity Range Testing (LQR)")
        
        all_passed = True
        details = []
        
        for v_test in v_range:
            self.v = v_test
            data = self.simulate_tracking(
                lambda t, v: ref_lane_change(t, v),
                Q, R, T=10.0
            )
            m = data['metrics']
            
            # LQR should adapt well to velocity
            lat_ok = m['lateral_rms'] < 0.5
            stable = m['lateral_max'] < 2.5
            
            passed = lat_ok and stable
            all_passed = all_passed and passed
            
            status = "✓" if passed else "✗"
            details.append(
                f"  {status} v={v_test:2d} m/s: "
                f"RMS={m['lateral_rms']:.3f}m, "
                f"Max={m['lateral_max']:.3f}m"
            )
        
        self.v = 5.0  # Reset
        
        result.passed = all_passed
        result.details = "\n".join(details)
        self.results.append(result)
        return result
    
    def test_q_weight_sensitivity(self, R=1.0):
        """Test 4: Sensitivity to Q weights."""
        result = TestResult("Q Weight Sensitivity")
        
        # Test different Q configurations
        q_configs = [
            ("Balanced", np.diag([10.0, 5.0])),
            ("High Lateral", np.diag([50.0, 5.0])),
            ("High Heading", np.diag([10.0, 20.0])),
            ("Conservative", np.diag([5.0, 2.0])),
        ]
        
        all_passed = True
        details = []
        
        for name, Q in q_configs:
            data = self.simulate_tracking(
                lambda t, v: ref_lane_change(t, v),
                Q, np.array([[R]]), T=10.0
            )
            m = data['metrics']
            
            # All should be stable and perform reasonably
            lat_ok = m['lateral_rms'] < 0.5
            stable = m['lateral_max'] < 2.0
            
            passed = lat_ok and stable
            all_passed = all_passed and passed
            
            status = "✓" if passed else "✗"
            details.append(
                f"  {status} {name:15s}: RMS={m['lateral_rms']:.3f}m"
            )
        
        result.passed = all_passed
        result.details = "\n".join(details)
        self.results.append(result)
        return result
    
    def test_parameter_robustness(self, Q, R, n_samples=100):
        """Test 5: Robustness to parameter variations."""
        result = TestResult("Parameter Robustness (Monte Carlo)")
        
        L_nominal = self.L
        v_nominal = self.v
        
        success_count = 0
        
        for i in range(n_samples):
            L_var = L_nominal * (1 + np.random.uniform(-0.15, 0.15))
            v_var = v_nominal * (1 + np.random.uniform(-0.20, 0.20))
            
            self.L = L_var
            self.v = v_var
            
            try:
                data = self.simulate_tracking(
                    lambda t, v: ref_lane_change(t, v),
                    Q, R, T=10.0
                )
                m = data['metrics']
                
                if m['lateral_rms'] < 0.8 and m['lateral_max'] < 2.0:
                    success_count += 1
                    
            except Exception:
                pass
        
        self.L = L_nominal
        self.v = v_nominal
        
        success_rate = success_count / n_samples
        result.passed = success_rate > 0.90  # LQR should be very robust
        
        result.details = (
            f"  Success rate: {success_rate*100:.1f}% "
            f"({success_count}/{n_samples})\n"
            f"  Parameter variations: L ±15%, v ±20%\n"
            f"  Threshold: {'✓ > 90%' if result.passed else '✗ < 90%'}"
        )
        
        self.results.append(result)
        return result
    
    def test_optimality_check(self, Q, R):
        """Test 6: Verify LQR optimality properties."""
        result = TestResult("LQR Optimality Verification")
        
        # LQR should have certain properties:
        # 1. Guaranteed stability
        # 2. Infinite gain margin
        # 3. At least 60° phase margin
        
        from lqr_stability_analysis import LQRStabilityAnalyzer
        
        analyzer = LQRStabilityAnalyzer(self.v, self.L)
        eig_result = analyzer.analyze_eigenvalues(Q, R)
        freq_result = analyzer.frequency_domain_analysis(Q, R)
        
        stable = eig_result['stable']
        gm = freq_result['gain_margin_db']
        pm = freq_result['phase_margin_deg']
        
        # LQR guarantees
        gm_ok = gm > 10  # Should have large gain margin
        pm_ok = pm > 50  # Should have good phase margin
        
        result.passed = stable and gm_ok and pm_ok
        result.details = (
            f"  Stability: {'✓ Stable' if stable else '✗ Unstable'}\n"
            f"  Gain Margin: {gm:.1f} dB "
            f"({'✓' if gm_ok else '✗'} > 10 dB)\n"
            f"  Phase Margin: {pm:.1f}° "
            f"({'✓' if pm_ok else '✗'} > 50°)"
        )
        
        self.results.append(result)
        return result
    
    def run_all_tests(self, Q=None, R=None):
        """Run complete test suite."""
        if Q is None:
            Q = np.diag([10.0, 5.0])
        if R is None:
            R = np.array([[1.0]])
        
        print("\n" + "="*70)
        print("  COMPREHENSIVE LQR CONTROLLER VALIDATION SUITE")
        print("="*70)
        print(f"\nTesting weights:")
        print(f"  Q = diag({Q[0,0]}, {Q[1,1]})")
        print(f"  R = {R[0,0]}")
        print(f"Vehicle params: L={self.L}m, v={self.v}m/s, dt={self.dt}s\n")
        
        start_time = time.time()
        
        tests = [
            ("Basic Tracking", lambda: self.test_basic_tracking(Q, R)),
            ("Disturbance Rejection", lambda: self.test_disturbance_rejection(Q, R)),
            ("Velocity Range", lambda: self.test_velocity_range(Q, R)),
            ("Q Weight Sensitivity", lambda: self.test_q_weight_sensitivity(R[0,0])),
            ("Parameter Robustness", lambda: self.test_parameter_robustness(Q, R, n_samples=100)),
            ("Optimality Check", lambda: self.test_optimality_check(Q, R)),
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
            print("🎉 ALL TESTS PASSED - LQR Controller validated! ✓")
        elif passed >= total * 0.8:
            print("⚠️  MOST TESTS PASSED - Minor tuning may help")
        else:
            print("❌ MULTIPLE FAILURES - Check Q/R weights")
        
        print("\n" + "="*70 + "\n")
        
        return self.results


# =====================
# Weight Comparison
# =====================

def compare_weight_matrices():
    """Compare different Q/R weight configurations."""
    
    print("\n" + "="*70)
    print("  COMPARATIVE ANALYSIS: DIFFERENT WEIGHT MATRICES")
    print("="*70 + "\n")
    
    tester = LQRPerformanceTester(L=2.5, v=5.0, dt=0.02)
    
    weight_sets = [
        ("Default", np.diag([10.0, 5.0]), 1.0),
        ("High Lateral", np.diag([50.0, 5.0]), 1.0),
        ("High Heading", np.diag([10.0, 20.0]), 1.0),
        ("Conservative", np.diag([5.0, 2.0]), 1.0),
        ("Aggressive", np.diag([20.0, 10.0]), 0.5),
    ]
    
    results_comparison = {}
    
    for name, Q, R_val in weight_sets:
        print(f"\nTesting: {name}")
        print(f"  Q = diag({Q[0,0]}, {Q[1,1]}), R = {R_val}")
        print("-" * 70)
        
        R = np.array([[R_val]])
        data = tester.simulate_tracking(
            lambda t, v: ref_lane_change(t, v),
            Q, R, T=12.0
        )
        
        m = data['metrics']
        results_comparison[name] = m
        
        print(f"  Lateral RMS: {m['lateral_rms']:.4f} m")
        print(f"  Lateral Max: {m['lateral_max']:.4f} m")
        print(f"  Heading RMS: {np.rad2deg(m['heading_rms']):.2f} °")
        print(f"  Settling time: {m['settling_time']:.2f} s")
        print(f"  Steering Max: {np.rad2deg(m['steering_max']):.1f} °")
    
    # Find best
    print("\n" + "="*70)
    print("  BEST PERFORMERS")
    print("="*70)
    
    best_lat = min(results_comparison.items(), key=lambda x: x[1]['lateral_rms'])
    best_settling = min(results_comparison.items(), key=lambda x: x[1]['settling_time'])
    
    print(f"\nBest Lateral Tracking: {best_lat[0]} "
          f"(RMS = {best_lat[1]['lateral_rms']:.4f} m)")
    print(f"Best Settling Time: {best_settling[0]} "
          f"(t_s = {best_settling[1]['settling_time']:.2f} s)")
    
    print("\n" + "="*70 + "\n")
    
    return results_comparison


# =====================
# Main
# =====================

if __name__ == "__main__":
    
    # Default weights
    Q = np.diag([10.0, 5.0])
    R = np.array([[1.0]])
    
    # Run full test suite
    tester = LQRPerformanceTester(L=2.5, v=5.0, dt=0.02)
    results = tester.run_all_tests(Q, R)
    
    # Comparative analysis
    print("\n" + "="*70)
    input("Press Enter to run comparative analysis...")
    comparison = compare_weight_matrices()
    
    print("\n✅ All validation tests complete!")
