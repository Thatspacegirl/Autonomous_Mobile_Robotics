# LQR Controller Stability Analysis Package

Complete stability analysis, testing, and validation suite for the LQR (Linear Quadratic Regulator) steering controller with guaranteed optimality and robustness properties.



##  Quick Start

### Run Complete Analysis (Recommended)
```bash
python run_lqr_analysis.py
```

This will:
- ✅ Verify LQR optimality and stability
- ✅ Analyze eigenvalues and closed-loop response
- ✅ Generate Bode plots with guaranteed margins
- ✅ Test Q/R weight sensitivity
- ✅ Run comprehensive validation tests
- ✅ Create visualization plots and dashboard

**Output files:**
- `lqr_eigenvalues.png` - Closed-loop pole locations
- `lqr_bode.png` - Frequency response analysis
- `lqr_velocity_stability.png` - Speed-dependent behavior
- `lqr_q_weights.png` - Weight sensitivity heatmap
- `lqr_dashboard.png` - **Complete dashboard (view this first!)**

## 🎯 Why LQR?

### Key Advantages Over PID

| Property | LQR | PID |
|----------|-----|-----|
| **Stability** | ✅ Guaranteed (if controllable) | ⚠️ Must verify |
| **Optimality** | ✅ Optimal for cost J | ❌ No guarantee |
| **Gain Margin** | ✅ Infinite upward (≥0.5×) | ⚠️ Depends on tuning |
| **Phase Margin** | ✅ ≥60° guaranteed | ⚠️ Depends on tuning |
| **Tuning** | ✅ Systematic (Q, R) | ⚠️ Trial-and-error |
| **Robustness** | ✅ Excellent | ⚠️ Variable |

### LQR Guarantees

If system is **controllable** and Q≥0, R>0, then LQR provides:
1. **Guaranteed stability**: All closed-loop poles in left half-plane
2. **Infinite gain margin**: Can scale K by [0.5, ∞) without instability
3. **Phase margin ≥ 60°**: Excellent robustness to delays/uncertainties
4. **Optimal performance**: Minimizes J = ∫(x'Qx + u'Ru)dt

## 📊 Understanding the Results

### 1. Eigenvalue Analysis

**What to look for:**
```
Closed-Loop Eigenvalues:
  λ₁ = -3.2 + 2.1j  → Damped oscillation
  λ₂ = -3.2 - 2.1j  → Complex conjugate pair

Damping Ratio: ζ = 0.84  ✓ Well-damped
Natural Frequency: ωₙ = 3.8 rad/s
```

**Interpretation:**
- **All Re(λ) < 0**: ✅ STABLE (LQR guarantees this)
- **ζ > 0.7**: Well-damped response, minimal overshoot
- **ζ < 0.5**: Underdamped, increase q_heading weight

### 2. Frequency Domain (Bode Plots)

**LQR Theoretical Guarantees:**

| Metric | LQR Guarantee | Typical |
|--------|---------------|---------|
| Gain Margin | Infinite upward | 15-25 dB |
| Phase Margin | ≥ 60° | 60-80° |

**Example results:**
```
Gain Margin: 18.5 dB  ✅
  → Can increase gain by 8.4× before instability

Phase Margin: 68°  ✅
  → 68° of phase lag margin (exceeds 60° guarantee)
```

### 3. Q/R Weight Effects

**Q Matrix** (State Weights):
```python
Q = [[q_lateral,    0       ],
     [0,        q_heading   ]]
```

**Effects:**
- **↑ q_lateral**: Tighter lateral tracking, more aggressive
- **↑ q_heading**: Better heading control, more damping
- **Ratio matters**: q_lateral/q_heading typically 1.5-3.0

**R Scalar** (Control Weight):
- **↑ R**: Gentler steering, smoother but slower
- **↓ R**: Aggressive steering, faster but more control effort

### 4. Validation Test Results

**Passing criteria:**

| Test | Target | Meaning |
|------|--------|---------|
| Basic Tracking | RMS < 0.3m | Excellent path following |
| Disturbance Rejection | t_settle < 4s | Fast recovery |
| Velocity Range | All stable | Robust to speed |
| Q Weight Sensitivity | All stable | Robust to tuning |
| Parameter Robustness | >90% success | Very robust |
| Optimality Check | GM>10dB, PM>50° | LQR guarantees hold |

## 🔧 Tuning Your LQR Controller

### Current Weights Not Optimal?

#### Default Configuration
```python
Q = np.diag([10.0, 5.0])  # [lateral, heading]
R = 1.0
```

#### Common Adjustments

**Problem: Response too slow**
```python
Q = np.diag([20.0, 10.0])  # ↑ Both weights
R = 1.0
# Result: Faster response, tighter tracking
```

**Problem: Oscillations / overshoot**
```python
Q = np.diag([10.0, 20.0])  # ↑ Heading weight
R = 1.0
# Result: More damping, less overshoot
```

**Problem: Too much steering activity**
```python
Q = np.diag([10.0, 5.0])
R = 2.0  # ↑ Control penalty
# Result: Gentler steering, smoother
```

**Problem: Loose tracking**
```python
Q = np.diag([50.0, 10.0])  # ↑ Lateral weight
R = 0.5  # ↓ Control penalty
# Result: Tighter path following
```

### Systematic Tuning Procedure

**Step 1: Bryson's Rule** (initial guess)
```python
q_lateral = 1 / acceptable_lateral_error²
q_heading = 1 / acceptable_heading_error²
r = 1 / acceptable_steering²

# Example:
# Accept ±0.5m lateral → q_lateral = 4
# Accept ±0.3rad heading → q_heading = 11
# Accept ±0.5rad steering → r = 4

Q = np.diag([4, 11])
R = 4
```

**Step 2: Normalize R = 1** (only ratios matter)
```python
scale = R  # R = 4 in example
Q_normalized = Q / scale  # [1, 2.75]
R_normalized = 1

# Round to nice values:
Q = np.diag([1, 3])
R = 1
```

**Step 3: Iterate based on response**
```python
# Run analysis
python run_lqr_analysis.py

# Check results:
# - Eigenvalues for damping
# - Step response for overshoot
# - Tracking errors for performance

# Adjust and repeat
```

### Quick Tuning Guide

| Observation | Action |
|-------------|--------|
| Too slow | ↑ Q (multiply by 2) |
| Oscillations | ↑ q_heading, ↓ q_lateral |
| Loose tracking | ↑ q_lateral |
| Too aggressive | ↑ R |
| Good balance | Done! ✓ |

## 🧪 Custom Testing

### Test Your Own Weights

```python
from lqr_stability_analysis import LQRStabilityAnalyzer

# Your custom weights
Q = np.diag([15.0, 8.0])
R = np.array([[1.5]])

# Quick stability check
analyzer = LQRStabilityAnalyzer(v=5.0, L=2.5)
result = analyzer.analyze_eigenvalues(Q, R)

if result['stable']:
    print("✓ Stable - K =", result['gain'].flatten())
    print(f"  Damping: {result['damping_ratios'][0]:.3f}")
else:
    print("✗ Unstable - check controllability")
```

### Compare Multiple Weight Sets

```python
from lqr_validation_tests import compare_weight_matrices

# Automatically tests and compares configurations
results = compare_weight_matrices()
```

### Test Custom Path

```python
from lqr_validation_tests import LQRPerformanceTester

def my_path(t, v):
    """Your custom reference path."""
    x = v * t
    y = 3.0 * np.sin(0.3 * t)
    # ... compute psi, kappa
    return x, y, psi, kappa

tester = LQRPerformanceTester(L=2.5, v=5.0)
Q = np.diag([10.0, 5.0])
R = np.array([[1.0]])

data = tester.simulate_tracking(my_path, Q, R, T=20.0)
print(f"Lateral RMS: {data['metrics']['lateral_rms']:.3f}m")
```

## 📚 Detailed Documentation

### LQR_STABILITY_GUIDE.md

**Comprehensive theoretical guide covering:**
1. ✅ LQR theory and optimization
2. ✅ Algebraic Riccati Equation (ARE) solution
3. ✅ Weight matrix selection and effects
4. ✅ Stability analysis methods
5. ✅ Robustness guarantees
6. ✅ Tuning guidelines
7. ✅ Comparison with PID

**Read this for:**
- Deep understanding of LQR optimality
- Mathematical foundations
- Weight selection methodology
- Advanced topics (gain scheduling, integral action)

## 🎯 Recommended Workflow

### For New LQR Controllers:

1. **Read LQR_STABILITY_GUIDE.md** (30 min)
   - Understand LQR theory
   - Learn about Q/R weight effects
   - Review tuning guidelines

2. **Run complete analysis** (2 min)
   ```bash
   python run_lqr_analysis.py
   ```

3. **Review dashboard** (5 min)
   - Check `lqr_dashboard.png`
   - Verify stability (should always be stable!)
   - Check damping and margins
   - Review test results

4. **Tune if needed** (iterative)
   - Adjust Q/R based on results
   - Re-run analysis
   - Compare performance

5. **Validate in simulation** (10 min)
   - Test with original tracking code
   - Verify visual performance
   - Test edge cases

### For Existing Controllers:

1. **Quick verification** (1 min)
   ```python
   from lqr_stability_analysis import LQRStabilityAnalyzer
   
   analyzer = LQRStabilityAnalyzer(v=5.0, L=2.5)
   result = analyzer.analyze_eigenvalues(Q, R)
   
   print(f"Stable: {result['stable']}")  # Should be True
   print(f"Damping: {result['damping_ratios'][0]:.3f}")
   ```

2. **Verify guarantees** (2 min)
   ```python
   freq = analyzer.frequency_domain_analysis(Q, R)
   print(f"GM: {freq['gain_margin_db']:.1f} dB (>10?)")
   print(f"PM: {freq['phase_margin_deg']:.1f}° (>60?)")
   ```

3. **Run validation** (3 min)
   ```bash
   python lqr_validation_tests.py
   ```

## ⚠️ Important Notes

### LQR Assumptions

These tools assume:
- **Linear dynamics**: Small errors (ẋ = Ax + Bu valid)
- **Time-invariant**: A, B don't change with time
- **Full state feedback**: Both e_y and e_ψ measured
- **No constraints**: Unlimited control authority

**For large errors or constraints:**
- Nonlinear effects matter
- Saturation impacts optimality
- Consider MPC (Model Predictive Control)

### When LQR Might Not Be Ideal

❌ **Model is highly uncertain**
- LQR requires accurate A, B matrices
- Consider robust control or adaptive methods

❌ **Cannot measure all states**
- Need state estimator (Kalman filter)
- Or use output feedback LQR

❌ **Hard constraints are critical**
- LQR doesn't handle constraints explicitly
- Use MPC or constrained LQR

✅ **For most path tracking:** LQR is excellent!

## 🔬 Advanced Features

### Gain Scheduling

Recompute K for different velocities:
```python
def compute_lqr_schedule(v_range, L, Q, R):
    """Pre-compute LQR gains for velocity range."""
    gains = []
    for v in v_range:
        analyzer = LQRStabilityAnalyzer(v, L)
        result = analyzer.analyze_eigenvalues(Q, R)
        gains.append(result['gain'])
    return gains

# Use during operation
v_schedule = [2, 5, 10, 15, 20]
K_schedule = compute_lqr_schedule(v_schedule, L, Q, R)

# Interpolate based on current speed
K_current = np.interp(v_current, v_schedule, K_schedule)
```

### Robustness Analysis

Monte Carlo with parameter uncertainty:
```python
from lqr_stability_analysis import LQRRobustnessAnalyzer

param_vars = {
    'L': 0.15,  # ±15% wheelbase
    'v': 0.20,  # ±20% velocity
}

result = LQRRobustnessAnalyzer.parameter_sensitivity(
    v, L, Q, R, param_vars, n_samples=500
)

print(f"Stability Rate: {result['stability_rate']*100:.1f}%")
# LQR should be >95% (very robust!)
```

### Q/R Weight Space Exploration

Visualize performance across weight space:
```python
from lqr_stability_analysis import LQRStabilityAnalyzer, LQRVisualizer

analyzer = LQRStabilityAnalyzer(v=5.0, L=2.5)

# Analyze Q weight space
result = analyzer.q_r_weight_analysis(
    q_lateral_range=[0, 2],   # 10^0 to 10^2
    q_heading_range=[0, 2],   # 10^0 to 10^2
    R_value=1.0
)

# Visualize
viz = LQRVisualizer()
viz.plot_q_weight_heatmap(result)
plt.show()
```

## 🐛 Troubleshooting

### "System is unstable"
```python
# Should never happen if system is controllable!
# Check:
from lqr_stability_analysis import LinearizedLateralDynamics

dyn = LinearizedLateralDynamics(v=5.0, L=2.5)
print(f"Controllable: {dyn.is_controllable()}")

# If not controllable (v=0?), fix velocity
```

### "Gain margin looks low"
```python
# LQR should have GM > 10 dB
# If not, verify:
# 1. Q is positive semi-definite
# 2. R is positive definite
# 3. ARE solution converged

import numpy as np
print("Q eigenvalues:", np.linalg.eigvals(Q))  # All ≥ 0?
print("R eigenvalues:", np.linalg.eigvals(R))  # All > 0?
```

### "Tracking performance is poor"
```python
# Increase Q/R ratio:
Q_new = Q * 2  # More aggressive
# or
R_new = R / 2  # Less control penalty

# Retest
```

### "Too much control activity"
```python
# Increase R:
R_new = R * 2  # More control penalty

# Or reduce Q:
Q_new = Q / 2
```

## 📞 Support & References

**Common Issues:**
- See LQR_STABILITY_GUIDE.md Section 7 (Troubleshooting)
- Check console output for specific failures
- Review generated dashboard for insights

**Key References:**
1. **Anderson & Moore** - *Optimal Control: Linear Quadratic Methods*
2. **Stengel** - *Optimal Control and Estimation*
3. **Åström & Murray** - *Feedback Systems*
4. **Rajamani** - *Vehicle Dynamics and Control*

---

## 📝 Summary Checklist

Before deploying your LQR controller:

- [ ] System is controllable: rank([B AB]) = 2
- [ ] Q ≥ 0 (positive semi-definite)
- [ ] R > 0 (positive definite)
- [ ] Closed-loop stable: all Re(λ) < 0 ✓
- [ ] Gain margin > 10 dB (LQR typically 15-25 dB)
- [ ] Phase margin > 60° (LQR guarantees ≥60°)
- [ ] Damping ratio ζ > 0.6 (well-damped)
- [ ] All validation tests pass
- [ ] Robustness: >90% Monte Carlo success
- [ ] Tested across velocity range
- [ ] Feedforward term included: δ = arctan(L·κ) - K·x
- [ ] Steering limits implemented
- [ ] Gain scheduling if v varies significantly

---

## 🎓 Key Takeaways

### LQR vs PID Summary

**Choose LQR when you want:**
- ✅ Guaranteed stability and robustness
- ✅ Optimal performance for given weights
- ✅ Systematic tuning procedure
- ✅ Excellent stability margins

**Choose PID when you want:**
- ✅ Simplest possible implementation
- ✅ No model required
- ✅ Most familiar to operators

**For Ackermann steering:** Both work well, but **LQR provides theoretical guarantees** that PID cannot match!

---

**Version:** 1.0  
**Last Updated:** 2024  
**Compatibility:** Python 3.7+, NumPy, SciPy, Matplotlib

---

*End of LQR Analysis README*
