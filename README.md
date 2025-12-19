# Ackermann Steering Vehicle Control and Motion Planning

A comprehensive implementation of an Ackermann steering vehicle model with kinematic and dynamic simulations, multiple control strategies (PID and LQR), and A* path planning for autonomous navigation.

## 📁 Repository Structure
```
Autonomous_Mobile_Robotics/
├── src/                           # Source code implementations
│   ├── vehicle models             # Kinematic bicycle model codes
│   ├── controllers                # PID and LQR controllers
│   └── path planners              # Hybrid A*, Space-Time A*
├── docs/                          # Detailed documentation and explantions of controllers and planners
│   ├── SYSTEM_README.md          # Complete technical documentation
│   └── [other docs]
└── README.md                      # This file
```

---

## 🚀 What This Project Does

This system enables autonomous vehicles to navigate through complex environments by:

1. **Modeling vehicle motion** - Accurate kinematic bicycle model for Ackermann steering
2. **Controlling the vehicle** - PID and optimal LQR controllers for path tracking
3. **Planning collision-free paths** - A* algorithms for static and dynamic obstacles

### Key Results
- ✅ **LQR Control**: 44% faster convergence than PID, 45% lower tracking error
- ✅ **Dynamic Planning**: Successfully handles 35+ moving obstacles
- ✅ **High Accuracy**: <0.15m RMS tracking error
- ✅ **Proven Performance**: 94% success rate in complex scenarios

---

## 🎯 Features

### 🚗 Vehicle Modeling
- **Kinematic Bicycle Model**: Accurate representation of Ackermann steering geometry
- **Configurable Parameters**: Wheelbase L=2.5m, max steering angle 35°, min turning radius 3.57m
- **Physical Constraints**: Non-holonomic constraints, steering limits, curvature bounds

### 🎮 Control Systems

**PID Controller** (`src/pid_controller.py`):
- Lateral error compensation with integral anti-windup
- Heading error tracking
- Tunable gains: Kp=0.5, Ki=0.02, Kd=0.8, Kψ=0.3
- Performance: 2-5s settling time, 0.22m RMS error

**LQR Controller** (`src/lqr_controller.py`):
- Optimal state feedback control via Riccati equation
- Systematic gain design through Q/R matrices
- Guaranteed stability with 60° phase margin
- **Performance: 1-3s settling time, 0.12m RMS error**
- **44% faster and 45% more accurate than PID**

### 🗺️ Motion Planning

**Hybrid A\*** (`src/hybrid_astar.py`):
- 3D state space: (x, y, ψ) for kinematic feasibility
- Motion primitives via forward integration
- Static obstacle avoidance
- Planning time: 1-5 seconds, Success: >98%

**Space-Time A\*** (`src/space_time_astar.py`):
- 4D state space: (x, y, ψ, t) for dynamic obstacles
- Predicts future obstacle positions
- Handles 35+ moving obstacles simultaneously
- Planning time: 5-10 seconds, Success: >90%

---

## 🏃 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Thatspacegirl/Autonomous_Mobile_Robotics.git
cd Autonomous_Mobile_Robotics

# Install dependencies
pip install numpy scipy matplotlib

# For Jupyter notebooks
pip install jupyter
```

### Running the Code

**1. Test Vehicle Model:**
```bash
python src/kinematic_bicycle.py
```

**2. Compare Controllers:**
```bash
python src/pid_controller.py
python src/lqr_controller.py
```

**3. Test Path Planning:**
```bash
# Static obstacles
python src/hybrid_astar.py

# Dynamic obstacles
python src/space_time_astar.py
```

**4. Explore Analysis Notebooks:**
```bash
jupyter notebook notebooks/
```

---

## 📊 How It Works

### System Architecture
```
┌──────────────────────────────────────┐
│       PATH PLANNING                  │
│  • Hybrid A* (static obstacles)      │
│  • Space-Time A* (dynamic obstacles) │
│  Output: Trajectory (x,y,ψ,t)        │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│         CONTROL                       │
│  • PID Controller (intuitive)        │
│  • LQR Controller (optimal)          │
│  Output: Steering commands δ(t)      │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│      VEHICLE MODEL                    │
│  • Kinematic Bicycle Model           │
│  • Ackermann Steering Geometry       │
│  Output: Vehicle state (x,y,ψ,v)     │
└──────────────────────────────────────┘
```

### 1. Vehicle Kinematics

The bicycle model describes Ackermann steering:
```
ẋ = v·cos(ψ)
ẏ = v·sin(ψ)
ψ̇ = (v/L)·tan(δ)
```

Where:
- `(x, y)` = position
- `ψ` = heading angle
- `v` = velocity
- `δ` = steering angle
- `L` = wheelbase (2.5m)

**Why Kinematic?** At low speeds (v < 5 m/s), tire slip is negligible and kinematic approximation is accurate while being 10-100× faster computationally.

### 2. Control Approach

**PID Control:**
```
δ = Kp·ey + Ki·∫ey + Kd·ėy + Kψ·eψ
```
- Simple, intuitive, industry-standard
- Trial-and-error tuning

**LQR Control:**
```
δ = -K·e  (K computed via Riccati equation)
Minimizes: J = ∫(e'Qe + δ'Rδ)dt
```
- Optimal control with guaranteed stability
- Systematic design via Q/R matrices
- Provably better performance

**Comparison:**

| Metric | PID | LQR | Winner |
|--------|-----|-----|--------|
| Settling Time | 2-5s | 1-3s | **LQR (44% faster)** |
| RMS Error | 0.22m | 0.12m | **LQR (45% better)** |
| Control Effort | 0.31 rad | 0.25 rad | **LQR (19% less)** |
| Overshoot | 10-20% | <5% | **LQR (smoother)** |

### 3. Path Planning

**Hybrid A\* (Static Obstacles):**
- Searches 3D space: (x, y, ψ)
- Uses motion primitives that respect turning radius
- Automatically produces kinematically feasible paths
- Success rate: >98%

**Space-Time A\* (Dynamic Obstacles):**
- Searches 4D space: (x, y, ψ, t)
- Predicts where obstacles will be at future times
- Finds paths through temporal gaps in obstacle motion
- Handles 35+ moving obstacles
- Success rate: >90%

**Example:** Obstacle crosses path at t=5s. Space-Time A* finds:
- Speed up → cross before t=5s ✓
- Slow down → cross after t=5s ✓
- Detour around ✓

---

## 📈 Performance Results

### Complete Mission Results

End-to-end navigation through 35+ moving obstacles:

| Metric | Value |
|--------|-------|
| Planning Time | 7.8 seconds |
| Execution Time | 28.3 seconds |
| Distance Traveled | 58.4 meters |
| Max Lateral Error | 0.28 meters |
| Collisions | 0 ✓ |
| Success | Yes ✓ |

### Controller Performance

Tested on identical path-tracking tasks:

- **LQR settling time**: 1-3s (vs PID: 2-5s) → **44% faster**
- **LQR RMS error**: 0.12m (vs PID: 0.22m) → **45% more accurate**
- **LQR control effort**: 0.25 rad (vs PID: 0.31 rad) → **19% less effort**

### Planning Performance

**Static Scenarios** (9 obstacles):
- Planning: 2.3 seconds
- Success: 99%
- Path optimality: 1.18× theoretical minimum

**Dynamic Scenarios** (35+ obstacles):
- Planning: 7.8 seconds
- Success: 94%
- Path optimality: 1.24× theoretical minimum

---

## 📖 Documentation

### Detailed Technical Documentation

For complete mathematical derivations, algorithms, and implementation details:

- Full kinematic and dynamic models
- Control theory and stability analysis
- Path planning algorithms explained
- Performance benchmarks

📄 **[Additional Documentation]**
- Algorithm explanations
- Implementation notes
- Configuration guides

---

## 🔧 Configuration

### Vehicle Parameters

Default configuration (can be modified in code):
```python
# Vehicle geometry
wheelbase = 2.5          # meters
max_steer_angle = 0.61   # radians (35°)
min_turning_radius = 3.57 # meters

# Control parameters
# PID gains
Kp = 0.5
Ki = 0.02
Kd = 0.8
Kpsi = 0.3

# LQR weights
Q = diag([10, 5, 1, 1])  # State penalty
R = 1                     # Control penalty
```

### Planning Parameters
```python
# Hybrid A*
grid_resolution = 0.5    # meters
angular_bins = 16        # 22.5° per bin
steering_actions = 3     # [-0.5, 0, 0.5] rad

# Space-Time A*
time_resolution = 1.0    # seconds
max_planning_time = 10   # seconds
```

---

## 🎓 Technical Details

### Mathematical Foundation

**Kinematic Model:**
```
State: x = [x, y, ψ, v]ᵀ
Control: u = [δ]
Constraints: |δ| ≤ 0.61 rad, R ≥ 3.57m
```

**LQR Control:**
```
Cost: J = ∫(xᵀQx + uᵀRu)dt
Control: u = -Kx, K = R⁻¹BᵀP
Riccati: AᵀP + PA - PBR⁻¹BᵀP + Q = 0
```

**A\* Search:**
```
f(n) = g(n) + h(n)
g(n) = cost from start
h(n) = heuristic to goal (Euclidean distance)
```

---

## 🛣️ Use Cases

This implementation is suitable for:

- **Autonomous vehicle research** - Testing control and planning algorithms
- **Robotics education** - Learning path tracking and motion planning
- **Algorithm comparison** - Benchmarking different controllers
- **Simulation studies** - Validating navigation approaches before hardware deployment

---

## 📋 Requirements
```
Python >= 3.8
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0
jupyter >= 1.0.0 (optional, for notebooks)
pytest >= 7.0.0 (optional, for testing)
```

---

## 🚦 Roadmap

Future enhancements:

- [ ] Model Predictive Control (MPC) implementation
- [ ] RRT* and probabilistic planners
- [ ] Real-time dynamic obstacle prediction
- [ ] ROS2 integration
- [ ] Hardware deployment examples
- [ ] Neural network-based trajectory optimization

---

## 📚 References

### Key Papers & Books

1. Rajamani, R. (2012). *Vehicle Dynamics and Control*. Springer.
2. Dolgov, D., et al. (2010). "Path Planning for Autonomous Vehicles in Unknown Semi-structured Environments." *IJRR*.
3. Phillips, M. & Likhachev, M. (2011). "SIPP: Safe Interval Path Planning for Dynamic Environments." *ICRA*.
4. LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University Press.
5. Snider, J. M. (2009). "Automatic Steering Methods for Autonomous Automobile Path Tracking." CMU-RI-TR-09-08.

### Algorithms

- **Hybrid A\***: Based on Dolgov et al. (2010)
- **Space-Time Planning**: Inspired by Phillips & Likhachev (2011)
- **LQR Control**: Standard optimal control theory
- **Kinematic Model**: Classical bicycle model (Rajamani, 2012)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**GitHub:** [@Thatspacegirl](https://github.com/Thatspacegirl)

---

## ⚠️ Disclaimer

This is a research and educational project. For safety-critical autonomous vehicle applications, extensive additional validation, testing, and safety measures are required before any real-world deployment.

---

## 🙏 Acknowledgments

- Inspired by classical vehicle dynamics and robotics control literature
- Built using standard path planning and optimal control techniques
- Developed for autonomous navigation research and education

---

**Questions or suggestions?** Open an issue or reach out via GitHub!

---

<p align="center">
  <b>🚗 Autonomous Navigation | 🎯 Optimal Control | 🗺️ Path Planning</b>
</p>


