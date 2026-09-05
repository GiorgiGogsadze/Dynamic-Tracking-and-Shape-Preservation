# Drone Swarm Choreography — Numerical Programming Final Project

**Author:** Giorgi Gogsadze

**Course:** Numerical Programming

**Instructor:** Prof. Ramaz Botchorishvili

**Date:** 27.01.2026

## Reports

Full write-ups with equations, figures, and per-case discussion are included in `reports/`:

- [Final_P1_Report_Giorgi_Gogsadze.pdf](Part1/P1_Report.pdf) — Static Formation on a Handwritten Input. [See Result](https://drive.google.com/file/d/1SNxTweYDmIoakz0uOpISYB7CD0rd3Htq/view?usp=sharing)
- [Final_P2_Report_Giorgi_Gogsadze.pdf](Part2/P2_Report.pdf) — Transition to New Year Greeting. [See Result](https://drive.google.com/file/d/1m9Im49hAMTX2ky-kI_PYogLwQ-ndluxa/view?usp=sharing)
- [Final_P3_Report_Giorgi_Gogsadze.pdf](Part3/P3_Report.pdf) — Dynamic Tracking and Shape Preservation. [See Result](https://drive.google.com/file/d/1eQHH6q2nLPhIDXzOSmhTRb8QVUaTtWR2/view?usp=sharing) [(input video)](https://drive.google.com/file/d/19X0yb2QDZV3NLnrhmJJN_dmzngm6jxO-/view?usp=sharing)

A three-stage numerical simulation pipeline that choreographs a swarm of drones — first converging on a handwritten signature, then morphing into a static holiday greeting, and finally coming alive to track and mimic the motion of an object in video. Every stage is built on the same core dynamical model (attraction + repulsion + damping, integrated with RK4), with each sub-problem extending the previous one.

## Table of Contents

- [Drone Swarm Choreography — Numerical Programming Final Project](#drone-swarm-choreography--numerical-programming-final-project)
  - [Reports](#reports)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Core Numerical Model](#core-numerical-model)
  - [Sub-Problem 1 — Static Formation from a Handwritten Input](#sub-problem-1--static-formation-from-a-handwritten-input)
  - [Sub-Problem 2 — Transition to a New Year Greeting](#sub-problem-2--transition-to-a-new-year-greeting)
  - [Sub-Problem 3 — Dynamic Tracking and Shape Preservation](#sub-problem-3--dynamic-tracking-and-shape-preservation)
    - [3.1 Initialization — edge detection](#31-initialization--edge-detection)
    - [3.2 Motion extraction — optical flow](#32-motion-extraction--optical-flow)
    - [3.3 Shape preservation via rigid-motion estimation](#33-shape-preservation-via-rigid-motion-estimation)
    - [3.4 Time synchronization](#34-time-synchronization)
    - [3.5 Algorithm summary](#35-algorithm-summary)
    - [3.6 Test cases](#36-test-cases)
  - [Numerical Methods](#numerical-methods)
  - [Results Summary](#results-summary)
  - [Known Limitations](#known-limitations)
  - [Setup \& Usage](#setup--usage)

## Overview

The project is split into three chained sub-problems, each producing a checkpoint that seeds the next:

| Stage  | Task                                                          | Model Type                          | Input                      | Output                     |
| ------ | ------------------------------------------------------------- | ----------------------------------- | -------------------------- | -------------------------- |
| **P1** | Converge from a grid to a handwritten signature               | IVP (defended over BVP)             | Photo of handwriting       | Static signature formation |
| **P2** | Morph from the signature into "Happy New Year!" text          | IVP                                 | Procedurally rendered text | Static greeting formation  |
| **P3** | Track a moving object from video while preserving swarm shape | IVP with Velocity Tracking (IVP-VT) | Video of a moving object   | Dynamic swarm motion       |

Each stage reuses the same attraction–repulsion–damping dynamics and RK4 integrator, so the swarm's behavior is numerically and visually continuous across the whole show — the final state of one sub-problem is the initial condition of the next.

## Core Numerical Model

Every sub-problem shares the same state representation and force structure.

**State.** For each drone `i = 1, …, N`, position `xᵢ(t) ∈ ℝᵈ` and velocity `vᵢ(t) ∈ ℝᵈ` (d = 2 or 3). The full system state is `y(t) = [x₁, …, x_N, v₁, …, v_N]`.

**Position update (with velocity saturation):**

```
ẋᵢ = vᵢ · min(1, v_max / ‖vᵢ‖)
```

**Velocity update (attraction–repulsion–damping):**

```
v̇ᵢ = (1/m) [ kp·(Tᵢ − xᵢ) + Σⱼ≠ᵢ f_rep(xᵢ, xⱼ) − kd·vᵢ ]
```

**Collision-avoidance force** (short-range, inverse-cube, zero beyond a safety radius):

```
f_rep(xᵢ, xⱼ) = k_rep · (xᵢ − xⱼ) / ‖xᵢ − xⱼ‖³   if ‖xᵢ − xⱼ‖ < R_safe
              = 0                                  otherwise
```

**Drone–target assignment.** Whenever drones need to be paired with a new set of target points, the pairing is solved as a linear assignment problem via the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`), minimizing total squared displacement:

```
min_π  Σᵢ ‖xᵢ − T_π(i)‖²
```

This guarantees globally optimal, non-crossing trajectories and is the reason morphing between formations looks smooth rather than tangled. It's an O(N³) operation, but since it's only solved once per formation change, it stays cheap even for a few hundred drones.

**Integrator.** All three sub-problems integrate this system with classical **4th-order Runge–Kutta (RK4)**:

```
y_{n+1} = y_n + (Δt/6)(k1 + 2k2 + 2k3 + k4)
```

with `Δt = 0.02`. Because RK4 is not A-stable, stability in practice comes from a combination of velocity saturation, damping, and repulsion clipping, not from the integrator alone.

**Default parameters** (P1/P2):

| Parameter | Meaning           | Value           |
| --------- | ----------------- | --------------- |
| N         | Number of drones  | 150 / 225 / 300 |
| d         | Spatial dimension | 2 or 3          |
| m         | Drone mass        | 1.0             |
| v_max     | Max velocity      | 50              |
| kp        | Attraction gain   | 3.5             |
| kd        | Damping gain      | 2.0             |
| k_rep     | Repulsion gain    | 18.0            |
| R_safe    | Safety radius     | 1.2             |
| Δt        | Time step         | 0.02            |

## Sub-Problem 1 — Static Formation from a Handwritten Input

**Goal:** move a swarm from an initial compact grid to a static shape matching a photo of handwriting (the author's signature).

**Why IVP instead of BVP.** The task is naturally a two-point boundary value problem (fixed start, fixed end configuration). Instead, it's solved as an initial value problem by encoding the target as an attracting equilibrium (`kp·(x_target − x)` plus damping). This avoids the cost and fragility of shooting-method BVP solvers on a high-dimensional nonlinear system, while achieving the same converged result, since damping and saturation make the dynamics asymptotically stable.

**Image preprocessing pipeline:**

1. **Adaptive thresholding** (`cv2.adaptiveThreshold`, Gaussian, inverse binary) — robust to uneven lighting, turns ink into white-on-black.
2. **Skeletonization** — morphological thinning down to 1-pixel-wide centerlines, so drones trace lines rather than clustering into blobs.
3. **Contour extraction** (`cv2.findContours`) — ordered pixel paths from the skeleton.
4. **Arc-length sampling** — N target points are sampled at even arc-length intervals along the contours, giving uniform drone spacing with no clustering at corners.
5. **Coordinate mapping** — image coordinates are rescaled into world coordinates, flipping the vertical axis (image top-left origin → world bottom-left origin).

**Test cases:**

- **Pen, normal size** — ✅ success. High contrast, continuous strokes, adequate scale.
- **Marker, normal size** — ✅ success. Skeletonization normalizes the thicker stroke width; adaptive thresholding absorbs uneven ink density.
- **Marker, small/cramped writing** — ❌ fails. Thick strokes relative to letter size blur corners and junctions, distorting the skeleton and producing inconsistent arc-length ordering — the swarm still converges to _some_ points, but the resulting shape is not legible.

**Quantitative validation:** mean position error `e(t) = (1/N) Σ ‖xᵢ(t) − Tᵢ‖` decreases monotonically in every tested case, converging below 0.3 world units.

## Sub-Problem 2 — Transition to a New Year Greeting

**Goal:** starting from the final signature formation of P1, morph the swarm into a static "Happy New Year!" greeting.

**Why the greeting is generated, not photographed.** Procedural text rendering guarantees topological consistency (no noise, no threshold tuning, fully deterministic), unlike sourcing the shape from an image.

**Text generation pipeline:**

1. Render the string with OpenCV using a handwriting-style font (`cv2.FONT_HERSHEY_SCRIPT_COMPLEX`) into a binary mask.
2. Skeletonize to a clean centerline.
3. Extract contours from the skeleton.
4. Sample uniformly by arc length to produce exactly N target points.
5. Map to world coordinates using the same transform as P1, preserving visual continuity between stages.

**Assignment:** the Hungarian algorithm re-solves the drone→target pairing at the start of this stage (current signature positions → new greeting targets), preventing trajectory crossings during the morph.

**Drone-count experiments** (2D and 3D both tested; 2D found fully sufficient):

| N       | Result               | Notes                                                                                                            |
| ------- | -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 150     | Insufficient density | Letters broken, gaps in curved strokes — a sampling problem, not an instability                                  |
| **225** | **Optimal**          | Clearly readable, uniform stroke spacing, no clustering, smooth convergence                                      |
| 300     | Over-sampling        | Very dense/smooth, but drones compete for near-identical targets; O(N²) pairwise repulsion cost rises noticeably |

**Conclusion:** shape quality here is governed by sampling density relative to text geometry, not by the control law or the integrator.

## Sub-Problem 3 — Dynamic Tracking and Shape Preservation

**Goal:** transition the swarm from the static greeting into the shape of an object detected in a video's first frame, then have the swarm dynamically follow the object's motion as a rigid formation (no chaotic deformation).

This stage upgrades the model to an **IVP with Velocity Tracking (IVP-VT)**, since the target is now a time-varying velocity field rather than a fixed point.

### 3.1 Initialization — edge detection

The first video frame seeds the swarm's shape (once, not per-frame):

1. Grayscale conversion.
2. **Canny edge detection**: `E = Canny(I, T_low, T_high)`.
3. Contour extraction from the edge map.
4. Uniform arc-length sampling to get N target points.

Edge detection is deliberately **not** re-run during playback — recomputing it every frame would cause contour flicker and undermine shape preservation without helping motion tracking.

### 3.2 Motion extraction — optical flow

- Dense motion between frames is estimated with **Farnebäck optical flow**, giving pixel-per-frame vectors `u(p, t)`.
- Flow is converted to world velocity using scale factors `sx = W_world/W_img`, `sy = H_world/H_img` and the video FPS: `Vx = ux·sx·FPS`, `Vy = −uy·sy·FPS` (the sign flip accounts for the image-to-world y-axis inversion).
- **Robustness:** the flow field is Gaussian-blurred to spread motion to nearby low-texture pixels (a "wake effect"), but a magnitude mask keeps strong, high-confidence flow sharp and only substitutes the smoothed version where flow is weak.

### 3.3 Shape preservation via rigid-motion estimation

Rather than letting each drone chase local optical flow directly (which shears and deforms the swarm), a **single rigid motion (translation + rotation)** is fit to the flow field and applied to the whole swarm:

```
centroid: c = (1/N) Σ xᵢ,   rᵢ = xᵢ − c
rigid model: vᵢ_rigid = u + ω·J·rᵢ,   J = [[0,-1],[1,0]]

ω = [ Σᵢ (J·rᵢ)·vᵢ_flow ] / [ Σᵢ |J·rᵢ|² + ε ]     (least-squares projection)
u = (1/N) Σᵢ (vᵢ_flow − ω·J·rᵢ)                     (mean residual)
```

The resulting `vᵢ_rigid` is saturated to `v_max` and becomes the _kinematic_ target: `ẋᵢ = v_rigid` enforces rigid shape motion directly, while the _velocity_ state still obeys the IVP-VT servo law:

```
v̇ᵢ = (1/m) [ kv·(vᵢ_rigid − vᵢ) + f_rep − kd·vᵢ ]
```

so `vᵢ` acts as a damped, collision-aware filtered version of the rigid target rather than the literal drive signal.

### 3.4 Time synchronization

Physics integrates at fixed `Δt` (e.g. 0.02 s) while video frames arrive every `Δt_video = 1/FPS ≈ 0.0333 s`. A time accumulator advances physics every `Δt` but only pulls a new flow frame once accumulated time exceeds `Δt_video`, avoiding double-counting or missed frames when the two rates don't match. Rigid velocities are optionally rescaled by `Δt_video / Δt` to keep motion magnitude consistent regardless of the ratio between simulation and video time steps.

### 3.5 Algorithm summary

1. Load state from the P2 checkpoint.
2. Compute the first-frame edge mask → initial target shape.
3. Morph from the greeting into the object shape over a few seconds (reusing the P1/P2 static model).
4. Per simulation step: refresh optical flow when video time advances → sample flow at drone positions → fit rigid `(u, ω)` → compute `v_rigid` → integrate the IVP-VT system with RK4 → record trajectories.

### 3.6 Test cases

| Case                         | Object motion                                         | Result                                              | Why                                                                                                                                                 |
| ---------------------------- | ----------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ring translating             | Left→right, slight downward drift                     | ✅ Success                                          | Clean edges, strong texture, slow translational motion — rigid-body assumption holds well                                                           |
| Cross translating + rotating | Combined translation and rotation                     | ✅ Success                                          | Strong directional features stabilize flow; motion exactly matches the assumed rigid model                                                          |
| Ring rotating out of plane   | Perspective distortion, briefly degenerates to a line | ⚠️ "Fails" as shape-tracking but succeeds by design | Edge detection only runs once at init, so transient visual distortion doesn't corrupt the swarm's shape — this is intentional robustness, not a bug |
| Two balls colliding          | Two independent, conflicting motions                  | ❌ Fails (expected)                                 | The single rigid-motion fit averages two incompatible flow fields toward ~zero net motion; multi-object tracking is explicitly out of scope         |

## Numerical Methods

- **Integrator:** 4th-order Runge–Kutta across all three sub-problems, chosen for O(Δt⁴) local truncation error and good behavior on smooth attraction–repulsion force fields. At Δt = 0.02, leading-order global error is on the order of 1.6×10⁻⁷.
- **Stability caveat:** classical RK4 is not A-stable; it's only stable within a bounded region of the complex plane for the linear test equation. Practical stability instead comes from:
  - velocity saturation (`v_max`),
  - damping (`−kd·v`),
  - repulsion clipping,
  - a small, fixed time step,
  - (P3 only) flow smoothing + strong-flow masking and a bounded simulation duration cap.
- **Assignment problem:** Hungarian algorithm (`linear_sum_assignment`), O(N³) worst case, solved once per formation change — negligible cost relative to the simulation loop for swarm sizes up to several hundred drones.

## Results Summary

- The pipeline reliably takes a swarm from a handwriting photo → a static holiday greeting → live tracking of a moving object, using one continuous, physically-inspired control law throughout.
- Shape quality in the static stages (P1, P2) is limited primarily by **image/text sampling density**, not by the ODE model or integrator.
- Dynamic tracking (P3) is reliable for a **single rigid object** under translation and/or rotation, even with handheld, non-ideal footage, and is intentionally robust to transient perspective distortion because shape is only initialized once, never re-derived from images during motion.
- The known failure mode — multiple independently moving objects — is a direct, expected consequence of fitting one global rigid-motion estimate per frame, not an implementation defect.

## Known Limitations

- **P1:** very thick strokes relative to the writing scale break skeleton fidelity and arc-length ordering.
- **P2:** greeting legibility depends on manually tuning N against text complexity (150 too sparse, 300 over-constrained, 225 sweet spot for this text/font).
- **P3:** single-object rigid-motion assumption only; no object segmentation or multi-target tracking; edge-based shape is fixed after initialization and cannot adapt to genuine non-rigid deformation.

## Setup & Usage

> Fill in with your actual entry points/CLI flags — placeholders below follow the parameter names used in the reports.

```bash
# install dependencies
pip install numpy opencv-python scipy matplotlib

# Sub-problem 1: handwriting -> static formation
python src/sub_problem_1.py --input inputs/my_name.jpg --n_drones 300 --dim 2

# Sub-problem 2: signature -> "Happy New Year!" greeting
python src/sub_problem_2.py --greeting_text "Happy New Year!" --n_drones 225 --dim 2

# Sub-problem 3: greeting -> video object tracking
python src/sub_problem_3.py --video inputs/my_rotation.mp4
```

Each stage reads the checkpoint saved by the previous stage as its initial condition, so run them in order (1 → 2 → 3) the first time through.
