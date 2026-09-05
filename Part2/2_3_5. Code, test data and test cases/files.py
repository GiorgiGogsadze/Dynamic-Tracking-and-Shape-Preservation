# === file: config.py ===
# Configuration Parameters

# Simulation Space
WORLD_W = 120.0
WORLD_H = 90.0
DEPTH = 40.0

# Resolution for Image/Video Processing
IMG_W = 800
IMG_H = 600

# Video Synchronization
VIDEO_FPS = 30.0    # Assumed FPS of input video
VIDEO_DT = 1.0 / VIDEO_FPS

# Drone Physics
MASS = 1.0
V_MAX = 50.0        # Increased to match potentially fast video motion
K_P = 3.5           # High P-Gain for tight formation holding
K_V = 8.0           # High Velocity Gain to minimize tracking lag
K_D = 2.0           # High Damping to prevent jitter on fine lines
K_REP = 18.0        # softer repulsion, more stable
R_SAFE = 1.2        # larger safety radius, smoother gradients
DT = 0.02           # Small timestep for stability

# Visualization
VIS_FPS = 30
GLOW_RADIUS = 2     # Small radius for high definition
COLOR_CORE = (255, 255, 255)
COLOR_GLOW = (0, 215, 255)   # Gold/Yellow-ish (BGR)

# === file: utils.py ===
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def assign_targets_hungarian(current_pos, target_pos):
    """
    Assigns N drones to N targets minimizing total Squared Euclidean distance.
    """
    n_drones = len(current_pos)
    n_targets = len(target_pos)

    # Resample if counts don't match
    if n_drones != n_targets:
        if n_targets == 0:
            return current_pos
        indices = np.linspace(0, n_targets - 1, n_drones).astype(int)
        target_pos = target_pos[indices]

    cost_matrix = cdist(current_pos, target_pos, 'sqeuclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    sorted_targets = np.zeros_like(current_pos)
    sorted_targets[row_ind] = target_pos[col_ind]

    return sorted_targets


def sample_path_by_arc_length(contours, n_points):
    """
    Samples N points evenly distributed along a list of contours (polylines).
    Prevents 'clustering' and ensures the whole signature is traced.
    """
    if not contours:
        return np.zeros((n_points, 2))

    # 1. Flatten into segments
    segments = []  # (start_pt, end_pt, length)
    total_len = 0.0

    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < 2:
            continue

        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        for i, d in enumerate(dists):
            if d > 0:
                segments.append((pts[i], pts[i + 1], d))
                total_len += d

    if total_len == 0:
        return np.zeros((n_points, 2))

    # 2. Walk the path
    step = total_len / float(n_points if n_points > 0 else 1)
    new_points = []

    current_dist = 0.0
    target_dist = 0.0

    for p_start, p_end, seg_len in segments:
        while target_dist <= current_dist + seg_len:
            # Interpolate
            ratio = (target_dist - current_dist) / seg_len
            pt = p_start + ratio * (p_end - p_start)
            new_points.append(pt)
            target_dist += step

            if len(new_points) >= n_points:
                break
        current_dist += seg_len
        if len(new_points) >= n_points:
            break

    # Pad if necessary
    while len(new_points) < n_points:
        new_points.append(new_points[-1] if new_points else [0, 0])

    return np.array(new_points)

# === file: preprocessing.py ===
import os
import cv2
import numpy as np
from config import *
from utils import sample_path_by_arc_length


def skeletonize_image(img):
    """
    Morphological thinning to get 1-pixel wide skeleton.
    Crucial for turning 'blobs' of ink into 'lines' of drones.
    """
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    skeleton = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    done = False
    while not done:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            done = True

    return skeleton


def _transform_to_world(pts, dim):
    world_pts = np.zeros((len(pts), dim))
    # Map Image pixels to World coords
    # Image (0,0) is Top-Left. World (0,0) is Bottom-Left.
    world_pts[:, 0] = (pts[:, 0] / IMG_W) * WORLD_W
    world_pts[:, 1] = ((IMG_H - pts[:, 1]) / IMG_H) * WORLD_H

    if dim == 3:
        world_pts[:, 2] = DEPTH / 2.0

    return world_pts


def extract_points_from_image(image_path, n_drones, dim=2):
    """
    Adaptive thresholding -> Skeletonization -> Contour Sampling.
    """
    if not image_path or not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Using fallback text.")
        return generate_text_points("Signature", n_drones, dim)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return generate_text_points("ERROR", n_drones, dim)

    img = cv2.resize(img, (IMG_W, IMG_H))

    # Adaptive Thresholding for robust handling of paper/lighting
    # Invert so ink is white (255)
    binary = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )

    # Skeletonize to get centerlines
    skeleton = skeletonize_image(binary)

    # Find contours on the skeleton to order the points
    contours, _ = cv2.findContours(
        skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # Sample equidistant points
    pts = sample_path_by_arc_length(contours, n_drones)

    return _transform_to_world(pts, dim)


def generate_text_points(text, n_drones, dim=2):
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    # Complex script simulates handwriting
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    scale = 3.0
    thickness = 2

    size = cv2.getTextSize(text, font, scale, thickness)[0]
    tx = (IMG_W - size[0]) // 2
    ty = (IMG_H + size[1]) // 2

    cv2.putText(mask, text, (tx, ty), font, scale, 255, thickness)

    # Skeletonize text too for consistency
    skeleton = skeletonize_image(mask)
    contours, _ = cv2.findContours(
        skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    pts = sample_path_by_arc_length(contours, n_drones)

    return _transform_to_world(pts, dim)

# === file: flow.py ===
import cv2
import numpy as np
from config import *


class OpticalFlowExtractor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.valid = self.cap.isOpened()
        self.prev_gray = None
        self.scale_x = WORLD_W / IMG_W
        self.scale_y = WORLD_H / IMG_H
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_frames / VIDEO_FPS

        if self.valid:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (IMG_W, IMG_H))
                self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.valid = False

    def get_first_frame_mask_points(self, n_drones, dim):
        if not self.valid:
            return np.zeros((n_drones, dim))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            return np.zeros((n_drones, dim))

        frame = cv2.resize(frame, (IMG_W, IMG_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        # Canny Edges for video objects
        edges = cv2.Canny(gray, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        from utils import sample_path_by_arc_length
        pts = sample_path_by_arc_length(contours, n_drones)

        from preprocessing import _transform_to_world
        return _transform_to_world(pts, dim)

    def read_next_flow(self):
        """
        Calculates flow with DILATION to prevent 'Loss of Lock'.
        """
        if not self.valid:
            return np.zeros((IMG_H, IMG_W, 2))
        ret, frame = self.cap.read()
        if not ret:
            return np.zeros((IMG_H, IMG_W, 2))

        frame = cv2.resize(frame, (IMG_W, IMG_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros((IMG_H, IMG_W, 2))
        
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, # type: ignore
            pyr_scale=0.5,
            levels=5,
            winsize=25,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        ) 
        self.prev_gray = gray

         # --- ROBUSTNESS ENHANCEMENT ---
        # Blur the flow field to spread velocity into the 'wake' of the object.
        # This helps drones that fall slightly behind to catch the current.
        k_size = 31
        flow_blur = cv2.GaussianBlur(flow, (k_size, k_size), 10)
        
        # Combine sharp flow (center) with blurred flow (edges)
        mag = np.linalg.norm(flow, axis=2)
        mask_strong = mag > 0.5
        
        flow_final = flow_blur.copy()
        flow_final[mask_strong] = flow[mask_strong]
        # -----------------------------
        
        # Correctly scale using VIDEO_FPS
        flow_world = np.zeros_like(flow_final)
        flow_world[:,:,0] = flow_final[:,:,0] * self.scale_x * VIDEO_FPS
        flow_world[:,:,1] = -flow_final[:,:,1] * self.scale_y * VIDEO_FPS

        return flow_world

    def sample_flow_at(self, flow_field, positions):
        px = (positions[:, 0] / WORLD_W) * IMG_W
        py = ((WORLD_H - positions[:, 1]) / WORLD_H) * IMG_H

        px = np.clip(px, 0, IMG_W - 1.01)
        py = np.clip(py, 0, IMG_H - 1.01)

        x0 = np.floor(px).astype(int)
        x1 = x0 + 1
        y0 = np.floor(py).astype(int)
        y1 = y0 + 1

        wx = (px - x0)[:, np.newaxis]
        wy = (py - y0)[:, np.newaxis]

        f00 = flow_field[y0, x0]
        f10 = flow_field[y0, x1]
        f01 = flow_field[y1, x0]
        f11 = flow_field[y1, x1]

        v_interp = (1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f10 + (1 - wx) * wy * f01 + wx * wy * f11

        if positions.shape[1] == 3:
            return np.column_stack([v_interp, np.zeros(len(v_interp))])
        return v_interp

# === file: dynamics.py ===
import numpy as np
from config import *


def velocity_saturation(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    mask = norm > 1e-6
    scale = np.ones_like(norm)
    scale[mask] = np.minimum(1.0, V_MAX / norm[mask])
    return v * scale


def compute_repulsion(x):
    N, D = x.shape
    r_vec = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(r_vec**2, axis=2)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)

    mask = dist < R_SAFE
    if not np.any(mask):
        return np.zeros_like(x)

    with np.errstate(divide='ignore'):
        # Inverse cube law for strong close-range repulsion to maintain separation
        factor = K_REP / (dist**3 + 1e-6)
    factor[~mask] = 0.0

    force = np.sum(r_vec * factor[:, :, np.newaxis], axis=1)
    F_MAX = 80.0
    force = np.clip(force, -F_MAX, F_MAX)
    return force


def dynamics_target(t, state, n, dim, targets):
    x = state[:n * dim].reshape(n, dim)
    v = state[n * dim:].reshape(n, dim)

    dx = velocity_saturation(v)

    # World boundary clamp
    x_next = x + dx * DT
    x_next[:, 0] = np.clip(x_next[:, 0], 0, WORLD_W)
    x_next[:, 1] = np.clip(x_next[:, 1], 0, WORLD_H)

    dx = (x_next - x) / DT

    f_att = K_P * (targets - x)
    f_rep = compute_repulsion(x)
    f_damp = -K_D * v

    dv = (f_att + f_rep + f_damp) / MASS
    return np.concatenate([dx.flatten(), dv.flatten()])


def dynamics_flow(t, state, n, dim, flow_field, flow_obj, ref_rel):
    # 1. unpack state
    x = state[:n * dim].reshape(n, dim)
    v = state[n * dim:].reshape(n, dim)

    # 2. sample optical flow at drone positions
    v_flow = flow_obj.sample_flow_at(flow_field, x)

    # 3. compute centroid and relative coordinates
    c = np.mean(x, axis=0)
    r = x - c

    # 4. estimate rigid motion from flow
    # (translation + rotation)
    Jr = np.column_stack([-r[:,1], r[:,0]])   # 2D rotation operator
    omega = np.sum(Jr * v_flow) / (np.sum(Jr * Jr) + 1e-8)
    u = np.mean(v_flow - omega * Jr, axis=0)

    # 5. rigid target velocity field
    v_rigid = u + omega * Jr
    v_rigid = velocity_saturation(v_rigid)
    time_scale = VIDEO_DT / DT
    v_rigid = v_rigid * time_scale

    # 6. forces
    f_track = K_V * (v_rigid - v)
    f_rep   = compute_repulsion(x)
    f_damp  = -K_D * v

    # 7. acceleration
    dv = (f_track + f_rep + f_damp) / MASS

    # 8. Rigid kinematics
    dx = v_rigid

    return np.concatenate([dx.flatten(), dv.flatten()])

# === file: solver.py ===
import numpy as np

def rk4_step(func, t, y, dt):
    """
    Fourth-order Runge-Kutta (RK4) integrator.

    Chosen for:
    - O(dt^4) local truncation error
    - Good stability for smooth ODEs
    - Balance between accuracy and computational cost

    With DT = 0.02, RK4 remains stable for the chosen gains
    (K_P, K_V, K_REP) without requiring implicit methods.
    """
    k1 = func(t, y)
    k2 = func(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = func(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = func(t + dt, y + dt*k3)
    
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# === file: visualizer.py ===
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from config import *


def save_video(trajectory, out_path, n_drones, dim):
    steps = trajectory.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore[attr-defined]
    out = cv2.VideoWriter(out_path, fourcc, VIS_FPS, (IMG_W, IMG_H))

    print(f"Exporting animation ({steps} frames)...")

    bg = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    for i in range(steps):
        frame = bg.copy()
        pts_world = trajectory[i]

        xs = (pts_world[:, 0] / WORLD_W) * IMG_W
        ys = IMG_H - (pts_world[:, 1] / WORLD_H) * IMG_H

        for j in range(n_drones):
            px, py = int(xs[j]), int(ys[j])
            if 0 <= px < IMG_W and 0 <= py < IMG_H:
                # Precision glow
                cv2.circle(frame, (px, py), GLOW_RADIUS + 1, COLOR_GLOW, -1)
                cv2.circle(frame, (px, py), 1, COLOR_CORE, -1)

        out.write(frame)
    out.release()
    print("Done.")


def plot_static(trajectory, out_dir, name, dim):
    fig = plt.figure(figsize=(12, 10))
    final = trajectory[-1]

    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(final[:, 0], final[:, 1], final[:, 2], s=2, c='black')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(final[:, 0], final[:, 1], s=5, c='black', marker='o')
        ax.set_aspect('equal')

    plt.title(f"{name} (Final Frame)")
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
    plt.close()

# === file: checkpoints.py ===
import numpy as np
import os
from config import *

from visualizer import plot_static, save_video

def save_checkpoint(path, state, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.savez(
        path,
        state=state,
        n_drones=args.n_drones,
        dim=args.dim,
        seed=args.seed,
        DT=DT,
        K_P=K_P,
        K_V=K_V,
        K_D=K_D,
        K_REP=K_REP,
        R_SAFE=R_SAFE,
    )

def load_checkpoint(path, args):
    data = np.load(path)
    print("Loaded parameters:")
    print(f"DT={data['DT']}, K_P={data['K_P']}, K_REP={data['K_REP']}")
    if data["n_drones"] != args.n_drones or data["dim"] != args.dim:
        raise ValueError(
            f"Checkpoint mismatch: expected "
            f"{args.n_drones} drones / dim {args.dim}, "
            f"got {data['n_drones']} / {data['dim']}"
        )
    return data["state"]

def export_stage(args, full_trajectory, video_skip, stage_name, file_name, start, end):
    arr = np.array(full_trajectory[start:end])

    if len(arr) < 2:
        return

    save_video(
        arr[::video_skip],
        os.path.join(args.out_dir, f"drone_show_{file_name}_{stage_name}.mp4"),
        args.n_drones,
        args.dim,
    )

    plot_static(
        arr[-1:],
        args.out_dir,
        f"final_positions_{file_name}_{stage_name}",
        args.dim,
    )

    print(f">>> Exported {stage_name} video and final image")


# === file: main.py ===
import argparse
import numpy as np
import os
from checkpoint import export_stage, load_checkpoint, save_checkpoint
from config import *
from utils import ensure_dir, assign_targets_hungarian
from preprocessing import extract_points_from_image, generate_text_points
from flow import OpticalFlowExtractor
from dynamics import dynamics_target, dynamics_flow
from solver import rk4_step

def run_project(args):
    ensure_dir(args.out_dir)
    np.random.seed(args.seed)
    ckpt_dir = "checkpoints"
    ckpt_p1 = os.path.join(ckpt_dir, "p1.npz")
    ckpt_p2 = os.path.join(ckpt_dir, "p2.npz")
    ckpt_p3 = os.path.join(ckpt_dir, "p3.npz")

    # ---------------------------------------------------
    # INITIAL STATE (fresh or resumed)
    # ---------------------------------------------------
    state = None

    if args.mode == "p3" and os.path.exists(ckpt_p2):
        print(">>> Loading state from P2 checkpoint")
        state = load_checkpoint(ckpt_p2, args)

    elif args.mode == "p2" and os.path.exists(ckpt_p1):
        print(">>> Loading state from P1 checkpoint")
        state = load_checkpoint(ckpt_p1, args)

    if state is None:
        print(">>> Initializing fresh state")
        side = int(np.ceil(np.sqrt(args.n_drones)))
        x_lin = np.linspace(WORLD_W * 0.4, WORLD_W * 0.6, side)
        y_lin = np.linspace(0, WORLD_H * 0.1, side)
        xv, yv = np.meshgrid(x_lin, y_lin)

        pos = np.zeros((args.n_drones, args.dim))
        limit = min(args.n_drones, len(xv.flatten()))
        pos[:limit, 0] = xv.flatten()[:limit]
        pos[:limit, 1] = yv.flatten()[:limit]

        if args.dim == 3:
            pos[:, 2] = DEPTH / 2.0

        vel = np.zeros_like(pos)
        state = np.concatenate([pos.flatten(), vel.flatten()])

    full_trajectory = []
    stage_marks = {}  # name -> end index
    
    def record(s):
        full_trajectory.append(s[: args.n_drones * args.dim].reshape(args.n_drones, args.dim))

    record(state)

    # Downsample video export if DT is very small
    video_skip = max(1, int(1.0 / (DT * VIS_FPS)))

    # 1. Handwritten
    if args.mode in ['p1', 'all']:
        print(">>> P1: Handwritten Formation (Precision)")
        target_pts = extract_points_from_image(args.image_path, args.n_drones, args.dim)
        curr_pos = state[: args.n_drones * args.dim].reshape(args.n_drones, args.dim)
        assigned = assign_targets_hungarian(curr_pos, target_pts)

        steps = int(8.0 / DT)
        for _ in range(steps):
            f = lambda t, y: dynamics_target(t, y, args.n_drones, args.dim, assigned)
            state = rk4_step(f, 0, state, DT)
            record(state)

        stage_marks["p1"] = len(full_trajectory)
        save_checkpoint(ckpt_p1, state, args)
        export_stage(
            args, full_trajectory, video_skip, "p1", os.path.splitext(os.path.basename(args.image_path))[0], start=0, end=stage_marks["p1"]
        )

    # 2. Greeting
    if args.mode in ['p2', 'all']:
        print(">>> P2: Greeting")
        target_pts = generate_text_points(args.greeting_text, args.n_drones, args.dim)
        curr_pos = state[: args.n_drones * args.dim].reshape(args.n_drones, args.dim)
        assigned = assign_targets_hungarian(curr_pos, target_pts)

        steps = int(6.0 / DT)
        for _ in range(steps):
            f = lambda t, y: dynamics_target(t, y, args.n_drones, args.dim, assigned)
            state = rk4_step(f, 0, state, DT)
            record(state)
        
        stage_marks["p2"] = len(full_trajectory)
        save_checkpoint(ckpt_p2, state, args)
        export_stage(
            args, full_trajectory, video_skip, "p2", f"{args.greeting_text}_{args.n_drones}", start=stage_marks.get("p1", 0), end=stage_marks["p2"]
        )

    # 3. Video
    if args.mode in ['p3', 'all']:
        print(">>> P3: Video Tracking")
        flow_proc = OpticalFlowExtractor(args.video_path)

        # Morph to shape
        shape_pts = flow_proc.get_first_frame_mask_points(args.n_drones, args.dim)
        curr_pos = state[: args.n_drones * args.dim].reshape(args.n_drones, args.dim)
        assigned = assign_targets_hungarian(curr_pos, shape_pts)

        # --- SHAPE PRESERVATION REFERENCE ---
        ref_centroid = np.mean(assigned, axis=0)
        ref_rel = assigned - ref_centroid
        # ----------------------------------

        for _ in range(int(3.0 / DT)):
            f = lambda t, y: dynamics_target(t, y, args.n_drones, args.dim, assigned)
            state = rk4_step(f, 0, state, DT)
            record(state)

        # Follow flow
        video_duration = flow_proc.video_duration
        MAX_SIM_TIME = 15.0  # safety cap

        sim_time = min(video_duration, MAX_SIM_TIME)
        max_steps = int(sim_time / DT)
        
        # Time Accumulator for Video Sync
        video_timer = 0.0
        current_field = np.zeros((IMG_H, IMG_W, 2))
        for _ in range(max_steps):
            video_timer += DT

            # Step video only when enough physics time has passed
            if video_timer >= VIDEO_DT:
                current_field = flow_proc.read_next_flow()
                video_timer -= VIDEO_DT
                if not flow_proc.valid and np.sum(np.abs(current_field)) == 0:
                    break

            f = lambda t, y: dynamics_flow(t, y, args.n_drones, args.dim, current_field, flow_proc, ref_rel)
            state = rk4_step(f, 0, state, DT)
            record(state)
        
        stage_marks["p3"] = len(full_trajectory)
        save_checkpoint(ckpt_p3, state, args)
        export_stage(
            args, full_trajectory, video_skip, "p3", os.path.splitext(os.path.basename(args.video_path))[0], start=stage_marks.get("p2", 0), end=stage_marks["p3"]
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='p3', choices=['p1', 'p2', 'p3', 'all'])
    parser.add_argument('--image_path', default='input/my_name_marker.jpg')
    parser.add_argument('--video_path', default='input/two_balls.mp4')
    parser.add_argument(
        '--n_drones',
        type=int,
        default=300,
        help="Higher count (300+) recommended for signatures",
    )
    parser.add_argument('--greeting_text', type=str, default="Happy New Year!")
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3])
    parser.add_argument('--out_dir', default='visualizations_2d')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.mode in ['p3', 'all'] and args.dim == 3:
        raise ValueError("P3 (video tracking) supports only dim=2")
    run_project(args)

main()
