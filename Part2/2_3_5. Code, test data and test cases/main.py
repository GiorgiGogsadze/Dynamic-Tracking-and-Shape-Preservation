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
from visualizer import plot_errors

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

    errors = []
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

            curr_pos = state[: args.n_drones * args.dim].reshape(args.n_drones, args.dim)
            curr_pos_norm = curr_pos / np.array([WORLD_W, WORLD_H])
            target_pts_norm = target_pts / np.array([WORLD_W, WORLD_H])
            diffs = curr_pos_norm - target_pts_norm        # shape (N, dim)
            distances = np.linalg.norm(diffs, axis=1)
            e_t = np.mean(distances)
            errors.append(e_t)

        stage_marks["p1"] = len(full_trajectory)
        save_checkpoint(ckpt_p1, state, args)
        export_stage(
            args, full_trajectory, video_skip, "p1", os.path.splitext(os.path.basename(args.image_path))[0], start=0, end=stage_marks["p1"]
        )
        errors = np.array(errors)
        print("Final mean error:", errors[-1])
        plot_errors(errors)

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
    parser.add_argument('--mode', default='all', choices=['p1', 'p2', 'p3', 'all'])
    parser.add_argument('--image_path', default='input/my_name_marker.jpg')
    parser.add_argument('--video_path', default='input/my_rotation.mp4')
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