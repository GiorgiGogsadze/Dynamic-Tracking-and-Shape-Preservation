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
