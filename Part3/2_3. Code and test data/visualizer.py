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

def plot_errors(errors):
    plt.figure()
    plt.plot(errors)
    plt.xlabel("Time step")
    plt.ylabel("Mean position error e(t)")
    plt.title("Convergence of Mean Position Error")
    plt.grid(True)
    plt.show()