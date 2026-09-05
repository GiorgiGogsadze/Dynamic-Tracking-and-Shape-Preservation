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
