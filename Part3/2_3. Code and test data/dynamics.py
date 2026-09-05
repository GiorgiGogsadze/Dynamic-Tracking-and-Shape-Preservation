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
