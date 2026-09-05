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
