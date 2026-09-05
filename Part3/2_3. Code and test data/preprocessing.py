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
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    thickness = 2

    # --- Measure text at scale = 1 ---
    base_size = cv2.getTextSize(text, font, 1.0, thickness)[0]

    # Desired fill ratio
    target_w = 0.95 * IMG_W
    target_h = 0.75 * IMG_H

    # Compute scale to fit both width & height
    scale_w = target_w / base_size[0]
    scale_h = target_h / base_size[1]
    scale = min(scale_w, scale_h)

    # Center text
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    tx = (IMG_W - size[0]) // 2
    ty = (IMG_H + size[1]) // 2

    cv2.putText(mask, text, (tx, ty), font, scale, 255, thickness)

    skeleton = skeletonize_image(mask)
    contours, _ = cv2.findContours(
        skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    pts = sample_path_by_arc_length(contours, n_drones)
    return _transform_to_world(pts, dim)

