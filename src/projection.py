import cv2
import numpy as np
import os
from numba import jit, prange

@jit(nopython=True, parallel=True)
def create_mapping_arrays(output_width, output_height, focal, w, h):
    x_coords = np.zeros((output_height, output_width), dtype=np.float32)
    y_coords = np.zeros((output_height, output_width), dtype=np.float32)
    valid_mask = np.zeros((output_height, output_width), dtype=np.bool_)
    for y_out in prange(output_height):
        for x_out in prange(output_width):
            longitude = (x_out / output_width - 0.5) * np.pi
            latitude = (y_out / output_height - 0.5) * (np.pi / 2)
            x_dir = np.cos(latitude) * np.sin(longitude)
            y_dir = np.sin(latitude)
            z_dir = np.cos(latitude) * np.cos(longitude)
            if z_dir > 0:
                x_flat = (x_dir * focal / z_dir) + (w / 2)
                y_flat = (-y_dir * focal / z_dir) + (h / 2)
                if 0 <= x_flat < w-1 and 0 <= y_flat < h-1:
                    x_coords[y_out, x_out] = x_flat
                    y_coords[y_out, x_out] = y_flat
                    valid_mask[y_out, x_out] = True
    return x_coords, y_coords, valid_mask

def flat_to_vr180_spherical_optimized(img, output_width=2048, field_of_view=140, x_coords=None, y_coords=None, valid_mask=None):
    h, w = img.shape[:2]
    output_height = output_width // 2
    if x_coords is None or y_coords is None or valid_mask is None:
        fov_rad = np.radians(field_of_view)
        focal = (w / 2) / np.tan(fov_rad / 2)
        x_coords, y_coords, valid_mask = create_mapping_arrays(output_width, output_height, focal, w, h)
    # remap requires mapping in float32
    result = cv2.remap(img, x_coords, y_coords, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    result[~valid_mask] = 0
    return result

def batch_project(folder_in, folder_out, output_width=2048, field_of_view=140):
    os.makedirs(folder_out, exist_ok=True)
    files = sorted([f for f in os.listdir(folder_in) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not files:
        return
    sample = cv2.imread(os.path.join(folder_in, files[0]))
    h, w = sample.shape[:2]
    fov_rad = np.radians(field_of_view)
    focal = (w / 2) / np.tan(fov_rad / 2)
    x_coords, y_coords, valid_mask = create_mapping_arrays(output_width, output_width//2, focal, w, h)
    for f in files:
        img = cv2.imread(os.path.join(folder_in, f))
        vr = flat_to_vr180_spherical_optimized(img, output_width, field_of_view, x_coords, y_coords, valid_mask)
        vr = cv2.flip(vr, 0)
        cv2.imwrite(os.path.join(folder_out, f), vr)
