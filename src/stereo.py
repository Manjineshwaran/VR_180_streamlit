import cv2
import numpy as np
import os

def generate_stereo_from_depth_frame(frame, depth, max_shift=30):
    H, W = frame.shape[:2]
    if depth is None:
        depth = np.ones((H, W), dtype=np.float32)
    else:
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth = depth.astype(np.float32) / 255.0

    shift = (1 - depth) * max_shift
    left_eye = np.zeros_like(frame)
    right_eye = np.zeros_like(frame)

    # faster raster approach
    xs = np.arange(W)
    for y in range(H):
        dx_row = shift[y].astype(np.int32)
        for x in range(W):
            dx = dx_row[x]
            nl = min(W-1, x + dx)
            nr = max(0, x - dx)
            left_eye[y, nl] = frame[y, x]
            right_eye[y, nr] = frame[y, x]

    mask_left = (left_eye.sum(axis=2) == 0).astype(np.uint8)
    left_eye = cv2.inpaint(left_eye, mask_left, 3, cv2.INPAINT_TELEA)
    mask_right = (right_eye.sum(axis=2) == 0).astype(np.uint8)
    right_eye = cv2.inpaint(right_eye, mask_right, 3, cv2.INPAINT_TELEA)
    return left_eye, right_eye

def batch_generate_stereo(frame_paths, depth_paths, left_out_dir, right_out_dir, max_shift=20):
    os.makedirs(left_out_dir, exist_ok=True)
    os.makedirs(right_out_dir, exist_ok=True)
    for fpath in frame_paths:
        fname = os.path.basename(fpath)
        frame = cv2.imread(fpath)
        dpath = depth_paths.get(fname)
        depth = cv2.imread(dpath, cv2.IMREAD_GRAYSCALE) if dpath and os.path.exists(dpath) else None
        l, r = generate_stereo_from_depth_frame(frame, depth, max_shift=max_shift)
        cv2.imwrite(os.path.join(left_out_dir, fname), l)
        cv2.imwrite(os.path.join(right_out_dir, fname), r)
