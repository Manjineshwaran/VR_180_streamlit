import cv2
import os
import numpy as np

def stack_lr(left_path, right_path, out_path, side_by_side=True):
    l = cv2.imread(left_path)
    r = cv2.imread(right_path)
    if l.shape != r.shape:
        raise ValueError("Left and right shapes differ")
    if side_by_side:
        out = np.hstack((l, r))
    else:
        out = np.vstack((l, r))
    cv2.imwrite(out_path, out)

def batch_stack(left_dir, right_dir, out_dir, side_by_side=True):
    os.makedirs(out_dir, exist_ok=True)
    left_files = sorted([f for f in os.listdir(left_dir) if f.endswith(".png")])
    right_files = sorted([f for f in os.listdir(right_dir) if f.endswith(".png")])
    for lf, rf in zip(left_files, right_files):
        stack_lr(os.path.join(left_dir, lf), os.path.join(right_dir, rf), os.path.join(out_dir, lf), side_by_side)
