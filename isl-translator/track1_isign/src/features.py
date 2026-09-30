"""Turn iSign .pose bytes into the paper's raw per-frame feature matrix.

We call the paper's own `load_pose_file` so the numbers are identical to what
their training code saw: POSE(33) + LEFT_HAND(21) + RIGHT_HAND(21) landmarks,
each (x, y, z, confidence) -> (frames, 300) float32.

Everything after loading (z-score, motion, padding to 500 frames) is applied at
training time by `preprocess_array`, so one cached feature file serves both the
spatial (300-d) and motion (600-d) models.
"""

import os
import tempfile

import numpy as np

from paper_pose_utils import (
    DEFAULT_COMPONENTS,
    add_motion_features,
    load_pose_file,
    normalize_pose,
    pad_or_truncate_pose,
)


def pose_bytes_to_features(pose_bytes):
    """Exactly `load_pose_file(path, DEFAULT_COMPONENTS, include_confidence=True)`."""
    fd, path = tempfile.mkstemp(suffix=".pose")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(pose_bytes)
        return load_pose_file(path, DEFAULT_COMPONENTS, True)
    finally:
        os.remove(path)


def preprocess_array(pose, max_length=500, normalization="zscore", pose_dim=300,
                     use_motion=False):
    """Same steps, in the same order, as the paper's `preprocess_pose` after the
    file has been loaded (see paper_pose_utils.preprocess_pose)."""
    base_pose_dim = pose_dim // 2 if use_motion else pose_dim
    if pose.size == 0:
        return np.zeros((max_length, pose_dim), dtype=np.float32), 0
    if pose.shape[1] < base_pose_dim:
        pose = np.pad(pose, ((0, 0), (0, base_pose_dim - pose.shape[1])))
    elif pose.shape[1] > base_pose_dim:
        pose = pose[:, :base_pose_dim]
    pose = normalize_pose(pose, normalization)
    if use_motion:
        pose = add_motion_features(pose)
    return pad_or_truncate_pose(pose, max_length)
