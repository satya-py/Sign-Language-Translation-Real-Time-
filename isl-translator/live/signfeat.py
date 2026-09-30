"""Keypoints -> model features. Used by BOTH training and the live app.

Fixes the flaw that made the INCLUDE pretrained model fail on a webcam: instead of
raw frame coordinates, every frame is expressed relative to the signer's own body
(origin between the shoulders, scale = shoulder width). Sitting close to a laptop
camera then looks the same as standing far from a studio camera.

Raw row layout (from record.py / extract_keypoints.py), 300 numbers per frame:
    pose 33 x (x, y, z, conf) | left hand 21 x (...) | right hand 21 x (...)
"""

import numpy as np

N_POSE, N_HAND = 33, 21
POSE_END, LEFT_END = N_POSE * 4, N_POSE * 4 + N_HAND * 4

# Face/arm points that matter for signing; the legs are useless here.
POSE_KEEP = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
L_SHOULDER, R_SHOULDER = 11, 12

SEQ_LEN = 32                      # every clip is resampled to this many frames
N_POINTS = len(POSE_KEEP) + 2 * N_HAND
FEATURE_DIM = N_POINTS * 2 * 2 + 3  # xy + velocity, plus 3 "part visible" flags


def split_parts(rows):
    """(frames, 300) -> pose, left, right as (frames, n, 4)."""
    rows = np.asarray(rows, dtype=np.float32)
    pose = rows[:, :POSE_END].reshape(len(rows), N_POSE, 4)
    left = rows[:, POSE_END:LEFT_END].reshape(len(rows), N_HAND, 4)
    right = rows[:, LEFT_END:].reshape(len(rows), N_HAND, 4)
    return pose, left, right


def body_frame(pose):
    """Origin between the shoulders, scale = shoulder distance (per frame)."""
    ls, rs = pose[:, L_SHOULDER, :2], pose[:, R_SHOULDER, :2]
    center = (ls + rs) / 2.0
    scale = np.linalg.norm(ls - rs, axis=1)
    # Frames without shoulders: fall back to the median of the valid ones.
    ok = np.isfinite(scale) & (scale > 1e-3)
    fallback = np.median(scale[ok]) if ok.any() else 0.25
    scale = np.where(ok, scale, fallback)
    center = np.where(np.isfinite(center), center, 0.5)
    return center, np.maximum(scale, 1e-3)


def normalize(rows):
    """(frames, 300) -> (frames, N_POINTS, 2) body-relative xy, plus visibility flags."""
    pose, left, right = split_parts(rows)
    center, scale = body_frame(pose)
    center = center[:, None, :]
    scale = scale[:, None, None]

    def rel(part):
        return (part[:, :, :2] - center) / scale

    points = np.concatenate([rel(pose[:, POSE_KEEP]), rel(left), rel(right)], axis=1)
    flags = np.stack([
        np.isfinite(pose[:, :, 0]).any(1),
        np.isfinite(left[:, :, 0]).any(1),
        np.isfinite(right[:, :, 0]).any(1),
    ], axis=1).astype(np.float32)
    return np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0), flags


def resample(points, flags, length=SEQ_LEN):
    """Stretch or squeeze a clip to a fixed number of frames (speed invariance)."""
    n = len(points)
    if n == length:
        return points, flags
    src = np.linspace(0, n - 1, num=length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None, None]
    out = points[lo] * (1 - w) + points[hi] * w
    fl = flags[lo] * (1 - w[:, :, 0]) + flags[hi] * w[:, :, 0]
    return out.astype(np.float32), fl.astype(np.float32)


def to_features(points, flags):
    """(T, P, 2) + (T, 3) -> (T, FEATURE_DIM) with velocity appended."""
    velocity = np.zeros_like(points)
    velocity[1:] = points[1:] - points[:-1]
    flat = points.reshape(len(points), -1)
    vel = velocity.reshape(len(points), -1)
    return np.concatenate([flat, vel, flags], axis=1).astype(np.float32)


def clip_to_features(rows, length=SEQ_LEN):
    """Full path: raw keypoint rows -> (length, FEATURE_DIM) model input."""
    points, flags = normalize(rows)
    points, flags = resample(points, flags, length)
    return to_features(points, flags)


def hand_speed(rows):
    """Per-frame wrist speed in body widths; used for segmentation and idle mining."""
    pose, left, right = split_parts(rows)
    center, scale = body_frame(pose)
    wrists = []
    for part, idx in ((pose, 15), (pose, 16)):
        wrists.append((part[:, idx, :2] - center) / scale[:, None])
    with np.errstate(invalid="ignore"):
        stacked = np.stack(wrists)
        w = np.where(np.isfinite(stacked).any(0), np.nanmean(stacked, axis=0), 0.0)
    w = np.nan_to_num(w, nan=0.0)
    speed = np.zeros(len(w), dtype=np.float32)
    speed[1:] = np.linalg.norm(w[1:] - w[:-1], axis=1)
    return speed


def hands_present(rows):
    _, left, right = split_parts(rows)
    return (np.isfinite(left[:, :, 0]).any(1) | np.isfinite(right[:, :, 0]).any(1))


def active_span(rows, start_speed=0.035, stop_speed=0.018, quiet_frames=7,
                min_frames=10, pad=3):
    """Find the moving part of a clip: the same cut the live app makes.

    Training on whole dataset clips while the live app classifies only the moving
    segment is a train/serve mismatch that costs a lot of accuracy, so both sides
    call this.
    """
    rows = np.asarray(rows, dtype=np.float32)
    speed = hand_speed(rows)
    hands = hands_present(rows)
    start = None
    best = None
    quiet = 0
    for i in range(len(rows)):
        moving = hands[i] and speed[i] > start_speed
        if start is None:
            if moving:
                start, quiet = i, 0
            continue
        quiet = quiet + 1 if (speed[i] < stop_speed or not hands[i]) else 0
        if quiet >= quiet_frames:
            span = (start, i - quiet + 1)
            if span[1] - span[0] >= min_frames and (best is None or
                                                    span[1] - span[0] > best[1] - best[0]):
                best = span
            start = None
    if start is not None:
        span = (start, len(rows))
        if span[1] - span[0] >= min_frames and (best is None or
                                                span[1] - span[0] > best[1] - best[0]):
            best = span
    if best is None:
        return 0, len(rows)
    lo = max(0, best[0] - pad)
    hi = min(len(rows), best[1] + pad)
    return lo, hi
