"""Static handshape features for ISL fingerspelling.

A word sign is a movement; a fingerspelled letter is a SHAPE held still. So these
features describe shape and placement in one frame, with no velocity at all:

  * every finger point relative to its own wrist, divided by that hand's size
    -> the same letter looks identical near or far from the camera;
  * where each wrist sits relative to the shoulders, divided by shoulder width
    -> "at the chin" and "at the chest" stay distinguishable;
  * hand size relative to the body, and the vector between the two wrists
    -> ISL fingerspelling is two-handed, so the relation between hands matters;
  * a present/absent flag per hand, because a missing hand is information too.

Input is the same 300-number row the live app already produces (record.py).
"""

import numpy as np

N_HAND = 21
WRIST = 0
FINGERTIPS = (4, 8, 12, 16, 20)

# 21*2 shape + 2 wrist placement + 1 size + 1 present, per hand, plus the
# wrist-to-wrist vector shared between them.
FEATURE_DIM = 2 * (N_HAND * 2 + 2 + 1 + 1) + 2


def _body_frame(pose):
    """Origin between the shoulders, unit = shoulder width."""
    left_shoulder, right_shoulder = pose[11, :2], pose[12, :2]
    center = (left_shoulder + right_shoulder) / 2.0
    width = float(np.linalg.norm(left_shoulder - right_shoulder))
    if not np.isfinite(center).all():
        center = np.array([0.5, 0.5], dtype=np.float32)
    if not np.isfinite(width) or width < 1e-3:
        width = 0.25
    return center.astype(np.float32), width


def _hand_block(hand, center, width):
    """(shape 42, wrist placement 2, size 1, present 1) for one hand."""
    if not np.isfinite(hand[:, 0]).any():
        return np.zeros(N_HAND * 2, np.float32), np.zeros(2, np.float32), 0.0, 0.0

    xy = np.nan_to_num(hand[:, :2], nan=0.0).astype(np.float32)
    wrist = xy[WRIST]
    relative = xy - wrist
    size = float(np.max(np.linalg.norm(relative, axis=1)))
    shape = relative / (size if size > 1e-4 else 1.0)
    placement = (wrist - center) / width
    return shape.reshape(-1), placement.astype(np.float32), size / width, 1.0


def row_to_features(row):
    """(300,) keypoint row -> (FEATURE_DIM,) float32."""
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    pose = row[:33 * 4].reshape(33, 4)
    left = row[33 * 4:33 * 4 + N_HAND * 4].reshape(N_HAND, 4)
    right = row[33 * 4 + N_HAND * 4:].reshape(N_HAND, 4)

    center, width = _body_frame(pose)
    l_shape, l_place, l_size, l_present = _hand_block(left, center, width)
    r_shape, r_place, r_size, r_present = _hand_block(right, center, width)

    if l_present and r_present:
        between = (r_place - l_place).astype(np.float32)
    else:
        between = np.zeros(2, np.float32)

    return np.concatenate([
        l_shape, l_place, [l_size, l_present],
        r_shape, r_place, [r_size, r_present],
        between,
    ]).astype(np.float32)


def has_hand(row):
    """A frame with no hand at all carries no letter; training must skip it."""
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    hands = row[33 * 4:]
    xs = hands.reshape(-1, 4)[:, 0]
    return bool(np.isfinite(xs).any())


def augment(row_features, rng):
    """Small camera and signer variation, applied in feature space.

    Rotation and scale act on the shape blocks and the placement vectors together,
    so a tilted camera or a taller signer still produces the same letter.
    """
    f = row_features.copy()
    angle = rng.uniform(-0.21, 0.21)                     # about +-12 degrees
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)

    def spin(block):
        return (block.reshape(-1, 2) @ rot.T).reshape(-1)

    for block in XY_BLOCKS:
        f[block] = spin(f[block])

    scale = rng.uniform(0.85, 1.15)
    for block in (LEFT_PLACE, RIGHT_PLACE, BETWEEN):
        f[block] *= scale

    if rng.random() < 0.3:                               # a left-handed signer
        f = mirror(f)

    f += rng.normal(0, 0.012, size=f.shape).astype(np.float32)
    return f.astype(np.float32)


# Where each block sits in the feature vector, so nothing has to count it out.
BLOCK = N_HAND * 2 + 4                      # shape 42 + place 2 + size 1 + present 1
LEFT_SHAPE, RIGHT_SHAPE = slice(0, N_HAND * 2), slice(BLOCK, BLOCK + N_HAND * 2)
LEFT_PLACE = slice(N_HAND * 2, N_HAND * 2 + 2)
RIGHT_PLACE = slice(BLOCK + N_HAND * 2, BLOCK + N_HAND * 2 + 2)
BETWEEN = slice(FEATURE_DIM - 2, FEATURE_DIM)
XY_BLOCKS = (LEFT_SHAPE, LEFT_PLACE, RIGHT_SHAPE, RIGHT_PLACE, BETWEEN)


def mirror(f):
    """Swap the hands and flip x: the same letter signed the other way round.

    Only the x of an actual coordinate is flipped. Flipping every other number in
    the vector, as this did before, also negated each hand's size - so half of the
    augmented training frames claimed a hand of negative size.
    """
    out = f.copy()
    left = out[:BLOCK].copy()
    out[:BLOCK] = out[BLOCK:2 * BLOCK]
    out[BLOCK:2 * BLOCK] = left
    for block in XY_BLOCKS:
        out[block][0::2] *= -1.0
    return out
