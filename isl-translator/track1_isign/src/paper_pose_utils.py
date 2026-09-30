# VERBATIM copy of inference/pose_utils.py from huggingface.co/manavdhamecha77/iSign-t5-pose-to-text
# (paper arXiv 2609.12993). Do not edit: this is the exact preprocessing the paper used.
"""Utilities for reading and preprocessing iSign binary .pose files."""

import logging
import json
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_COMPONENTS = (
    "POSE_LANDMARKS",
    "LEFT_HAND_LANDMARKS",
    "RIGHT_HAND_LANDMARKS",
)


def load_pose_file(
    pose_path: str,
    components: Optional[Sequence[str]] = DEFAULT_COMPONENTS,
    include_confidence: bool = True,
) -> np.ndarray:

    try:
        with open(pose_path, "rb") as f:
            raw = f.read()

        if raw.lstrip().startswith((b"{", b"[")):
            payload = json.loads(raw.decode("utf-8"))
            frames = (
                payload.get("data", [])
                if isinstance(payload, dict)
                else payload
            )

            rows = [
                np.asarray(
                    frame.get("keypoints", []),
                    dtype=np.float32,
                ).reshape(-1)
                for frame in frames
            ]

            return (
                np.asarray(rows, dtype=np.float32)
                if rows
                else np.array([])
            )

    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        FileNotFoundError,
        OSError,
    ) as exc:
        logger.error(
            "Failed to read pose file %s: %s",
            pose_path,
            exc,
        )
        return np.array([])

    try:
        from pose_format import Pose

        pose = Pose.read(raw)

    except (
        FileNotFoundError,
        ImportError,
        ValueError,
        OSError,
    ) as exc:
        logger.error(
            "Failed to load pose file %s: %s",
            pose_path,
            exc,
        )
        return np.array([])

    data = np.asarray(
        pose.body.data,
        dtype=np.float32,
    )

    confidence = np.asarray(
        pose.body.confidence,
        dtype=np.float32,
    )

    if data.ndim != 4 or data.shape[1] == 0:
        logger.warning(
            "Unexpected pose shape in %s: %s",
            pose_path,
            data.shape,
        )
        return np.array([])

    data = data[:, 0]

    confidence = (
        confidence[:, 0]
        if confidence.ndim == 3
        else None
    )

    wanted = (
        set(components)
        if components is not None
        else None
    )

    selected = []

    offset = 0

    for component in pose.header.components:

        count = len(component.points)

        if (
            wanted is None
            or component.name in wanted
        ):

            component_data = data[
                :,
                offset:offset + count,
                :,
            ]

            if (
                include_confidence
                and confidence is not None
            ):

                component_data = np.concatenate(
                    [
                        component_data,
                        confidence[
                            :,
                            offset:offset + count,
                            None,
                        ],
                    ],
                    axis=2,
                )

            selected.append(component_data)

        offset += count

    if not selected:

        logger.warning(
            "No requested components found in %s",
            pose_path,
        )

        return np.array([])

    return np.concatenate(
        selected,
        axis=1,
    ).reshape(
        data.shape[0],
        -1,
    )


def normalize_pose(
    pose: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:

    if pose.size == 0:
        return pose

    if method == "zscore":

        mean = np.nanmean(
            pose,
            axis=0,
            keepdims=True,
        )

        std = np.nanstd(
            pose,
            axis=0,
            keepdims=True,
        )

        return np.nan_to_num(
            (pose - mean)
            / np.where(
                std == 0,
                1.0,
                std,
            )
        )

    if method == "minmax":

        min_val = np.nanmin(
            pose,
            axis=0,
            keepdims=True,
        )

        max_val = np.nanmax(
            pose,
            axis=0,
            keepdims=True,
        )

        return np.nan_to_num(
            (pose - min_val)
            / np.where(
                max_val == min_val,
                1.0,
                max_val - min_val,
            )
        )

    raise ValueError(
        f"Unknown normalization method: {method}"
    )


def pad_or_truncate_pose(
    pose: np.ndarray,
    max_length: int,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, int]:

    if pose.size == 0:

        width = (
            pose.shape[1]
            if pose.ndim > 1
            else 0
        )

        return (
            np.full(
                (max_length, width),
                pad_value,
                dtype=np.float32,
            ),
            0,
        )

    length = min(
        pose.shape[0],
        max_length,
    )

    padded = np.full(
        (max_length, pose.shape[1]),
        pad_value,
        dtype=pose.dtype,
    )

    padded[:length] = pose[:length]

    return padded, length


def add_motion_features(
    pose: np.ndarray,
) -> np.ndarray:
    """
    Concatenate frame-wise velocity with pose features.

    velocity[t] = pose[t] - pose[t - 1]

    First frame velocity is zero.
    """

    if pose.size == 0:
        return pose

    velocity = np.zeros_like(
        pose,
        dtype=np.float32,
    )

    velocity[1:] = (
        pose[1:] - pose[:-1]
    )

    return np.concatenate(
        [
            pose,
            velocity,
        ],
        axis=1,
    )


def preprocess_pose(
    pose_path: str,
    max_length: int,
    normalization: str = "zscore",
    components: Optional[
        Sequence[str]
    ] = DEFAULT_COMPONENTS,
    include_confidence: bool = True,
    pose_dim: int = 600,
    use_motion: bool = True,
) -> Tuple[np.ndarray, int]:

    pose = load_pose_file(
        pose_path,
        components,
        include_confidence,
    )

    base_pose_dim = (
        pose_dim // 2
        if use_motion
        else pose_dim
    )

    if pose.size == 0:

        return (
            np.zeros(
                (
                    max_length,
                    pose_dim,
                ),
                dtype=np.float32,
            ),
            0,
        )

    if pose.shape[1] < base_pose_dim:

        pose = np.pad(
            pose,
            (
                (0, 0),
                (
                    0,
                    base_pose_dim
                    - pose.shape[1],
                ),
            ),
        )

    elif pose.shape[1] > base_pose_dim:

        pose = pose[
            :,
            :base_pose_dim,
        ]

    pose = normalize_pose(
        pose,
        normalization,
    )

    if use_motion:

        pose = add_motion_features(
            pose,
        )

    return pad_or_truncate_pose(
        pose,
        max_length,
    )


def create_attention_mask(
    actual_lengths: List[int],
    max_length: int,
) -> np.ndarray:

    mask = np.zeros(
        (
            len(actual_lengths),
            max_length,
        ),
        dtype=np.int64,
    )

    for index, length in enumerate(
        actual_lengths
    ):

        mask[
            index,
            :length,
        ] = 1

    return mask