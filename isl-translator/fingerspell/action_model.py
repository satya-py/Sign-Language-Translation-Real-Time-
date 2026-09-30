"""Runs the notebook's action.h5 fingerspelling model without TensorFlow.

"Action Detection Refined.ipynb" trains three dense layers on one frame of hand
landmarks:

    126 numbers  ->  Dense(128, relu)  ->  Dropout(0.3)  ->  Dense(64, relu)
                 ->  Dense(36, softmax)

Dropout does nothing at inference, so the whole model is two matrix multiplies
and a softmax - about 25k multiply-adds. Installing TensorFlow (roughly 500 MB,
and a numpy version of its own choosing) into the project venv to perform them
would have been the expensive way to get the same numbers, and this machine has
under a gigabyte of memory to spare. The weights are read straight out of the
HDF5 file with h5py and applied with numpy instead, which is exact: the same
floats, the same order of operations.

    python action_model.py            # check the weights load and time a forward pass
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "action.h5"

# The training order from cell 22 of the notebook. Output index i means ACTIONS[i];
# if this list ever stops matching label_map, every prediction is mislabelled
# while the model itself is perfectly fine.
ACTIONS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

N_HAND = 21
FEATURE_DIM = 2 * N_HAND * 3        # left hand xyz, then right hand xyz


class ActionModel:
    """The notebook's model, in numpy."""

    def __init__(self, path=None):
        import h5py                                    # only needed to read the file

        self.path = Path(path or DEFAULT_MODEL)
        if not self.path.exists():
            raise FileNotFoundError(f"no model at {self.path}")
        self.layers = []
        with h5py.File(self.path, "r") as handle:
            weights = handle["model_weights"]
            # Keras writes the layers in order; keep that order, and keep only
            # the ones that actually hold weights (Dropout holds none).
            names = list(weights.attrs.get("layer_names", list(weights.keys())))
            names = [n.decode() if isinstance(n, bytes) else n for n in names]
            for name in names:
                group = weights[name]
                if name not in group:
                    continue
                inner = group[name]
                kernel = next((k for k in inner if "kernel" in k), None)
                bias = next((k for k in inner if "bias" in k), None)
                if kernel is None:
                    continue
                self.layers.append((np.array(inner[kernel], dtype=np.float32),
                                    np.array(inner[bias], dtype=np.float32)))
        if len(self.layers) != 3:
            raise ValueError(f"expected 3 dense layers in {self.path}, "
                             f"found {len(self.layers)}")
        if self.layers[0][0].shape[0] != FEATURE_DIM:
            raise ValueError(f"the model takes {self.layers[0][0].shape[0]} inputs, "
                             f"not the {FEATURE_DIM} this code builds")
        self.labels = list(ACTIONS)
        if self.layers[-1][1].shape[0] != len(self.labels):
            raise ValueError(f"the model has {self.layers[-1][1].shape[0]} outputs but "
                             f"{len(self.labels)} labels are listed; the label list "
                             "must match the training order exactly")

    def probabilities(self, features):
        """One 126-long vector (or a batch of them) -> class probabilities."""
        x = np.asarray(features, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None]
        for i, (kernel, bias) in enumerate(self.layers):
            x = x @ kernel + bias
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)                  # relu
        x = x - x.max(axis=-1, keepdims=True)           # softmax, stably
        np.exp(x, out=x)
        x /= x.sum(axis=-1, keepdims=True)
        return x[0] if single else x

    def predict(self, features, top_k=3):
        probs = self.probabilities(features)
        order = np.argsort(-probs)[:top_k]
        return [(self.labels[i], float(probs[i])) for i in order]


def hand_features(results):
    """MediaPipe Hands results -> the 126 numbers the model was trained on.

    Copied from the notebook's extract_hand_keypoints (cell 25), including the
    detail that matters most: a hand that is not seen stays all zeros, and which
    half of the vector a hand lands in comes from MediaPipe's own Left/Right
    label, not from where it is on screen.
    """
    left = np.zeros(N_HAND * 3, dtype=np.float32)
    right = np.zeros(N_HAND * 3, dtype=np.float32)
    if results.multi_hand_landmarks:
        for landmarks, handedness in zip(results.multi_hand_landmarks,
                                         results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in landmarks.landmark],
                              dtype=np.float32).flatten()
            if handedness.classification[0].label == "Left":
                left = coords
            else:
                right = coords
    return np.concatenate([left, right])


def features_from_holistic_row(row, mirrored=True):
    """The project's own 300-number frame -> the same 126 numbers.

    Only used to score this model against frames that were recorded for the other
    letter model, so the two can be compared at all. It is an approximation:
    Holistic and Hands are different networks and their landmarks differ slightly.

    `mirrored` mirrors x, because the notebook collected its data from the
    flipped preview (cell 25: cv2.flip(frame, 1)) while the project's own
    recordings are unmirrored.
    """
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    pose_end = 33 * 4
    out = []
    for start in (pose_end, pose_end + N_HAND * 4):
        block = row[start:start + N_HAND * 4].reshape(N_HAND, 4)
        xyz = block[:, :3].copy()
        if not np.isfinite(xyz).all():
            out.append(np.zeros(N_HAND * 3, dtype=np.float32))
            continue
        if mirrored:
            xyz[:, 0] = 1.0 - xyz[:, 0]
        out.append(xyz.flatten())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    import time
    model = ActionModel(args.model)
    shapes = " -> ".join(str(k.shape) for k, _ in model.layers)
    print(f"{args.model}\nlayers: {shapes}\n{len(model.labels)} classes: "
          f"{' '.join(model.labels)}")

    batch = np.random.rand(1000, FEATURE_DIM).astype(np.float32)
    start = time.perf_counter()
    probs = model.probabilities(batch)
    took = (time.perf_counter() - start) * 1000
    print(f"\n1000 frames in {took:.1f} ms ({took/1000:.3f} ms each); "
          f"probabilities sum to {probs.sum(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
