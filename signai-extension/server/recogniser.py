"""Loads the ISL recogniser this project already has, without touching it.

The extension needs three things from the existing work, and imports all three
read-only from ``isl-translator/live``:

  * ``record.landmarks_to_row``  - a frame of MediaPipe Holistic as 300 numbers;
  * ``live_demo.SignSpotter``    - where one sign starts and ends in a stream;
  * ``live_demo.Recogniser``     - those frames as a word;
  * ``sentence.build``           - the words as one English sentence, through an
                                   LLM when one is configured and through the
                                   rule grammar otherwise.

Nothing here writes to that directory. Set SIGNAI_LIVE_DIR to point somewhere
else, or SIGNAI_MODEL to a different checkpoint.

A note on the architecture, because the brief asked for "GCN+BiLSTM": the model
that is actually trained here is SignNet - a depthwise 1D CNN with a small
transformer encoder over body-relative keypoints, 0.82M parameters, measured at
95.4% on 1185 clips. Swapping in a graph-convolution + BiLSTM model would mean
training one from scratch, and an untrained model of any shape recognises
nothing. The loader below takes any object with ``labels`` and ``predict(rows)``,
so a different architecture can replace it the day it is trained.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LIVE = HERE.parent.parent / "isl-translator" / "live"
LIVE_DIR = Path(os.environ.get("SIGNAI_LIVE_DIR", DEFAULT_LIVE)).resolve()


class PipelineUnavailable(RuntimeError):
    """Raised with an explanation the popup can show the user."""


def _import_pipeline():
    if not LIVE_DIR.is_dir():
        raise PipelineUnavailable(
            f"the ISL pipeline was not found at {LIVE_DIR}. "
            "Set SIGNAI_LIVE_DIR to the folder holding live_demo.py."
        )
    if str(LIVE_DIR) not in sys.path:
        sys.path.insert(0, str(LIVE_DIR))
    try:
        import mediapipe as mp                     # noqa: F401
        from live_demo import Recogniser, SignSpotter
        from record import landmarks_to_row
        import sentence as sentence_builder
    except Exception as error:                     # ImportError, DLL errors, ...
        raise PipelineUnavailable(
            f"could not load the ISL pipeline from {LIVE_DIR}: {error}"
        ) from error
    return mp, Recogniser, SignSpotter, landmarks_to_row, sentence_builder


class Pipeline:
    """One loaded model, shared by every session."""

    def __init__(self):
        mp, Recogniser, SignSpotter, landmarks_to_row, sentence_builder = _import_pipeline()
        self.mp = mp
        self.SignSpotter = SignSpotter
        self.landmarks_to_row = landmarks_to_row
        self.sentence = sentence_builder

        model_path = os.environ.get("SIGNAI_MODEL", str(LIVE_DIR / "models" / "signs.pt"))
        if not Path(model_path).exists():
            raise PipelineUnavailable(f"no model file at {model_path}")
        # CPU by default. The classifier is 0.82M parameters - a few milliseconds
        # either way - while a CUDA context costs several hundred megabytes of
        # system memory, and memory is what this machine runs out of. Set
        # SIGNAI_DEVICE=cuda to override.
        device = os.environ.get("SIGNAI_DEVICE", "cpu")
        import torch
        torch.set_num_threads(int(os.environ.get("SIGNAI_TORCH_THREADS", "1")))
        self.recogniser = Recogniser(model_path, device=device)
        self.model_path = model_path
        self.device = device
        # MediaPipe's heavier model is the accurate one the classifier was trained
        # against; complexity 0 is the fallback when the graph cannot be built at
        # all, which on a full machine shows up as "failed to setup XNNPACK
        # runtime" and an aborted process rather than an exception.
        self.complexity = int(os.environ.get("SIGNAI_COMPLEXITY", "1"))
        self._holistic = None

    @property
    def labels(self):
        return [l for l in self.recogniser.labels if l != "idle"]

    def _build_holistic(self, complexity):
        return self.mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
            model_complexity=complexity)

    def new_holistic(self):
        """The tracker, shared by every session.

        One per session was the obvious design and the wrong one here: each graph
        holds its own TFLite arenas, and a second one on a machine with a gigabyte
        free fails inside XNNPACK - which aborts the process instead of raising.
        Frames are already serialised onto a single thread, so sharing costs
        nothing as long as one person signs at a time.
        """
        if self._holistic is None:
            try:
                self._holistic = self._build_holistic(self.complexity)
            except Exception as error:
                if self.complexity == 0:
                    raise PipelineUnavailable(
                        f"MediaPipe could not start: {error}. Close other programs "
                        "and try again - it needs a few hundred megabytes free."
                    ) from error
                self.complexity = 0
                self._holistic = self._build_holistic(0)
        return self._holistic

    def close_holistic(self):
        """Only at shutdown: sessions share the tracker, so none of them owns it."""
        if self._holistic is not None:
            try:
                self._holistic.close()
            finally:
                self._holistic = None

    def new_spotter(self):
        return self.SignSpotter()

    def predict(self, segment, top_k=3):
        return self.recogniser.predict(segment, top_k=top_k)

    def build_sentence(self, glosses, use_llm=True, context="video call"):
        text, source = self.sentence.build(glosses, use_llm=use_llm, context=context)
        return text, source
