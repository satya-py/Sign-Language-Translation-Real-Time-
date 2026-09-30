"""Live fingerspelling with the notebook's action.h5 model.

The notebook classifies a single frame; a video stream needs two more rules, the
same two the other letter recogniser uses:

  * a letter is a pose HELD, so it is only reported once the same guess has been
    steady for several frames;
  * the same letter may repeat ("ANNA"), so a repeat needs a real break first -
    the hand leaving, or the shape changing to something else.

Everything about how a frame becomes 126 numbers is copied from the notebook,
including the two details that silently ruin it if they are changed:

  * the frame is MIRRORED before MediaPipe sees it (cell 25: cv2.flip(frame, 1)),
    because that is how the training data was collected;
  * which half of the vector a hand lands in comes from MediaPipe's own
    Left/Right label, which it assigns as if looking in a mirror.

    python action_infer.py --self-test      # check it loads and runs on one frame
"""

from __future__ import annotations

import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from action_model import ACTIONS, ActionModel, hand_features  # noqa: E402


class ActionSpeller:
    """Frames in, letters out, using action.h5."""

    def __init__(self, model_path=None, window=8, agree=4, threshold=0.50,
                 gap_frames=6, max_hands=2, mirror=True):
        self.model = ActionModel(model_path)
        self.labels = list(ACTIONS)
        self.window, self.agree, self.threshold = window, agree, threshold
        self.gap_frames = gap_frames
        self.mirror = mirror
        import mediapipe as mp
        self._mp = mp
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.reset()

    # ------------------------------------------------------------- state

    def reset(self):
        self.recent = deque(maxlen=self.window)
        self.letters = []
        self.last_committed = None
        self.since_gap = 0

    @property
    def word(self):
        return "".join(self.letters)

    def backspace(self):
        if self.letters:
            self.letters.pop()
        self.last_committed = None
        self.recent.clear()
        return self.word

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass

    # ------------------------------------------------------------- reading

    def features_from_frame(self, frame_bgr):
        """A BGR frame -> the 126 numbers, or None when no hand is in it."""
        import cv2
        image = cv2.flip(frame_bgr, 1) if self.mirror else frame_bgr
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        return hand_features(results)

    def classify(self, frame_bgr, top_k=3):
        """One frame -> [(letter, probability), ...], empty with no hand in view."""
        features = self.features_from_frame(frame_bgr)
        if features is None:
            return []
        return self.model.predict(features, top_k=top_k)

    def update(self, frame_bgr):
        """Feed one frame. Returns (committed_letter_or_None, current, probability)."""
        top = self.classify(frame_bgr, top_k=1)
        label, prob = top[0] if top else (None, 0.0)

        if label is None or prob < self.threshold:
            # No hand, or an unsure frame: this is the break that lets a letter
            # repeat. It is not counted as a vote - MediaPipe drops the hand for
            # a frame or two at a time, and counting those blanks made letters
            # that are hard to see impossible to spell at all.
            self.since_gap += 1
            if self.since_gap >= self.gap_frames:
                self.last_committed = None
                self.recent.clear()
            return None, label, prob

        self.since_gap = 0
        self.recent.append(label)
        steady, votes = Counter(self.recent).most_common(1)[0]
        if votes >= self.agree and steady != self.last_committed:
            self.last_committed = steady
            self.letters.append(steady)
            self.recent.clear()
            return steady, label, prob
        return None, label, prob


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--self-test", action="store_true",
                        help="run one blank frame through the whole path")
    args = parser.parse_args()

    speller = ActionSpeller(args.model)
    print(f"loaded {speller.model.path}: {len(speller.labels)} classes")
    if args.self_test:
        import cv2                                   # noqa: F401  (import check)
        blank = np.zeros((480, 640, 3), np.uint8)
        committed, current, prob = speller.update(blank)
        print(f"blank frame -> committed {committed}, current {current}, prob {prob:.2f} "
              "(no hand in it, so nothing is expected)")
        print("the whole path runs: mediapipe hands -> 126 features -> dense model")
    speller.close()


if __name__ == "__main__":
    main()
