"""Live fingerspelling: frames in, letters out.

Fingerspelling is different from word signs in two ways, and both shape this code:

  * a letter is a pose HELD, not a movement, so we classify every frame and only
    accept a letter once it has been steady for a while;
  * the same letter can repeat ("ANNA"), so a repeat is only allowed after a real
    break — the hand leaving, or the shape changing to something else.

Without those two rules the buffer fills with dozens of copies of one letter.
"""

import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import letterfeat  # noqa: E402
from train_letters import LetterNet  # noqa: E402

DEFAULT_MODEL = HERE / "models" / "letters.pt"


class LetterRecogniser:
    """Rolling-window letter spotter over single frames."""

    # Measured by replaying the held-out frames of all 37 letters as a stream
    # (tune_letters.py): 8/4/0.50 with a 6-frame gap commits 35 of them and nothing
    # wrong, where the shipped 8/6/0.60 committed only 27 - ten letters could never
    # be spelled at all, which is what made fingerspelling feel broken. The gap is
    # 6 frames (~0.4 s) because MediaPipe drops the hand for a frame or two at a
    # time; 4 was short enough that those dropouts reset the vote, and 10 was long
    # enough to swallow the pause between a doubled letter.
    def __init__(self, model_path=None, device=None, window=8, agree=4,
                 threshold=0.50, gap_frames=6):
        blob = torch.load(str(model_path or DEFAULT_MODEL), map_location="cpu",
                          weights_only=False)
        self.labels = blob["labels"]
        self.val_accuracy = blob.get("val_accuracy")
        self.samples = blob.get("samples")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LetterNet(blob["feature_dim"], len(self.labels)).to(self.device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

        self.window, self.agree, self.threshold = window, agree, threshold
        self.gap_frames = gap_frames
        self.reset()

    def reset(self):
        self.recent = deque(maxlen=self.window)
        self.letters = []
        self.last_committed = None
        self.since_gap = 0

    @torch.no_grad()
    def classify(self, row):
        """One frame -> (label, probability), or (None, 0.0) with no hand in view."""
        if not letterfeat.has_hand(row):
            return None, 0.0
        x = torch.from_numpy(letterfeat.row_to_features(row))[None].to(self.device)
        probs = torch.softmax(self.model(x), dim=-1)[0].cpu().numpy()
        best = int(np.argmax(probs))
        return self.labels[best], float(probs[best])

    def update(self, row):
        """Feed one frame. Returns (committed_letter_or_None, current_label, prob)."""
        label, prob = self.classify(row)

        if label is None or prob < self.threshold:
            # Hand gone or unsure: this is the break that lets a letter repeat.
            # It is NOT counted into the window: MediaPipe loses the hand in about a
            # third of frames on letters where the hands overlap (C, S, V), and
            # counting those blanks as votes made those letters impossible to spell.
            self.since_gap += 1
            if self.since_gap >= self.gap_frames:
                self.last_committed = None
                self.recent.clear()          # a real break ends the current letter
            return None, label, prob

        self.since_gap = 0
        self.recent.append(label)
        counts = Counter(l for l in self.recent if l is not None)
        if not counts:
            return None, label, prob
        steady, votes = counts.most_common(1)[0]
        if votes >= self.agree and steady != self.last_committed:
            self.last_committed = steady
            self.letters.append(steady)
            self.recent.clear()
            return steady, label, prob
        return None, label, prob

    @property
    def word(self):
        """What has been spelled so far, e.g. ['S','A','T','Y','A'] -> 'SATYA'."""
        return "".join(self.letters)

    def backspace(self):
        if self.letters:
            self.letters.pop()
        self.last_committed = None
        self.recent.clear()
        return self.word
