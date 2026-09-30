"""Choose the fingerspelling commit rule from data, not by feel.

A letter is only reported once the same shape has held for a few frames. Too strict
and letters can never be spelled at all - the shipped rule (window 8, 6 agreeing
frames, probability 0.60) committed only 27 of the 37 letters when their held-out
frames were replayed, so ten letters were simply impossible. Too loose and one hold
produces several letters.

This replays the last frames of every letter as a stream and counts what each rule
would spell.

    python tune_letters.py
    python tune_letters.py --fps 14        # match the web app's frame rate
"""

import argparse
import collections
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from letter_infer import LetterRecogniser  # noqa: E402

RULES = [(8, 6, 0.60), (8, 5, 0.50), (8, 4, 0.50), (8, 4, 0.40),
         (6, 3, 0.45), (10, 5, 0.45), (12, 6, 0.45)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "data" / "letters"))
    p.add_argument("--held-out", type=int, default=6,
                   help="frames from the end of each recording to judge on")
    p.add_argument("--fps", type=float, default=14.0)
    args = p.parse_args()

    by_label = collections.defaultdict(list)
    for path in sorted(glob.glob(str(Path(args.data) / "*" / "*.npz"))):
        data = np.load(path, allow_pickle=True)
        by_label[str(data["label"])].append(np.asarray(data["keypoints"], np.float32))
    if not by_label:
        sys.exit(f"no recordings under {args.data}")

    print(f"{len(by_label)} letters, judging on the last {args.held_out} frames of each\n")
    print(f"{'rule':<32}{'right':>7}{'wrong':>7}{'never':>7}{'delay':>10}")
    best = None
    for window, agree, threshold in RULES:
        right = wrong = never = 0
        delays = []
        for label, rows in by_label.items():
            held = rows[-args.held_out:]
            rec = LetterRecogniser(window=window, agree=agree, threshold=threshold)
            for seen, row in enumerate(held * 3, start=1):
                committed, _, _ = rec.update(row)
                if committed:
                    delays.append(seen)
                    break
            if not rec.letters:
                never += 1
            elif rec.letters[0] == label:
                right += 1
            else:
                wrong += 1
        delay = sum(delays) / max(1, len(delays)) / args.fps
        print(f"window {window} agree {agree} p>={threshold:.2f}   {right:>6}{wrong:>7}"
              f"{never:>7}{delay:>9.2f}s")
        score = right - 2 * wrong                 # a wrong letter has to be deleted
        if best is None or score > best[0]:
            best = (score, window, agree, threshold, right, wrong, never)

    _, window, agree, threshold, right, wrong, never = best
    print(f"\nbest: window={window} agree={agree} threshold={threshold} "
          f"({right} right, {wrong} wrong, {never} never committed)")
    print("set them as the defaults in letter_infer.py LetterRecogniser.__init__")


if __name__ == "__main__":
    main()
