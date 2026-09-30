"""Which letters are weak, and why - so re-recording is aimed instead of guessed.

Two different failures look the same when spelling live:

  * MediaPipe never found a hand in the frame, so no letter can be read at all
    (ISL letters where the hands overlap do this; L, S, B and D are the worst here);
  * the hand was found but the classifier reads it as another letter.

The first is fixed by recording the letter again with the hands further apart and
the whole hand inside the frame; the second by recording more takes of it.

    python check_letters.py
"""

import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import letterfeat                            # noqa: E402
from letter_infer import LetterRecogniser    # noqa: E402


def main():
    frames = defaultdict(list)
    for path in sorted(glob.glob(str(HERE / "data" / "letters" / "*" / "*.npz"))):
        data = np.load(path, allow_pickle=True)
        frames[str(data["label"])].append(np.asarray(data["keypoints"], np.float32))
    if not frames:
        sys.exit("no recordings yet: python record_letters.py")

    rec = LetterRecogniser()
    print(f"{len(frames)} letters, {sum(len(v) for v in frames.values())} frames\n")
    print(f"{'letter':<8}{'hand found':>12}{'read as':>26}{'verdict':>28}")
    todo = []
    for label in sorted(frames):
        rows = frames[label]
        found = [r for r in rows if letterfeat.has_hand(r)]
        guesses = Counter(g for g, p in (rec.classify(r) for r in found) if g)
        top = guesses.most_common(2)
        reads = ", ".join(f"{g} {100*c/max(1,len(found)):.0f}%" for g, c in top)
        share = len(found) / len(rows)
        if share < 0.85:
            verdict = "record again, hands apart"
            todo.append(label)
        elif not top or top[0][0] != label:
            verdict = f"confused with {top[0][0] if top else '?'}"
            todo.append(label)
        elif top[0][1] / max(1, len(found)) < 0.8:
            verdict = "record a few more takes"
            todo.append(label)
        else:
            verdict = "fine"
        print(f"{label:<8}{len(found)}/{len(rows):<10}{reads:>26}{verdict:>28}")

    if todo:
        print(f"\nworth recording again ({len(todo)}): {', '.join(todo)}")
        print("python record_letters.py --labels " + ",".join(todo))
    else:
        print("\nevery letter is detected and read correctly")


if __name__ == "__main__":
    main()
