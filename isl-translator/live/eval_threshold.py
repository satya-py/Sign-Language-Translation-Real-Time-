"""Pick the acceptance rule from data instead of by feel.

The live app reports a sign when the classifier's top probability clears a
threshold. Too high and signs vanish silently - the signer signs and nothing
appears, which is the worst failure in front of an audience. Too low and wrong
signs enter the sentence.

This replays the held-out clips (the same split the training used, so these clips
never trained the model) and counts the three outcomes for each candidate rule.

    python eval_threshold.py
    python eval_threshold.py --model models/signs.pt --webcam-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import train as T                        # noqa: E402
from live_demo import Recogniser         # noqa: E402

RULES = [(0.50, 0.0), (0.45, 0.0), (0.40, 0.0), (0.35, 0.0), (0.30, 0.0),
         (0.40, 1.8), (0.35, 1.6), (0.30, 1.6)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(HERE / "models" / "signs.pt"))
    p.add_argument("--webcam-only", action="store_true",
                   help="judge only on takes recorded from a webcam (the demo domain)")
    args = p.parse_args()

    clips = T.load_clips([T.KEYPOINT_DIR, T.RECORDING_DIR])
    for c in clips:
        c["webcam"] = c["name"].startswith("web_")
    clips = T.crop_to_sign(clips)
    _, val = T.split_clips(clips)
    if args.webcam_only:
        val = [c for c in val if c["webcam"]]
    if not val:
        sys.exit("no held-out clips to judge on")

    rec = Recogniser(args.model)
    scored = {}
    for tta in (False, True):
        scored[tta] = [(c["label"], rec.predict(c["rows"], top_k=2, tta=tta)) for c in val]

    print(f"{len(val)} held-out clips "
          f"({sum(c['webcam'] for c in val)} from a webcam), {len(rec.labels)} classes\n")
    print(f"{'rule':<34}{'right':>12}{'wrong':>12}{'silent':>12}")
    best = None
    for tta in (False, True):
        for th, margin in RULES:
            ok = bad = mute = 0
            for label, top in scored[tta]:
                (l1, p1), (_, p2) = top[0], top[1]
                if p1 < th or (margin and p1 < margin * p2):
                    mute += 1
                elif l1 == label:
                    ok += 1
                else:
                    bad += 1
            n = len(val)
            name = f"tta={str(tta):<5} p>={th:.2f}" + (f" and p1>={margin}xp2" if margin else "")
            print(f"{name:<34}{ok:>6}{100*ok/n:>5.0f}%{bad:>6}{100*bad/n:>5.0f}%"
                  f"{mute:>6}{100*mute/n:>5.0f}%")
            # A wrong sign has to be cleared from the sentence; a silent one only
            # has to be repeated. Score them 2:1 against being right.
            score = ok - 2 * bad - mute
            if best is None or score > best[0]:
                best = (score, tta, th, margin, ok, bad, mute)

    _, tta, th, margin, ok, bad, mute = best
    rule = f"threshold {th:.2f}" + (f", margin {margin}x" if margin else "")
    print(f"\nbest: {rule}, tta={tta}  ->  right {ok}, wrong {bad}, silent {mute}")
    print("set it in backend/app.py CONFIG['threshold'] (and live_demo.py --threshold)")


if __name__ == "__main__":
    main()
