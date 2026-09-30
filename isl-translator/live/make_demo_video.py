"""Build sentence videos by joining single-sign clips, then run the demo on them.

Two uses:
  * a real multi-sign test (one clip only ever shows one sign);
  * a backup demo video for the presentation, in case the live camera misbehaves.

    python make_demo_video.py --sentence i,sick,today
    python make_demo_video.py --all           # build every example sentence
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).parent
VIDEO_DIR = HERE / "data" / "videos"
OUT_DIR = HERE / "demo"

SENTENCES = [
    ["hello", "how_are_you"],
    ["i", "sick", "today"],
    ["doctor", "hospital"],
    ["i", "school", "tomorrow"],
    ["friend", "house", "good"],
    ["thank_you"],
]


def build(words, out_path, seed=None, size=(640, 480), fps=25.0, rest_frames=12):
    """Join one clip per word, with a short still gap so signs stay separable."""
    rng = random.Random(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if hasattr(cv2, "VideoWriter_fourcc") \
        else cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        sys.exit(f"cannot write {out_path}")
    used = []
    for word in words:
        clips = sorted((VIDEO_DIR / word).glob("*.MOV")) + sorted((VIDEO_DIR / word).glob("*.mp4"))
        if not clips:
            sys.exit(f"no video for '{word}' in {VIDEO_DIR}")
        clip = rng.choice(clips)
        used.append(clip.name)
        cap = cv2.VideoCapture(str(clip))
        last = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last = cv2.resize(frame, size)
            writer.write(last)
        cap.release()
        for _ in range(rest_frames):      # hands at rest between signs
            if last is not None:
                writer.write(last)
    writer.release()
    return used


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sentence", default=None, help="comma separated words")
    p.add_argument("--all", action="store_true", help="build every example sentence")
    p.add_argument("--model", default=str(HERE / "models" / "signs.pt"))
    p.add_argument("--annotate", action="store_true",
                   help="also save a version with the predictions drawn on it")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = SENTENCES if args.all else [args.sentence.split(",")] if args.sentence else SENTENCES[:1]

    results = []
    for words in todo:
        name = "_".join(words)
        raw = OUT_DIR / f"{name}.mp4"
        used = build(words, raw, seed=args.seed)
        cmd = [sys.executable, str(HERE / "live_demo.py"), "--model", args.model,
               "--video", str(raw), "--headless", "--no-llm"]
        if args.annotate:
            cmd += ["--save", str(OUT_DIR / f"{name}_annotated.mp4")]
        out = subprocess.run(cmd, capture_output=True, text=True)
        signs = [l for l in out.stdout.splitlines() if l.startswith("sign:")]
        sentence = next((l.split(":", 1)[1].strip() for l in out.stdout.splitlines()
                         if l.startswith("SENTENCE:")), "(none)")
        print(f"\n{' + '.join(words)}  ->  {raw.name}")
        for s in signs:
            print("   ", s)
        print("    SENTENCE:", sentence)
        results.append({"words": words, "video": str(raw), "clips": used,
                        "detected": signs, "sentence": sentence})

    ok = sum(1 for r in results
             if [s.split()[1] for s in r["detected"]] == r["words"])
    print(f"\n{ok}/{len(results)} sentences recognised exactly")


if __name__ == "__main__":
    main()
