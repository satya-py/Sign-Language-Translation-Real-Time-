"""Learn a sign from a real signer, then record yourself doing it.

Left half: an INCLUDE reference video (deaf signer) looping.
Right half: your webcam. Press SPACE and copy the sign.

    python fetch_reference.py --index                       # once
    python fetch_reference.py --words hello,i,you,doctor    # download references
    python learn_record.py --signer satya --words hello,i,you,doctor

Keys:  SPACE start/stop a take   N next word   B previous word
       R next reference clip     S slower/faster reference
       D delete last take        Q quit
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from record import DATA, count_takes, landmarks_to_row, save_take  # noqa: E402

REF_DIR = HERE / "reference"
PANEL_W, PANEL_H = 480, 360


def reference_clips(word):
    d = REF_DIR / word
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir()
                   if p.suffix.lower() in (".mov", ".mp4", ".avi")])


def fit(img, w, h):
    """Resize keeping aspect ratio, pad with black."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    resized = cv2.resize(img, (max(1, int(iw * scale)), max(1, int(ih * scale))))
    canvas = np.zeros((h, w, 3), np.uint8)
    y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def label(img, text, color=(230, 230, 230), y=26, scale=0.7, thick=2):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signer", required=True)
    p.add_argument("--words", default=None,
                   help="comma separated; default = every word with a reference video")
    p.add_argument("--takes", type=int, default=15)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--auto", type=int, default=0, help="testing only: N automatic 2 s takes")
    args = p.parse_args()

    if args.words:
        words = args.words.split(",")
    else:
        words = sorted(d.name for d in REF_DIR.iterdir()
                       if d.is_dir() and reference_clips(d.name)) if REF_DIR.exists() else []
    if not words:
        sys.exit("No reference videos. Run: python fetch_reference.py --index  "
                 "then --words hello,i,you,...")
    missing = [w for w in words if not reference_clips(w)]
    if missing:
        print(f"note: no reference video for {missing} (they will show a blank panel)")

    import mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5, model_complexity=1)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        sys.exit(f"camera {args.camera} did not open")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window = "learn and record"
    if not args.auto:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, PANEL_W * 2, PANEL_H + 60)

    idx, clip_i, ref_cap = 0, 0, None
    recording, frames, t_start, message, last_saved = False, [], 0.0, "", None
    ref_delay, next_ref_frame, ref_img = 0.06, 0.0, None
    saved_total, auto_done = 0, 0

    def open_reference():
        nonlocal ref_cap, ref_img
        if ref_cap is not None:
            ref_cap.release()
            ref_cap = None
        ref_img = None
        clips = reference_clips(words[idx])
        if clips:
            ref_cap = cv2.VideoCapture(str(clips[clip_i % len(clips)]))

    open_reference()
    print(f"words: {words}\nrecordings go to {DATA}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            word = words[idx]
            now = time.time()

            if recording:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                frames.append(landmarks_to_row(holistic.process(rgb)))

            if args.auto:
                if not recording:
                    recording, frames, t_start = True, [], now
                elif now - t_start > 2.0:
                    save_take(frames, word, args.signer, 25.0)
                    saved_total += 1
                    auto_done += 1
                    recording = False
                    print(f"[auto] {word}: saved {len(frames)} frames")
                    if auto_done >= args.auto:
                        break
                continue

            # Play the reference video on a loop, independent of camera speed.
            if ref_cap is not None and now >= next_ref_frame:
                got, rimg = ref_cap.read()
                if not got:
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    got, rimg = ref_cap.read()
                if got:
                    ref_img = rimg
                next_ref_frame = now + ref_delay

            left = fit(ref_img, PANEL_W, PANEL_H) if ref_img is not None \
                else np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            label(left, "COPY THIS", (120, 220, 255))
            if ref_img is None:
                label(left, "no reference video", (120, 120, 255), y=PANEL_H // 2, scale=0.6, thick=1)

            right = fit(cv2.flip(frame, 1), PANEL_W, PANEL_H)
            label(right, "YOU", (80, 255, 120))
            if recording:
                cv2.circle(right, (PANEL_W - 30, 28), 12, (60, 60, 255), -1)
                label(right, f"REC {now - t_start:.1f}s", (60, 60, 255), y=56, scale=0.6, thick=2)

            canvas = np.zeros((PANEL_H + 60, PANEL_W * 2, 3), np.uint8)
            canvas[:PANEL_H, :PANEL_W] = left
            canvas[:PANEL_H, PANEL_W:] = right
            done = count_takes(word)
            bar = f"{word.upper().replace('_', ' ')}   ({idx + 1}/{len(words)})   takes {done}/{args.takes}"
            label(canvas, bar, (255, 255, 255), y=PANEL_H + 26, scale=0.8, thick=2)
            label(canvas, message or "SPACE record | N next | B back | R other clip | S speed | D delete | Q quit",
                  (170, 170, 170), y=PANEL_H + 50, scale=0.45, thick=1)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not recording:
                    recording, frames, t_start, message = True, [], now, ""
                else:
                    recording = False
                    if len(frames) >= 8:
                        last_saved = save_take(frames, word, args.signer, 25.0)
                        saved_total += 1
                        message = f"saved {len(frames)} frames"
                        if count_takes(word) >= args.takes and idx + 1 < len(words):
                            idx, clip_i, message = idx + 1, 0, message + " -> next word"
                            open_reference()
                    else:
                        message = "too short, not saved"
            elif key in (ord("n"), ord("N")):
                idx, clip_i, recording, message = (idx + 1) % len(words), 0, False, ""
                open_reference()
            elif key in (ord("b"), ord("B")):
                idx, clip_i, recording, message = (idx - 1) % len(words), 0, False, ""
                open_reference()
            elif key in (ord("r"), ord("R")):
                clip_i += 1
                open_reference()
            elif key in (ord("s"), ord("S")):
                ref_delay = {0.06: 0.12, 0.12: 0.25, 0.25: 0.06}[ref_delay]
                message = f"reference speed {'slow' if ref_delay > 0.1 else 'normal'}"
            elif key in (ord("d"), ord("D")) and last_saved and last_saved.exists():
                last_saved.unlink()
                saved_total -= 1
                message, last_saved = f"deleted {last_saved.name}", None
            elif key in (ord("q"), 27):
                break
            elif cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        holistic.close()
        if ref_cap is not None:
            ref_cap.release()
        if not args.auto:
            cv2.destroyAllWindows()
        summary = {w: count_takes(w) for w in words}
        DATA.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(DATA / "summary.json", "w"), indent=1)
        print(f"saved {saved_total} takes this session")
        print("takes per word:", summary)


if __name__ == "__main__":
    main()
