"""Live ISL demo: webcam (or a video file) -> signs -> English sentence.

How it works, per frame:
  1. MediaPipe Holistic gives body + hand keypoints;
  2. a sign starts when the hands move and ends when they settle (no fixed window,
     so signs are never cut in half);
  3. the finished segment goes to the trained classifier, which can also answer
     "idle", and low-confidence guesses are dropped;
  4. after a pause the collected glosses become an English sentence.

    python live_demo.py                          # webcam
    python live_demo.py --video clip.mp4         # uploaded-video demo
    python live_demo.py --model models/signs.pt

Keys:  Q quit   C clear the sentence   S finish the sentence now
"""

import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import sentence as sentence_builder  # noqa: E402
import signfeat  # noqa: E402
from record import landmarks_to_row  # noqa: E402
from train import SignNet  # noqa: E402

DEFAULT_MODEL = HERE / "models" / "signs.pt"


class SignSpotter:
    """Turns a stream of keypoint rows into finished sign segments.

    Speed here is how far the wrists move between frames, so in principle it
    depends on the frame rate - the browser sends about 13 a second against the
    dataset's 25. Scaling the thresholds by the measured rate was tried and made it
    worse (56% of the signer's own takes recognised, against 75%): it cut segments
    while the sign was still being made. The fixed values below were re-measured on
    those takes and are the best of the settings tried.
    """

    def __init__(self, start_speed=0.035, stop_speed=0.018, quiet_frames=7,
                 min_frames=10, max_frames=70, max_seconds=5.0):
        self.start_speed, self.stop_speed = start_speed, stop_speed
        self.quiet_frames, self.min_frames, self.max_frames = quiet_frames, min_frames, max_frames
        self.max_seconds = max_seconds
        self.recent = deque(maxlen=5)
        self.buffer = []
        self.quiet = 0
        self.active = False
        self.fps = None
        self._last_time = None

    def update(self, row, now=None):
        """Feed one frame. Returns a finished segment (list of rows) or None.

        Pass a clock and the length limit follows the real frame rate, so a slow
        connection cannot hand the classifier fifteen seconds of movement as one
        sign. Five seconds was chosen by measurement: shorter cuts real signing
        short (2.8 s took recognition of the signer's own takes from 96% to 90%,
        because they hold a sign while recording). The speed thresholds are
        deliberately NOT scaled with the frame rate - that was tried and measured
        worse, 75% against 96%, as it ended segments mid-sign.
        """
        start_speed, stop_speed = self.start_speed, self.stop_speed
        quiet_frames, min_frames = self.quiet_frames, self.min_frames
        max_frames = self.max_frames
        if now is not None:
            if self._last_time is not None:
                gap = now - self._last_time
                if 0.005 < gap < 1.0:
                    rate = 1.0 / gap
                    self.fps = rate if self.fps is None else 0.9 * self.fps + 0.1 * rate
            self._last_time = now
            if self.fps:
                max_frames = min(max_frames,
                                 max(min_frames + 4, round(self.max_seconds * self.fps)))
        self.recent.append(row)
        speed = 0.0
        if len(self.recent) >= 2:
            speed = float(signfeat.hand_speed(np.array(self.recent))[-1])
        hands = bool(signfeat.hands_present(np.array([row]))[0])

        if not self.active:
            if hands and speed > start_speed:
                self.active = True
                self.buffer = list(self.recent)      # include the run-up frames
                self.quiet = 0
            return None

        self.buffer.append(row)
        self.quiet = self.quiet + 1 if (speed < stop_speed or not hands) else 0
        if self.quiet >= quiet_frames or len(self.buffer) >= max_frames:
            segment, self.active = self.buffer, False
            self.buffer, self.quiet = [], 0
            return segment if len(segment) >= min_frames else None
        return None

    def flush(self):
        """Hand over a sign that is still in progress.

        A sign is normally cut when the hands go still. If the signer stops the
        camera, presses "Finish sentence", or simply holds the last sign while the
        sentence pause runs out, that final sign used to be thrown away - 10% of
        the signer's takes ended that way, each one a sign the app never reported.
        """
        segment, self.active = self.buffer, False
        self.buffer, self.quiet = [], 0
        return segment if len(segment) >= self.min_frames else None

    @property
    def state(self):
        return "signing" if self.active else "waiting"


class Recogniser:
    def __init__(self, model_path, device=None):
        blob = torch.load(model_path, map_location="cpu", weights_only=False)
        self.labels = blob["labels"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SignNet(blob["feature_dim"], len(self.labels)).to(self.device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.val_accuracy = blob.get("val_accuracy")

    @torch.no_grad()
    def predict(self, rows, top_k=3, tta=False):
        """Classify one segment.

        Averaging over trimmed views (tta=True) was meant to steady the guess, but
        measured on 166 held-out clips it costs accuracy - 93% correct against 96%
        for the single view, and it silenced three times as many signs by pushing
        them under the threshold. Trimming a segment the spotter already trimmed
        removes the start of the sign. It stays available for comparison.
        """
        rows = np.asarray(rows, dtype=np.float32)
        # The training clips are cropped to the moving part, so crop the same way
        # here. Without it, a segment that begins with the hands still is read as
        # "idle" with high confidence - 5% of real takes were lost that way.
        lo, hi = signfeat.active_span(rows)
        if hi - lo >= 10:
            rows = rows[lo:hi]
        views = [rows]
        if tta and len(rows) >= 16:
            cut = max(1, len(rows) // 10)
            views += [rows[cut:], rows[:-cut], rows[cut:-cut]]
        batch = np.stack([signfeat.clip_to_features(v) for v in views])
        logits = self.model(torch.from_numpy(batch).to(self.device))
        probs = torch.softmax(logits, dim=-1).mean(0).cpu().numpy()
        order = np.argsort(-probs)[:top_k]
        return [(self.labels[i], float(probs[i])) for i in order]


def draw(frame, glosses, sentence, source, status, last_pred, fps, threshold):
    h, w = frame.shape[:2]
    panel = frame[:118].copy()
    cv2.rectangle(panel, (0, 0), (w, 118), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.6, frame[:118], 0.4, 0, frame[:118])

    words = " ".join(g.upper().replace("_", " ") for g in glosses) or "..."
    cv2.putText(frame, words[-60:], (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(frame, words[-60:], (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 120), 2)

    if sentence:
        cv2.putText(frame, sentence[:70], (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(frame, sentence[:70], (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(frame, source, (w - 70, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150, 150, 150), 1)

    colour = (80, 200, 255) if status == "signing" else (170, 170, 170)
    cv2.putText(frame, f"[{status}]", (14, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)
    if last_pred:
        label, prob = last_pred
        weak = prob < threshold
        cv2.putText(frame, f"{label} {prob*100:.0f}%" + ("  (too low, ignored)" if weak else ""),
                    (120, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (90, 90, 255) if weak else (120, 220, 120), 1)
    cv2.putText(frame, f"{fps:.0f} fps | Q quit  C clear  S sentence", (w - 330, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--video", default=None, help="run on a video file instead of the webcam")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.50,
                   help="ignore predictions below this confidence")
    p.add_argument("--pause", type=float, default=2.5,
                   help="seconds of stillness that finish a sentence")
    p.add_argument("--no-llm", action="store_true", help="always use the rule-based sentences")
    p.add_argument("--save", default=None, help="write the annotated video here")
    p.add_argument("--headless", action="store_true", help="no window (for testing)")
    args = p.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"model not found: {args.model}\nTrain one first:  python train.py")
    rec = Recogniser(args.model)
    print(f"model: {len(rec.labels)} classes {rec.labels}")
    if rec.val_accuracy:
        print(f"validation accuracy at training time: {rec.val_accuracy*100:.1f}%")

    import mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5, model_complexity=1)
    source_is_file = args.video is not None
    cap = cv2.VideoCapture(args.video) if source_is_file else \
        cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        sys.exit(f"could not open {'video ' + args.video if source_is_file else 'the camera'}")
    if not source_is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    writer = None
    spotter = SignSpotter()
    glosses, sentence_text, source = [], "", ""
    gloss_times = {}                 # when each sign entered the current sentence
    last_pred, last_sign_time = None, time.time()
    fps, fps_mark, fps_count = 0.0, time.time(), 0
    lock = threading.Lock()

    def finish_sentence():
        nonlocal sentence_text, source, glosses
        with lock:
            words = list(glosses)
        if not words:
            return
        text, src = sentence_builder.build(words, use_llm=not args.no_llm)
        with lock:
            sentence_text, source, glosses = text, src, []
            gloss_times.clear()
        print(f"SENTENCE: {text}   [{src}]", flush=True)

    window = "ISL live translator"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 720)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if source_is_file:
                    break
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            row = landmarks_to_row(holistic.process(rgb))

            segment = spotter.update(row)
            if segment is not None:
                top = rec.predict(segment)
                label, prob = top[0]
                last_pred = (label, prob)
                if label != "idle" and prob >= args.threshold:
                    with lock:
                        now = time.time()
                        repeat = (glosses and (glosses[-1] == label or
                                  (label in glosses and now - gloss_times.get(label, 0) < 8.0)))
                        if not repeat:
                            glosses.append(label)
                            gloss_times[label] = now
                    last_sign_time = time.time()
                    print(f"sign: {label} {prob*100:.0f}%  "
                          f"(next: {top[1][0]} {top[1][1]*100:.0f}%)", flush=True)

            if glosses and time.time() - last_sign_time > args.pause and spotter.state == "waiting":
                threading.Thread(target=finish_sentence, daemon=True).start()
                last_sign_time = time.time()

            fps_count += 1
            if time.time() - fps_mark >= 0.5:
                fps = fps_count / (time.time() - fps_mark)
                fps_mark, fps_count = time.time(), 0

            shown = frame if source_is_file else cv2.flip(frame, 1)
            with lock:
                shown = draw(shown.copy(), glosses, sentence_text, source, spotter.state,
                             last_pred, fps, args.threshold)
            if args.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if hasattr(cv2, "VideoWriter_fourcc") \
                        else cv2.VideoWriter.fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, 20.0,
                                             (shown.shape[1], shown.shape[0]))
                writer.write(shown)
            if args.headless:
                continue

            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("c"), ord("C")):
                with lock:
                    glosses, sentence_text = [], ""
            if key in (ord("s"), ord("S")):
                finish_sentence()
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if glosses:
            finish_sentence()
        cap.release()
        holistic.close()
        if writer is not None:
            writer.release()
            print("saved annotated video to", args.save)
        if not args.headless:
            cv2.destroyAllWindows()
        print("final sentence:", sentence_text or "(none)")


if __name__ == "__main__":
    main()
