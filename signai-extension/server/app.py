"""SignAI backend: JPEG frames in, English sentences out.

    python server/app.py                  # http://127.0.0.1:5001
    python server/app.py --port 5050 --no-llm

The Chrome extension posts short chunks of frames (about a second at a time)
rather than one request per frame; per-frame HTTP would spend more time in
request overhead than in MediaPipe. Each browser session keeps its own tracker
and its own sign spotter here, because both carry state from frame to frame.

Endpoints
    GET  /api/health          model, vocabulary, how many sessions are open
    POST /api/session         start one, returns {"session": "..."}
    POST /api/chunk           {"session", "frames": [dataURL, ...]}
                              -> {"glosses", "sentence", "final", "confidence"}
    POST /api/finish          close the sentence now, whatever is in hand
    POST /api/reset           throw away the words collected so far
    POST /api/close           end the session and free its tracker
"""

from __future__ import annotations

import argparse
import base64
import binascii
import faulthandler
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from recogniser import Pipeline, PipelineUnavailable

# A native crash inside MediaPipe or torch kills the process with no Python
# traceback at all - which is exactly what "the server just exited" looks like.
# This makes it print the stack it died on.
faulthandler.enable()

app = Flask(__name__)
# The extension's origin is chrome-extension://<id>, which changes per install,
# and this server only ever listens on the loopback interface.
CORS(app, resources={r"/api/*": {"origins": "*"}})

CONFIG = {
    "threshold": 0.35,     # matches the live app: measured, not guessed
    "pause": 2.0,          # seconds of stillness that end a sentence
    "repeat_window": 8.0,  # the same sign may not re-enter a sentence this fast
    "use_llm": True,
    "session_ttl": 600.0,  # a browser that goes away must not leak a tracker
}

PIPELINE: Pipeline | None = None
PIPELINE_ERROR = ""


class Worker:
    """One thread that owns everything MediaPipe and torch touch.

    A MediaPipe graph belongs to the thread that built it. Flask's development
    server hands each request to whichever worker thread happens to be free, so a
    tracker created while answering /api/session was then used from a different
    thread on the next request - and MediaPipe does not fail politely at that. It
    takes the process down, with no Python traceback, which is what "the server
    exited by itself" was.

    Every call below therefore runs here, one at a time, on the thread that
    created the tracker.
    """

    def __init__(self):
        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="signai-pipeline",
                                       daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            job, done, box = self.jobs.get()
            try:
                box.append(("ok", job()))
            except BaseException as error:      # reported to the caller instead
                box.append(("error", error))
            finally:
                done.set()

    def run(self, job, timeout=180.0):
        """Run job() on the pipeline thread and hand back what it returned."""
        if threading.current_thread() is self.thread:
            return job()                        # already there; do not deadlock
        done, box = threading.Event(), []
        self.jobs.put((job, done, box))
        if not done.wait(timeout):
            raise TimeoutError("the recogniser did not answer in time")
        kind, value = box[0]
        if kind == "error":
            raise value
        return value


WORKER = Worker()


@dataclass
class Session:
    """Everything one browser tab needs, and a lock so it stays consistent."""

    id: str
    holistic: object
    spotter: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    glosses: list = field(default_factory=list)
    gloss_times: dict = field(default_factory=dict)
    sentence: str = ""
    last_sign: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    confidence: float = 0.0
    last_guess: list = field(default_factory=list)   # top 3 of the last segment
    accepted: bool = False                           # was that guess used?
    shoulders: float = 1.0                           # share of frames with shoulders
    hands: float = 0.0                               # share of frames with a hand
    shoulder_width: float = 0.0                      # how wide they are in the frame

    def close(self):
        """Sessions share the tracker, so ending one must not close it."""
        self.holistic = None


SESSIONS: dict[str, Session] = {}
SESSIONS_LOCK = threading.Lock()


def pipeline() -> Pipeline:
    global PIPELINE, PIPELINE_ERROR
    if PIPELINE is None:
        PIPELINE = Pipeline()          # raises PipelineUnavailable with a reason
        PIPELINE_ERROR = ""
    return PIPELINE


def sweep_sessions():
    """Drop sessions nobody has posted to for a while."""
    now = time.time()
    with SESSIONS_LOCK:
        stale = [s for s in SESSIONS.values() if now - s.touched > CONFIG["session_ttl"]]
        for session in stale:
            SESSIONS.pop(session.id, None)
    for session in stale:
        WORKER.run(session.close)


def decode(frame: str):
    """A data URL from the browser canvas -> a BGR image."""
    if not isinstance(frame, str):
        return None
    payload = frame.split(",", 1)[1] if "," in frame else frame
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return image


def accept(session: Session, segment) -> bool:
    """Classify one segment and add it to the sentence if it is worth adding."""
    pipe = pipeline()
    top = pipe.predict(segment, top_k=3)
    label, prob = top[0]
    session.confidence = float(prob)
    # Keep the guess even when it is refused: "read headache at 13%" tells the
    # signer what to do, where a silent screen tells them nothing.
    session.last_guess = [{"label": l, "prob": round(float(p), 3)} for l, p in top]
    session.accepted = False
    if label == "idle" or prob < CONFIG["threshold"]:
        return False
    now = time.time()
    # The hand settling after a sign is easily read as that sign again, and a
    # sentence almost never repeats a word within a few seconds.
    recent = now - session.gloss_times.get(label, 0.0) < CONFIG["repeat_window"]
    if session.glosses and (session.glosses[-1] == label or
                            (label in session.glosses and recent)):
        return False
    session.glosses.append(label)
    session.gloss_times[label] = now
    session.last_sign = now
    session.accepted = True
    return True


def close_sentence(session: Session):
    """Turn the collected words into a sentence and start a new one."""
    if not session.glosses:
        return ""
    text, _source = pipeline().build_sentence(
        session.glosses, use_llm=CONFIG["use_llm"])
    session.sentence = text
    session.glosses = []
    session.gloss_times = {}
    session.last_sign = time.time()
    return text


def memory_free_gb():
    """Free physical memory, or None where it cannot be read cheaply."""
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / (1024 ** 3)
    except Exception:
        return None


@app.get("/api/health")
def health():
    try:
        pipe = pipeline()
    except PipelineUnavailable as error:
        return jsonify({"ok": False, "detail": str(error)}), 503
    return jsonify({
        "ok": True,
        "model": pipe.model_path,
        "vocabulary": len(pipe.labels),
        "signs": pipe.labels,
        "sessions": len(SESSIONS),
        "threshold": CONFIG["threshold"],
        "use_llm": CONFIG["use_llm"],
        "device": getattr(pipe, "device", "cpu"),
        "tracker_complexity": getattr(pipe, "complexity", 1),
        "memory_free_gb": round(memory_free_gb() or 0, 2),
    })


@app.post("/api/session")
def open_session():
    try:
        pipe = pipeline()
    except PipelineUnavailable as error:
        return jsonify({"ok": False, "detail": str(error)}), 503
    sweep_sessions()
    session = Session(id=uuid.uuid4().hex,
                      holistic=WORKER.run(pipe.new_holistic),
                      spotter=pipe.new_spotter())
    with SESSIONS_LOCK:
        SESSIONS[session.id] = session
    return jsonify({"session": session.id, "vocabulary": len(pipe.labels)})


def require_session(payload):
    session_id = (payload or {}).get("session")
    with SESSIONS_LOCK:
        return SESSIONS.get(session_id)


@app.post("/api/chunk")
def chunk():
    payload = request.get_json(silent=True) or {}
    session = require_session(payload)
    if session is None:
        return jsonify({"ok": False, "detail": "unknown session"}), 404
    frames = payload.get("frames") or []
    if not isinstance(frames, list):
        return jsonify({"ok": False, "detail": "frames must be a list"}), 400

    pipe = pipeline()
    # Decoding JPEG is plain OpenCV and safe anywhere; everything after it is not.
    images = [img for img in (decode(f) for f in frames[:40]) if img is not None]

    def track():
        """Runs on the pipeline thread: tracker, spotter and classifier."""
        added_here = False
        shoulders = hands = 0
        for image in images:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            row = pipe.landmarks_to_row(session.holistic.process(rgb))
            # Every feature is measured from the middle of the shoulders and
            # divided by the shoulder width. With the shoulders out of frame that
            # scale is guesswork and the whole sign is read against the wrong
            # yardstick - which is what "13% confidence" on a sign the model knows
            # at 95% actually means.
            left, right = row[11 * 4:11 * 4 + 2], row[12 * 4:12 * 4 + 2]
            if np.isfinite(left).all() and np.isfinite(right).all():
                shoulders += 1
                # How much of the frame the signer's shoulders span - the one
                # number that says how far away they are, in the same units the
                # training clips were measured in.
                span = float(np.linalg.norm(left - right))
                session.shoulder_width = (0.8 * session.shoulder_width + 0.2 * span
                                          if session.shoulder_width else span)
            if np.isfinite(row[33 * 4:]).any():
                hands += 1
            segment = session.spotter.update(row, now=time.time())
            if segment is not None and accept(session, segment):
                added_here = True
        if images:
            # Smooth it: one frame where an arm crosses the shoulder is not news.
            session.shoulders = 0.6 * session.shoulders + 0.4 * (shoulders / len(images))
            session.hands = 0.6 * session.hands + 0.4 * (hands / len(images))
        return added_here

    with session.lock:
        session.touched = time.time()
        added = WORKER.run(track)

        final = False
        if added:
            # Show the sentence growing instead of making the signer wait for the
            # pause: on a call, a line that appears two seconds late is a line
            # spoken over.
            session.sentence, _ = WORKER.run(
                lambda: pipe.build_sentence(session.glosses, use_llm=False))
        elif session.glosses and time.time() - session.last_sign > CONFIG["pause"]:
            # A sign still in the hands when the pause runs out belongs to this
            # sentence, not the next one.
            def finish_up():
                leftover = session.spotter.flush() if session.spotter.active else None
                if leftover is not None:
                    accept(session, leftover)
                return close_sentence(session)

            WORKER.run(finish_up)
            final = True

        return jsonify({
            "glosses": list(session.glosses),
            "sentence": session.sentence,
            "building": bool(session.glosses) and not final,
            "final": final,
            "confidence": round(session.confidence, 3),
            "guess": session.last_guess,
            "accepted": session.accepted,
            "framing": framing_advice(session),
        })


def framing_advice(session: Session):
    """What to change about the camera, in the words the signer needs.

    Ordered by how much damage each fault does. Missing shoulders is first
    because every feature is measured from the middle of the shoulders and
    divided by the shoulder width: without them the sign is compared against the
    wrong yardstick, and a sign the model knows at 95% comes back at 13%.

    The hands rule is deliberately slack. A signer rests their hands between
    signs, so a low share of frames with a hand in them is normal; only a stream
    where the hands are almost never seen is worth mentioning.

    The distance thresholds are not invented. Across the training data the
    signer's shoulders span 0.26 to 0.49 of the frame in their own webcam takes
    and 0.12 to 0.16 in the INCLUDE studio clips. Sitting far closer than that -
    head filling the frame, as in a video call - puts every hand position outside
    anything the model was trained on, and a sign it knows at 95% comes back at
    13%.
    """
    width = session.shoulder_width
    if width > 0.60:
        return {"ok": False, "shoulders": round(session.shoulders, 2),
                "hands": round(session.hands, 2), "width": round(width, 2),
                "detail": "You are much closer to the camera than when you taught "
                          "these signs. Move back until your head, shoulders and "
                          "both hands are all in frame."}
    if 0 < width < 0.10:
        return {"ok": False, "shoulders": round(session.shoulders, 2),
                "hands": round(session.hands, 2), "width": round(width, 2),
                "detail": "You are very far from the camera. Move closer until "
                          "your shoulders fill about a third of the frame."}
    if session.shoulders < 0.6:
        return {"ok": False, "shoulders": round(session.shoulders, 2),
                "hands": round(session.hands, 2),
                "detail": "Sit back until your shoulders are in frame. Signs are "
                          "measured against your shoulder width, so with the "
                          "shoulders cut off even a sign the model knows well "
                          "reads as noise."}
    if session.hands < 0.15:
        return {"ok": False, "shoulders": round(session.shoulders, 2),
                "hands": round(session.hands, 2), "width": round(width, 2),
                "detail": "Your hands are hardly ever visible. Move into better "
                          "light and keep both hands inside the frame."}
    return {"ok": True, "detail": "", "shoulders": round(session.shoulders, 2),
            "hands": round(session.hands, 2), "width": round(width, 2)}


@app.post("/api/finish")
def finish():
    session = require_session(request.get_json(silent=True) or {})
    if session is None:
        return jsonify({"ok": False, "detail": "unknown session"}), 404
    def finish_up():
        leftover = session.spotter.flush() if session.spotter.active else None
        if leftover is not None:
            accept(session, leftover)
        return close_sentence(session)

    with session.lock:
        session.touched = time.time()
        text = WORKER.run(finish_up)
    return jsonify({"sentence": text, "final": bool(text), "glosses": []})


@app.post("/api/reset")
def reset():
    session = require_session(request.get_json(silent=True) or {})
    if session is None:
        return jsonify({"ok": False, "detail": "unknown session"}), 404
    with session.lock:
        session.glosses = []
        session.gloss_times = {}
        session.sentence = ""
        session.spotter = WORKER.run(pipeline().new_spotter)
    return jsonify({"ok": True})


@app.post("/api/close")
def close():
    payload = request.get_json(silent=True) or {}
    with SESSIONS_LOCK:
        session = SESSIONS.pop(payload.get("session", ""), None)
    if session is not None:
        WORKER.run(session.close)            # free the tracker where it was made
    return jsonify({"ok": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--threshold", type=float, default=CONFIG["threshold"])
    parser.add_argument("--pause", type=float, default=CONFIG["pause"])
    parser.add_argument("--no-llm", action="store_true",
                        help="use the rule grammar only, never the LLM")
    args = parser.parse_args()
    CONFIG["threshold"] = args.threshold
    CONFIG["pause"] = args.pause
    CONFIG["use_llm"] = not args.no_llm

    try:
        pipe = WORKER.run(pipeline)          # load it on the thread that will use it
        print(f"model: {pipe.model_path}\n{len(pipe.labels)} signs loaded")
    except PipelineUnavailable as error:
        print(f"WARNING: {error}\nThe server will start and report this on /api/health.")

    free = memory_free_gb()
    if free is not None:
        print(f"{free:.1f} GB of memory free")
        if free < 1.5:
            print("WARNING: MediaPipe needs a few hundred megabytes to build its "
                  "graph, and aborts the whole process when it cannot get them "
                  "(\"failed to setup XNNPACK runtime\"). Close the other backend "
                  "(isl-translator) and spare Chrome tabs, or start this server "
                  "with SIGNAI_COMPLEXITY=0 for the lighter tracker.")

    print(f"SignAI backend on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False,
            use_reloader=False)


if __name__ == "__main__":
    main()
