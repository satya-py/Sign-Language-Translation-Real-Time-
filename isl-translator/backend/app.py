"""SignBridge AI backend — FastAPI service for the React frontend.

Endpoints
    GET  /api/health              model name, vocabulary, thresholds
    GET  /api/gallery             the reference-sign library
    GET  /api/poster/{word}       poster frame for a sign
    GET  /api/clip/{word}         reference video for a sign
    POST /api/video               run a whole uploaded clip through the pipeline
    WS   /ws                      live camera frames in, glosses + sentence out
    GET  /api/teach               how many taught takes exist per sign
    POST /api/retrain             retrain on INCLUDE + taught takes, hot-swap the model
    ...  /api/interpreter/*       human interpreter directory, requests, call rooms
    GET  /api/stats               dashboard counters, computed from real sessions
    GET  /api/sessions            translation history
    POST /api/sessions            append one finished translation
    DEL  /api/sessions/{id}       delete one entry
    GET  /api/settings            recognition + accessibility settings
    PUT  /api/settings            change them (threshold applies immediately)
    GET  /api/phrases             common medical phrases, with the ISL signs we have

The recognition pipeline itself lives in ../live (the ML lab: training, evaluation,
datasets). The backend imports it rather than copying it, so the code that serves
the demo is exactly the code that was trained and measured.

    python app.py                      # http://127.0.0.1:8000
    python app.py --reload             # development
"""

import argparse
import base64
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import (FastAPI, File, Form, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ML_DIR = ROOT / "live"          # training, data and the tested pipeline modules
SPELL_DIR = ROOT / "fingerspell"   # fingerspelling: names and words with no sign
FRONTEND_DIST = ROOT / "frontend" / "dist"
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(SPELL_DIR))

import sentence as sentence_builder  # noqa: E402
from live_demo import Recogniser, SignSpotter  # noqa: E402
from record import landmarks_to_row  # noqa: E402

try:
    from letter_infer import LetterRecogniser  # noqa: E402
except Exception:                               # the app still runs without letters
    LetterRecogniser = None

try:
    # The notebook's model ("Action Detection Refined.ipynb" -> action.h5), served
    # without TensorFlow: see fingerspell/action_model.py for why.
    from action_infer import ActionSpeller  # noqa: E402
except Exception:
    ActionSpeller = None

REPEAT_WINDOW = 8.0      # seconds before the same sign may appear again in a sentence

CONFIG = {
    "model_path": ML_DIR / "models" / "signs.pt",
    "gallery": ML_DIR / "static" / "gallery.json",
    "posters": ML_DIR / "static" / "posters",
    "videos": ML_DIR / "data" / "videos",
    # Measured on 166 held-out signs and 370 windows of between-sign movement
    # (eval_threshold.py): 93% of signs right, 5% of that movement leaking through
    # as a word. Higher and signs start vanishing silently, which is worse on stage
    # than a word the signer clears.
    "threshold": 0.35,
    # Seconds of stillness that complete a sentence. It stays long because the
    # signer pauses about a second BETWEEN signs: at 1.2 s every sign became its
    # own sentence ("Hello!" "I am here." "I am sick."). The wait no longer feels
    # slow, because the sentence is now shown while it is still being built.
    "pause": 2.0,
    "use_llm": True,
    # Which fingerspelling model the app uses. "letters" is the one trained in
    # this project from the signer's own recordings; "action" is the notebook's
    # action.h5. See /api/fingerspell for what each one measured.
    "spell_engine": "letters",
}
# The signer's own first take, kept as the reference clip for the ~100 signs no
# open dataset covers. It is the most honest reference we can show: it is exactly
# what the model was trained on.
OWN_CLIPS = ML_DIR / "data" / "reference_self"
DATA_DIR = HERE / "data"
SESSIONS_FILE = DATA_DIR / "sessions.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
STATE = {"model": None, "letters": None, "action": None}
TRAINING = {"running": False, "last": None, "log": "", "started": 0}

# Phrases a doctor says most often. Each one lists the signs we can actually show
# from the reference library, so "Show in ISL" never promises a sign we lack.
MEDICAL_PHRASES = [
    {"text": "Where does it hurt?", "signs": ["bad"]},
    {"text": "Do you have a fever?", "signs": ["fever", "hot"]},
    {"text": "Do you have a cough?", "signs": ["cough"]},
    {"text": "Do you have a headache?", "signs": ["headache"]},
    {"text": "Is your throat sore?", "signs": ["sore_throat"]},
    {"text": "Do you have trouble breathing?", "signs": ["trouble_breathing"]},
    {"text": "Do you feel nauseous?", "signs": ["nausea"]},
    {"text": "Does your stomach hurt?", "signs": ["stomach_ache"]},
    {"text": "Do you feel tired?", "signs": ["tired"]},
    {"text": "Since when have you been sick?", "signs": ["sick", "today"]},
    {"text": "Take this medicine.", "signs": ["medicine"]},
    {"text": "Take the medicine in the morning and at night.",
     "signs": ["medicine", "morning", "night"]},
    {"text": "Please sit down on the bed.", "signs": ["bed"]},
    {"text": "The doctor is coming.", "signs": ["doctor"]},
    {"text": "Do you feel weak?", "signs": ["weak"]},
    {"text": "Are you feeling cold?", "signs": ["cold"]},
    {"text": "You are healthy now.", "signs": ["healthy", "good"]},
    {"text": "Come to the hospital tomorrow.", "signs": ["hospital", "tomorrow"]},
    {"text": "Thank you, get well soon.", "signs": ["thank_you", "good"]},
]

app = FastAPI(title="SignBridge AI", version="1.0")

import interpreter  # noqa: E402  (human-interpreter fallback feature)
app.include_router(interpreter.router)
# Who may call this API from a browser. Locally that is the Vite dev server;
# hosted, it is wherever the frontend was deployed, which is not known until it
# is deployed - so it comes from the environment:
#
#     ALLOWED_ORIGINS=https://signbridge.vercel.app,https://www.example.com
#
# A browser refuses a cross-origin response that does not name its own origin, so
# getting this wrong shows up as every request failing with nothing in the server
# log at all.
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
_origins += ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # Vercel gives every deployment its own subdomain; without this only the
    # production URL would work and every preview build would be blocked.
    allow_origin_regex=os.environ.get("ALLOWED_ORIGIN_REGEX",
                                      r"https://.*\.vercel\.app"),
    allow_methods=["*"], allow_headers=["*"],
)


def model():
    if STATE["model"] is None:
        STATE["model"] = Recogniser(str(CONFIG["model_path"]))
    return STATE["model"]


def letter_model():
    """Fingerspelling recogniser, loaded on first use (None if not trained yet)."""
    if LetterRecogniser is None:
        return None
    if STATE["letters"] is None:
        path = SPELL_DIR / "models" / "letters.pt"
        if not path.exists():
            return None
        STATE["letters"] = LetterRecogniser(path)
    return STATE["letters"]


def action_model():
    """The notebook's model, with its own MediaPipe Hands graph.

    Built only when someone actually spells with it and closed again afterwards:
    it is a second MediaPipe graph next to the Holistic one the words need, and
    on a machine with a gigabyte free the second graph is the one that fails.
    """
    if ActionSpeller is None:
        return None
    if STATE["action"] is None:
        path = SPELL_DIR / "action.h5"
        if not path.exists():
            return None
        try:
            STATE["action"] = ActionSpeller(path)
        except Exception:
            return None
    return STATE["action"]


def release_action_model():
    speller = STATE.get("action")
    if speller is not None:
        speller.close()
        STATE["action"] = None


def spell_engine(name=None):
    """Whichever letter model is selected: 'letters' (PyTorch) or 'action' (h5)."""
    choice = name or CONFIG.get("spell_engine", "letters")
    if choice == "action":
        speller = action_model()
        if speller is not None:
            return speller, "action"
    speller = letter_model()
    if speller is not None:
        return speller, "letters"
    return None, choice


def new_holistic():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5, model_complexity=1)


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1), encoding="utf-8")


def settings():
    stored = read_json(SETTINGS_FILE, {})
    return {"threshold": CONFIG["threshold"], "pause": CONFIG["pause"],
            "use_llm": CONFIG["use_llm"], "speech": True, "language": "en-IN",
            "large_text": False, "high_contrast": False, "reduced_motion": False,
            **stored}


def apply_saved_settings():
    """Put the stored settings into CONFIG at startup.

    Without this they only took effect after someone pressed Save in the UI again:
    the file said 0.5 while the running server used its own default, so the
    threshold shown in Settings was not the one deciding anything."""
    stored = read_json(SETTINGS_FILE, {})
    for key in ("threshold", "pause"):
        if key in stored:
            CONFIG[key] = float(stored[key])
    if "use_llm" in stored:
        CONFIG["use_llm"] = bool(stored["use_llm"])


apply_saved_settings()


def read_gallery():
    path = Path(CONFIG["gallery"])
    if not path.exists():
        raise HTTPException(503, "gallery not built: run live/make_gallery.py")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- API

@app.get("/api/health")
def health():
    rec = model()
    return {"status": "ok", "signs": [l for l in rec.labels if l != "idle"],
            "classes": len(rec.labels), "threshold": CONFIG["threshold"],
            "pause_seconds": CONFIG["pause"],
            "validation_accuracy": rec.val_accuracy}


@app.get("/api/gallery")
def gallery():
    return JSONResponse(read_gallery())


@app.get("/api/poster/{word}")
def poster(word: str):
    path = Path(CONFIG["posters"]) / f"{Path(word).name}.jpg"
    if not path.exists():
        raise HTTPException(404, "no poster for that sign")
    return FileResponse(path, media_type="image/jpeg")


MEDIA = {".webm": "video/webm", ".mp4": "video/mp4", ".MOV": "video/mp4"}


def own_clip(word: str):
    """The signer's own reference take for a sign, if one was recorded."""
    folder = OWN_CLIPS / Path(word).name
    if not folder.is_dir():
        return None
    clips = sorted(folder.glob("*.webm")) + sorted(folder.glob("*.mp4"))
    return clips[0] if clips else None


@app.get("/api/clip/{word}")
def clip(word: str):
    """Reference video. INCLUDE clips are H.264 inside .MOV, which browsers play
    when served as video/mp4, so nothing is re-encoded."""
    mine = own_clip(word)
    entry = next((e for e in read_gallery() if e["word"] == word), None)
    if entry is None:
        if mine is None:
            raise HTTPException(404, "unknown sign")
        return FileResponse(mine, media_type=MEDIA.get(mine.suffix, "video/mp4"))
    folder = ML_DIR / "data" / entry.get("dir", "videos") / word
    path = folder / entry["file"]
    if not path.exists():
        raise HTTPException(404, "video file missing")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/teach/video")
async def teach_video(word: str = Form(...), file: UploadFile = File(...)):
    """Keep a take as the reference clip for a sign no dataset covers.

    The browser records it with MediaRecorder, so whatever it sends is a format
    that same browser can play back. Only the first take is kept - it is meant as
    a reminder of how the sign was performed, not a second copy of the data."""
    name = re.sub(r"[^a-z0-9_]", "", (word or "").lower())
    if not name:
        raise HTTPException(400, "which sign?")
    if name not in sentence_builder.LEXICON and name not in set(model().labels):
        raise HTTPException(404, "not a sign this project knows")
    blob = await file.read()
    if len(blob) < 2048:
        raise HTTPException(400, "recording too short to keep")
    if len(blob) > 20_000_000:
        raise HTTPException(413, "recording too large")
    suffix = ".webm" if "webm" in (file.content_type or "") else ".mp4"
    folder = OWN_CLIPS / name
    folder.mkdir(parents=True, exist_ok=True)
    existing = own_clip(name)
    if existing is not None:
        return {"word": name, "kept": False, "reason": "a reference take already exists"}
    path = folder / f"take{suffix}"
    path.write_bytes(blob)
    return {"word": name, "kept": True, "bytes": len(blob)}


@app.delete("/api/teach/video/{word}")
def teach_video_delete(word: str):
    """Recorded a bad reference take? Drop it and record another."""
    path = own_clip(re.sub(r"[^a-z0-9_]", "", word.lower()))
    if path is None:
        raise HTTPException(404, "no own take for that sign")
    path.unlink()
    return {"word": word, "deleted": True}


@app.post("/api/video")
async def video(file: UploadFile = File(...)):
    """Same pipeline on an uploaded clip: the safe fallback for a demo."""
    rec = model()
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    tmp = Path(tempfile.mkdtemp()) / f"upload{suffix}"
    tmp.write_bytes(await file.read())

    holistic = new_holistic()
    spotter = SignSpotter()
    glosses, segments, alternatives = [], [], {}
    cap = cv2.VideoCapture(str(tmp))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        segment = spotter.update(landmarks_to_row(holistic.process(rgb)))
        if segment is not None:
            top = rec.predict(segment, top_k=3)
            label, prob = top[0]
            segments.append({"label": label, "prob": round(prob, 3),
                             "alternatives": [l for l, _ in top[1:]]})
            if label != "idle" and prob >= CONFIG["threshold"]:
                if not glosses or glosses[-1] != label:
                    alternatives[len(glosses)] = [l for l, _ in top[1:]]
                    glosses.append(label)
    cap.release()
    holistic.close()

    text, source = sentence_builder.build(glosses, use_llm=CONFIG["use_llm"],
                                          alternatives=alternatives)
    return {"glosses": glosses, "sentence": text, "source": source, "segments": segments}


@app.get("/api/sessions")
def list_sessions(limit: int = 100, mode: str = "all"):
    items = read_json(SESSIONS_FILE, [])
    if mode != "all":
        items = [i for i in items if i.get("mode") == mode]
    return list(reversed(items))[:limit]


@app.post("/api/sessions")
async def add_session(entry: dict):
    """Append one finished translation so History and the dashboard show real data."""
    items = read_json(SESSIONS_FILE, [])
    record = {
        "id": f"s{int(time.time() * 1000)}",
        "at": time.strftime("%Y-%m-%d %H:%M"),
        "mode": entry.get("mode", "general"),
        "glosses": entry.get("glosses", []),
        "sentence": entry.get("sentence", ""),
        "source": entry.get("source", ""),
        "seconds": round(float(entry.get("seconds", 0)), 1),
    }
    items.append(record)
    write_json(SESSIONS_FILE, items[-500:])
    return record


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    items = read_json(SESSIONS_FILE, [])
    kept = [i for i in items if i.get("id") != session_id]
    write_json(SESSIONS_FILE, kept)
    return {"deleted": len(items) - len(kept)}


@app.get("/api/stats")
def stats():
    """Dashboard counters. Everything here is measured, nothing is invented."""
    items = read_json(SESSIONS_FILE, [])
    rec = model()
    signs_used = {g for i in items for g in i.get("glosses", [])}
    return {
        "sessions": len(items),
        "signs_recognised": sum(len(i.get("glosses", [])) for i in items),
        "vocabulary": len([l for l in rec.labels if l != "idle"]),
        "vocabulary_used": len(signs_used),
        "model_accuracy": round((rec.val_accuracy or 0) * 100, 1),
        "live_accuracy": 95.4,          # eval_live.py, 1185 clips, 86 signs, th 0.35
        "by_mode": {m: sum(1 for i in items if i.get("mode") == m)
                    for m in ("general", "healthcare", "learning")},
        "recent": list(reversed(items))[:5],
    }


@app.get("/api/settings")
def get_settings():
    return settings()


@app.put("/api/settings")
async def put_settings(update: dict):
    current = settings()
    current.update({k: v for k, v in update.items() if k in current})
    write_json(SETTINGS_FILE, current)
    CONFIG["threshold"] = float(current["threshold"])
    CONFIG["pause"] = float(current["pause"])
    CONFIG["use_llm"] = bool(current["use_llm"])
    return current


@app.get("/api/phrases")
def phrases():
    known = {e["word"] for e in read_gallery()}
    return [{**p, "signs": [s for s in p["signs"] if s in known]} for p in MEDICAL_PHRASES]


# Teaching order: grammar words unlock whole sentence shapes, so they come first.
# A sign's own "tier" in the vocabulary wins; this is the fallback by kind.
TEACH_TIER = {"answer": 1, "negation": 1, "verb": 1, "polite": 1,
              "question": 2, "pronoun": 2, "command": 1, "condition": 2, "info": 2,
              "symptom": 3, "thing": 3, "person": 3, "place": 3, "body": 3,
              "state": 4, "time": 4, "severity": 4, "duration": 4, "modal": 2,
              "possessive": 2}
TIER_NAME = {1: "Teach these first - the demo needs them",
             2: "Questions, people and history",
             3: "Things, places, symptoms and body parts",
             4: "Extra detail",
             5: "Nice to have"}


def recorded_counts():
    root = ML_DIR / "recordings"
    counts = {}
    if root.exists():
        for d in root.iterdir():
            if d.is_dir():
                n = len(list(d.glob("*.npz")))
                if n:
                    counts[d.name] = n
    return counts


@app.get("/api/teach")
def teach_status():
    """Everything teachable: what the model knows, what has takes recorded, and
    every word the sentence engine is already prepared for.

    The vocabulary file is the source of truth, so a sign with no reference video
    (WANT, NO, WHERE ...) is still offered here - those are exactly the words the
    model is missing."""
    rec = model()
    in_model = {l for l in rec.labels if l != "idle"}
    videos = {g["word"] for g in read_gallery()}
    own = {d.name for d in OWN_CLIPS.iterdir() if own_clip(d.name)} if OWN_CLIPS.is_dir() else set()
    counts = recorded_counts()

    signs = []
    for word, (kind, entry) in sentence_builder.LEXICON.items():
        signs.append({
            "word": word, "kind": kind,
            "in_model": word in in_model,
            "has_video": word in videos or word in own,
            "own_video": word in own,
            "takes": counts.get(word, 0),
            "tier": int(entry.get("tier", TEACH_TIER.get(kind, 4))),
        })
    # anything recorded or known that the vocabulary does not list yet
    for word in sorted((in_model | set(counts)) - {s["word"] for s in signs}):
        signs.append({"word": word, "kind": "other", "in_model": word in in_model,
                      "has_video": word in videos or word in own,
                      "own_video": word in own, "takes": counts.get(word, 0),
                      "tier": 4})

    signs.sort(key=lambda s: (s["in_model"], s["tier"], s["word"]))
    return {"signs": signs, "counts": counts, "recommended": 5,
            "tiers": TIER_NAME, "training": TRAINING["running"], "last": TRAINING["last"],
            "totals": {"in_model": len(in_model),
                       "missing": sum(1 for s in signs if not s["in_model"]),
                       "takes": sum(counts.values())}}


@app.post("/api/retrain")
async def retrain(payload: dict | None = None):
    """Retrain on INCLUDE plus everything taught here, then hot-swap the model.

    This is how the demo adapts to a new camera, a new signer, and to signs that
    INCLUDE does not contain at all (the symptom vocabulary).
    """
    if TRAINING["running"]:
        raise HTTPException(409, "training already running")
    # 300 epochs with 87 classes; 80 left the model underfitted, which showed up as
    # signs that used to work going quiet after a new one was taught. On this
    # machine it takes about five minutes.
    epochs = int((payload or {}).get("epochs", 300))
    device = (payload or {}).get("device", "cpu")   # the server already holds the GPU

    def run():
        TRAINING.update({"running": True, "started": time.time(), "log": ""})

        class Tee:
            """Training prints its progress; forward it to the UI line by line."""
            def write(self, text):
                if text.strip():
                    TRAINING["log"] = (TRAINING["log"] + text)[-8000:]
                sys.__stdout__.write(text)

            def flush(self):
                sys.__stdout__.flush()

        try:
            # Free the GPU first: on a 4 GB card the server's own CUDA context is
            # enough to starve training.
            STATE["model"] = None
            import torch
            torch.cuda.empty_cache()

            import train as trainer
            argv = ["--epochs", str(epochs), "--device", device,
                    "--out", str(CONFIG["model_path"])]
            with contextlib.redirect_stdout(Tee()):
                result = trainer.main(argv)
            TRAINING["last"] = {
                "at": time.strftime("%Y-%m-%d %H:%M"), "ok": True,
                "summary": f"{result['clips']} clips, {len(result['labels'])} classes, "
                           f"validation {result['val_accuracy'] * 100:.1f}%"
                           + (f" - GOT WORSE: {', '.join(result['regressed'][:4])}"
                              if result.get("regressed") else ""),
                "regressed": result.get("regressed", []),
                "seconds": round(time.time() - TRAINING["started"], 1)}
            model()                                   # load the new weights
        except BaseException as e:                    # SystemExit too: argparse exits
            TRAINING["log"] += f"{chr(10)}{type(e).__name__}: {e}"
            TRAINING["last"] = {"at": time.strftime("%Y-%m-%d %H:%M"), "ok": False,
                                "summary": f"{type(e).__name__}: {e}",
                                "seconds": round(time.time() - TRAINING["started"], 1)}
        finally:
            TRAINING["running"] = False
            STATE["model"] = None                     # reload lazily either way

    threading.Thread(target=run, daemon=True).start()
    return {"started": True}


@app.post("/api/retrain/revert")
def retrain_revert():
    """Put the previous model back.

    Every retrain copies the model it replaces to models/signs_previous.pt first, so
    a retrain that made things worse an hour before a demo is one click to undo."""
    if TRAINING["running"]:
        raise HTTPException(409, "training is running")
    current = Path(CONFIG["model_path"])
    backup = current.with_name(current.stem + "_previous.pt")
    if not backup.exists():
        raise HTTPException(404, "no previous model to go back to")
    spare = current.with_name(current.stem + "_reverted.pt")
    shutil.copy2(current, spare)          # so the revert itself can be undone
    shutil.copy2(backup, current)
    STATE["model"] = None
    rec = model()
    TRAINING["last"] = {"at": time.strftime("%Y-%m-%d %H:%M"), "ok": True,
                        "summary": f"reverted to the previous model "
                                   f"({len(rec.labels) - 1} signs)", "seconds": 0}
    return {"reverted": True, "signs": len(rec.labels) - 1,
            "val_accuracy": rec.val_accuracy}


@app.get("/api/retrain/status")
def retrain_status():
    rec = model()
    return {"running": TRAINING["running"], "last": TRAINING["last"],
            "log": TRAINING["log"][-2000:],
            "signs": len([l for l in rec.labels if l != "idle"])}


@app.get("/api/fingerspell")
def fingerspell_status():
    """Which letter models exist, how good each one measured, and which is in use."""
    engines = []

    letters = letter_model()
    if letters is not None:
        engines.append({
            "id": "letters", "name": "Letters model (this project)",
            "letters": letters.labels, "count": len(letters.labels),
            "accuracy": letters.val_accuracy, "samples": letters.samples,
            "measured_on": "held-out frames from the end of each of your "
                           "recordings (train_letters.py --split time)",
        })

    action_path = SPELL_DIR / "action.h5"
    if ActionSpeller is not None and action_path.exists():
        entry = {"id": "action", "name": "action.h5 (Action Detection Refined.ipynb)",
                 "count": 36, "accuracy": None, "samples": None,
                 "measured_on": "not measured on held-out data: the MP_Data folder "
                                "it trained on is not in this project. Run "
                                "fingerspell/action_check.py to measure it on your "
                                "own camera."}
        loaded = STATE.get("action")
        if loaded is not None:
            entry["letters"] = loaded.labels
        return_entry = entry
        engines.append(return_entry)

    if not engines:
        return {"available": False,
                "reason": "no letter model yet: record letters and run "
                          "fingerspell/train_letters.py"}

    current = CONFIG.get("spell_engine", "letters")
    if not any(e["id"] == current for e in engines):
        current = engines[0]["id"]
    active = next(e for e in engines if e["id"] == current)
    return {"available": True, "engine": current, "engines": engines,
            # kept so older pages keep working
            "letters": active.get("letters", []),
            "validation_accuracy": active.get("accuracy"),
            "samples": active.get("samples")}


@app.put("/api/fingerspell")
def fingerspell_choose(payload: dict):
    """Switch letter model. The graphs are heavy, so the unused one is freed."""
    choice = (payload or {}).get("engine", "letters")
    if choice not in ("letters", "action"):
        raise HTTPException(400, "engine must be 'letters' or 'action'")
    if choice == "action" and action_model() is None:
        raise HTTPException(503, "action.h5 could not be loaded")
    if choice != "action":
        release_action_model()
    CONFIG["spell_engine"] = choice
    stored = read_json(SETTINGS_FILE, {})
    stored["spell_engine"] = choice
    write_json(SETTINGS_FILE, stored)
    return {"engine": choice}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    rec = model()
    holistic = new_holistic()
    spotter = SignSpotter()
    glosses, sentence_text, last_pred, source = [], "", None, ""
    last_sign, fps_mark, fps_count, fps = time.time(), time.time(), 0, 0.0
    teaching, teach_rows = None, []          # sign being taught, and its frames
    finished = []                            # signs behind the sentence on screen
    speller = None                           # set while fingerspelling a name
    gloss_times = {}                         # when each sign entered this sentence
    alternatives = {}                        # index -> runner-up guesses, for the LLM
    await ws.send_text(json.dumps({"type": "hello",
                                   "vocab": [l for l in rec.labels if l != "idle"]}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            spoke = False

            if msg.get("cmd") == "clear":
                glosses, sentence_text, source, finished = [], "", "", []
                gloss_times, alternatives = {}, {}
                continue
            elif msg.get("cmd") == "spell_start":
                speller, engine = spell_engine(msg.get("engine"))
                if speller is None:
                    await ws.send_text(json.dumps({"type": "spell_unavailable"}))
                else:
                    speller.reset()
                    await ws.send_text(json.dumps({"type": "spelling", "letters": [],
                                                   "word": "", "current": None,
                                                   "engine": engine}))
                continue
            elif msg.get("cmd") == "spell_backspace":
                if speller:
                    speller.backspace()
                    await ws.send_text(json.dumps({"type": "spelling",
                                                   "letters": speller.letters,
                                                   "word": speller.word, "current": None}))
                continue
            elif msg.get("cmd") == "spell_stop":
                word = speller.word if speller else ""
                letters = list(speller.letters) if speller else []
                if speller:
                    speller.reset()
                if isinstance(speller, object) and ActionSpeller is not None \
                        and isinstance(speller, ActionSpeller):
                    release_action_model()      # free the second MediaPipe graph
                speller = None
                await ws.send_text(json.dumps({"type": "spelled", "word": word,
                                               "letters": letters}))
                continue
            elif msg.get("cmd") == "teach_start":
                teaching, teach_rows = msg.get("word"), []
            elif msg.get("cmd") == "teach_stop":
                saved = 0
                if teaching and len(teach_rows) >= 8:
                    out_dir = ML_DIR / "recordings" / teaching
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    np.savez_compressed(out_dir / f"web_{stamp}.npz",
                                        keypoints=np.asarray(teach_rows, dtype=np.float32),
                                        label=teaching, signer=msg.get("signer", "web"),
                                        fps=14.0)
                    saved = len(list(out_dir.glob("*.npz")))
                await ws.send_text(json.dumps({"type": "taught", "word": teaching,
                                               "frames": len(teach_rows), "takes": saved}))
                teaching, teach_rows = None, []
                continue
            elif msg.get("cmd") == "finish":
                # the sign still in the signer's hands counts too
                leftover = spotter.flush()
                if leftover is not None:
                    label, prob = rec.predict(leftover, top_k=1)[0]
                    if label != "idle" and prob >= CONFIG["threshold"] and (
                            not glosses or glosses[-1] != label):
                        glosses.append(label)
                        gloss_times[label] = time.time()
                if glosses:
                    sentence_text, source = sentence_builder.build(
                        glosses, use_llm=CONFIG["use_llm"])
                    finished = list(glosses)
                    glosses, spoke = [], True
            elif msg.get("frame"):
                raw = base64.b64decode(msg["frame"].split(",", 1)[1])
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    # The two letter models want different things: the project's
                    # own reads the 300-number Holistic row, while action.h5 runs
                    # its own MediaPipe Hands over the frame itself. Giving the
                    # second one a row would be 126 numbers of nonsense.
                    if speller is not None and ActionSpeller is not None                             and isinstance(speller, ActionSpeller):
                        committed, current, prob = speller.update(frame)
                        await ws.send_text(json.dumps({
                            "type": "spelling", "letters": speller.letters,
                            "word": speller.word, "committed": committed,
                            "engine": "action",
                            "current": {"label": current, "prob": round(prob, 3)}
                                       if current else None}))
                        continue
                    row = landmarks_to_row(holistic.process(rgb))
                    if speller is not None:
                        committed, current, prob = speller.update(row)
                        await ws.send_text(json.dumps({
                            "type": "spelling", "letters": speller.letters,
                            "word": speller.word, "committed": committed,
                            "current": {"label": current, "prob": round(prob, 3)}
                                       if current else None}))
                        continue
                    if teaching is not None:
                        teach_rows.append(row)
                        await ws.send_text(json.dumps({"type": "teaching",
                                                       "word": teaching,
                                                       "frames": len(teach_rows)}))
                        continue
                    segment = spotter.update(row, now=time.time())
                    if segment is not None:
                        top = rec.predict(segment, top_k=3)
                        label, prob = top[0]
                        last_pred = {"label": label, "prob": round(prob, 3)}
                        if label != "idle" and prob >= CONFIG["threshold"]:
                            now = time.time()
                            # A sentence almost never contains the same sign twice in
                            # quick succession; without this, the hand settling after a
                            # sign is often read as that sign again.
                            recent = now - gloss_times.get(label, 0) < REPEAT_WINDOW
                            if not (glosses and (glosses[-1] == label or
                                                 (label in glosses and recent))):
                                alternatives[len(glosses)] = [l for l, _ in top[1:]]
                                glosses.append(label)
                                gloss_times[label] = now
                                # Show the sentence as it grows. Waiting for the pause
                                # meant the screen showed the previous sentence while
                                # the signer was already making the next sign.
                                sentence_text, _ = sentence_builder.build(
                                    glosses, use_llm=False)
                                source = "building"
                            elif not glosses:
                                alternatives[0] = [l for l, _ in top[1:]]
                                glosses.append(label)
                                gloss_times[label] = now
                            last_sign = now
                    # A sign held while the pause runs out was being thrown away;
                    # take it before the sentence is closed.
                    if (glosses and spotter.active
                            and time.time() - last_sign > CONFIG["pause"]):
                        leftover = spotter.flush()
                        if leftover is not None:
                            label, prob = rec.predict(leftover, top_k=1)[0]
                            if (label != "idle" and prob >= CONFIG["threshold"]
                                    and glosses[-1] != label):
                                glosses.append(label)
                                gloss_times[label] = time.time()
                    if glosses and time.time() - last_sign > CONFIG["pause"] \
                            and spotter.state == "waiting":
                        sentence_text, source = sentence_builder.build(
                            glosses, use_llm=CONFIG["use_llm"],
                            alternatives=alternatives, context=msg.get("mode", ""))
                        finished = list(glosses)    # keep them on screen with the sentence
                        glosses, spoke, last_sign = [], True, time.time()
                        gloss_times, alternatives = {}, {}
                    fps_count += 1
                    if time.time() - fps_mark >= 1.0:
                        fps = fps_count / (time.time() - fps_mark)
                        fps_mark, fps_count = time.time(), 0

            await ws.send_text(json.dumps({
                "type": "update", "glosses": glosses, "sentence": sentence_text,
                "source": source, "state": spotter.state, "last": last_pred,
                "said": finished, "fps": round(fps, 1), "speak": spoke}))
    except WebSocketDisconnect:
        pass
    finally:
        holistic.close()


# Serve the built React app when it exists, so the demo runs from one URL.
# React Router owns the paths, so any non-API request falls back to index.html;
# without that, opening /learn directly (or refreshing it) would 404.
if not FRONTEND_DIST.exists():
    # Deployed, the page lives on Vercel and this process only serves the API -
    # and a bare 404 at / makes a working backend look broken. Say what this is
    # and whether the model actually loaded.
    @app.get("/", response_class=HTMLResponse)
    def index_placeholder():
        try:
            signs = len(model().labels) - 1
            status = f"<p><b>Model loaded:</b> {signs} signs.</p>"
        except Exception as error:
            status = f"<p><b>The model did not load:</b> {error}</p>"
        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>SignBridge API</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:40rem;margin:4rem auto;
padding:0 1rem;color:#0d1b2a}} code{{background:#eef2f6;padding:2px 6px;border-radius:5px}}
a{{color:#0f9b8e}}</style></head><body>
<h1>SignBridge AI — API</h1>
<p>This is the backend. The web app is deployed separately; point it here with
<code>VITE_API_BASE</code>.</p>
{status}
<p>Try <a href="/api/health">/api/health</a>, <a href="/api/gallery">/api/gallery</a>,
<a href="/docs">/docs</a>.</p>
<p>To serve the app from this process instead, build the frontend
(<code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>) and restart.</p>
</body></html>"""


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html", media_type="text/html")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # A host that runs this (Render, Fly, a VM) sets PORT and expects the process
    # to listen on 0.0.0.0. Not doing so is the usual reason a deployment is
    # reported unhealthy while its own logs look perfectly fine.
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    p.add_argument("--model", default=None, help="override the model path")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--no-llm", action="store_true", help="always use the rule-based sentences")
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    if args.model:
        CONFIG["model_path"] = Path(args.model)
    if args.threshold is not None:
        CONFIG["threshold"] = args.threshold
    if args.no_llm:
        CONFIG["use_llm"] = False
    if not Path(CONFIG["model_path"]).exists():
        sys.exit(f"model not found: {CONFIG['model_path']}\n"
                 f"train one first:  python {ML_DIR / 'train.py'}")

    # A port left behind by an earlier run is the most common start-up failure,
    # and uvicorn only reports it as a raw socket error. Say what to do instead.
    import socket
    probe = socket.socket()
    try:
        probe.bind((args.host, args.port))
    except OSError:
        stop = ('powershell "Get-NetTCPConnection -LocalPort %d -State Listen | '
                'ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"' % args.port)
        sys.exit(chr(10).join([
            "Port %d is already in use - SignBridge is probably still running." % args.port,
            "  See it:   http://%s:%d" % (args.host, args.port),
            "  Stop it:  %s" % stop,
            "  Or use another port:  python app.py --port %d" % (args.port + 1),
        ]))
    finally:
        probe.close()

    model()      # load before serving so the first request is fast
    print(f"SignBridge AI backend on http://{args.host}:{args.port}")
    print("frontend:", "built app served from /" if FRONTEND_DIST.exists()
          else "not built yet — run the Vite dev server (npm run dev)")

    import uvicorn
    uvicorn.run("app:app" if args.reload else app, host=args.host, port=args.port,
                reload=args.reload, log_level="warning")


if __name__ == "__main__":
    main()
