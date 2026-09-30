"""Web app: browser camera -> FastAPI -> signs -> English sentence (+ speech).

The browser only captures frames; MediaPipe, the classifier and the sentence
builder all run in Python, exactly as in training. That keeps live input identical
to the training data, which is what makes the model work outside the dataset.

    python server.py                 # then open http://127.0.0.1:8000

Pages live in static/ and follow the SignBridge AI design system (DESIGN.md):
    /        camera, live translation, doctor's reply
    /learn   every sign as a reference video to copy
"""

import argparse
import base64
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import sentence as sentence_builder  # noqa: E402
from live_demo import Recogniser, SignSpotter  # noqa: E402
from record import landmarks_to_row  # noqa: E402

app = FastAPI(title="ISL live translator")
VIDEO_DIR = HERE / "data" / "videos"
STATIC = HERE / "static"
STATE = {"model": None, "use_llm": True}


def get_model():
    if STATE["model"] is None:
        STATE["model"] = Recogniser(STATE["model_path"])
    return STATE["model"]


def new_holistic():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5, model_complexity=1)


@app.get("/")
def index():
    return FileResponse(STATIC / "app.html", media_type="text/html")


@app.get("/learn")
def learn():
    return FileResponse(STATIC / "learn.html", media_type="text/html")


@app.get("/static/{name}")
def static_file(name: str):
    """Stylesheet and any other page asset (see DESIGN.md for the design system)."""
    path = STATIC / Path(name).name
    if not path.exists():
        raise HTTPException(404, "not found")
    types = {".css": "text/css", ".js": "text/javascript", ".json": "application/json",
             ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg"}
    return FileResponse(path, media_type=types.get(path.suffix, "application/octet-stream"))


@app.get("/gallery")
def gallery():
    path = STATIC / "gallery.json"
    if not path.exists():
        raise HTTPException(404, "run make_gallery.py first")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/poster/{word}")
def poster(word: str):
    path = STATIC / "posters" / f"{Path(word).name}.jpg"
    if not path.exists():
        raise HTTPException(404, "no poster")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/clip/{word}")
def clip(word: str):
    """The reference video. INCLUDE clips are H.264 in .MOV, which browsers play
    when served as video/mp4, so no conversion is needed."""
    entries = json.loads((STATIC / "gallery.json").read_text(encoding="utf-8"))
    entry = next((e for e in entries if e["word"] == word), None)
    if entry is None:
        raise HTTPException(404, "unknown sign")
    path = VIDEO_DIR / word / entry["file"]
    if not path.exists():
        raise HTTPException(404, "video missing")
    return FileResponse(path, media_type="video/mp4")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    rec = get_model()
    holistic = new_holistic()
    spotter = SignSpotter()
    glosses, sentence_text, last_pred = [], "", None
    last_sign, fps_mark, fps_count, fps = time.time(), time.time(), 0, 0.0
    await ws.send_text(json.dumps({"vocab": [l for l in rec.labels if l != "idle"]}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            spoke = False
            if msg.get("cmd") == "clear":
                glosses, sentence_text = [], ""
            elif msg.get("cmd") == "finish":
                if glosses:
                    sentence_text, _ = sentence_builder.build(glosses, use_llm=STATE["use_llm"])
                    glosses, spoke = [], True
            elif msg.get("frame"):
                raw = base64.b64decode(msg["frame"].split(",", 1)[1])
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    segment = spotter.update(landmarks_to_row(holistic.process(rgb)))
                    if segment is not None:
                        label, prob = rec.predict(segment)[0]
                        last_pred = {"label": label, "prob": prob}
                        if label != "idle" and prob >= STATE["threshold"]:
                            if not glosses or glosses[-1] != label:
                                glosses.append(label)
                            last_sign = time.time()
                    if glosses and time.time() - last_sign > STATE["pause"] \
                            and spotter.state == "waiting":
                        sentence_text, _ = sentence_builder.build(glosses, use_llm=STATE["use_llm"])
                        glosses, spoke, last_sign = [], True, time.time()
                    fps_count += 1
                    if time.time() - fps_mark >= 1.0:
                        fps, fps_mark, fps_count = fps_count / (time.time() - fps_mark), time.time(), 0
            await ws.send_text(json.dumps({"glosses": glosses, "sentence": sentence_text,
                                           "state": spotter.state, "last": last_pred,
                                           "fps": fps, "spoke": spoke}))
    except WebSocketDisconnect:
        pass
    finally:
        holistic.close()


@app.post("/video")
async def video(file: UploadFile = File(...)):
    """Same pipeline on an uploaded clip (the safe fallback demo)."""
    rec = get_model()
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    tmp = Path(tempfile.mkdtemp()) / f"upload{suffix}"
    tmp.write_bytes(await file.read())
    holistic = new_holistic()
    spotter = SignSpotter()
    glosses, detail = [], []
    cap = cv2.VideoCapture(str(tmp))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        segment = spotter.update(landmarks_to_row(holistic.process(rgb)))
        if segment is not None:
            label, prob = rec.predict(segment)[0]
            detail.append({"label": label, "prob": round(prob, 3)})
            if label != "idle" and prob >= STATE["threshold"]:
                if not glosses or glosses[-1] != label:
                    glosses.append(label)
    cap.release()
    holistic.close()
    text, src = sentence_builder.build(glosses, use_llm=STATE["use_llm"])
    return JSONResponse({"glosses": glosses, "sentence": text, "source": src,
                         "segments": detail})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(HERE / "models" / "signs.pt"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--pause", type=float, default=2.0)
    p.add_argument("--no-llm", action="store_true")
    args = p.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"model not found: {args.model}\nTrain one first:  python train.py")
    STATE.update({"model_path": args.model, "threshold": args.threshold,
                  "pause": args.pause, "use_llm": not args.no_llm})
    get_model()
    import uvicorn
    print(f"open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
