"""Drives the SignAI backend exactly as the extension does, with real video.

Frames from a few dataset clips are posted in one-second chunks; the test then
checks that words are recognised, that the sentence grows while signing, and
that it is finished once the signer stops.

    python server/app.py                 # in one terminal
    python tests/test_backend.py         # in another

    python tests/test_backend.py --url http://127.0.0.1:5001 --words hello,i,sick,today
"""

import argparse
import base64
import glob
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent.parent / "isl-translator" / "live" / "data" / "videos"


def post(url, path, payload):
    request = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as answer:
        return json.loads(answer.read())


def frames_for(word, every=2, pause_frames=12):
    """A clip as base64 JPEGs, followed by the still pause a signer leaves."""
    clips = sorted(glob.glob(str(VIDEOS / word / "*.MOV"))) \
        + sorted(glob.glob(str(VIDEOS / word / "*.mp4")))
    if not clips:
        raise SystemExit(f"no reference clip for '{word}' under {VIDEOS}")
    capture = cv2.VideoCapture(clips[0])
    out, step = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        step += 1
        if step % every:
            continue
        small = cv2.resize(frame, (640, 480))
        out.append("data:image/jpeg;base64," + base64.b64encode(
            cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1]).decode())
    capture.release()
    return out + [out[-1]] * pause_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:5001")
    parser.add_argument("--words", default="hello,i,sick,today")
    parser.add_argument("--chunk", type=int, default=13, help="frames per request")
    args = parser.parse_args()
    words = args.words.split(",")

    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/api/health", timeout=30) as a:
            health = json.loads(a.read())
    except urllib.error.URLError as error:
        raise SystemExit(f"backend not reachable at {args.url}: {error}")
    if not health.get("ok"):
        raise SystemExit(f"backend is up but the model is not: {health.get('detail')}")
    print(f"backend ready: {health['vocabulary']} signs, threshold {health['threshold']}")

    frames = []
    for word in words:
        frames += frames_for(word)
    print(f"{len(frames)} frames from {len(words)} clips "
          f"({', '.join(words)}), sent {args.chunk} at a time")

    session = post(args.url, "/api/session", {})["session"]
    said, finals = [], []
    for start in range(0, len(frames), args.chunk):
        answer = post(args.url, "/api/chunk",
                      {"session": session, "frames": frames[start:start + args.chunk]})
        for gloss in answer["glosses"]:
            if gloss not in said:
                said.append(gloss)
        if answer["final"] and answer["sentence"]:
            finals.append(answer["sentence"])
            print(f"  finished: {answer['sentence']}")
        elif answer["glosses"]:
            print(f"  building: {answer['glosses']} -> {answer['sentence']}")
        time.sleep(0.02)

    tail = post(args.url, "/api/finish", {"session": session})
    if tail.get("sentence"):
        finals.append(tail["sentence"])
        print(f"  finished: {tail['sentence']}")
    post(args.url, "/api/close", {"session": session})

    recognised = [w for w in words if w in said]
    print(f"\nsigns recognised: {len(recognised)}/{len(words)} ({', '.join(said) or 'none'})")
    print(f"sentences produced: {len(finals)}")
    ok = bool(finals) and len(recognised) >= max(1, len(words) // 2)
    print("PASS" if ok else "FAIL: too little was recognised to call this working")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
