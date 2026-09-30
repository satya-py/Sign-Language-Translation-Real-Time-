"""Prepare the "learn the signs" page data: one reference clip + poster per word.

The INCLUDE clips are H.264 inside .MOV, which browsers play when served as
video/mp4, so nothing is re-encoded. We only pick the shortest clip per word (fast
to load) and save a small poster frame so the page stays light until a video is
clicked.

    python make_gallery.py
"""

import json
from pathlib import Path

import cv2

HERE = Path(__file__).parent
VIDEO_DIR = HERE / "data" / "videos"
# Signs we can demonstrate but cannot yet recognise: only one demo video exists,
# which is far too little to train a class. They appear in the library marked
# "teach me", and become recognisable once a few takes are recorded.
REFERENCE_ONLY_DIR = HERE / "data" / "reference_only"
# Signs the signer recorded themselves, kept as the reference for the words no open
# dataset covers (PAIN, HEAD, WATER, WHERE ...). See backend /api/teach/video.
OWN_DIR = HERE / "data" / "reference_self"
STATIC = HERE / "static"
POSTERS = STATIC / "posters"


def main():
    POSTERS.mkdir(parents=True, exist_ok=True)
    model_labels = None
    model_path = HERE / "models" / "signs.pt"
    if model_path.exists():
        import torch
        model_labels = [l for l in torch.load(model_path, map_location="cpu",
                                              weights_only=False)["labels"] if l != "idle"]

    word_dirs = []
    if VIDEO_DIR.exists():
        word_dirs += [(d, True) for d in sorted(VIDEO_DIR.iterdir()) if d.is_dir()]
    if REFERENCE_ONLY_DIR.exists():
        word_dirs += [(d, False) for d in sorted(REFERENCE_ONLY_DIR.iterdir()) if d.is_dir()]
    if OWN_DIR.exists():
        word_dirs += [(d, False) for d in sorted(OWN_DIR.iterdir()) if d.is_dir()]

    gallery = []
    for word_dir, trainable in word_dirs:
        word = word_dir.name
        in_model = bool(model_labels and word in model_labels)
        clips = (sorted(word_dir.glob("*.MOV")) + sorted(word_dir.glob("*.mp4"))
                 + sorted(word_dir.glob("*.webm")))
        if not clips:
            continue
        clip = min(clips, key=lambda p: p.stat().st_size)   # smallest = quickest to load

        cap = cv2.VideoCapture(str(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)            # middle of the sign
        ok, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        if ok:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (360, int(h * 360 / w)))
            cv2.imwrite(str(POSTERS / f"{word}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 75])
        if word_dir.parent == OWN_DIR:
            source = "your own take"
        elif trainable:
            source = "INCLUDE" if in_model else "INCLUDE reference"
        else:
            source = "symptom demo"
        gallery.append({"word": word, "file": clip.name, "dir": word_dir.parent.name,
                        "seconds": round(n / fps, 1), "poster": ok,
                        "in_model": in_model, "source": source})

    (STATIC / "gallery.json").write_text(json.dumps(gallery, indent=1), encoding="utf-8")
    missing = [g["word"] for g in gallery if not g["poster"]]
    recognised = sum(1 for g in gallery if g["in_model"])
    print(f"{len(gallery)} signs in the gallery ({recognised} recognised, "
          f"{len(gallery) - recognised} demonstration only) -> {STATIC / 'gallery.json'}")
    if missing:
        print("no poster for:", missing)
    if model_labels:
        absent = [l for l in model_labels if l not in {g['word'] for g in gallery}]
        if absent:
            print("model knows these but has no video:", absent)


if __name__ == "__main__":
    main()
