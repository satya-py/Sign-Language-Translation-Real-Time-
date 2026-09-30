# Fingerspelling

Two models can spell letters here. They were built differently, they measure
differently, and the app lets you switch between them on the Fingerspell page.

| | `models/letters.pt` | `action.h5` |
|---|---|---|
| Built by | `train_letters.py` in this project | `Action Detection Refined.ipynb` |
| Framework | PyTorch | Keras (served here without TensorFlow, see below) |
| Classes | 37 (A–Z, 0–9, 10) | 36 (0–9, A–Z) |
| Input per frame | 94 shape features from a **Holistic** row | 126 raw coordinates from **MediaPipe Hands** |
| Training data | 1110 frames, 30 per letter, from your webcam | `MP_Data`, up to 200 per class — **not in this repository** |
| Measured | **94.6%** on held-out frames (`--split time`) | **not measured on held-out data** — see below |

## Why there are two, and which to trust

`letters.pt` learns from features that are scale- and position-free: each finger
point is measured from its own wrist and divided by the hand's size, and each
wrist is placed against the shoulders. That is why it survives you sitting closer
or further from the camera.

`action.h5` learns from the raw landmark coordinates as MediaPipe reports them,
which carry where your hand was on screen and how big it looked. That is simpler
and can work well at the distance it was recorded at, but it has no way to know
that the same letter closer to the lens is the same letter.

**The honest position on `action.h5`'s accuracy:** the notebook reports an
accuracy computed on a 10% split of frames drawn from the same sittings as the
training frames, which flatters any model trained this way — two frames of one
hold are nearly the same picture. The folder it trained on (`MP_Data`) is not in
this repository, so that number cannot be recomputed here.

What *can* be measured is how it does on frames of your hands recorded for the
other model. Feeding those through it, with the hand blocks in the order it
expects, gives **33.2%** over 930 frames of 36 letters. That number is a lower
bound, not a verdict: those frames come from MediaPipe **Holistic**, and
`action.h5` was trained on **Hands** landmarks — different networks, slightly
different points. It is evidence that the two do not agree, not proof of how the
model behaves on its own input.

To measure it properly, on your camera, in about three minutes:

```bash
cd "D:\new Sign language\isl-translator\fingerspell" && "D:\new Sign language\INCLUDE\venv\Scripts\python.exe" action_check.py
```

Hold each letter, press SPACE, and it judges ten frames of it. At the end it
prints per-letter accuracy and the confusions, and writes `action_check.json`.
Do the same for the other model with `check_letters.py`, and put both numbers in
the repository README — a measured number from your own camera is worth more than
either model's training-time claim.

## Serving `action.h5` without TensorFlow

The model is three dense layers:

```
126 -> Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dense(36, softmax)
```

Dropout does nothing at inference, so that is two matrix multiplies and a
softmax — about 25k multiply-adds, 0.002 ms a frame. Installing TensorFlow into
the project venv to perform them would cost about 500 MB and its own choice of
numpy version, on a machine that has under a gigabyte of memory free while the
app runs. `action_model.py` reads the weights out of the HDF5 file with h5py and
applies them with numpy instead.

That is not an approximation. Checked against Keras loading the same file, the
largest difference across a batch of inputs is **2.7e-10**, and the top class is
the same on every row.

## The two details that silently ruin either model

**Mirroring.** The notebook collected its data from the flipped preview
(`cv2.flip(frame, 1)` in cell 25), so `action.h5` expects a mirrored frame;
`action_infer.py` flips before MediaPipe sees the frame, and `record_letters.py`
in this project deliberately does not, because the web app sends the camera image
as it is. Getting this backwards swaps left and right hands and reads every
letter wrong while looking perfectly healthy in testing — this project shipped
that bug once, and it is why `mirror_dataset.py` exists.

**Which hand is which.** MediaPipe Hands labels handedness as if it were looking
in a mirror. On the same frames, its "Left" is Holistic's right. `action_model.py`
follows MediaPipe's own label, exactly as the notebook did.

## Files

| File | What it does |
|---|---|
| `action_model.py` | loads `action.h5` weights, runs them in numpy |
| `action_infer.py` | live spelling with it: MediaPipe Hands, steady-frame voting, repeat handling |
| `action_check.py` | measures it on your camera, letter by letter |
| `action_live.py` | the notebook's own live loop (needs `action_venv`, uses TensorFlow) |
| `letterfeat.py`, `train_letters.py`, `letter_infer.py` | the project's own letter model |
| `tune_letters.py` | picks the commit rule (window, votes, threshold) from data |
| `check_letters.py` | finds which letters are weak and why |
| `mirror_dataset.py` | one-off: flips recordings into the web app's orientation |

## Switching model in the app

The Fingerspell page has a **Which letter model** row; the choice is saved and
also available over the API:

```bash
curl -X PUT http://127.0.0.1:8000/api/fingerspell -H "Content-Type: application/json" -d "{\"engine\":\"action\"}"
```

Only one is loaded at a time. Each carries its own MediaPipe graph, and on this
machine a second graph is what runs the memory out.

## `action_venv`

`action_live.py` and the notebook need TensorFlow, which lives in `action_venv`.
Two package versions had to be pinned for it to import at all:

```bash
action_venv\Scripts\python.exe -m pip install "scipy<1.14" "keras==3.4.1"
```

scipy 1.14+ and keras 3.15 are built against numpy 2, while mediapipe pins numpy
1.26 — with the newer pair, `import tensorflow` fails before it reaches the model.
The rest of the project never imports TensorFlow, so `INCLUDE/venv` is untouched.
