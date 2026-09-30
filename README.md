# SignBridge AI — real-time Indian Sign Language translator

A deaf patient signs to a webcam; the doctor reads an English sentence and speaks
back. Everything runs on the machine in front of you: MediaPipe finds the body,
a small classifier reads each sign, and a slot grammar turns the signs into one
sentence.

There are three parts in this repository:

| Folder | What it is |
|---|---|
| [`isl-translator/`](isl-translator) | the translator: model, training, FastAPI backend, React frontend |
| [`signai-extension/`](signai-extension) | a Chrome extension that speaks the translation into a video call and types it into that call's chat |
| `Action Detection Refined.ipynb` | the notebook that trained the fingerspelling model in `isl-translator/fingerspell/action.h5` |

## What it can do

* **87 signs** recognised live from a webcam, no gloves, no depth camera.
* **Sentences, not words**: `I PAIN HEAD 2 DAY` becomes *"I have a headache for
  2 days."* — 81 sentence cases are checked by `live/test_sentences.py`.
* **Fingerspelling** for names and anything with no sign, 37 letters and digits.
* **Teach it yourself**: record five takes of a new sign in the browser and
  retrain in about five minutes, without leaving the app.
* **Human interpreter fallback** when the machine is not enough.
* **Works inside a video call** through the Chrome extension.

## Measured, not claimed

Every number here came from a script in the repository, and each can be re-run.

| What | Result | How |
|---|---|---|
| First sign correct through the live pipeline | **95.4%** (1185 clips, 87 signs) | `live/eval_live.py` |
| The signer's own webcam takes, end to end | **96% right, 1% wrong, 3% missed** (272 takes) | `live/eval_live.py` |
| Movement *between* signs reported as a word | **14%** (was 56% before the idle set was enlarged) | `live/train.py` |
| Sentence construction | **81/81 cases exact, no sign dropped** | `live/test_sentences.py` |
| Fingerspelling letters | **94.6%** on held-out frames | `fingerspell/train_letters.py --split time` |
| Letters that can actually be committed live | **35 of 37**, 0.35 s each | `fingerspell/tune_letters.py` |
| Speed | 22 fps desktop, ~14 fps through the browser | |

Two things are deliberately *not* claimed: the notebook's `action.h5` accuracy
(the data it trained on is not in this repository — see
[`isl-translator/fingerspell/README.md`](isl-translator/fingerspell/README.md)),
and any accuracy for a signer the model has never seen. Sign recognition is
signer-dependent; five takes from the person who will demo it is worth more than
any amount of tuning.

## Running it locally

The backend needs Python 3.12, and about 1 GB of free memory.

```bash
pip install -r isl-translator/backend/requirements.txt
```

```bash
cd isl-translator/frontend && npm install && npm run build
```

```bash
cd isl-translator/backend && python app.py
```

Then open <http://127.0.0.1:8000>. The backend serves the built frontend, so one
process is the whole app.

The reference videos of each sign (11 GB of the INCLUDE dataset) are **not** in
the repository. The Learn page still shows a poster for every sign; to fetch the
clips themselves:

```bash
cd isl-translator/live && python fetch_reference.py --words hello,i,you --per-word 2
```

## Hosting it

The frontend goes to Vercel, the backend to Render. Full step-by-step commands
are in [`DEPLOY.md`](DEPLOY.md), including the one thing that decides whether it
works at all:

> MediaPipe builds its tracking graph through XNNPACK, which needs a few hundred
> megabytes in one piece. When it cannot get them it **aborts the process**
> instead of raising an error. With torch alongside it, the backend needs about
> **1 GB** to answer its first frame — so Render's 512 MB free and starter plans
> will start it, pass the health check, and then die on the first camera frame.

## How it works

```
webcam ──► MediaPipe Holistic ──► 300 numbers a frame
                                      │
                        body-relative features (origin between the
                        shoulders, scale = shoulder width) ──► SignNet
                                      │                    (1D CNN + transformer,
                        sign spotter cuts one sign            0.82M parameters)
                                      │
                             glosses ──► slot grammar (+ an LLM when a key is set)
                                      │
                                 "I have a headache for 2 days."
```

The choices that mattered most, with what each one was worth:

* **Body-relative features.** The shipped INCLUDE model used raw pixel
  coordinates and was useless on a webcam. Measuring every point from the
  shoulders, in shoulder widths, is what made it work at all.
* **Training on the moving part of a clip.** Dataset clips begin and end with the
  signer standing still; the live app never sends those frames. Cropping to the
  movement removed a whole class of confident nonsense.
* **Mining the movement between signs.** Without examples of a hand dropping and
  rising, the classifier calls it a word: 56% of between-sign movement came back
  as a sign. With them, 14%.
* **Capping class weights.** A sign with 5 takes had six times the pull of one
  with 20, so each newly taught sign quietly outvoted everything taught before —
  which is what "the old signs stopped working" turned out to be.
* **Test-time augmentation was removed.** It sounded right and measured worse:
  93% against 96%, and it silenced three times as many signs.

## Credits and licences

* **INCLUDE** (AI4Bharat), CC BY 4.0 — the sign dataset the word model learned
  from. <https://zenodo.org/records/4010759>
* **MediaPipe** (Google), Apache 2.0 — pose and hand tracking.
* The models, code and recordings in this repository are the author's own work.

The `INCLUDE/` folder of the original dataset project is not included here; it is
downloaded, not authored, and is 8.5 GB.
