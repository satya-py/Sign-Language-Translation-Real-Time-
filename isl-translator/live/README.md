# ISL Live Translator — doctor/patient demo

37 health-related signs, recognised live from a webcam and turned into English sentences.

Real-time Indian Sign Language → English for a hospital conversation.
Camera → MediaPipe keypoints → sign segmentation → classifier → English sentence → speech.

```
webcam ──► MediaPipe Holistic ──► SignSpotter ──► SignNet ──► sentence.py ──► text + speech
           (pose + 2 hands)       (cuts each     (classifier   (ISL order →
                                   sign out)      + "idle")     English)
```

## Run the demo

```bash
# web app (recommended for the presentation)
python server.py                       # then open http://127.0.0.1:8000
                                       #  /       camera + translation + doctor reply
                                       #  /learn  every sign as a video to copy

# desktop window
python live_demo.py

# on a recorded clip (safe fallback demo)
python live_demo.py --video demo/i_sick_today.mp4
```

Use the venv: `D:\new Sign language\INCLUDE\venv\Scripts\python.exe`.

## Measured performance

| What | Result |
|---|---|
| Sign accuracy through the **live** pipeline | **95.4%** (1185 clips, 86 signs + idle) |
| The signer's own webcam takes, end to end | **96% right, 1% wrong, 3% missed** (272 takes) |
| Movement *between* signs reported as a word | **14%** (550 windows; it was 56% before the idle set was enlarged) |

Enlarging the idle set further (`--idle-still 30 --idle-move 120`) trades the other
way: 93% of takes recognised and 11% of between-sign movement let through. The
shipped setting prefers catching the sign, because a signer repeating a sign that
was missed reads worse on stage than a stray word that can be cleared.
| Processing speed | 22 fps desktop, 20 fps through the web app |
| Delay from end of sign to word on screen | **~0.31 s** |
| Classifier size / training time | 0.80M parameters / 45 s on an RTX 3050 |

Re-measure any time: `python eval_live.py --model models/signs.pt`

## Why this works where the INCLUDE pretrained model failed

| Problem in the old model | Fix here |
|---|---|
| Raw 1920×1080 pixel coordinates → a webcam looks completely different | Body-relative features: origin between the shoulders, scale = shoulder width (`signfeat.py`) |
| No "not signing" class → it named a sign constantly | `idle` class, mined from still moments **and** from hand-transition moments |
| Fixed rolling window cut signs in half | `SignSpotter` starts on movement and ends on stillness |
| Almost no augmentation | Rotation ±15°, scale, shift, speed 0.75–1.35×, mirror, dropped landmarks, jitter |
| Trained on whole clips, used on segments | Training clips are cropped to the moving part, the same cut the live app makes |

## Files

| File | What it does |
|---|---|
| `signfeat.py` | Keypoints → body-relative features. Used by training **and** the live app. |
| `fetch_reference.py` | Builds the word index from INCLUDE on Zenodo (HTTP range, no full download). |
| `fetch_videos.py` | Downloads only the videos for the chosen words. |
| `extract_keypoints.py` | Videos → keypoint files (MediaPipe Holistic). |
| `train.py` | Trains `SignNet`; mines the idle class; reports per-class accuracy. |
| `eval_live.py` | Honest test: replays clips through the **live** pipeline. |
| `live_demo.py` | Desktop demo (webcam or video file). |
| `server.py` | Web app: browser camera → FastAPI → sentence + speech. |
| `sentence.py` | Glosses → English, medical templates, optional LLM. |
| `record.py` / `learn_record.py` | Record your own signs (with a reference video to copy). |
| `make_demo_video.py` | Joins single-sign clips into sentence videos for testing and backup demos. |
| `make_gallery.py` | Prepares the "learn the signs" page (one reference clip + poster per sign). |

### Where the reference clips come from

INCLUDE covers 225 words, and only 64 of the 164 words the sentence engine handles.
Nothing open covers PAIN, HEAD, WATER, WHERE or the rest of the hospital vocabulary —
CISLR has them but is gated and not redistributable — so those signs get their
reference from the person teaching them: the Teach page records the first take with
`MediaRecorder` and `POST /api/teach/video` keeps it under
`live/data/reference_self/<word>/`. Whoever records the next takes copies that clip,
which is the point: the reference then matches exactly what the model learned. A bad
take is replaced with `DELETE /api/teach/video/<word>`.

Fuzzy matching against the INCLUDE word list is deliberately *not* used — it offers
"paint" for PAIN and "waiter" for WATER, and a wrong reference is worse than none.
| `conversation_demo.py` | Builds and tests a whole scripted doctor-patient conversation. |

## Add more signs

```bash
python fetch_reference.py --index --categories Home,Society     # extend the word index
python fetch_reference.py --list                                # see available words
python fetch_reference.py --words pain,head --per-word 3         # download clips for words
python make_gallery.py                                          # rebuild the library page
python fetch_videos.py --words medicine,patient,bed             # download them
python extract_keypoints.py                                     # keypoints
python train.py --epochs 80                                     # retrain (about 1 min)
python eval_live.py                                             # check accuracy
```

Words that exist in INCLUDE are listed by `--list`. Words INCLUDE does **not** have
(water, help, pain, fever, yes, no, sorry, please) must be recorded with `record.py`.

## Sentence building

Works with no API key (templates). For better sentences, set a key before starting:

```bash
setx GROQ_API_KEY "gsk_..."        # free tier, fast
setx ANTHROPIC_API_KEY "sk-ant-..."
```

Examples produced without any key:

```
I SICK TODAY              -> I am sick today.
I HOT                     -> I have a fever.
MEDICINE MORNING NIGHT    -> Take the medicine in the morning and at night.
MOTHER HOSPITAL YESTERDAY -> My mother was at the hospital yesterday.
PATIENT BED WEAK          -> The patient was in bed and weak.
```

## Honest limits

* Only the trained signs are recognised; anything else is ignored or guessed.
* Training data is INCLUDE (studio video, deaf signers). A very different camera angle,
  bad lighting or a very fast signer will lower accuracy.
* ISL has no single sign for *fever*, *pain*, *yes*, *no*, so those are read from the
  nearest sign (HOT = fever, BAD = not feeling well). This is stated on screen.
* This is an assistive demo, not a medical device. It must not be used for real
  clinical decisions without a human interpreter.
