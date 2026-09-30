# SignBridge AI — real-time Indian Sign Language translator

A deaf patient signs in front of a webcam; the doctor reads the English sentence and
types a reply back. 37 health-related signs, recognised live.

```
isl-translator/
├── frontend/     React + Vite user interface (DESIGN.md design system)
├── backend/      FastAPI service: REST + WebSocket
├── live/         ML lab: word signs — data, training, evaluation, pipeline
├── fingerspell/  letters A-Z and 0-10: recorder, features, trainer, live speller
└── track1_isign/ research track: pose-to-text reproduction of arXiv 2609.12993
```

## Run it

```bash
# 1. build the interface once
cd frontend && npm install && npm run build

# 2. start the server
cd ../backend && "D:\new Sign language\INCLUDE\venv\Scripts\python.exe" app.py
```

Then open **http://127.0.0.1:8000**.

## Pages

| Page | Path | What it does |
|---|---|---|
| Landing | `/` | the pitch, with live numbers from the running model |
| Login / Sign up | `/login`, `/signup` | **demo profile only** — a name kept in this browser, no password asked or stored |
| Dashboard | `/app/dashboard` | quick actions and real counters from your own sessions |
| Live translation | `/app/live` | camera, detected signs, confidence, sentence, transcript |
| Healthcare | `/app/healthcare` | patient ↔ doctor, common phrases, **Show in ISL** reference clips |
| General conversation | `/app/conversation` | everyday two-way conversation |
| Learn signs | `/app/learn` | 37 reference videos by category, with sentences to try |
| History | `/app/history` | every finished sentence, searchable, exportable |
| Settings | `/app/settings` | confidence threshold, pause, speech, accessibility, devices |
| Teach the model | `/app/teach` | record your own takes, retrain in about a minute |
| Fingerspell | `/app/fingerspell` | spell names, places and untaught words letter by letter |
| Interpreter Connect | `/app/interpreter` | invite a human ISL interpreter (normal or emergency) |
| Find an interpreter | `/app/interpreter/find` | directory with search, language and availability filters |
| Emergency | `/app/interpreter/emergency` | finds whoever is on call, with a clear safety warning |
| Interpreter call | `/app/interpreter/call/:room` | three-person call with AI captions running alongside |
| Interpreter desk | `/app/interpreter/dashboard` | the other side: availability, incoming requests, stats |

## Human interpreter, and why it is a fallback

AI translation covers a fixed vocabulary. When that is not enough — unusual words,
consent, an emergency — the user invites a human interpreter into the same conversation.
The AI keeps translating during the call and its captions are shared with everyone, so the
interpreter corrects the machine instead of replacing it. Media is peer to peer; the backend
only relays messages, so a second browser (or a second laptop on the network) becomes the
real interpreter.

## Where the earlier symptom work went

The nine symptom signs from the previous project (cough, fever, headache, nausea, runny nose,
sore throat, stomach ache, tired, trouble breathing) are in `live/data/reference_only/`.
They appear in the sign library and in Teach marked **"not recognised yet"**: one demo video
each is enough to *show* a sign, nowhere near enough to *train* one. Record about five takes
of each in Teach and they become recognisable.

## Measured

| | |
|---|---|
| Sign accuracy through the live pipeline | **97.1%** (630 clips, 37 signs + idle) |
| Speed | 22 fps desktop, 20 fps through the browser |
| Delay from end of sign to word on screen | ~0.31 s |
| Model | 0.80M parameters, trains in 45 s on an RTX 3050 |
| Fingerspelling | 37 letters (A-Z, 0-10), **96.7%** on held-out frames, 955 usable frames |

Details, training commands and honest limits: [live/README.md](live/README.md).
Bengali demo guide: [live/DEMO_GUIDE_BN.md](live/DEMO_GUIDE_BN.md).

## Fingerspelling

Names, places and anything the word model was never taught are spelled letter by letter.

```bash
python fingerspell/record_letters.py --signer yourname    # capture frames per letter
python fingerspell/train_letters.py                       # about 15 s, writes models/letters.pt
```

A letter is a shape held still, not a movement, so the live speller classifies every
frame and accepts a letter only once it has been steady for several frames. A repeated
letter (the two N's in ANNA) needs a real break — drop the hand for a moment — otherwise
one held shape would fill the buffer. Verified by replaying recorded frames: SATYA, ANNA,
RAM and A1 all spell correctly, and holding A for 40 frames yields "A", not "AAAA".

**Limits:** the letter data came from one signer in one session, so accuracy for a
different person will be lower, and look-alike shapes (H/K, L/X, Z/C) are where it slips.
