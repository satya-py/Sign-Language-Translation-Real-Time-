# SignAI — Universal ISL Real-Time Translator (Chrome extension)

Sign Indian Sign Language to your webcam during a video call. SignAI recognises the
signs, builds an English sentence, **speaks it aloud** so the other people hear it,
and **types it into that call's chat box** — on Google Meet, Zoom, Teams, Discord,
Slack, Jitsi, Webex, Whereby, Skype, BigBlueButton or a custom WebRTC app.

Nothing in `isl-translator/` or `INCLUDE/` is modified. This folder is self-contained
and reads the trained model from `isl-translator/live/` at run time.

```
signai-extension/
  extension/        the Chrome extension (load this unpacked)
  server/           the local Flask backend
  tests/            an injection test bench and a backend test
```

## 1. Start the backend

```bash
cd "D:\new Sign language\signai-extension\server" && "D:\new Sign language\INCLUDE\venv\Scripts\python.exe" app.py
```

It listens on `http://127.0.0.1:5001` and prints how many signs it loaded, plus how
much memory is free. Check it:

```bash
curl http://127.0.0.1:5001/api/health
```

**Memory matters on this machine.** MediaPipe builds its tracking graph through
XNNPACK, which needs a few hundred megabytes in one piece; when it cannot get them
it does not raise an exception, it **aborts the whole process** - the server simply
disappears, printing `failed to setup XNNPACK runtime`. With 7.3 GB of RAM, Chrome
and the main `isl-translator` backend already leave about a gigabyte, which is the
edge of what works. So:

* do not run `isl-translator/backend/app.py` and this server at the same time;
* the server already keeps itself light - the classifier runs on the **CPU** (a CUDA
  context costs more memory than the model), torch uses one thread, and all sessions
  share one tracker;
* if it still aborts, start it with the lighter tracker:

```bash
set SIGNAI_COMPLEXITY=0 && "D:\new Sign language\INCLUDE\venv\Scripts\python.exe" app.py
```

| Variable | Default | What it does |
|---|---|---|
| `SIGNAI_COMPLEXITY` | `1` | MediaPipe model complexity; `0` is lighter and less accurate |
| `SIGNAI_DEVICE` | `cpu` | `cuda` to use the GPU instead |
| `SIGNAI_TORCH_THREADS` | `1` | torch CPU threads |
| `SIGNAI_MODEL` | `isl-translator/live/models/signs.pt` | a different checkpoint |
| `SIGNAI_LIVE_DIR` | `isl-translator/live` | where the pipeline lives |

## 2. Load the extension

1. Open `chrome://extensions`
2. Turn on **Developer mode** (top right)
3. **Load unpacked** → choose `D:\new Sign language\signai-extension\extension`
4. Pin SignAI to the toolbar

## 3. Use it in a call

1. Open your video call (Meet, Zoom, Teams…) and click **SignAI** in the toolbar.
2. Press **Use this tab** — Chrome asks for permission for that one site.
3. Press **Start translating**. The first time, a tab opens asking for the camera:
   press **Allow camera** there and choose **Allow** in Chrome's prompt, then come
   back and press **Start translating** again.
4. Sign. The sentence grows in the popup as each sign is recognised; about two
   seconds of stillness finishes it, and then it is spoken and posted.

You can close the popup — capture keeps running in an offscreen document, which is
the point: Chrome destroys a popup as soon as you click back into the call.

## How the chat injection works

`content.js` is injected **only** when something is being sent — there is no content
script in the manifest, no overlay, no observer. Until you post, the page is
untouched. It then tries three tiers in order:

| Tier | What it looks at | Example |
|---|---|---|
| 1 | The element you already focused | you clicked in the chat box yourself |
| 2 | Selectors for known calling apps | `textarea[name="chatTextInput"]` (Meet), `div[data-tid="ckeditor-editor"]` (Teams), `div[data-slate-editor]` (Discord), `div[data-qa="message_input"]` (Slack), `textarea#usermsg` (Jitsi), `div[data-test="chat-input-area"]` (Webex) |
| 3 | Anything editable whose label, placeholder, test id or class says chat, message, composer, conversation or comment | a custom WebRTC app |

Open shadow roots are searched too, and a search box is actively scored *down* so it
is never mistaken for a composer.

Two details decide whether this works on real sites:

* **React-controlled inputs.** `el.value = text` is silently discarded by React,
  Vue and Angular, because they compare against their own tracked value. SignAI
  writes through `HTMLTextAreaElement.prototype`'s own setter and then dispatches a
  real `InputEvent`, which is what those frameworks listen for.
* **Rich text editors.** Teams (CKEditor), Discord (Slate) and Slack (Quill) ignore
  text inserted into the DOM. SignAI focuses the editor, collapses the selection to
  the end, fires `beforeinput`, and uses `document.execCommand('insertText')` —
  deprecated, and still the only call that produces the same internal transaction as
  a keystroke. A Range insertion is the fallback.

Sending: the nearest `button[type="submit"]`, `button[aria-label*="send"]`,
`[data-tooltip*="send"]`… is clicked; if there is none, Enter is dispatched as
`keydown`/`keypress`/`keyup`. If no composer is visible at all, SignAI clicks the
chat panel button first (`button[aria-label*="chat" i]` and friends) and looks again.

### Measured on the test bench

`tests/mock_calls.html` reproduces how each app builds its composer, including the
failure cases. Serve it and drive it, or open it in the browser and call
`window.__signaiInsert({text: 'hello'})` from the console:

| Case | Result |
|---|---|
| Google Meet — textarea + send button | delivered, tier 2 |
| Zoom — textarea, Enter to send | delivered, tier 2 |
| Teams — CKEditor contenteditable | delivered, tier 2 |
| Discord — Slate contenteditable | delivered, tier 2 |
| React-controlled textarea | delivered, tier 2 (a plain `.value =` loses this one) |
| Chat panel closed | panel opened, then delivered |
| Custom WebRTC app, unknown selectors | delivered, tier 3 (heuristic) |
| Page with only a search box | refused, search box left empty |
| Chat box already focused | delivered, tier 1 |

## The voice

The Web Speech API (`speechSynthesis`) speaks each finished sentence, preferring an
`en-IN` voice and falling back to any English one. It plays on your **speakers**, so
your microphone carries it to the call. On headphones only you hear it — post to the
chat as well, or use a virtual audio cable if your meeting tool supports one.

## What the backend does

`server/app.py` (Flask) keeps one MediaPipe tracker and one sign spotter per browser
session. The extension posts about a second of JPEG frames per request, because one
HTTP round trip per frame at 14 fps costs more in overhead than in recognition.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | model path, vocabulary, open sessions |
| `POST /api/session` | start a session |
| `POST /api/chunk` | `{session, frames[]}` → `{glosses, sentence, final, confidence}` |
| `POST /api/finish` | close the sentence now, including a sign still in hand |
| `POST /api/reset` | drop the words collected so far |
| `POST /api/close` | end the session, free the tracker |

Sentence construction reuses `isl-translator/live/sentence.py`: an LLM when one is
configured, and the slot grammar otherwise (`--no-llm` forces the grammar).

```bash
cd "D:\new Sign language\signai-extension" && "D:\new Sign language\INCLUDE\venv\Scripts\python.exe" tests/test_backend.py
```

replays real dataset clips through the same endpoints. On this machine it recognises
3 of 4 clips and produces 2 sentences — the same behaviour the live app shows.

## Troubleshooting

**“Camera blocked: Permission dismissed”.** Chrome cannot ask for the camera from a
toolbar popup: the prompt takes focus, Chrome destroys the popup, and the request
dies — Chrome reports that as *dismissed*, not as a real refusal. SignAI therefore
never prompts from the popup; pressing **Start translating** without a grant opens
`permission.html` in a tab, which asks properly. The grant belongs to the
extension's origin, so the offscreen document (which can never show a prompt) uses
it afterwards. If you pressed **Block** at some point, open
`chrome://settings/content/camera`, remove this extension from the blocked list, and
allow it again.

**“backend not reachable”.** `server/app.py` is not running, or it is on another
port — set the address under Settings in the popup.

**The server prints `failed to setup XNNPACK runtime` and exits.** It ran out of
memory building the MediaPipe graph; that failure aborts the process instead of
raising. Close the main `isl-translator` backend and a few Chrome tabs, or start
this one with `SIGNAI_COMPLEXITY=0`. `/api/health` reports the free memory it sees,
and the server warns at startup when less than 1.5 GB is free.

**Nothing is posted into the chat.** Press **Use this tab** while the call tab is in
front, then **Post now** to test. If it says no chat box was found, open the call's
chat panel once and try again.

**The sentence appears but nobody hears it.** The voice plays through your speakers;
on headphones only you hear it. Keep **Post into the call chat** on as well.

## Honest limitations

* **The model is not GCN+BiLSTM.** The brief asked for one; what is trained in this
  project is SignNet — a depthwise 1D CNN with a small transformer encoder over
  body-relative keypoints, 0.82M parameters, measured at 95.4% on 1185 clips. An
  untrained graph network would recognise nothing, so the working model is used.
  `server/recogniser.py` accepts any object with `labels` and `predict(rows)`, so a
  GCN+BiLSTM can replace it the day it is trained.
* **One person in frame.** MediaPipe Holistic tracks a single person; if somebody
  else is in shot it may track them, and confidence collapses.
* The backend must run locally — the extension never sends video anywhere else.
* Chrome 116+ (the offscreen document API).
* Sites can change their DOM at any time; tier 3 is what keeps it working when they
  do, but a platform selector may need updating after a redesign.

## Permissions, and why each one exists

| Permission | Why |
|---|---|
| `offscreen` | holds the camera and the voice while the popup is closed |
| `scripting` | injects `content.js` into the call tab, only when sending |
| `activeTab`, `tabs` | knows which tab is the call |
| `storage` | remembers your settings |
| camera (asked in a tab) | the webcam frames; they go only to your local backend |
| `tts` | fallback voice if `speechSynthesis` is unavailable |
| `http://127.0.0.1/*` | the local backend |
| optional `http(s)://*/*` | requested per site, only when you press **Use this tab** |

No analytics, no remote calls: the only network traffic is to your own backend.
