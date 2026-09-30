# SignBridge AI — backend (FastAPI)

Serves the API, the live WebSocket and (once built) the React app.

```bash
"D:\new Sign language\INCLUDE\venv\Scripts\python.exe" app.py
# http://127.0.0.1:8000
```

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | model status, the list of known signs, thresholds |
| `GET /api/gallery` | reference-sign library (word, clip, length) |
| `GET /api/poster/{word}` | poster frame for a sign |
| `GET /api/clip/{word}` | reference video (H.264 in .MOV, served as video/mp4) |
| `POST /api/video` | run a whole uploaded clip through the pipeline |
| `WS /ws` | live camera frames in → glosses + sentence out |
| `GET /*` | the built React app (SPA fallback to index.html) |

Options: `--port`, `--model`, `--threshold`, `--no-llm`, `--reload`.

**Where the intelligence lives:** the recognition pipeline is imported from `../live`
(`signfeat.py`, `live_demo.py`, `sentence.py`). The backend deliberately does not copy
that code, so what runs in the demo is exactly what was trained and measured
(97.1% on 630 clips — see `../live/README.md`).

Frames arrive as 640×480 JPEGs at about 14 fps. MediaPipe, the classifier and the
sentence builder all run here in Python, which keeps live input identical to the
training data.
