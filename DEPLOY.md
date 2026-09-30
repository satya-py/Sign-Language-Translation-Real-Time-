# Deploying: backend on Render, frontend on Vercel

Read the memory note first — it decides whether this works at all.

## Before anything: the memory the backend needs

MediaPipe builds its tracking graph through XNNPACK, which needs a few hundred
megabytes in one contiguous block. When it cannot get them it does not raise an
exception — it **aborts the process**, printing `failed to setup XNNPACK
runtime`. With torch loaded alongside it, this backend needs about **1 GB** of
memory before it can answer its first camera frame.

| Render plan | Memory | What happens |
|---|---|---|
| Free | 512 MB | starts, `/api/health` answers, dies on the first frame |
| Starter | 512 MB | the same |
| **Standard** | **2 GB** | **works** |

That is not a guess: the same failure was reproduced locally, with the stack
trace, on a machine with 1 GB free. If a paid plan is not an option, the honest
alternative is to run the backend locally and host only the frontend — set
`VITE_API_BASE=http://127.0.0.1:8000` in a local build, or keep using
<http://127.0.0.1:8000>, where one process serves both.

Free Render instances also sleep after 15 minutes and take ~50 s to wake. For a
demo in front of judges, open the page a minute early.

---

## 1. Push the repository

Nothing is committed yet; `git init` has been run and `.gitignore` is in place.
From `D:\new Sign language`:

```bash
git status --short | head -20
```

That should list code, models (3 MB) and recordings — and none of the videos,
venvs or the `INCLUDE/` folder. Then:

```bash
git add -A
```

```bash
git commit -m "SignBridge AI: ISL translator, Chrome extension and fingerspelling"
```

Create an **empty** repository on GitHub (no README, no .gitignore — this
repository has both), then:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
```

```bash
git branch -M main
```

```bash
git push -u origin main
```

If the push is rejected because the remote is not empty, pull first with
`git pull --rebase origin main` and push again.

---

## 2. Backend on Render

1. Render dashboard → **New** → **Blueprint** → connect the repository. Render
   reads [`render.yaml`](render.yaml) and proposes `signbridge-backend` as a
   Docker service on the Standard plan.
2. Create it. The first build takes about 10 minutes — torch is 200 MB.
3. When it is live, note the URL, for example
   `https://signbridge-backend.onrender.com`.
4. Check it:

```bash
curl https://signbridge-backend.onrender.com/api/health
```

It should answer with the model path, `"vocabulary": 87` and the free memory it
sees.

Doing it by hand instead of by blueprint: **New → Web Service**, runtime
**Docker**, Dockerfile path `isl-translator/Dockerfile`, Docker context
`isl-translator`, health check path `/api/health`, plan Standard.

### Environment variables to set on Render

| Name | Value | Why |
|---|---|---|
| `ALLOWED_ORIGINS` | your Vercel URL, e.g. `https://signbridge.vercel.app` | without it the browser blocks every response and the server log shows nothing |
| `GROQ_API_KEY` *(optional)* | your key | an LLM then polishes the sentences; without it the rule grammar is used, which is what the measured numbers were taken on |

Add `ALLOWED_ORIGINS` after step 3 below, once Vercel has given you the URL, then
let Render redeploy.

---

## 3. Frontend on Vercel

1. Vercel → **Add New** → **Project** → import the repository.
2. **Root Directory**: `isl-translator/frontend` — this matters; the repository
   root has no `package.json`.
3. Framework preset: **Vite**. Build command and output directory come from
   [`vercel.json`](isl-translator/frontend/vercel.json).
4. **Environment Variables** → add:

   | Name | Value |
   |---|---|
   | `VITE_API_BASE` | `https://signbridge-backend.onrender.com` (no trailing slash) |

   Vite reads this at **build** time, so changing it later needs a redeploy.
5. Deploy, then copy the URL into `ALLOWED_ORIGINS` on Render.

Or from the command line:

```bash
cd "D:\new Sign language\isl-translator\frontend" && npx vercel --prod
```

---

## 4. Check the two halves are talking

Open the Vercel URL and look at the dashboard: it should say **backend ready**
with the number of signs. If it does not:

| What you see | What it is | Fix |
|---|---|---|
| Every request fails, nothing in the Render log | CORS | `ALLOWED_ORIGINS` on Render must be the exact Vercel origin, `https://…`, no trailing slash |
| Requests go to the Vercel domain instead of Render | `VITE_API_BASE` was empty at build time | set it and **redeploy** — a Vite variable is compiled in, not read at run time |
| The page loads but the camera never connects | the WebSocket | the backend must be on `https://`; a page on `https://` cannot open a `ws://` socket |
| Works, then stops after a few minutes | a free instance went to sleep | Standard plan, or open the page early |
| Render log ends with `failed to setup XNNPACK runtime` | memory | Standard plan (2 GB); see the top of this file |

## 5. The Chrome extension

The extension is not hosted; it is loaded from disk and talks to a backend on
`http://127.0.0.1:5001` by default. To use it against the Render backend, open
its popup → Settings → **Backend** and paste the Render URL. Its own server
(`signai-extension/server/app.py`) is only needed when running everything
locally.

---

## What is deliberately not deployed

* **The 11 GB of reference videos.** The Learn page shows a poster for every
  sign; the clips are fetched on demand with `live/fetch_reference.py` when
  someone wants them locally.
* **The signer's own reference takes** (`live/data/reference_self/`) — they are
  video of a person's face and hands, and belong to that person.
* **Training data beyond the keypoints.** `live/recordings/` (23 MB of numbers,
  no video) is committed so anyone can retrain; the videos they came from are
  not.
