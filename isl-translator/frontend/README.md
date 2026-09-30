# SignBridge AI — frontend (React + Vite)

Implements the SignBridge design system from `DESIGN.md`: Plus Jakarta Sans for
headings, Atkinson Hyperlegible Next for everything that must be read quickly,
royal-blue primary, clinical-teal AI states, 48px hit targets and visible focus rings.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api and /ws to the backend
npm run build    # writes dist/, which the backend then serves at http://127.0.0.1:8000
```

| File | Purpose |
|---|---|
| `src/layouts/MarketingLayout.jsx` | public pages: landing, login, sign up |
| `src/layouts/AppLayout.jsx` | signed-in app: sidebar + topbar |
| `src/pages/Landing.jsx` | hero, features, how it works, live numbers |
| `src/pages/Auth.jsx` | login / sign up (demo profile, no password) |
| `src/pages/Dashboard.jsx` | quick actions and real counters |
| `src/pages/LiveTranslation.jsx` | camera, glosses, sentence, transcript (also General conversation) |
| `src/pages/Healthcare.jsx` | patient ↔ doctor, phrases, Show in ISL |
| `src/pages/Learn.jsx` | sign library by category |
| `src/pages/History.jsx` | past sessions, search, export, delete |
| `src/pages/Settings.jsx` | recognition, accessibility, devices |
| `src/components/CameraPanel.jsx` | camera viewport + HUD, shared by every translating page |
| `src/useTranslator.js` | WebSocket + camera frame pump (640×480, ~14 fps) |
| `src/store.jsx` | demo profile (localStorage) and settings (backend) |
| `src/styles.css`, `src/shell.css` | the design system |

**Accounts:** login and sign up create a *demo profile*: a display name kept in this
browser. No password is requested, nothing is sent to a server. A hackathon demo has
nowhere safe to store real credentials, so it does not pretend to.

Run the backend first; in development Vite proxies `/api` and `/ws` to port 8000, so
the app always talks to a same-origin URL and there are no CORS surprises.
