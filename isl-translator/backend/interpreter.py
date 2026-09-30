"""Human interpreter connect: directory, invitations and call rooms.

The human interpreter is a FALLBACK for the moments AI translation is not enough
(unusual vocabulary, emergencies, consent conversations). AI keeps running during
the call and its captions are shown to everyone, so the interpreter corrects rather
than replaces it.

State is deliberately small and in-memory plus a JSON file: this is a demo, so an
interpreter "registers" from the same machine or another browser on the LAN. Nothing
here is a real telehealth platform, and the UI says so.
"""

import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api/interpreter", tags=["interpreter"])

DATA_DIR = Path(__file__).resolve().parent / "data"
INTERPRETERS_FILE = DATA_DIR / "interpreters.json"

# Seed directory. These are demo profiles for the hackathon, not real people:
# the UI labels them as such, and any of them can be "taken over" by opening the
# interpreter dashboard, which is how a real second person joins the demo.
SEED = [
    {"id": "int-ananya", "name": "Ananya Sharma", "title": "ISL Interpreter",
     "languages": ["Indian Sign Language", "English", "Hindi"], "experience_years": 7,
     "available": True, "emergency": True, "rating": 4.9, "calls": 412, "demo": True},
    {"id": "int-rahul", "name": "Rahul Banerjee", "title": "ISL Interpreter",
     "languages": ["Indian Sign Language", "English", "Bengali"], "experience_years": 5,
     "available": True, "emergency": True, "rating": 4.8, "calls": 289, "demo": True},
    {"id": "int-meera", "name": "Meera Iyer", "title": "ISL Interpreter (medical)",
     "languages": ["Indian Sign Language", "English", "Hindi", "Tamil"],
     "experience_years": 9, "available": False, "emergency": False, "rating": 5.0,
     "calls": 640, "demo": True},
    {"id": "int-sameer", "name": "Sameer Khan", "title": "ISL Interpreter",
     "languages": ["Indian Sign Language", "English", "Hindi", "Urdu"],
     "experience_years": 4, "available": True, "emergency": False, "rating": 4.7,
     "calls": 151, "demo": True},
]

STATE = {
    "interpreters": {},
    "requests": {},      # request_id -> request
    "rooms": {},         # room_id -> room
    "sockets": {},       # room_id -> [WebSocket]
}


def _save():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INTERPRETERS_FILE.write_text(json.dumps(list(STATE["interpreters"].values()), indent=1),
                                 encoding="utf-8")


def load():
    STATE["interpreters"] = {i["id"]: i for i in SEED}
    try:
        for stored in json.loads(INTERPRETERS_FILE.read_text(encoding="utf-8")):
            STATE["interpreters"][stored["id"]] = stored
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return STATE["interpreters"]


load()


# ------------------------------------------------------------------ directory

@router.get("/list")
def list_interpreters(q: str = "", language: str = "", availability: str = "all"):
    items = list(STATE["interpreters"].values())
    if q:
        needle = q.lower()
        items = [i for i in items if needle in i["name"].lower()
                 or any(needle in l.lower() for l in i["languages"])]
    if language:
        items = [i for i in items if any(language.lower() in l.lower() for l in i["languages"])]
    if availability == "available":
        items = [i for i in items if i["available"]]
    elif availability == "emergency":
        items = [i for i in items if i["available"] and i["emergency"]]
    return sorted(items, key=lambda i: (not i["available"], -i["rating"]))


@router.post("/register")
async def register(profile: dict):
    """Anyone can take a shift as an interpreter (used by the interpreter dashboard)."""
    interpreter_id = profile.get("id") or f"int-{uuid.uuid4().hex[:6]}"
    record = {
        "id": interpreter_id,
        "name": profile.get("name", "Interpreter"),
        "title": profile.get("title", "ISL Interpreter"),
        "languages": profile.get("languages") or ["Indian Sign Language", "English"],
        "experience_years": int(profile.get("experience_years", 1)),
        "available": bool(profile.get("available", True)),
        "emergency": bool(profile.get("emergency", True)),
        "rating": float(profile.get("rating", 5.0)),
        "calls": int(profile.get("calls", 0)),
        "demo": False,
    }
    STATE["interpreters"][interpreter_id] = record
    _save()
    return record


@router.put("/{interpreter_id}/availability")
async def set_availability(interpreter_id: str, payload: dict):
    person = STATE["interpreters"].get(interpreter_id)
    if not person:
        raise HTTPException(404, "unknown interpreter")
    person["available"] = bool(payload.get("available", True))
    _save()
    return person


# --------------------------------------------------------------- invitations

@router.post("/request")
async def create_request(payload: dict):
    """A user invites an interpreter into a conversation."""
    interpreter_id = payload.get("interpreter_id")
    if interpreter_id and interpreter_id not in STATE["interpreters"]:
        raise HTTPException(404, "unknown interpreter")
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    room_id = f"room-{uuid.uuid4().hex[:8]}"
    record = {
        "id": request_id,
        "room": room_id,
        "interpreter_id": interpreter_id,          # None = any available interpreter
        "kind": payload.get("kind", "general"),    # general | healthcare | emergency
        "from": payload.get("from", "Guest"),
        "participants": payload.get("participants", ["Deaf user", "Doctor"]),
        "note": payload.get("note", ""),
        "status": "waiting",
        "created": time.time(),
        "at": time.strftime("%H:%M"),
    }
    STATE["requests"][request_id] = record
    return record


@router.get("/requests")
def list_requests(interpreter_id: str = "", status: str = ""):
    items = list(STATE["requests"].values())
    if interpreter_id:
        items = [r for r in items if r["interpreter_id"] in (None, "", interpreter_id)]
    if status:
        items = [r for r in items if r["status"] == status]
    return sorted(items, key=lambda r: -r["created"])


@router.post("/requests/{request_id}/respond")
async def respond(request_id: str, payload: dict):
    record = STATE["requests"].get(request_id)
    if not record:
        raise HTTPException(404, "unknown request")
    accept = bool(payload.get("accept"))
    record["status"] = "accepted" if accept else "declined"
    record["interpreter_id"] = payload.get("interpreter_id") or record["interpreter_id"]
    record["responded"] = time.strftime("%H:%M")
    if accept:
        person = STATE["interpreters"].get(record["interpreter_id"], {})
        STATE["rooms"][record["room"]] = {
            "id": record["room"], "kind": record["kind"], "started": time.time(),
            "interpreter": person.get("name", "Interpreter"),
            "participants": record["participants"], "captions": [],
        }
    return record


@router.get("/rooms/{room_id}")
def room(room_id: str):
    found = STATE["rooms"].get(room_id)
    if not found:
        raise HTTPException(404, "room not found")
    return {**found, "minutes": round((time.time() - found["started"]) / 60, 1)}


@router.post("/rooms/{room_id}/caption")
async def add_caption(room_id: str, payload: dict):
    """AI captions keep flowing during a human call; everyone sees the same text."""
    found = STATE["rooms"].get(room_id)
    if not found:
        raise HTTPException(404, "room not found")
    entry = {"at": time.strftime("%H:%M:%S"), "who": payload.get("who", "ai"),
             "text": payload.get("text", ""), "confidence": payload.get("confidence")}
    found["captions"].append(entry)
    found["captions"] = found["captions"][-200:]
    return entry


@router.get("/stats")
def stats():
    rooms = list(STATE["rooms"].values())
    return {
        "interpreters": len(STATE["interpreters"]),
        "available": sum(1 for i in STATE["interpreters"].values() if i["available"]),
        "waiting_requests": sum(1 for r in STATE["requests"].values() if r["status"] == "waiting"),
        "calls_completed": sum(1 for r in STATE["requests"].values() if r["status"] == "accepted"),
        "active_calls": len(rooms),
        "total_minutes": round(sum((time.time() - r["started"]) / 60 for r in rooms), 1),
    }


# ------------------------------------------------------ live room signalling

@router.websocket("/ws/{room_id}")
async def room_socket(ws: WebSocket, room_id: str):
    """Relays chat, captions and WebRTC offers between everyone in one room.

    The server never touches the media: it only passes messages along, so a real
    peer-to-peer call works between two browsers, and the demo still works with one.
    """
    await ws.accept()
    STATE["sockets"].setdefault(room_id, []).append(ws)
    peers = STATE["sockets"][room_id]
    await ws.send_text(json.dumps({"type": "joined", "room": room_id, "peers": len(peers)}))
    try:
        while True:
            raw = await ws.receive_text()
            for peer in list(peers):
                if peer is not ws:
                    try:
                        await peer.send_text(raw)
                    except Exception:
                        pass
    except WebSocketDisconnect:
        pass
    finally:
        if ws in peers:
            peers.remove(ws)
        for peer in list(peers):
            try:
                await peer.send_text(json.dumps({"type": "peer-left", "peers": len(peers)}))
            except Exception:
                pass
