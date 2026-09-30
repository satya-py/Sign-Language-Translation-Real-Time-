/**
 * The camera end of SignAI.
 *
 * Grabs frames at about 14 a second - the rate the recogniser was measured at -
 * and posts them to the local backend in short chunks rather than one request
 * per frame: one HTTP round trip per frame at 14 fps spends more time in request
 * overhead than in MediaPipe.
 *
 * Also owns the voice. speechSynthesis needs a document, and this is the only
 * document the extension keeps alive while the user is in their call.
 */

const FRAME_MS = 70;            // ~14 frames a second
const CHUNK_MS = 900;           // send what has been collected about once a second
const JPEG_QUALITY = 0.7;       // measurably enough for MediaPipe at 640x480

const video = document.getElementById('camera');
const canvas = document.getElementById('frame');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

let stream = null;
let session = null;
let backendUrl = '';
let running = false;
let pending = [];               // frames waiting to be sent
let inFlight = false;
let frameTimer = null;
let chunkTimer = null;
let framesThisSecond = 0;
let fpsMark = Date.now();
let fps = 0;
let backoff = 0;                // grows while the backend is unreachable

function report(patch) {
  chrome.runtime.sendMessage({ type: 'offscreen-update', ...patch }).catch(() => {});
}

function stopped(error) {
  chrome.runtime.sendMessage({ type: 'offscreen-stopped', error: error || '' })
    .catch(() => {});
}

// --------------------------------------------------------------- the camera

async function startCamera() {
  if (stream) return;
  // The permission prompt cannot be shown from here, so the popup asks for it
  // first; by the time this runs the grant already exists for the extension.
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 20 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
}

function grabFrame() {
  if (!running || !stream || video.readyState < 2) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  pending.push(canvas.toDataURL('image/jpeg', JPEG_QUALITY));
  framesThisSecond += 1;
  const now = Date.now();
  if (now - fpsMark >= 1000) {
    fps = Math.round((framesThisSecond * 1000) / (now - fpsMark));
    framesThisSecond = 0;
    fpsMark = now;
  }
  // If the backend stalls, drop the oldest frames rather than growing forever:
  // a sign made ten seconds ago is no longer worth translating.
  if (pending.length > 60) pending = pending.slice(-40);
}

// --------------------------------------------------------------- the backend

async function call(path, body) {
  const response = await fetch(backendUrl.replace(/\/$/, '') + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) throw new Error(`backend said ${response.status}`);
  return response.json();
}

async function openSession() {
  const answer = await call('/api/session', {});
  session = answer.session;
  report({
    session,
    backend: 'ready',
    vocabulary: answer.vocabulary || 0,
    error: '',
  });
}

async function sendChunk() {
  if (!running || inFlight || !pending.length) return;
  if (backoff && Date.now() < backoff) return;
  const frames = pending;
  pending = [];
  inFlight = true;
  try {
    if (!session) await openSession();
    const answer = await call('/api/chunk', { session, frames });
    backoff = 0;
    report({
      glosses: answer.glosses || [],
      sentence: answer.sentence || '',
      building: !!answer.building,
      final: !!answer.final,
      confidence: answer.confidence || 0,
      guess: answer.guess || [],
      accepted: !!answer.accepted,
      framing: answer.framing || { ok: true, detail: '' },
      backend: 'ready',
      fps,
      error: '',
    });
  } catch (err) {
    // Keep the camera running: the backend often comes back a second later, and
    // stopping the capture would make the user re-grant the camera.
    backoff = Date.now() + 2000;
    session = null;
    report({ backend: 'unreachable', error: err.message, fps });
  } finally {
    inFlight = false;
  }
}

// ----------------------------------------------------------------- the voice

let voices = [];

function loadVoices() {
  voices = speechSynthesis.getVoices();
  return voices;
}
speechSynthesis.onvoiceschanged = loadVoices;
loadVoices();

/** Indian English first, then any English, then whatever exists. */
function pickVoice(voiceUri) {
  if (!voices.length) loadVoices();
  if (voiceUri) {
    const chosen = voices.find((v) => v.voiceURI === voiceUri);
    if (chosen) return chosen;
  }
  return voices.find((v) => v.lang === 'en-IN')
      || voices.find((v) => /en[-_]IN/i.test(v.lang))
      || voices.find((v) => /^en[-_]GB/i.test(v.lang))
      || voices.find((v) => v.lang.startsWith('en'))
      || voices[0]
      || null;
}

function speak(text, rate, voiceUri) {
  if (!text) return;
  // Cancel first: two sentences spoken over each other are unintelligible, and
  // the point of the voice is that the people in the call understand it.
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = pickVoice(voiceUri);
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  } else {
    utterance.lang = 'en-IN';
  }
  utterance.rate = rate || 1.0;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;
  speechSynthesis.speak(utterance);
}

// -------------------------------------------------------------------- control

async function start(message) {
  backendUrl = message.backendUrl;
  try {
    await startCamera();
  } catch (err) {
    stopped(`camera: ${err.message}`);
    return;
  }
  running = true;
  pending = [];
  session = null;
  frameTimer = setInterval(grabFrame, FRAME_MS);
  chunkTimer = setInterval(sendChunk, CHUNK_MS);
  try {
    await openSession();
  } catch (err) {
    report({ backend: 'unreachable', error: err.message });
  }
}

function stop() {
  running = false;
  clearInterval(frameTimer);
  clearInterval(chunkTimer);
  frameTimer = chunkTimer = null;
  pending = [];
  stopCamera();
  speechSynthesis.cancel();
  if (session) call('/api/close', { session }).catch(() => {});
  session = null;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.target !== 'offscreen') return false;
  (async () => {
    switch (msg.type) {
      case 'start':
        await start(msg);
        sendResponse({ ok: true });
        break;
      case 'stop':
        stop();
        sendResponse({ ok: true });
        break;
      case 'finish':
        try {
          const answer = await call('/api/finish', { session });
          report({
            glosses: [], sentence: answer.sentence || '', final: !!answer.sentence,
            building: false, backend: 'ready',
          });
        } catch (err) {
          report({ backend: 'unreachable', error: err.message });
        }
        sendResponse({ ok: true });
        break;
      case 'clear':
        pending = [];
        try {
          await call('/api/reset', { session });
        } catch { /* a reset that fails is not worth interrupting the user for */ }
        sendResponse({ ok: true });
        break;
      case 'speak':
        speak(msg.text, msg.rate, msg.voiceUri);
        sendResponse({ ok: true });
        break;
      case 'voices':
        sendResponse({
          voices: loadVoices().map((v) => ({
            name: v.name, lang: v.lang, voiceURI: v.voiceURI, default: v.default,
          })),
        });
        break;
      default:
        sendResponse({ ok: false });
    }
  })();
  return true;
});
