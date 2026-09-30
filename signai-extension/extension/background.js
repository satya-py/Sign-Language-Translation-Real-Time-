/**
 * SignAI service worker: the one place that knows what is going on.
 *
 * It owns three things and nothing else:
 *   - the offscreen document, which holds the camera and talks to the backend
 *     (the popup cannot hold it: a popup is destroyed the moment it loses focus,
 *     which is exactly when the user clicks back into their video call);
 *   - the settings, in chrome.storage.local;
 *   - what happens to a finished sentence: speak it, post it into the call, or
 *     both.
 *
 * A service worker is itself evicted after ~30 s idle, so anything that must
 * survive lives in storage, never in a module variable alone.
 */

const DEFAULTS = {
  backendUrl: 'http://127.0.0.1:5001',
  autoSpeak: true,
  autoPost: true,
  speakRate: 1.0,
  voiceUri: '',            // empty = pick the best Indian English voice available
  prefix: '[ISL] ',        // what the chat message is prefixed with
  minConfidence: 0.35,
};

const STATE = {
  running: false,
  session: null,
  glosses: [],
  sentence: '',
  building: false,
  lastFinal: '',
  confidence: 0,
  guess: [],               // top three of the last segment, accepted or not
  accepted: false,
  framing: { ok: true, detail: '' },
  fps: 0,
  backend: 'unknown',      // unknown | ready | unreachable
  vocabulary: 0,
  callTabId: null,         // where translated text is posted
  lastPost: null,          // {ok, detail, at}
  error: '',
};

// ---------------------------------------------------------------- settings

async function getSettings() {
  const stored = await chrome.storage.local.get('settings');
  return { ...DEFAULTS, ...(stored.settings || {}) };
}

async function setSettings(patch) {
  const next = { ...(await getSettings()), ...patch };
  await chrome.storage.local.set({ settings: next });
  return next;
}

// ------------------------------------------------------- offscreen document

const OFFSCREEN_PATH = 'offscreen.html';

async function hasOffscreen() {
  // getContexts is the only reliable check; hasDocument() was removed.
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
  });
  return contexts.length > 0;
}

let creating = null;                       // guards against two parallel creates

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  if (creating) return creating;
  creating = chrome.offscreen.createDocument({
    url: OFFSCREEN_PATH,
    reasons: ['USER_MEDIA'],
    justification:
      'Holds the webcam stream and reads translated sentences aloud while the '
      + 'popup is closed and the user is in their video call.',
  });
  try {
    await creating;
  } finally {
    creating = null;
  }
}

async function closeOffscreen() {
  if (await hasOffscreen()) await chrome.offscreen.closeDocument();
}

// ------------------------------------------------------------- broadcasting

function broadcast() {
  // The popup may be closed; that rejection is expected and means nothing.
  chrome.runtime.sendMessage({ type: 'state', state: STATE }).catch(() => {});
}

async function saveState() {
  await chrome.storage.session.set({ state: STATE });
}

async function restoreState() {
  const stored = await chrome.storage.session.get('state');
  if (stored.state) Object.assign(STATE, stored.state, { error: '' });
}

// ------------------------------------------------------------- chat posting

/**
 * Post one sentence into the chat of a tab.
 *
 * content.js is injected here and only here - never declaratively - so a page
 * the user never sends to is never touched at all.
 */
async function postToTab(tabId, text, settings) {
  const target = { tabId, allFrames: true };
  try {
    const results = await chrome.scripting.executeScript({
      target,
      files: ['content.js'],
    });
    if (!results.length) throw new Error('nothing was injected');
    const outcome = await chrome.tabs.sendMessage(tabId, {
      type: 'signai-insert',
      text,
      prefix: settings.prefix,
      autoSend: true,
    });
    return outcome || { ok: false, detail: 'no answer from the page' };
  } catch (err) {
    return { ok: false, detail: err.message };
  }
}

async function permissionForTab(tab) {
  if (!tab || !tab.url) return false;
  let origin;
  try {
    origin = new URL(tab.url).origin + '/*';
  } catch {
    return false;
  }
  if (!/^https?:/.test(origin)) return false;
  return chrome.permissions.contains({ origins: [origin] });
}

async function deliver(sentence) {
  const settings = await getSettings();
  if (!sentence) return;
  STATE.lastFinal = sentence;

  if (settings.autoSpeak) {
    await ensureOffscreen();
    chrome.runtime.sendMessage({
      target: 'offscreen',
      type: 'speak',
      text: sentence,
      rate: settings.speakRate,
      voiceUri: settings.voiceUri,
    }).catch(() => {});
  }

  if (settings.autoPost) {
    const tabId = STATE.callTabId;
    if (tabId == null) {
      STATE.lastPost = { ok: false, detail: 'no call tab chosen', at: Date.now() };
    } else {
      const outcome = await postToTab(tabId, sentence, settings);
      STATE.lastPost = { ...outcome, at: Date.now() };
    }
  }
  await saveState();
  broadcast();
}

// ------------------------------------------------------------------ messages

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // Messages addressed to the offscreen document are not ours to answer.
  if (msg?.target === 'offscreen') return false;

  (async () => {
    switch (msg?.type) {
      case 'getState': {
        await restoreState();
        sendResponse({ state: STATE, settings: await getSettings() });
        break;
      }
      case 'setSettings': {
        const settings = await setSettings(msg.patch || {});
        sendResponse({ settings });
        break;
      }
      case 'useThisTab': {
        STATE.callTabId = msg.tabId ?? null;
        await saveState();
        broadcast();
        sendResponse({ ok: true, tabId: STATE.callTabId });
        break;
      }
      case 'start': {
        const settings = await getSettings();
        await ensureOffscreen();
        STATE.running = true;
        STATE.error = '';
        await saveState();
        chrome.runtime.sendMessage({
          target: 'offscreen',
          type: 'start',
          backendUrl: settings.backendUrl,
          minConfidence: settings.minConfidence,
        }).catch(() => {});
        broadcast();
        sendResponse({ ok: true });
        break;
      }
      case 'stop': {
        STATE.running = false;
        chrome.runtime.sendMessage({ target: 'offscreen', type: 'stop' }).catch(() => {});
        await closeOffscreen();
        await saveState();
        broadcast();
        sendResponse({ ok: true });
        break;
      }
      case 'finishSentence': {
        chrome.runtime.sendMessage({ target: 'offscreen', type: 'finish' }).catch(() => {});
        sendResponse({ ok: true });
        break;
      }
      case 'clear': {
        STATE.glosses = [];
        STATE.sentence = '';
        chrome.runtime.sendMessage({ target: 'offscreen', type: 'clear' }).catch(() => {});
        await saveState();
        broadcast();
        sendResponse({ ok: true });
        break;
      }
      case 'speakNow': {
        await ensureOffscreen();
        const settings = await getSettings();
        chrome.runtime.sendMessage({
          target: 'offscreen',
          type: 'speak',
          text: msg.text || STATE.lastFinal || STATE.sentence,
          rate: settings.speakRate,
          voiceUri: settings.voiceUri,
        }).catch(() => {});
        sendResponse({ ok: true });
        break;
      }
      case 'postNow': {
        const settings = await getSettings();
        const text = msg.text || STATE.lastFinal || STATE.sentence;
        if (STATE.callTabId == null) {
          sendResponse({ ok: false, detail: 'choose the call tab first' });
          break;
        }
        const outcome = await postToTab(STATE.callTabId, text, settings);
        STATE.lastPost = { ...outcome, at: Date.now() };
        await saveState();
        broadcast();
        sendResponse(outcome);
        break;
      }
      case 'cameraGranted': {
        // The permission tab got the grant; nothing to do but let the popup know
        // it can start now.
        STATE.error = '';
        await saveState();
        broadcast();
        sendResponse({ ok: true });
        break;
      }
      case 'voices': {
        await ensureOffscreen();
        const answer = await chrome.runtime.sendMessage({
          target: 'offscreen', type: 'voices',
        }).catch(() => null);
        sendResponse(answer || { voices: [] });
        break;
      }

      // ---- coming back from the offscreen document
      case 'offscreen-update': {
        Object.assign(STATE, {
          glosses: msg.glosses ?? STATE.glosses,
          sentence: msg.sentence ?? STATE.sentence,
          building: !!msg.building,
          confidence: msg.confidence ?? STATE.confidence,
          guess: msg.guess ?? STATE.guess,
          accepted: !!msg.accepted,
          framing: msg.framing ?? STATE.framing,
          fps: msg.fps ?? STATE.fps,
          backend: msg.backend ?? STATE.backend,
          vocabulary: msg.vocabulary ?? STATE.vocabulary,
          session: msg.session ?? STATE.session,
          error: msg.error || '',
        });
        if (msg.final && msg.sentence) await deliver(msg.sentence);
        else { await saveState(); broadcast(); }
        sendResponse({ ok: true });
        break;
      }
      case 'offscreen-stopped': {
        STATE.running = false;
        STATE.error = msg.error || '';
        await saveState();
        broadcast();
        sendResponse({ ok: true });
        break;
      }
      default:
        sendResponse({ ok: false, detail: 'unknown message' });
    }
  })();
  return true;                       // the handler above answers asynchronously
});

// A tab that closes must not keep being the target of chat posts.
chrome.tabs.onRemoved.addListener(async (tabId) => {
  if (STATE.callTabId === tabId) {
    STATE.callTabId = null;
    await saveState();
    broadcast();
  }
});
