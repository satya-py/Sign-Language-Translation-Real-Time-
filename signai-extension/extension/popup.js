/**
 * The whole user interface. Everything lives in this popup on purpose: the
 * extension never draws anything on a web page, so a call is never covered by a
 * panel the other participants can see in a screen share.
 *
 * The popup is a view, not a brain. It asks the service worker for state, sends
 * it commands, and re-renders - so closing it (which Chrome does the moment the
 * user clicks back into the call) stops nothing.
 */

const $ = (id) => document.getElementById(id);
let settings = null;
let state = null;

// ------------------------------------------------------------------ helpers

function send(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (answer) => {
      // A closed receiver sets lastError; reading it keeps Chrome quiet.
      void chrome.runtime.lastError;
      resolve(answer || {});
    });
  });
}

async function currentTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

function shortUrl(url) {
  try { return new URL(url).host; } catch { return url || ''; }
}

// ------------------------------------------------------------------ render

function render() {
  if (!state || !settings) return;

  $('runDot').classList.toggle('on', state.running);
  $('runDot').title = state.running ? 'translating' : 'not translating';
  $('startBtn').textContent = state.running ? 'Stop translating' : 'Start translating';
  $('startBtn').classList.toggle('stop', state.running);

  const backend = {
    ready: `backend ready · ${state.vocabulary || 0} signs`,
    unreachable: 'backend not reachable — is server/app.py running?',
    unknown: 'backend not checked yet',
  }[state.backend] || state.backend;
  $('backendLine').textContent = backend;

  const sentence = state.sentence || (state.running
    ? 'Sign to your camera…'
    : 'Start translating, then sign to your camera.');
  $('sentence').textContent = sentence;
  $('sentence').classList.toggle('building', !!state.building);
  // A finished sentence stays on screen; dim it so it is never mistaken for a
  // translation of the sign being made right now.
  $('sentence').classList.toggle('stale', !state.building && !!state.sentence);

  $('glosses').innerHTML = '';
  for (const gloss of state.glosses || []) {
    const chip = document.createElement('span');
    chip.className = 'gloss';
    chip.textContent = gloss.replace(/_/g, ' ');
    $('glosses').append(chip);
  }

  const confidence = Math.round((state.confidence || 0) * 100);
  $('confidenceBar').style.width = `${confidence}%`;
  // Say what was read AND whether it counted. A bare percentage next to an old
  // sentence reads as if that sentence were the new one.
  const guess = state.guess?.[0];
  if (!guess) {
    $('confidenceText').textContent = 'no sign read yet';
  } else if (state.accepted) {
    $('confidenceText').textContent =
      `read ${guess.label.replace(/_/g, ' ')} at ${confidence}%`;
  } else {
    const runnerUp = state.guess[1]
      ? `, then ${state.guess[1].label.replace(/_/g, ' ')}` : '';
    $('confidenceText').textContent =
      `read ${guess.label.replace(/_/g, ' ')} at ${confidence}%${runnerUp}`
      + ' — too unsure to use, so nothing was added';
  }
  $('framing').hidden = !(state.framing && state.framing.ok === false);
  $('framing').textContent = state.framing?.detail || '';

  // Always show how you are framed, next to how you were framed when you taught
  // the signs. Those two numbers explain most bad recognition on their own.
  const width = state.framing?.width;
  $('cameraLine').textContent = width
    ? `Your shoulders fill ${Math.round(width * 100)}% of the frame `
      + '(you taught these signs at 26–49%).'
    : '';
  $('fpsChip').textContent = state.fps ? `${state.fps} fps` : '—';

  // A camera error from the offscreen document means the same thing: the grant
  // is missing, and only a tab can ask for it.
  if (state.error && /camera/i.test(state.error) && !$('error').dataset.sticky) {
    $('error').hidden = false;
    $('error').textContent = `${state.error} — press Start translating to allow it.`;
  } else {
    $('error').hidden = !state.error;
    $('error').textContent = state.error || '';
  }

  $('autoSpeak').checked = settings.autoSpeak;
  $('autoPost').checked = settings.autoPost;
  $('rate').value = settings.speakRate;
  $('backendUrl').value = settings.backendUrl;
  $('prefix').value = settings.prefix;
  $('minConfidence').value = settings.minConfidence;
  $('minConfidenceValue').textContent = Number(settings.minConfidence).toFixed(2);

  if (state.lastPost) {
    const when = new Date(state.lastPost.at).toLocaleTimeString();
    $('postResult').textContent = state.lastPost.ok
      ? `sent at ${when} — ${state.lastPost.detail}`
      : `could not send at ${when} — ${state.lastPost.detail}`;
  }
}

async function refreshTabLine() {
  const tab = await currentTab();
  if (state?.callTabId == null) {
    $('tabLine').textContent = `No call tab chosen (you are on ${shortUrl(tab?.url)}).`;
    return;
  }
  try {
    const chosen = await chrome.tabs.get(state.callTabId);
    $('tabLine').textContent = `Posting into ${shortUrl(chosen.url)}`;
  } catch {
    $('tabLine').textContent = 'The chosen call tab was closed.';
  }
}

async function loadVoices() {
  const { voices = [] } = await send({ type: 'voices' });
  const select = $('voice');
  select.innerHTML = '';
  const auto = document.createElement('option');
  auto.value = '';
  auto.textContent = 'Best Indian English voice';
  select.append(auto);
  for (const voice of voices) {
    const option = document.createElement('option');
    option.value = voice.voiceURI;
    option.textContent = `${voice.name} (${voice.lang})`;
    select.append(option);
  }
  select.value = settings.voiceUri || '';
}

async function refresh() {
  const answer = await send({ type: 'getState' });
  state = answer.state;
  settings = answer.settings;
  render();
  await refreshTabLine();
}

// ------------------------------------------------------------------ actions

async function cameraState() {
  try {
    const result = await navigator.permissions.query({ name: 'camera' });
    return result.state;                       // granted | prompt | denied
  } catch {
    return 'unknown';
  }
}

function askForCameraInATab(reason) {
  // Never call getUserMedia from here. Showing the prompt takes focus away from
  // the popup, Chrome destroys the popup, and the request dies as
  // "Permission dismissed" - which looks to the user like the camera is broken.
  chrome.tabs.create({ url: chrome.runtime.getURL('permission.html') });
  $('error').hidden = false;
  $('error').textContent = reason;
}

$('startBtn').addEventListener('click', async () => {
  if (state.running) {
    await send({ type: 'stop' });
    await refresh();
    return;
  }
  const camera = await cameraState();
  if (camera === 'denied') {
    askForCameraInATab('The camera is blocked for this extension. The tab that '
                     + 'just opened explains how to unblock it.');
    return;
  }
  if (camera !== 'granted') {
    askForCameraInATab('Chrome needs to ask for the camera in a tab, not in this '
                     + 'popup. Allow it there, then press Start translating again.');
    return;
  }
  $('error').hidden = true;
  await send({ type: 'start' });
  await refresh();
});

$('finishBtn').addEventListener('click', () => send({ type: 'finishSentence' }));
$('clearBtn').addEventListener('click', async () => { await send({ type: 'clear' }); refresh(); });
$('speakBtn').addEventListener('click', () => send({ type: 'speakNow' }));

$('postBtn').addEventListener('click', async () => {
  const answer = await send({ type: 'postNow' });
  $('postResult').textContent = answer.ok
    ? `sent — ${answer.detail}`
    : `could not send — ${answer.detail || 'unknown reason'}`;
});

$('useTabBtn').addEventListener('click', async () => {
  const tab = await currentTab();
  if (!tab?.url || !/^https?:/.test(tab.url)) {
    $('postResult').textContent = 'Open the video call tab first.';
    return;
  }
  const origin = `${new URL(tab.url).origin}/*`;
  // Ask only for the site the user is actually calling on, at the moment they
  // ask for it - never for every site up front.
  const granted = await chrome.permissions.request({ origins: [origin] });
  if (!granted) {
    $('postResult').textContent = `Permission for ${shortUrl(tab.url)} was declined.`;
    return;
  }
  await send({ type: 'useThisTab', tabId: tab.id });
  await refresh();
  const probe = await chrome.tabs.sendMessage(tab.id, { type: 'signai-probe' })
    .catch(() => null);
  $('postResult').textContent = probe?.ok
    ? `chat box found (${probe.element})`
    : 'no chat box found yet — open the chat panel in the call';
});

for (const [id, key] of [['autoSpeak', 'autoSpeak'], ['autoPost', 'autoPost']]) {
  $(id).addEventListener('change', async (event) => {
    const answer = await send({ type: 'setSettings', patch: { [key]: event.target.checked } });
    settings = answer.settings;
  });
}

$('voice').addEventListener('change', async (event) => {
  const answer = await send({ type: 'setSettings', patch: { voiceUri: event.target.value } });
  settings = answer.settings;
});

$('rate').addEventListener('change', async (event) => {
  const answer = await send({ type: 'setSettings', patch: { speakRate: Number(event.target.value) } });
  settings = answer.settings;
});

$('minConfidence').addEventListener('input', (event) => {
  $('minConfidenceValue').textContent = Number(event.target.value).toFixed(2);
});

$('saveBtn').addEventListener('click', async () => {
  const answer = await send({
    type: 'setSettings',
    patch: {
      backendUrl: $('backendUrl').value.trim() || 'http://127.0.0.1:5001',
      prefix: $('prefix').value,
      minConfidence: Number($('minConfidence').value),
    },
  });
  settings = answer.settings;
  $('postResult').textContent = 'Settings saved. Restart translating to use them.';
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === 'state') {
    state = msg.state;
    render();
  }
});

refresh().then(loadVoices);
