/**
 * The camera grant, taken in a tab instead of the popup.
 *
 * getUserMedia from a toolbar popup fails with "Permission dismissed": showing
 * the prompt moves focus off the popup, Chrome destroys the popup, and the
 * pending request dies with it. A normal extension page has no such problem, and
 * the grant it obtains belongs to the extension's origin - so the offscreen
 * document, which can never show a prompt of its own, can use it afterwards.
 */

const status = document.getElementById('status');
const preview = document.getElementById('preview');

async function currentState() {
  try {
    const result = await navigator.permissions.query({ name: 'camera' });
    return result.state;                     // granted | prompt | denied
  } catch {
    return 'unknown';
  }
}

async function refresh() {
  const state = await currentState();
  if (state === 'granted') {
    status.textContent = 'The camera is allowed. You can close this tab and press '
                       + 'Start translating in the SignAI popup.';
  } else if (state === 'denied') {
    status.textContent = 'The camera is blocked for this extension. Open '
                       + 'chrome://settings/content/camera, remove the block, then '
                       + 'press Allow camera again.';
  }
  return state;
}

document.getElementById('allow').addEventListener('click', async () => {
  status.textContent = 'Waiting for your answer to the Chrome prompt…';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    // Show it briefly: seeing yourself is the clearest proof it worked, and it
    // also tells the user which camera Chrome picked.
    preview.hidden = false;
    preview.srcObject = stream;
    status.textContent = 'Camera allowed. Go back to your call and press '
                       + 'Start translating.';
    chrome.runtime.sendMessage({ type: 'cameraGranted' }).catch(() => {});
    setTimeout(() => {
      stream.getTracks().forEach((track) => track.stop());
      preview.srcObject = null;
      preview.hidden = true;
    }, 2500);
  } catch (error) {
    const state = await currentState();
    status.textContent = state === 'denied'
      ? 'Chrome has this extension on the blocked list. Open '
        + 'chrome://settings/content/camera, remove it, and try again.'
      : `The camera was not allowed: ${error.name} — ${error.message}. `
        + 'Press Allow camera and choose Allow in the prompt at the top of this tab.';
  }
});

document.getElementById('close').addEventListener('click', () => window.close());

refresh();
