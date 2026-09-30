/**
 * The only code SignAI ever runs on a web page - and it runs nothing until the
 * user (or a finished sentence with auto-post on) asks it to.
 *
 * It is injected on demand with chrome.scripting.executeScript, never through a
 * content_scripts entry in the manifest, so a page the user never posts to is
 * never touched: no overlay, no observer, no listener, no DOM change.
 *
 * Its whole job: find the chat box of whatever video call this page is, put the
 * translated sentence in it the way that page's own framework expects, and send.
 *
 *   - Tier 1: the element the user already has focused.
 *   - Tier 2: selectors for the calling apps people actually use.
 *   - Tier 3: anything editable that calls itself chat, message, composer,
 *             conversation or comment.
 *
 * Every tier ends in the same two problems, which is where most naive versions
 * fail: React and friends ignore a plain `.value = x` assignment, and rich text
 * editors (Teams, Discord, Slack) ignore everything except a real input event.
 */

(() => {
  if (window.__signaiContentLoaded) return;      // injected again: do not stack
  window.__signaiContentLoaded = true;

  const CHAT_WORDS = /(chat|message|composer|conversation|comment|send a message|type a message)/i;
  const SEND_WORDS = /(send|post|submit)/i;

  // ------------------------------------------------------------ dom helpers

  /** querySelectorAll that also walks into open shadow roots. */
  function deepQueryAll(selector, root = document, out = [], depth = 0) {
    if (depth > 8) return out;
    try {
      out.push(...root.querySelectorAll(selector));
    } catch { /* an invalid selector must not kill the whole search */ }
    const hosts = root.querySelectorAll('*');
    for (const host of hosts) {
      if (host.shadowRoot) deepQueryAll(selector, host.shadowRoot, out, depth + 1);
    }
    return out;
  }

  function isVisible(el) {
    if (!el || !el.isConnected) return false;
    // checkVisibility covers display:none, visibility, content-visibility and
    // zero opacity in one call, including on ancestors.
    if (typeof el.checkVisibility === 'function') {
      if (!el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) return false;
    } else {
      const style = getComputedStyle(el);
      if (style.display === 'none' || style.visibility === 'hidden'
          || style.opacity === '0') return false;
    }
    // Any real box will do. An absolute minimum width was tried and rejected:
    // in a narrow window a perfectly usable chat box is only a few pixels wide
    // to this measurement, and the composer then looks invisible.
    const box = el.getBoundingClientRect();
    return box.width * box.height > 0;
  }

  function isEditable(el) {
    if (!el) return false;
    const tag = el.tagName;
    if (tag === 'TEXTAREA') return !el.disabled && !el.readOnly;
    if (tag === 'INPUT') {
      const type = (el.getAttribute('type') || 'text').toLowerCase();
      return ['text', 'search', ''].includes(type) && !el.disabled && !el.readOnly;
    }
    return el.isContentEditable === true;
  }

  function describe(el) {
    const bits = [el.tagName.toLowerCase()];
    if (el.id) bits.push(`#${el.id}`);
    const label = el.getAttribute?.('aria-label') || el.getAttribute?.('placeholder');
    if (label) bits.push(`"${label.slice(0, 40)}"`);
    return bits.join(' ');
  }

  /** Every attribute a page might describe its chat box with, as one string. */
  function signature(el) {
    const attrs = ['aria-label', 'placeholder', 'data-testid', 'data-tid', 'data-qa',
                   'name', 'title', 'id', 'class', 'aria-placeholder', 'data-test'];
    const parts = attrs.map((a) => el.getAttribute?.(a) || '');
    // A label that sits next to the box counts too.
    const labelled = el.getAttribute?.('aria-labelledby');
    if (labelled) {
      for (const id of labelled.split(/\s+/)) {
        parts.push(document.getElementById(id)?.textContent || '');
      }
    }
    return parts.join(' ');
  }

  // --------------------------------------------------------- finding the box

  // Tier 2: the apps people actually call each other on. Order matters only in
  // that the first visible match wins.
  const PLATFORM_SELECTORS = [
    // Google Meet
    'textarea[name="chatTextInput"]',
    'textarea[aria-label*="chat" i]',
    'textarea[aria-label*="send a message" i]',
    // Zoom (web client)
    'textarea.chat-box__chat-textarea',
    'div.chat-rtf-box__editor[contenteditable="true"]',
    'textarea[placeholder*="chat" i]',
    // Microsoft Teams
    'div[data-tid="ckeditor-editor"][contenteditable="true"]',
    'div[data-tid="ckeditor"][contenteditable="true"]',
    'div[id="new-message-input"][contenteditable="true"]',
    // Discord
    'div[role="textbox"][contenteditable="true"][data-slate-editor="true"]',
    'div[class*="slateTextArea"][contenteditable="true"]',
    // Slack (including huddles)
    'div[data-qa="message_input"][contenteditable="true"]',
    'div.ql-editor[contenteditable="true"]',
    // Jitsi Meet
    'textarea#usermsg',
    'input#chatinput',
    'textarea[placeholder*="type a message" i]',
    // Webex
    'div[data-test="chat-input-area"]',
    'div[contenteditable="true"][aria-label*="message" i]',
    // Whereby, Skype, BigBlueButton and the long tail
    'textarea[id="message-input"]',
    'textarea[placeholder*="type" i]',
    'input[placeholder*="chat" i]',
    'textarea[aria-label*="message" i]',
    'div[role="textbox"][contenteditable="true"]',
  ];

  function tier1() {
    const active = document.activeElement;
    if (active && isEditable(active) && isVisible(active)) {
      // A focused search box is not a chat box; only trust focus when the
      // element looks like a composer or the page gives us nothing else.
      return { el: active, tier: 1, why: 'the focused input' };
    }
    // Focus may live inside an iframe's document that we are also injected into.
    return null;
  }

  function tier2() {
    for (const selector of PLATFORM_SELECTORS) {
      for (const el of deepQueryAll(selector)) {
        if (isEditable(el) && isVisible(el)) {
          return { el, tier: 2, why: `matched ${selector}` };
        }
      }
    }
    return null;
  }

  function tier3() {
    const candidates = deepQueryAll('textarea, input[type="text"], input[type="search"], '
                                  + '[contenteditable="true"], [role="textbox"]');
    let best = null;
    for (const el of candidates) {
      if (!isEditable(el) || !isVisible(el)) continue;
      const text = signature(el);
      let score = 0;
      if (CHAT_WORDS.test(text)) score += 10;
      if (/type|write|say/i.test(text)) score += 2;
      if (el.tagName === 'TEXTAREA' || el.isContentEditable) score += 2;
      // A composer sits at the bottom of its panel, and is wider than it is tall.
      const box = el.getBoundingClientRect();
      if (box.top > window.innerHeight * 0.5) score += 3;
      if (box.width > box.height * 3) score += 1;
      if (/search|find|url|address/i.test(text)) score -= 12;
      if (score <= 0) continue;
      if (!best || score > best.score) best = { el, score, tier: 3, why: `heuristic score ${score}` };
    }
    return best;
  }

  function findChatBox() {
    const focused = tier1();
    // Prefer focus only when it really looks like a chat box; otherwise let the
    // platform selectors have their say and fall back to focus afterwards.
    if (focused && CHAT_WORDS.test(signature(focused.el))) return focused;
    return tier2() || focused || tier3();
  }

  // ------------------------------------------------------ opening the panel

  const CHAT_BUTTON_SELECTORS = [
    'button[aria-label*="chat" i]',
    'button[data-tooltip*="chat" i]',
    'button[aria-label*="messages" i]',
    'div[role="button"][aria-label*="chat" i]',
    'button[data-tid*="chat" i]',
    'button[title*="chat" i]',
  ];

  /** Open the chat panel if it is closed, and say whether anything was clicked. */
  function openChatPanel() {
    for (const selector of CHAT_BUTTON_SELECTORS) {
      for (const button of deepQueryAll(selector)) {
        if (!isVisible(button)) continue;
        const pressed = button.getAttribute('aria-pressed');
        const expanded = button.getAttribute('aria-expanded');
        if (pressed === 'true' || expanded === 'true') return false;   // already open
        button.click();
        return true;
      }
    }
    return false;
  }

  // ----------------------------------------------------------- writing text

  /**
   * Set the value of an input or textarea so that React, Vue and Angular notice.
   *
   * React tracks the last value it wrote on the DOM node itself; a plain
   * `el.value = x` leaves that tracker untouched, React compares, sees no
   * change, and throws the text away the moment anything re-renders. Going
   * through the prototype's own setter is what makes the frameworks accept it.
   */
  function setNativeValue(el, value) {
    const proto = el instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (setter) setter.call(el, value);
    else el.value = value;
  }

  function fireInput(el, text) {
    el.dispatchEvent(new InputEvent('input', {
      bubbles: true, composed: true, data: text, inputType: 'insertText',
    }));
    el.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
  }

  function writeIntoTextarea(el, text) {
    el.focus();
    const existing = el.value || '';
    const separator = existing && !/\s$/.test(existing) ? ' ' : '';
    setNativeValue(el, existing + separator + text);
    try {
      el.setSelectionRange(el.value.length, el.value.length);
    } catch { /* inputs of some types refuse a selection; harmless */ }
    fireInput(el, text);
  }

  function writeIntoContentEditable(el, text) {
    el.focus();
    // Put the caret at the end, or the text lands wherever the caret last was.
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(el);
    range.collapse(false);
    selection.removeAllRanges();
    selection.addRange(range);

    el.dispatchEvent(new InputEvent('beforeinput', {
      bubbles: true, composed: true, cancelable: true,
      inputType: 'insertText', data: text,
    }));

    let inserted = false;
    try {
      // Deprecated, and still the only call that Slate, Quill, CKEditor and
      // Draft all understand, because it produces the same internal transaction
      // as a keystroke.
      inserted = document.execCommand('insertText', false, text);
    } catch { inserted = false; }

    if (!inserted) {
      range.insertNode(document.createTextNode(text));
      range.collapse(false);
      selection.removeAllRanges();
      selection.addRange(range);
    }

    el.dispatchEvent(new InputEvent('input', {
      bubbles: true, composed: true, inputType: 'insertText', data: text,
    }));
  }

  function write(el, text) {
    if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') writeIntoTextarea(el, text);
    else writeIntoContentEditable(el, text);
  }

  // ------------------------------------------------------------- sending it

  function findSendButton(el) {
    // Look near the composer first: a "send" button on the other side of the
    // page belongs to something else.
    const scopes = [];
    let node = el;
    for (let i = 0; i < 6 && node; i += 1) {
      node = node.parentElement || node.getRootNode?.()?.host;
      if (node) scopes.push(node);
    }
    scopes.push(document);
    const selectors = [
      'button[type="submit"]',
      'button[aria-label*="send" i]',
      'button[data-tooltip*="send" i]',
      'button[data-testid*="send" i]',
      'button[data-tid*="send" i]',
      'div[role="button"][aria-label*="send" i]',
      'button[title*="send" i]',
    ];
    for (const scope of scopes) {
      for (const selector of selectors) {
        let found = [];
        try { found = [...scope.querySelectorAll(selector)]; } catch { found = []; }
        for (const button of found) {
          if (isVisible(button) && !button.disabled) return button;
        }
      }
    }
    return null;
  }

  function pressEnter(el) {
    const base = {
      key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
      bubbles: true, cancelable: true, composed: true,
    };
    el.dispatchEvent(new KeyboardEvent('keydown', base));
    el.dispatchEvent(new KeyboardEvent('keypress', base));
    el.dispatchEvent(new KeyboardEvent('keyup', base));
  }

  function send(el) {
    const button = findSendButton(el);
    if (button) {
      button.click();
      return `clicked ${describe(button)}`;
    }
    pressEnter(el);
    return 'pressed Enter';
  }

  // --------------------------------------------------------------- the entry

  async function insert({ text, prefix = '[ISL] ', autoSend = true }) {
    if (!text) return { ok: false, detail: 'nothing to insert' };
    const message = `${prefix}${text}`;

    let target = findChatBox();
    if (!target) {
      // The chat panel is probably closed - open it and look again, because on
      // Meet and Teams the composer does not exist until the panel is open.
      if (openChatPanel()) {
        await new Promise((resolve) => setTimeout(resolve, 700));
        target = findChatBox();
      }
    }
    if (!target) {
      return { ok: false, detail: 'no chat box found on this page', host: location.host };
    }

    write(target.el, message);
    const sent = autoSend ? send(target.el) : 'left in the box';
    return {
      ok: true,
      tier: target.tier,
      detail: `${target.why}; ${sent}`,
      element: describe(target.el),
      host: location.host,
    };
  }

  // In the test page there is no extension runtime; the functions below are
  // still exercised directly through window.__signaiInsert.
  const runtime = (typeof chrome !== 'undefined' && chrome.runtime?.onMessage)
    ? chrome.runtime : null;

  runtime?.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg?.type === 'signai-insert') {
      insert(msg).then(sendResponse);
      return true;
    }
    if (msg?.type === 'signai-probe') {
      const found = findChatBox();
      sendResponse(found
        ? { ok: true, tier: found.tier, element: describe(found.el), host: location.host }
        : { ok: false, detail: 'no chat box visible', host: location.host });
      return true;
    }
    return false;
  });

  // Exposed for the test page in tests/, which drives the same code in a normal
  // browser tab where chrome.runtime does not exist.
  window.__signaiInsert = insert;
  window.__signaiFind = findChatBox;
})();
