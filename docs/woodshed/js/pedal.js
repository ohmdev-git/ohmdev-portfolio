// Foot switch / remote: Bluetooth page-turner pedals (they type keys like PageDown or
// ArrowDown) and, where the browser supports Web MIDI, MIDI pedals. Any key or MIDI
// message can be taught to any action with "Set" (learn mode).
import { $, esc, store, toast } from './util.js?v=202611261434';
import { t as tr } from './i18n.js?v=202611261434';

const KEY = 'woodshed:pedal';
export const ACTION_IDS = ['play', 'restartA', 'toA', 'loop', 'back', 'fwd', 'prevSec', 'nextSec', 'tap', 'met', 'slower', 'faster', 'prevSong', 'nextSong'];
// Most page-turner pedals send Up/Down arrows or PageUp/PageDown out of the box.
export const DEFAULTS = { PageDown: 'play', ArrowDown: 'play', PageUp: 'restartA', ArrowUp: 'restartA' };
const NEVER = new Set(['Escape', 'Tab', 'Shift', 'Control', 'Alt', 'Meta', 'AltGraph', 'CapsLock', 'Fn', 'FnLock', 'OS', 'Hyper', 'Super', 'Unidentified', 'Process', 'Dead', 'NumLock', 'ScrollLock']);
const MAX_BINDINGS = 40;
const ICON = { play: '⏯', restartA: '⏮', toA: '↩', loop: '⟲', back: '−2s', fwd: '+2s', prevSec: '⇤', nextSec: '⇥', tap: '♩', met: '♪', slower: '−5%', faster: '+5%', prevSong: '⏪', nextSong: '⏩' };

let app, ACT, isTyping;
let P = { on: false, map: {}, midi: false };
let learning = null, learnT = null, hudT = null, lastPress = '';
let midiAccess = null;
const ccState = {};

/* ---------- storage ---------- */
function sane(o) {
  const out = { on: false, map: {}, midi: false };
  if (!o || typeof o !== 'object') return { ...out, map: { ...DEFAULTS } };
  out.on = o.on === true; out.midi = o.midi === true;
  const m = o.map && typeof o.map === 'object' && !Array.isArray(o.map) ? o.map : null;
  if (!m) out.map = { ...DEFAULTS };
  else for (const k of Object.keys(m).slice(0, MAX_BINDINGS)) if (typeof k === 'string' && k.length <= 40 && ACTION_IDS.includes(m[k])) out.map[k] = m[k];
  return out;
}
const save = () => store.set(KEY, P);

/* ---------- key ids + names ---------- */
// Named keys (PageDown, ArrowUp, Enter, MediaPlayPause…) by name; printable keys by
// physical position so a Thai/English layout switch doesn't break a binding.
export function keyId(e) {
  const k = e.key || '';
  if (k.length > 1 && !NEVER.has(k)) return k;
  if (k === ' ' || e.code === 'Space') return 'Space';
  if (k.length === 1) return e.code ? 'code:' + e.code : 'k:' + k;
  return '';
}
const ARROWS = { ArrowUp: '↑', ArrowDown: '↓', ArrowLeft: '←', ArrowRight: '→' };
export function keyName(id) {
  let m;
  if (ARROWS[id]) return ARROWS[id];
  if (id === 'PageUp') return 'Page Up';
  if (id === 'PageDown') return 'Page Down';
  if (id === 'Space') return 'Space';
  if ((m = /^midi:n(\d+)$/.exec(id))) return 'MIDI ' + tr('pd.note') + ' ' + m[1];
  if ((m = /^midi:cc(\d+)$/.exec(id))) return 'MIDI CC ' + m[1];
  if ((m = /^midi:pc(\d+)$/.exec(id))) return 'MIDI PC ' + m[1];
  if ((m = /^code:Key([A-Z])$/.exec(id))) return m[1];
  if ((m = /^code:Digit(\d)$/.exec(id))) return m[1];
  if ((m = /^code:Numpad(.+)$/.exec(id))) return 'Num ' + m[1];
  if ((m = /^code:(.+)$/.exec(id))) return m[1];
  if ((m = /^k:(.)$/.exec(id))) return m[1].toUpperCase();
  if (/^Media/.test(id)) return id.replace(/^Media/, '').replace(/([a-z])([A-Z])/g, '$1 $2');
  return id.replace(/([a-z])([A-Z])/g, '$1 $2');
}

/* ---------- running an action ---------- */
function hud(text) {
  const el = $('pdHud'); if (!el) return;
  el.textContent = text; el.hidden = false; el.classList.remove('go'); void el.offsetWidth; el.classList.add('go');
  clearTimeout(hudT); hudT = setTimeout(() => { el.hidden = true; el.classList.remove('go'); }, 900);
}
function fire(id, src) {
  const a = P.map[id];
  if (!a || !ACT[a]) return false;
  lastPress = id;
  let r;
  try { r = ACT[a](); } catch (e) { console.error(e); }
  if (r !== false) hud((ICON[a] || '') + '  ' + tr('pd.a.' + a));
  renderStatus();
  return true;
}

/* ---------- learn mode ---------- */
function startLearn(action) {
  stopLearn(true);
  learning = action;
  clearTimeout(learnT); learnT = setTimeout(() => { stopLearn(); toast(tr('pd.learnTimeout')); }, 12000);
  render();
}
function stopLearn(quiet) {
  clearTimeout(learnT);
  if (learning == null) return;
  learning = null;
  if (!quiet) render();
}
function assign(id) {
  const a = learning;
  if (!a || !id) return;
  const prev = P.map[id];
  if (!(id in P.map) && Object.keys(P.map).length >= MAX_BINDINGS) { stopLearn(); toast(tr('pd.tooMany')); return; }
  P.map[id] = a;
  if (!P.on) P.on = true;
  save();
  stopLearn();
  toast(prev && prev !== a ? tr('pd.moved', { k: keyName(id), a: tr('pd.a.' + a), b: tr('pd.a.' + prev) }) : tr('pd.assigned', { k: keyName(id), a: tr('pd.a.' + a) }));
  render();
}

/* ---------- keyboard ---------- */
function onKey(e) {
  if (learning) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); stopLearn(); return; }
    const id = keyId(e);
    if (!id || NEVER.has(e.key)) return;          // wait for a real key
    e.preventDefault(); e.stopImmediatePropagation();
    if (e.repeat) return;
    assign(id);
    return;
  }
  if (!P.on || e.ctrlKey || e.metaKey || e.altKey) return;
  if (isTyping(e.target)) return;
  const id = keyId(e);
  if (!id || !P.map[id]) return;
  e.preventDefault(); e.stopImmediatePropagation();
  if (e.repeat) return;                          // a held pedal fires once
  fire(id, 'key');
}

/* ---------- Web MIDI ---------- */
export const midiSupported = () => typeof navigator !== 'undefined' && typeof navigator.requestMIDIAccess === 'function';
export function midiId(data) {
  if (!data || data.length < 2) return '';
  const st = data[0] & 0xf0, d1 = data[1] & 0x7f, d2 = data.length > 2 ? data[2] & 0x7f : 0;
  if (st === 0x90 && d2 > 0) return 'midi:n' + d1;
  if (st === 0xc0) return 'midi:pc' + d1;
  if (st === 0xb0) {
    const ch = data[0] & 0x0f, k = ch + ':' + d1, was = ccState[k] || 0;
    ccState[k] = d2;
    return d2 >= 64 && was < 64 ? 'midi:cc' + d1 : '';   // fire on the press edge only
  }
  return '';
}
function onMidi(ev) {
  const id = midiId(ev.data);
  if (!id) return;
  if (learning) { assign(id); return; }
  if (P.on) fire(id, 'midi');
}
function hookInputs() {
  if (!midiAccess) return;
  midiAccess.inputs.forEach(inp => { inp.onmidimessage = onMidi; });
  renderStatus();
}
async function connectMidi(user) {
  if (!midiSupported()) { if (user) toast(tr('pd.midiNo')); return; }
  try {
    midiAccess = await navigator.requestMIDIAccess({ sysex: false });
    midiAccess.onstatechange = hookInputs;
    hookInputs();
    if (!P.midi) { P.midi = true; save(); }
    if (user) toast(midiCount() ? tr('pd.midiOk', { n: midiCount() }) : tr('pd.midiNone'));
  } catch (e) {
    midiAccess = null;
    if (P.midi) { P.midi = false; save(); }
    if (user) toast(tr('pd.midiDenied'));
  }
  render();
}
function midiCount() { let n = 0; if (midiAccess) midiAccess.inputs.forEach(i => { if (i.state !== 'disconnected') n++; }); return n; }
function midiNames() { const a = []; if (midiAccess) midiAccess.inputs.forEach(i => { if (i.state !== 'disconnected') a.push(i.name || 'MIDI'); }); return a; }

/* ---------- UI ---------- */
function renderStatus() {
  const el = $('pdStatus'); if (!el) return;
  const parts = [];
  if (midiAccess) { const n = midiNames(); parts.push(n.length ? tr('pd.midiDevs', { n: n.join(', ') }) : tr('pd.midiNone')); }
  if (lastPress && P.map[lastPress]) parts.push(tr('pd.last', { k: keyName(lastPress), a: tr('pd.a.' + P.map[lastPress]) }));
  el.textContent = parts.join(' · ');
  el.hidden = !parts.length;
}
export function render() {
  if (!$('pdCard')) return;
  $('pdOn').checked = P.on;
  $('pdBody').hidden = !P.on && !learning;
  const byAction = {};
  for (const [k, a] of Object.entries(P.map)) (byAction[a] = byAction[a] || []).push(k);
  $('pdList').innerHTML = ACTION_IDS.map(a => {
    const keys = byAction[a] || [], L = learning === a;
    return `<li class="pd${L ? ' learn' : ''}${keys.length ? ' has' : ''}" data-a="${a}">
      <span class="pn">${esc(tr('pd.a.' + a))}</span>
      <span class="pk">${L ? `<em>${esc(tr('pd.pressNow'))}</em>` : keys.map(k => `<button class="pkey" data-unbind="${esc(k)}" title="${esc(tr('pd.remove'))}" aria-label="${esc(tr('pd.removeX', { k: keyName(k) }))}">${esc(keyName(k))}<i>×</i></button>`).join('')}</span>
      <button class="btn small${L ? ' on' : ''}" data-learn="${a}">${esc(L ? tr('cancel') : tr('set'))}</button>
    </li>`;
  }).join('');
  const mb = $('pdMidi');
  mb.hidden = !midiSupported();
  mb.textContent = midiAccess ? tr('pd.midiOn') : tr('pd.midiConnect');
  mb.disabled = !!midiAccess;
  $('pdMidiNo').hidden = midiSupported();
  renderStatus();
}

export const _state = () => ({ P: JSON.parse(JSON.stringify(P)), learning });

export function init(a, actions, typingTest) {
  app = a; ACT = actions; isTyping = typingTest || (() => false);
  P = sane(store.get(KEY, null));
  document.addEventListener('keydown', onKey, true);
  $('pdOn').addEventListener('change', () => { P.on = $('pdOn').checked; if (!P.on) stopLearn(true); save(); render(); $('pdOn').blur(); });
  $('pdList').addEventListener('click', e => {
    const l = e.target.closest('[data-learn]'), u = e.target.closest('[data-unbind]');
    if (l) { const act = l.dataset.learn; if (learning === act) stopLearn(); else startLearn(act); }
    else if (u) { delete P.map[u.dataset.unbind]; if (lastPress === u.dataset.unbind) lastPress = ''; save(); render(); }
  });
  $('pdReset').addEventListener('click', () => { stopLearn(true); P.map = { ...DEFAULTS }; save(); render(); toast(tr('pd.resetDone')); });
  $('pdMidi').addEventListener('click', () => connectMidi(true));
  app.on('lang', render);
  // reconnect a MIDI pedal silently only if the browser already granted it
  if (P.midi && midiSupported() && navigator.permissions && navigator.permissions.query) {
    navigator.permissions.query({ name: 'midi' }).then(s => { if (s.state === 'granted') connectMidi(false); }).catch(() => {});
  }
  render();
}
