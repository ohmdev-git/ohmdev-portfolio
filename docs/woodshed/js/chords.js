// Chord sheet: chords placed on the song's timeline. A "now / next" display follows
// the playhead; chords are tapped in while the song plays, typed as text, or edited
// on the lane. Stored per song as woodshed:chords:<video id>.
import { $, esc, store, fmt, fmt2, toast, uid, debounce, clamp } from './util.js?v=202611261719';
import { t as tr } from './i18n.js?v=202611261719';
import * as Met from './metronome.js?v=202611261719';
import * as Theory from './theory.js?v=202611261719';

export const IDS = ['', 'm', '5', '7', 'maj7', 'm7', 'mMaj7', '6', 'm6', '69', 'm69', 'sus2', 'sus4', '7sus4', 'add9', 'madd9', '9', 'maj9', 'm9', '11', 'm11', '13', 'maj13', 'm13', 'dim', 'dim7', 'm7b5', 'aug', '7#5', 'maj7#5', '7b5', '7b9', '7#9', '9#5', '9#11'];
const ALIAS = {
  maj: '', M: '', major: '', ma: '', min: 'm', mi: 'm', minor: 'm', '-': 'm',
  M7: 'maj7', ma7: 'maj7', Maj7: 'maj7', j7: 'maj7', M9: 'maj9', Maj9: 'maj9', M13: 'maj13', Maj13: 'maj13',
  mmaj7: 'mMaj7', mM7: 'mMaj7', mMa7: 'mMaj7', 'm+7': 'mMaj7', minmaj7: 'mMaj7', '-maj7': 'mMaj7', '-M7': 'mMaj7',
  '-7': 'm7', min7: 'm7', mi7: 'm7', '-6': 'm6', '-9': 'm9', '-11': 'm11', '-13': 'm13', min9: 'm9', min6: 'm6', min11: 'm11',
  '+': 'aug', '#5': 'aug', '+5': 'aug', aug5: 'aug', '+7': '7#5', '7+': '7#5', aug7: '7#5', '7+5': '7#5', '7aug': '7#5', 'maj7+5': 'maj7#5', 'M7#5': 'maj7#5', 'M7+5': 'maj7#5',
  o: 'dim', '°': 'dim', o7: 'dim7', '°7': 'dim7', dim5: 'dim', ø: 'm7b5', ø7: 'm7b5', h7: 'm7b5', hdim: 'm7b5', 'm7-5': 'm7b5', '-7b5': 'm7b5', min7b5: 'm7b5',
  '7-5': '7b5', '7-9': '7b9', '7+9': '7#9', '9+5': '9#5', '9+11': '9#11', '7#11': '9#11',
  sus: 'sus4', '7sus': '7sus4', '4': 'sus4', '2': 'sus2', add2: 'add9', madd2: 'madd9', '-add9': 'madd9', '6/9': '69', 'm6/9': 'm69', '-69': 'm69',
  dom: '7', dom7: '7', Δ: 'maj7', Δ7: 'maj7', Δ9: 'maj9', mΔ7: 'mMaj7', '-Δ7': 'mMaj7', p: '5', power: '5'
};
const LET = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const MAX = 400;
const LAG = 0.15;              // reaction time when tapping chords in while the song plays (real seconds)

/* ---------- parsing ---------- */
function quality(q) {
  q = q.replace(/[()\s]/g, '');
  if (IDS.includes(q)) return q;
  if (q in ALIAS) return ALIAS[q];
  const q2 = q.replace(/^min(?=\d|M|maj)/, 'm').replace(/^-(?=\d|M|maj)/, 'm').replace(/^Maj(?=\d)/, 'maj');
  if (IDS.includes(q2)) return q2;
  if (q2 in ALIAS) return ALIAS[q2];
  return null;
}
// "Am7", "F#m7b5", "Bb/D", "C6/9", "Ebmaj7", "N.C." → {n, r, ty, b} (r = -1 for "no chord")
export function parseChord(tok) {
  let s = String(tok || '').trim().replace(/♯/g, '#').replace(/♭/g, 'b');
  if (!s || s.length > 16) return null;
  if (/^n\.?c\.?$/i.test(s)) return { n: 'N.C.', r: -1, ty: '', b: -1 };
  const m = /^([A-Ga-g])([#b]?)(.*)$/.exec(s);
  if (!m) return null;
  const L = m[1].toUpperCase(), acc = m[2];
  const r = (LET[L] + (acc === '#' ? 1 : acc === 'b' ? 11 : 0)) % 12;
  let rest = m[3], b = -1, bn = '';
  const sm = /^(.*?)\/([A-Ga-g])([#b]?)$/.exec(rest);
  if (sm) {
    const BL = sm[2].toUpperCase();
    b = (LET[BL] + (sm[3] === '#' ? 1 : sm[3] === 'b' ? 11 : 0)) % 12; bn = BL + sm[3]; rest = sm[1];
  }
  const ty = quality(rest);
  if (ty == null) return null;
  return { n: L + acc + ty + (b >= 0 && b !== r ? '/' + bn : ''), r, ty, b: b === r ? -1 : b };
}
// split text into chord tokens; "%" or "/" repeats the previous chord
export function parseLine(text) {
  const out = [], bad = [];
  for (const tok of String(text || '').split(/[\s,|]+/).filter(Boolean)) {
    if ((tok === '%' || tok === '/' || tok === '-') && out.length) { out.push({ ...out[out.length - 1] }); continue; }
    const c = parseChord(tok);
    if (c) out.push(c); else bad.push(tok);
  }
  return { chords: out, bad };
}

/* ---------- state ---------- */
let app, S;
let C = [];                    // sorted by t
let sel = null, cur = -2, undo = [], drag = null, linkPending = null, clearArm = 0;
const UI = { open: false, sev: false };
const key = vid => 'woodshed:chords:' + (vid || S.vid);
const sig = L => L.map(c => c.t.toFixed(1) + c.n).join('|');
function clean(v) {
  if (!Array.isArray(v)) return [];
  const out = [];
  for (const x of v) {
    if (!x || !isFinite(x.t) || x.t < 0) continue;
    const p = parseChord(x.n); if (!p) continue;
    out.push({ id: typeof x.id === 'string' && x.id ? x.id.slice(0, 24) : uid(), t: +x.t, ...p });
    if (out.length >= MAX) break;
  }
  return out.sort((a, b) => a.t - b.t);
}
function load() { C = S.vid ? clean(store.get(key(), [])) : []; sel = null; cur = -2; undo = []; linkPending = null; }
const writeC = (vid, list) => {
  if (!vid) return;
  if (!list.length) { store.del ? store.del(key(vid)) : localStorage.removeItem(key(vid)); return; }
  store.set(key(vid), list.map(({ id, t, n }) => ({ id, t: +t.toFixed(2), n })));
};
const saveSoon = debounce(writeC, 250);
function save() { saveSoon.flush(); writeC(S.vid, C); }
function snapshot() { undo.push(C.map(c => ({ ...c }))); if (undo.length > 30) undo.shift(); }
function changed(soon) { C.sort((a, b) => a.t - b.t); if (soon) saveSoon(S.vid, C); else save(); cur = -2; renderAll(); }
export const list = () => C.map(c => ({ ...c }));

/* ---------- finding the chord at a time ---------- */
function idxAt(t) {               // last chord starting at or before t
  let lo = 0, hi = C.length - 1, ans = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (C[m].t <= t + 0.02) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}
const loopOn = () => S.loop && S.A != null && S.B != null;

// called by the player clock (~30 ms)
export function tick(t) {
  if (!C.length) return;
  const i = idxAt(t);
  let nextT, next;
  if (i + 1 < C.length) { next = C[i + 1]; nextT = next.t; }
  if (loopOn() && t < S.B + 0.05 && (nextT == null || nextT >= S.B)) {   // the loop wraps before the next change
    const j = idxAt(S.A); next = j >= 0 ? C[j] : null; nextT = S.B;
    if (next && i >= 0 && next.id === C[i].id) next = null;
  }
  const from = i >= 0 ? C[i].t : 0;
  const k = i + ':' + (next ? next.id : '-');
  if (k !== cur) {
    cur = k;
    $('csCur').textContent = i >= 0 ? C[i].n : '–';
    $('csCur').disabled = !(i >= 0 && C[i].r >= 0);
    $('csNext').textContent = next ? next.n : '–';
    $('csLane').querySelectorAll('.cs.cur').forEach(x => x.classList.remove('cur'));
    if (i >= 0) { const el = $('csLane').querySelector(`[data-cs="${CSS.escape(C[i].id)}"]`); if (el) el.classList.add('cur'); }
  }
  const f = nextT != null && nextT > from ? clamp((t - from) / (nextT - from), 0, 1) : 0;
  $('csFill').style.width = (f * 100).toFixed(1) + '%';
}

/* ---------- editing ---------- */
function place(p) {
  if (!app.ok() || !S.dur) { toast(tr(S.vid ? 'waitLoad' : 'loadSongFirst')); return; }
  snapshot();
  if (sel) {
    const x = C.find(c => c.id === sel);
    if (x) { Object.assign(x, p); changed(); flash(x.id); return; }
    sel = null;
  }
  if (C.length >= MAX) { undo.pop(); toast(tr('cs.many')); return; }
  const playing = S.state === 1;
  let t = app.now() - (playing ? LAG * S.rate : 0);
  t = clamp(t, 0, Math.max(0, S.dur - 0.05));
  const near = C.find(c => Math.abs(c.t - t) < 0.25 * Math.max(1, S.rate));
  if (near) { Object.assign(near, p); changed(); flash(near.id); return; }
  const x = { id: uid(), t, ...p };
  C.push(x); changed(); flash(x.id);
  if (!playing && C.length === 1) toast(tr('cs.firstTip'));
}
function flash(id) {
  const el = $('csLane').querySelector(`[data-cs="${CSS.escape(id)}"]`);
  if (el) { el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); }
}
function del(id) { snapshot(); C = C.filter(c => c.id !== id); if (sel === id) sel = null; changed(); }
function nudge(id, d) {
  const i = C.findIndex(c => c.id === id); if (i < 0) return;
  snapshot();
  const lo = i > 0 ? C[i - 1].t + 0.05 : 0, hi = i + 1 < C.length ? C[i + 1].t - 0.05 : Math.max(0, S.dur - 0.05);
  C[i].t = clamp(C[i].t + d, lo, Math.max(lo, hi));
  changed(true); app.userSeek(C[i].t);
}
function toPlayhead(id) {
  const x = C.find(c => c.id === id); if (!x || !app.ok()) return;
  snapshot();
  const t = clamp(app.now(), 0, Math.max(0, S.dur - 0.05));
  const clash = C.find(c => c.id !== id && Math.abs(c.t - t) < 0.05);
  if (clash) C = C.filter(c => c !== clash);
  x.t = t; changed();
}
function doUndo() {
  if (!undo.length) { toast(tr('cs.noUndo')); return; }
  C = undo.pop(); if (sel && !C.some(c => c.id === sel)) sel = null; changed();
  toast(tr('cs.undone'));
}
function clearAll() {
  if (!C.length) return;
  if (Date.now() - clearArm > 3000) { clearArm = Date.now(); toast(tr('cs.clearAgain')); $('csClear').classList.add('arm'); setTimeout(() => $('csClear') && $('csClear').classList.remove('arm'), 3000); return; }
  clearArm = 0; $('csClear').classList.remove('arm');
  snapshot(); C = []; sel = null; changed(); toast(tr('cs.cleared'));
}
// typed chords fill the A–B loop: one per bar when the song's tempo is set, else evenly
function fillLoop(text) {
  if (!app.ok() || !S.dur) { toast(tr(S.vid ? 'waitLoad' : 'loadSongFirst')); return false; }
  const { chords, bad } = parseLine(text);
  if (bad.length) { toast(tr('cs.bad', { x: bad.slice(0, 3).join(' ') })); return false; }
  if (!chords.length) { toast(tr('cs.typeSome')); return false; }
  if (S.A == null || S.B == null) { toast(tr('cs.needAB')); return false; }
  const A = S.A, B = S.B;
  let times = [];
  if (Met.hasBpm()) {
    const bar = Met.countBeats(1) * 60 / Met.bpm();
    for (let k = 0; A + k * bar < B - 0.05 && k < MAX; k++) times.push(A + k * bar);
  } else times = chords.map((_, k) => A + k * (B - A) / chords.length);
  const add = times.map((t, k) => ({ id: uid(), t, ...chords[k % chords.length] }));
  snapshot();
  C = C.filter(c => c.t < A - 0.01 || c.t >= B - 0.01).concat(add).slice(0, MAX);
  sel = null; changed();
  toast(tr(Met.hasBpm() ? 'cs.filledBars' : 'cs.filledEven', { n: add.length }));
  return true;
}

/* ---------- share link ---------- */
export function linkPart() {
  if (!C.length) return '';
  const s = C.map(c => Math.round(c.t * 10).toString(36) + '.' + c.n).join(',');
  const enc = encodeURIComponent(s);
  return enc.length <= 1500 ? enc : null;          // null: too long to share in a link
}
function decodeLink(s) {
  const out = [];
  for (const part of String(s || '').split(',').slice(0, MAX)) {
    const m = /^([0-9a-z]{1,6})\.(.+)$/.exec(part); if (!m) continue;
    const t = parseInt(m[1], 36) / 10, p = parseChord(m[2]);
    if (p && isFinite(t)) out.push({ id: uid(), t, ...p });
  }
  return out.sort((a, b) => a.t - b.t);
}
// chords that came with a shared link: take them if the song has none, else offer
export function fromLink(s) {
  const L = decodeLink(s);
  if (!L.length || !S.vid) return;
  if (!C.length) { C = L; save(); cur = -2; renderAll(); toast(tr('cs.fromLink', { n: L.length })); return; }
  if (sig(L) === sig(C)) return;
  linkPending = L; setOpen(true); renderAll();
}

/* ---------- rendering ---------- */
function keyChords() {
  const D = Theory.data(), TH = Theory._state;
  if (!D || !TH || !TH.key) return { label: '', list: [] };
  const K = D.keys[TH.key.m][TH.key.r];
  const src = UI.sev ? K.sev : K.triads;
  return { label: tr(TH.key.m === 'major' ? 'th.keyMaj' : 'th.keyMin', { k: K.tonic }), list: src.map((c, i) => ({ n: c.s, rn: K.rn[i] })) };
}
function renderEditor() {
  $('csEdBtn').setAttribute('aria-pressed', UI.open);
  $('csEdBtn').classList.toggle('on', UI.open);
  $('csEd').hidden = !UI.open;
  if (!UI.open) return;
  const K = keyChords();
  $('csKey').textContent = K.label;
  $('csSev').querySelectorAll('[data-sev]').forEach(b => b.classList.toggle('on', (b.dataset.sev === '1') === UI.sev));
  const used = [...new Set(C.map(c => c.n))].filter(n => !K.list.some(k => k.n === n));
  $('csPal').innerHTML = K.list.map(k => `<button class="kc" data-n="${esc(k.n)}"><small>${esc(k.rn)}</small><b>${esc(k.n)}</b></button>`).join('') +
    used.map(n => `<button class="kc used" data-n="${esc(n)}"><small>${esc(tr('cs.inSong'))}</small><b>${esc(n)}</b></button>`).join('') +
    `<button class="kc nc" data-n="N.C."><small>${esc(tr('cs.break'))}</small><b>N.C.</b></button>`;
  const x = sel && C.find(c => c.id === sel);
  $('csSel').hidden = !x;
  $('csPalHint').textContent = tr(x ? 'cs.palChange' : S.state === 1 ? 'cs.palTapLive' : 'cs.palTap');
  if (x) {
    $('csSelName').textContent = x.n;
    $('csSelT').textContent = fmt2(x.t);
    $('csSelChart').disabled = x.r < 0;
  }
  $('csUndo').disabled = !undo.length;
  $('csClear').disabled = !C.length;
  const lk = $('csLink');
  lk.hidden = !linkPending;
  if (linkPending) $('csLinkMsg').textContent = tr('cs.linkHas', { n: linkPending.length, m: C.length });
}
function renderLane() {
  const lane = $('csLane');
  const has = !!(S.dur && C.length);
  lane.classList.toggle('has', has);
  lane.classList.toggle('edit', UI.open);
  $('csNow').hidden = !C.length;
  if (!has) { lane.innerHTML = ''; return; }
  const W = lane.clientWidth || 1, v = app.view();
  const vis = C.filter((c, i) => c.t < v.e && (i + 1 >= C.length || C[i + 1].t > v.s));
  lane.innerHTML = vis.map((c, k) => {
    const p = Math.max(0, app.pct(Math.max(c.t, v.s)));
    const nx = k + 1 < vis.length ? app.pct(vis[k + 1].t) : 100;
    const w = Math.max(0, (nx - p) / 100 * W - 2);
    return `<button class="cs${c.id === sel ? ' sel' : ''}${c.r < 0 ? ' nc' : ''}" data-cs="${esc(c.id)}" style="left:${p}%;width:${Math.max(w, 2).toFixed(1)}px" title="${esc(c.n + ' · ' + fmt(c.t))}">${w >= 18 ? `<span>${esc(c.n)}</span>` : ''}</button>`;
  }).join('');
  cur = -2;
}
function renderAll() { renderLane(); renderEditor(); if (app.ok()) tick(app.now()); }
function setOpen(o) {
  UI.open = !!o; if (!UI.open) sel = null;
  store.set('woodshed:chordEd', { open: UI.open, sev: UI.sev });
  renderAll();
}

/* ---------- lane pointer: tap = jump (and select while editing), drag = move (editing) ---------- */
function laneT(x) { const r = $('csLane').getBoundingClientRect(), v = app.view(); return v.s + clamp((x - r.left) / r.width, 0, 1) * (v.e - v.s); }
function onDown(e) {
  const b = e.target.closest('[data-cs]'); if (!b || e.button > 0) return;
  // chips can be slivers in the full-song view: pick the chord whose span holds the pointer
  const i = idxAt(laneT(e.clientX) - 0.02), x = i >= 0 ? C[i] : C.find(c => c.id === b.dataset.cs); if (!x) return;
  drag = { id: x.id, x0: e.clientX, t0: x.t, moved: false, snap: false };
  if (UI.open) { try { $('csLane').setPointerCapture(e.pointerId); } catch (err) {} e.preventDefault(); }
}
function onMove(e) {
  if (!drag || !UI.open) return;
  if (!drag.moved && Math.abs(e.clientX - drag.x0) < 6) return;
  if (!drag.moved) { drag.moved = true; snapshot(); }
  const i = C.findIndex(c => c.id === drag.id); if (i < 0) return;
  const lo = i > 0 ? C[i - 1].t + 0.05 : 0, hi = i + 1 < C.length ? C[i + 1].t - 0.05 : Math.max(0, S.dur - 0.05);
  C[i].t = clamp(laneT(e.clientX), lo, Math.max(lo, hi));
  const el = $('csLane').querySelector(`[data-cs="${CSS.escape(drag.id)}"]`);
  if (el) el.style.left = app.pct(C[i].t) + '%';
  $('csSelT').textContent = fmt2(C[i].t);
}
function onUp() {
  if (!drag) return;
  const d = drag; drag = null;
  const x = C.find(c => c.id === d.id); if (!x) return;
  if (d.moved) { sel = x.id; changed(); app.userSeek(x.t); return; }
  if (UI.open) { sel = sel === x.id ? null : x.id; renderLane(); renderEditor(); }
  app.userSeek(x.t);
}

/* ---------- init ---------- */
export const _state = () => ({ C: list(), sel, undo: undo.length, open: UI.open, link: linkPending ? linkPending.length : 0 });
export function init(a) {
  app = a; S = a.S;
  const u = store.get('woodshed:chordEd', null);
  if (u && typeof u === 'object') { UI.open = u.open === true; UI.sev = u.sev === true; }
  app.on('video', () => { saveSoon.flush(); load(); renderAll(); });
  app.on('timeline', renderLane);
  app.on('dur', renderLane);
  app.on('theory', renderEditor);
  app.on('lang', renderAll);
  if (window.ResizeObserver) new ResizeObserver(() => renderLane()).observe($('csLane'));
  window.addEventListener('pagehide', () => saveSoon.flush());

  $('csEdBtn').addEventListener('click', () => setOpen(!UI.open));
  $('csDone').addEventListener('click', () => setOpen(false));
  $('csCur').addEventListener('click', () => {
    const i = idxAt(app.now()); if (i >= 0 && C[i].r >= 0) Theory.showChord(C[i].r, C[i].ty);
  });
  $('csPal').addEventListener('click', e => { const b = e.target.closest('[data-n]'); if (b) { const p = parseChord(b.dataset.n); if (p) place(p); } });
  $('csSev').addEventListener('click', e => { const b = e.target.closest('[data-sev]'); if (!b) return; UI.sev = b.dataset.sev === '1'; store.set('woodshed:chordEd', { open: UI.open, sev: UI.sev }); renderEditor(); });
  const D0 = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
  $('csRoot').innerHTML = D0.map((n, i) => `<option value="${i}">${n}</option>`).join('');
  $('csType').innerHTML = IDS.map(id => `<option value="${esc(id)}">${esc(id === '' ? 'maj' : id)}</option>`).join('');
  $('csAddOther').addEventListener('click', () => {
    const n = D0[+$('csRoot').value] + $('csType').value, p = parseChord(n); if (p) place(p);
  });
  ['csRoot', 'csType'].forEach(id => $(id).addEventListener('change', e => e.target.blur()));
  $('csSelPrev').addEventListener('click', () => sel && nudge(sel, -0.1));
  $('csSelNext').addEventListener('click', () => sel && nudge(sel, 0.1));
  $('csSelHere').addEventListener('click', () => sel && toPlayhead(sel));
  $('csSelDel').addEventListener('click', () => sel && del(sel));
  $('csSelChart').addEventListener('click', () => { const x = sel && C.find(c => c.id === sel); if (x) Theory.showChord(x.r, x.ty); });
  $('csSelX').addEventListener('click', () => { sel = null; renderLane(); renderEditor(); });
  $('csUndo').addEventListener('click', doUndo);
  $('csClear').addEventListener('click', clearAll);
  $('csForm').addEventListener('submit', e => { e.preventDefault(); if (fillLoop($('csText').value)) { $('csText').value = ''; $('csText').blur(); } });
  $('csLinkUse').addEventListener('click', () => { if (!linkPending) return; snapshot(); C = linkPending; linkPending = null; sel = null; changed(); toast(tr('cs.linkUsed')); });
  $('csLinkKeep').addEventListener('click', () => { linkPending = null; renderEditor(); });
  const lane = $('csLane');
  lane.addEventListener('pointerdown', onDown);
  lane.addEventListener('pointermove', onMove);
  lane.addEventListener('pointerup', onUp);
  lane.addEventListener('pointercancel', () => { if (drag && drag.moved) { const d = drag; drag = null; changed(); } drag = null; });
  load(); renderAll();
}
