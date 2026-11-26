// Practice: fretboard note quiz + ear training. Sounds use the plucked-string synth;
// chords are voiced with the same generator as the chord charts so they sound like a guitar.
import { $, esc, store } from './util.js?v=202611261719';
import { t as tr } from './i18n.js?v=202611261719';
import { TUNINGS, generate } from './voicings.js?v=202611261719';
import { boardSVG } from './fretboard.js?v=202611261719';
import * as Pluck from './pluck.js?v=202611261719';

const FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
const SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const NATURAL = new Set([0, 2, 4, 5, 7, 9, 11]);
const ROUND = 60;          // seconds in a timed round
const rnd = n => Math.floor(Math.random() * n);
const pick = a => a[rnd(a.length)];

let app, theoryState;
const P = { tab: 'quiz' };

/* ======================= fretboard quiz ======================= */
const Q = { mode: 'name', range: 12, nat: true, strings: [0, 1, 2, 3, 4, 5], best: {}, cur: null, lock: false,
  run: false, end: 0, score: 0, miss: 0, tick: null, marks: [], last: '', flash: null };
const qKey = () => Q.mode + ':' + Q.range + ':' + (Q.nat ? 'n' : 'a');
const tuning = () => (theoryState && TUNINGS[theoryState.tuning] ? theoryState.tuning : 'std');
const lefty = () => !!(theoryState && theoryState.lefty);
const tun = () => TUNINGS[tuning()];
const saveQ = () => store.set('woodshed:quiz', { mode: Q.mode, range: Q.range, nat: Q.nat, strings: Q.strings, best: Q.best });
const strOrd = s => tr('qz.str' + (6 - s));   // 6th string = low E

function loadQ() {
  const v = store.get('woodshed:quiz', null) || {};
  if (v.mode === 'find' || v.mode === 'name') Q.mode = v.mode;
  if ([5, 12, 15].includes(v.range)) Q.range = v.range;
  if (typeof v.nat === 'boolean') Q.nat = v.nat;
  if (Array.isArray(v.strings)) { const s = [...new Set(v.strings.filter(x => Number.isInteger(x) && x >= 0 && x < 6))].sort(); if (s.length) Q.strings = s; }
  if (v.best && typeof v.best === 'object') for (const k in v.best) if (/^(name|find):(5|12|15):[na]$/.test(k) && Number.isInteger(v.best[k]) && v.best[k] >= 0) Q.best[k] = Math.min(v.best[k], 999);
}
function cellsFor(s) {
  const out = [];
  for (let f = 0; f <= Q.range; f++) { const pc = (tun()[s] + f) % 12; if (!Q.nat || NATURAL.has(pc)) out.push({ s, f, pc }); }
  return out;
}
function newQ() {
  clearTimeout(Q.flash); Q.lock = false; Q.marks = [];
  let c, tries = 0;
  do {
    const s = pick(Q.strings), cells = cellsFor(s);
    c = cells.length ? pick(cells) : { s, f: 0, pc: tun()[s] % 12 };
    if (Q.mode === 'find') c = { s: c.s, pc: c.pc, f: -1 };
    tries++;
  } while (tries < 20 && (c.s + ':' + c.f + ':' + c.pc) === Q.last);
  Q.last = c.s + ':' + c.f + ':' + c.pc;
  Q.cur = c;
  renderQ();
}
function midiAt(s, f) { return tun()[s] + f; }
function answerName(pc) {
  if (!Q.cur || Q.lock || Q.mode !== 'name') return;
  const c = Q.cur, ok = pc === c.pc;
  Pluck.play([midiAt(c.s, c.f)]);
  Q.lock = true;
  if (ok) Q.score++; else Q.miss++;
  Q.marks = [{ s: c.s, f: c.f, label: FLAT[c.pc], kind: ok ? 'tone' : 'root' }];
  renderQ({ good: c.pc, bad: ok ? null : pc });
  Q.flash = setTimeout(() => { if (Q.cur === c) newQ(); }, ok ? 550 : 1300);
}
function answerCell(s, f) {
  if (!Q.cur || Q.lock || Q.mode !== 'find') return;
  const c = Q.cur, pc = (tun()[s] + f) % 12, ok = s === c.s && pc === c.pc;
  Pluck.play([midiAt(s, f)]);
  if (ok) {
    Q.score++; Q.lock = true;
    Q.marks = [];   // every place this note sits on the string
    for (let g = 0; g <= Q.range; g++) if ((tun()[c.s] + g) % 12 === c.pc) Q.marks.push({ s: c.s, f: g, label: FLAT[c.pc], kind: g === f ? 'tone' : 'scale' });
    renderQ();
    Q.flash = setTimeout(() => { if (Q.cur === c) newQ(); }, 700);
  } else {
    Q.miss++;
    Q.marks = Q.marks.filter(m => m.kind !== 'root').concat([{ s, f, label: FLAT[pc], kind: 'root' }]);
    renderQ();
    clearTimeout(Q.flash);
    Q.flash = setTimeout(() => { if (Q.cur === c) { Q.marks = Q.marks.filter(m => m.kind !== 'root'); renderQ(); } }, 900);
  }
}
function startRound() {
  if (Q.run) { stopRound(false); return; }
  Q.run = true; Q.score = 0; Q.miss = 0; Q.end = Date.now() + ROUND * 1000;
  $('qzResult').hidden = true;
  clearInterval(Q.tick); Q.tick = setInterval(tickRound, 200);
  newQ(); tickRound();
}
function tickRound() {
  if (!Q.run) return;
  const left = Math.max(0, Math.ceil((Q.end - Date.now()) / 1000));
  $('qzTime').textContent = left + 's';
  if (left <= 0) stopRound(true);
}
function stopRound(finished) {
  clearInterval(Q.tick); Q.tick = null;
  if (!Q.run) return;
  Q.run = false;
  if (finished) {
    const k = qKey(), prev = Q.best[k] || 0, rec = Q.score > prev;
    if (rec) { Q.best[k] = Q.score; saveQ(); }
    const tot = Q.score + Q.miss;
    $('qzResult').hidden = false;
    $('qzResult').textContent = (rec && Q.score > 0 ? tr('qz.newBest', { n: Q.score }) : tr('qz.done', { n: Q.score, b: Math.max(prev, Q.score) })) +
      (tot ? ' · ' + tr('qz.acc', { p: Math.round(Q.score / tot * 100) }) : '');
  }
  renderQ();
}
function promptText() {
  const c = Q.cur; if (!c) return '';
  if (Q.mode === 'name') return tr('qz.whatNote');
  return tr('qz.findOn', { n: FLAT[c.pc] + (SHARP[c.pc] !== FLAT[c.pc] ? ' / ' + SHARP[c.pc] : ''), s: strOrd(c.s), o: FLAT[tun()[c.s] % 12] });
}
function renderQ(fb) {
  const c = Q.cur;
  const off = new Set([0, 1, 2, 3, 4, 5].filter(s => !Q.strings.includes(s)));
  let marks = Q.marks.slice();
  if (c && Q.mode === 'name' && !Q.lock) marks = [{ s: c.s, f: c.f, label: '?', kind: 'q' }];
  const b = boardSVG({ tuning: tuning(), lefty: lefty(), frets: Q.range, marks, hits: Q.mode === 'find', off });
  const svg = $('qzFb');
  svg.innerHTML = b.svg; svg.setAttribute('viewBox', `0 0 ${b.W} ${b.H}`);
  svg.classList.toggle('r5', Q.range === 5); svg.classList.toggle('r12', Q.range === 12);
  svg.classList.toggle('find', Q.mode === 'find');
  if (c && Q.mode === 'find') svg.querySelectorAll(`.hit[data-s="${c.s}"]`).forEach(r => r.classList.add('target'));
  $('qzPrompt').textContent = promptText();
  $('qzScore').textContent = tr('qz.score', { n: Q.score, m: Q.miss });
  $('qzBest').textContent = tr('qz.best', { n: Q.best[qKey()] || 0 });
  $('qzTime').hidden = !Q.run;
  $('qzStart').textContent = Q.run ? tr('qz.stop') : tr('qz.start');
  $('qzStart').classList.toggle('on', Q.run);
  $('qzNotes').hidden = Q.mode !== 'name';
  $('qzNotes').innerHTML = (Q.nat ? [0, 2, 4, 5, 7, 9, 11] : [...Array(12).keys()]).map(pc => {
    const cls = fb && fb.good === pc ? ' good' : fb && fb.bad === pc ? ' bad' : '';
    const lbl = SHARP[pc] === FLAT[pc] ? FLAT[pc] : `${SHARP[pc]}<small>${FLAT[pc]}</small>`;
    return `<button data-pc="${pc}" class="${cls.trim()}">${lbl}</button>`;
  }).join('');
  $('qzNotes').classList.toggle('nat', Q.nat);
  // controls
  $('qzMode').querySelectorAll('[data-m]').forEach(x => x.classList.toggle('on', x.dataset.m === Q.mode));
  $('qzRange').querySelectorAll('[data-r]').forEach(x => x.classList.toggle('on', +x.dataset.r === Q.range));
  $('qzNat').checked = Q.nat;
  const names = tun().map(m => FLAT[m % 12]);
  $('qzStr').innerHTML = `<span>${esc(tr('qz.strings'))}</span>` + [5, 4, 3, 2, 1, 0].map(s =>
    `<button data-s="${s}" class="${Q.strings.includes(s) ? 'on' : ''}" aria-pressed="${Q.strings.includes(s)}" title="${esc(strOrd(s))}">${esc(s === 5 ? names[s].toLowerCase() : names[s])}</button>`).join('');
}
function changeQ(fn) { fn(); if (Q.run) stopRound(false); saveQ(); newQ(); }

/* ======================= ear training ======================= */
const IV = { 1: 'm2', 2: 'M2', 3: 'm3', 4: 'M3', 5: 'P4', 6: 'TT', 7: 'P5', 8: 'm6', 9: 'M6', 10: 'm7', 11: 'M7', 12: 'P8' };
const IV_LVL = { 1: [3, 4, 5, 7, 12], 2: [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] };
const CH = { maj: [0, 4, 7], m: [0, 3, 7], dim: [0, 3, 6], aug: [0, 4, 8], sus2: [0, 2, 7], sus4: [0, 5, 7], 7: [0, 4, 7, 10], maj7: [0, 4, 7, 11], m7: [0, 3, 7, 10] };
const CH_LVL = { 1: ['maj', 'm'], 2: ['maj', 'm', 'dim', 'aug', 'sus2', 'sus4'], 3: ['maj', 'm', 'dim', 'aug', 'sus2', 'sus4', '7', 'maj7', 'm7'] };
const DEG = { 0: '1', 1: 'b2', 2: '2', 3: 'b3', 4: '3', 5: '4', 6: '#4', 7: '5', 8: 'b6', 9: '6', 10: 'b7', 11: '7' };
const DEG_LVL = { 1: [0, 4, 7], 2: [0, 2, 4, 5, 7, 9, 11], 3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] };
const LO = 40, HI = 76;   // E2 … E5, a guitar's range

const E = { ex: 'iv', lvl: 1, dir: 'up', best: {}, cur: null, answered: false, streak: 0, right: 0, total: 0, next: null };
const eKey = () => E.ex + ':' + E.lvl + (E.ex === 'iv' ? ':' + E.dir : '');
const saveE = () => store.set('woodshed:ear', { ex: E.ex, lvl: E.lvl, dir: E.dir, best: E.best });
function loadE() {
  const v = store.get('woodshed:ear', null) || {};
  if (['iv', 'ch', 'deg'].includes(v.ex)) E.ex = v.ex;
  if ([1, 2, 3].includes(v.lvl)) E.lvl = v.lvl;
  if (['up', 'down', 'harm'].includes(v.dir)) E.dir = v.dir;
  if (v.best && typeof v.best === 'object') for (const k in v.best) if (/^(iv:[123]:(up|down|harm)|ch:[123]|deg:[123])$/.test(k) && Number.isInteger(v.best[k]) && v.best[k] >= 0) E.best[k] = Math.min(v.best[k], 9999);
}
const choices = () => (E.ex === 'iv' ? IV_LVL[E.lvl] : E.ex === 'ch' ? CH_LVL[E.lvl] : DEG_LVL[E.lvl]);
const label = a => (E.ex === 'iv' ? tr('ear.iv.' + IV[a]) : E.ex === 'ch' ? tr('ear.ch.' + a) : DEG[a]);
const sub = a => (E.ex === 'iv' ? IV[a] : E.ex === 'ch' ? (a === 'maj' ? 'C' : 'C' + a) : '');

// chord voicing on a standard-tuned guitar, low on the neck
function chordMidis(rootPc, iv) {
  const vs = generate(TUNINGS.std, rootPc, iv, { maxFret: 12, limit: 10 }).filter(v => {
    const f = v.frets.filter(x => x > 0); return !f.length || Math.min(...f) <= 7;
  });
  const v = vs.length ? vs[rnd(Math.min(3, vs.length))] : null;
  if (!v) { const r = 48 + rootPc; return iv.map(i => r + i); }
  return v.frets.map((f, s) => (f >= 0 ? TUNINGS.std[s] + f : -1)).filter(m => m >= 0);
}
function seqFor(cur, ans) {
  if (cur.kind === 'iv') {
    const a = cur.root, b = cur.root + ans;
    if (cur.dir === 'harm') return [{ m: [a, b], t: 0, gap: 0 }];
    const [x, y] = cur.dir === 'down' ? [b, a] : [a, b];
    return [{ m: [x], t: 0 }, { m: [y], t: 0.75 }];
  }
  if (cur.kind === 'ch') return [{ m: cur.voice[ans] || (cur.voice[ans] = chordMidis(cur.root, CH[ans])), t: 0, gap: 0.03 }];
  // scale degree: I IV V I in the key, then the note
  const r = cur.root;
  return [
    { m: [r - 12, r, r + 4, r + 7], t: 0, gap: 0.02 },
    { m: [r - 7, r, r + 5, r + 9], t: 0.7, gap: 0.02 },
    { m: [r - 5, r - 1, r + 2, r + 7], t: 1.4, gap: 0.02 },
    { m: [r - 12, r, r + 4, r + 7], t: 2.1, gap: 0.02 },
    { m: [r + ans], t: 3.3 }
  ];
}
function ask() {
  clearTimeout(E.next);
  const ch = choices();
  let ans = pick(ch);
  if (E.cur && ch.length > 1 && ans === E.cur.ans && Math.random() < 0.7) ans = pick(ch.filter(x => x !== ans));
  let cur;
  if (E.ex === 'iv') {
    const dir = E.dir;
    cur = { kind: 'iv', ans, dir, root: 43 + rnd(HI - 43 - 12) };      // room for an octave above
  } else if (E.ex === 'ch') cur = { kind: 'ch', ans, root: rnd(12), voice: {} };
  else cur = { kind: 'deg', ans, root: 52 + rnd(8) };                      // key E … B, note stays ≤ 70
  E.cur = cur; E.answered = false;
  hear();
  renderE();
}
function hear(alt) {
  if (!E.cur) return false;
  return Pluck.playSeq(seqFor(E.cur, alt == null ? E.cur.ans : alt));
}
function answerE(a) {
  if (!E.cur) return;
  if (E.answered) { hear(a); return; }                      // after answering: compare by ear
  const ok = String(a) === String(E.cur.ans);
  E.answered = true; E.total++;
  if (ok) {
    E.right++; E.streak++;
    const k = eKey(); if (E.streak > (E.best[k] || 0)) { E.best[k] = E.streak; saveE(); }
    E.next = setTimeout(() => { if (E.answered) ask(); }, 1000);
  } else E.streak = 0;
  renderE({ good: E.cur.ans, bad: ok ? null : a });
}
function renderE(fb) {
  $('earEx').querySelectorAll('[data-x]').forEach(x => x.classList.toggle('on', x.dataset.x === E.ex));
  $('earLvl').querySelectorAll('[data-l]').forEach(x => x.classList.toggle('on', +x.dataset.l === E.lvl));
  $('earDir').hidden = E.ex !== 'iv';
  $('earDir').querySelectorAll('[data-d]').forEach(x => x.classList.toggle('on', x.dataset.d === E.dir));
  $('earPlay').textContent = E.cur ? tr('ear.next') : tr('ear.start');
  $('earReplay').disabled = !E.cur;
  $('earHint').textContent = !E.cur ? tr('ear.hint.' + E.ex) : E.answered ? tr('ear.compare') : tr('ear.which.' + E.ex);
  $('earAns').className = 'earans ' + E.ex;
  $('earAns').innerHTML = choices().map(a => {
    const st = fb && String(fb.good) === String(a) ? ' good' : fb && fb.bad != null && String(fb.bad) === String(a) ? ' bad' : '';
    const sb = sub(a);
    return `<button data-a="${esc(a)}" class="${st.trim()}"${E.cur ? '' : ' disabled'}>${esc(label(a))}${E.ex === 'iv' ? `<small>${esc(sb)}</small>` : ''}</button>`;
  }).join('');
  const acc = E.total ? Math.round(E.right / E.total * 100) : 0;
  $('earStat').textContent = tr('ear.stat', { s: E.streak, b: E.best[eKey()] || 0, r: E.right, t: E.total, p: acc });
  const msg = $('earMsg');
  if (fb) { msg.hidden = false; msg.className = 'earmsg ' + (fb.bad == null ? 'ok' : 'no'); msg.textContent = fb.bad == null ? tr('ear.right', { a: label(fb.good) }) : tr('ear.wrong', { a: label(fb.good) }); }
  else msg.hidden = true;
}
function changeE(fn) { fn(); saveE(); clearTimeout(E.next); E.cur = null; E.answered = false; E.streak = 0; renderE(); }

/* ======================= shell ======================= */
function showTab(t) {
  P.tab = t === 'ear' ? 'ear' : 'quiz';
  store.set('woodshed:practice', { tab: P.tab });
  document.querySelectorAll('#practice [data-pr]').forEach(b => { b.classList.toggle('on', b.dataset.pr === P.tab); b.setAttribute('aria-selected', b.dataset.pr === P.tab); });
  $('prQuiz').hidden = P.tab !== 'quiz';
  $('prEar').hidden = P.tab !== 'ear';
  if (P.tab !== 'quiz' && Q.run) stopRound(false);
}
export const _state = () => ({ Q: { mode: Q.mode, range: Q.range, nat: Q.nat, strings: Q.strings.slice(), cur: Q.cur, score: Q.score, miss: Q.miss, run: Q.run, best: { ...Q.best } }, E: { ex: E.ex, lvl: E.lvl, dir: E.dir, cur: E.cur && { ...E.cur }, streak: E.streak, right: E.right, total: E.total, best: { ...E.best }, answered: E.answered } });
export function _endRound() { Q.end = Date.now() - 1; tickRound(); }

export function init(a, thState) {
  app = a; theoryState = thState;
  loadQ(); loadE();
  const pv = store.get('woodshed:practice', null);
  showTab(pv && pv.tab);

  document.querySelectorAll('#practice [data-pr]').forEach(b => b.addEventListener('click', () => showTab(b.dataset.pr)));
  $('qzMode').addEventListener('click', e => { const b = e.target.closest('[data-m]'); if (b && b.dataset.m !== Q.mode) changeQ(() => { Q.mode = b.dataset.m; }); });
  $('qzRange').addEventListener('click', e => { const b = e.target.closest('[data-r]'); if (b && +b.dataset.r !== Q.range) changeQ(() => { Q.range = +b.dataset.r; }); });
  $('qzNat').addEventListener('change', () => changeQ(() => { Q.nat = $('qzNat').checked; }));
  $('qzStr').addEventListener('click', e => {
    const b = e.target.closest('[data-s]'); if (!b) return;
    const s = +b.dataset.s;
    if (Q.strings.includes(s)) { if (Q.strings.length === 1) return; changeQ(() => { Q.strings = Q.strings.filter(x => x !== s); }); }
    else changeQ(() => { Q.strings = [...Q.strings, s].sort(); });
  });
  $('qzNotes').addEventListener('click', e => { const b = e.target.closest('[data-pc]'); if (b) answerName(+b.dataset.pc); });
  $('qzFb').addEventListener('click', e => { const r = e.target.closest('.hit'); if (r) answerCell(+r.dataset.s, +r.dataset.f); });
  $('qzStart').addEventListener('click', startRound);
  $('qzSkip').addEventListener('click', () => { if (Q.cur && !Q.lock) { Q.miss++; } newQ(); });

  $('earEx').addEventListener('click', e => { const b = e.target.closest('[data-x]'); if (b && b.dataset.x !== E.ex) changeE(() => { E.ex = b.dataset.x; }); });
  $('earLvl').addEventListener('click', e => { const b = e.target.closest('[data-l]'); if (b && +b.dataset.l !== E.lvl) changeE(() => { E.lvl = +b.dataset.l; }); });
  $('earDir').addEventListener('click', e => { const b = e.target.closest('[data-d]'); if (b && b.dataset.d !== E.dir) changeE(() => { E.dir = b.dataset.d; }); });
  $('earPlay').addEventListener('click', ask);
  $('earReplay').addEventListener('click', () => hear());
  $('earAns').addEventListener('click', e => { const b = e.target.closest('[data-a]'); if (b && !b.disabled) answerE(isNaN(+b.dataset.a) ? b.dataset.a : +b.dataset.a); });

  // follow the theory panel's tuning / left-handed setting
  ['thTuning', 'thLefty'].forEach(id => { const el = $(id); if (el) el.addEventListener('change', () => setTimeout(() => { if (Q.run) stopRound(false); newQ(); }, 0)); });
  app.on('lang', () => { renderQ(); renderE(); });
  newQ(); renderE();
}
