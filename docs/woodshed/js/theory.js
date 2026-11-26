// Chords / scales / song key: fretboard diagrams for every position, chord boxes, diatonic chords.
// Data is built from tonal (MIT) and chords-db (MIT) by tools/build-theory.mjs.
import { $, esc, store, toast } from './util.js?v=202611261434';
import { t as tr } from './i18n.js?v=202611261434';
import { generate, TUNINGS, boxBase, sounds } from './voicings.js?v=202611261434';
import * as Pluck from './pluck.js?v=202611261434';

let app, S, D = null, DBV = null;
const VER = new URL(import.meta.url).search;
const TH = { tab: 'chords', root: 0, type: '', scale: 'major', pos: 0, labels: 'notes', tuning: 'std', lefty: false, sel: 0, key: { r: 0, m: 'major' }, sev: false, keySaved: false };
const FRETS = 15;
const GROUPS = ['basic', 'seventh', 'sixth', 'sus', 'ext', 'dimaug', 'alt'];
const SCALE_GROUPS = ['common', 'mode', 'minor', 'sym', 'jazz'];

const saveG = () => store.set('woodshed:theory', { tab: TH.tab, root: TH.root, type: TH.type, scale: TH.scale, labels: TH.labels, tuning: TH.tuning, lefty: TH.lefty, sev: TH.sev });
const keyKey = () => 'woodshed:key:' + S.vid;

/* ---------- data ---------- */
async function load() {
  if (D) return true;
  try {
    const [a, b] = await Promise.all([
      fetch(new URL('../data/theory.json' + VER, import.meta.url)).then(r => { if (!r.ok) throw 0; return r.json(); }),
      fetch(new URL('../data/voicings.json' + VER, import.meta.url)).then(r => { if (!r.ok) throw 0; return r.json(); })
    ]);
    D = a; DBV = b; return true;
  } catch (e) { $('thMsg').hidden = false; $('thMsg').textContent = tr('th.loadFail'); return false; }
}
const rootName = i => D.roots[i];
const typeOf = id => D.types.find(x => x.id === id) || D.types[0];
const scaleOf = id => D.scales.find(x => x.id === id) || D.scales[0];
const chordSym = (ri, id) => rootName(ri) + id;
// spelled note name for a pitch class, using the chord/scale's own spelling when it has one
function speller(names) {
  const byPc = {};
  const pcOf = n => { const L = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[n[0]]; let a = 0; for (const c of n.slice(1)) a += c === '#' ? 1 : c === 'b' ? -1 : 0; return ((L + a) % 12 + 12) % 12; };
  for (const n of names || []) byPc[pcOf(n)] = n;
  const flat = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
  return pc => byPc[pc] || flat[pc];
}

/* ---------- voicings ---------- */
const vcache = new Map();
function voicingsFor(ri, id) {
  const k = TH.tuning + ':' + ri + ':' + id;
  if (vcache.has(k)) return vcache.get(k);
  const tun = TUNINGS[TH.tuning];
  let curated = [];
  // chords-db is for standard tuning; Eb tuning = same shapes, chord a half step lower
  if (TH.tuning === 'std') curated = (DBV[rootName(ri)] || {})[id] || [];
  else if (TH.tuning === 'eb') curated = (DBV[rootName((ri + 1) % 12)] || {})[id] || [];
  const list = curated.map(([frets, fingers, barre]) => ({ frets, fingers, barre, curated: true }));
  const seen = new Set(list.map(v => v.frets.join()));
  for (const v of generate(tun, ri, typeOf(id).iv)) if (!seen.has(v.frets.join())) { seen.add(v.frets.join()); list.push(v); }
  const pos = v => { const f = v.frets.filter(x => x > 0); return f.length ? Math.min(...f) : 0; };
  list.sort((a, b) => pos(a) - pos(b) || (b.curated ? 1 : 0) - (a.curated ? 1 : 0));
  vcache.set(k, list);
  return list;
}

/* ---------- fretboard ---------- */
const W = 1000, H = 210, L = 52, RGT = 14, TOP = 26, GAP = 30;
const fx = n => L + (W - L - RGT) * (1 - Math.pow(2, -n / 12)) / (1 - Math.pow(2, -FRETS / 12));
const mirror = x => (TH.lefty ? W - x : x);
const noteX = f => (f === 0 ? L - 20 : (fx(f - 1) + fx(f)) / 2);
const sy = s => TOP + (5 - s) * GAP;           // high e on top
function drawBoard(marks) {
  const tun = TUNINGS[TH.tuning];
  const spell = speller([]);
  let g = '';
  // board
  const x0 = Math.min(mirror(L), mirror(fx(FRETS))), x1 = Math.max(mirror(L), mirror(fx(FRETS)));
  g += `<rect class="wood" x="${x0}" y="${TOP - 12}" width="${x1 - x0}" height="${GAP * 5 + 24}" rx="4"/>`;
  for (const f of [3, 5, 7, 9, 15]) g += `<circle class="inlay" cx="${mirror(noteX(f))}" cy="${TOP + GAP * 2.5}" r="6"/>`;
  g += `<circle class="inlay" cx="${mirror(noteX(12))}" cy="${TOP + GAP * 1.5}" r="6"/><circle class="inlay" cx="${mirror(noteX(12))}" cy="${TOP + GAP * 3.5}" r="6"/>`;
  for (let f = 1; f <= FRETS; f++) g += `<line class="fret" x1="${mirror(fx(f))}" y1="${TOP - 12}" x2="${mirror(fx(f))}" y2="${TOP + GAP * 5 + 12}"/>`;
  g += `<line class="nut" x1="${mirror(L)}" y1="${TOP - 12}" x2="${mirror(L)}" y2="${TOP + GAP * 5 + 12}"/>`;
  for (let s = 0; s < 6; s++) {
    g += `<line class="str" style="stroke-width:${0.8 + (5 - s) * 0.32}px" x1="${x0}" y1="${sy(s)}" x2="${x1}" y2="${sy(s)}"/>`;
    g += `<text class="slab" x="${TH.lefty ? W - 12 : 12}" y="${sy(s) + 4}">${spell(tun[s] % 12)}</text>`;
  }
  for (const f of [1, 3, 5, 7, 9, 12, 15]) g += `<text class="fnum" x="${mirror(noteX(f))}" y="${H - 6}">${f}</text>`;
  // notes
  for (const m of marks) {
    const x = mirror(m.f < 0 ? L - 20 : noteX(m.f)), y = sy(m.s);
    if (m.f < 0) { g += `<text class="mute" x="${x}" y="${y + 4}">×</text>`; continue; }
    g += `<g class="n ${m.kind}${m.dim ? ' dim' : ''}"><circle cx="${x}" cy="${y}" r="11"/><text x="${x}" y="${y + 3.8}">${esc(m.label)}</text></g>`;
  }
  $('fb').innerHTML = g;
  $('fb').setAttribute('viewBox', `0 0 ${W} ${H}`);
}
// every place on the neck a set of pitch classes appears
function allMarks(rootPc, pcs, labelOf, kindOf) {
  const tun = TUNINGS[TH.tuning], out = [];
  for (let s = 0; s < 6; s++) for (let f = 0; f <= FRETS; f++) {
    const pc = (tun[s] + f) % 12;
    if (pcs.has(pc)) out.push({ s, f, pc, label: labelOf(pc), kind: kindOf(pc) });
  }
  return out;
}

/* ---------- chord box ---------- */
function chordBox(v, i, name) {
  const w = 96, h = 118, ox = 18, oy = 26, sw = 13, fh = 17;
  const base = boxBase(v.frets);
  let g = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">`;
  for (let s = 0; s < 6; s++) { const x = ox + (TH.lefty ? 5 - s : s) * sw; g += `<line class="bs" x1="${x}" y1="${oy}" x2="${x}" y2="${oy + fh * 5}"/>`; }
  for (let f = 0; f <= 5; f++) g += `<line class="${f === 0 && base === 1 ? 'bn' : 'bf'}" x1="${ox}" y1="${oy + f * fh}" x2="${ox + 5 * sw}" y2="${oy + f * fh}"/>`;
  if (base > 1) g += `<text class="bb" x="${ox - 5}" y="${oy + fh / 2 + 4}">${base}</text>`;
  if (v.barre) {
    const on = v.frets.map((f, s) => (f === v.barre ? s : -1)).filter(s => s >= 0);
    const a = Math.min(...on), b = Math.max(...on);
    const xa = ox + (TH.lefty ? 5 - b : a) * sw, xb = ox + (TH.lefty ? 5 - a : b) * sw;
    g += `<rect class="bar" x="${xa - 5}" y="${oy + (v.barre - base + 0.5) * fh - 5}" width="${xb - xa + 10}" height="10" rx="5"/>`;
  }
  v.frets.forEach((f, s) => {
    const x = ox + (TH.lefty ? 5 - s : s) * sw;
    if (f < 0) g += `<text class="bx" x="${x}" y="${oy - 8}">×</text>`;
    else if (f === 0) g += `<circle class="bo" cx="${x}" cy="${oy - 11}" r="4"/>`;
    else if (!(v.barre && f === v.barre && v.fingers[s] === 1)) g += `<circle class="bd" cx="${x}" cy="${oy + (f - base + 0.5) * fh}" r="5.6"/>` + (v.fingers[s] ? `<text class="bfi" x="${x}" y="${oy + (f - base + 0.5) * fh + 3.4}">${v.fingers[s]}</text>` : '');
  });
  g += '</svg>';
  return `<button class="vbox${i === TH.sel ? ' on' : ''}" data-v="${i}" aria-pressed="${i === TH.sel}" aria-label="${esc(name + ' · ' + tr('th.shape', { n: i + 1 }))}">${g}<span>${v.curated ? tr('th.common') : tr('th.fret', { n: base })}</span></button>`;
}

/* ---------- renderers ---------- */
function renderRoots() {
  $('thRoots').innerHTML = D.roots.map((r, i) => `<button class="${i === curRoot() ? 'on' : ''}" data-r="${i}">${r}</button>`).join('');
}
const curRoot = () => (TH.tab === 'key' ? TH.key.r : TH.root);
function renderTypes() {
  $('thTypes').innerHTML = GROUPS.map(g => `<div class="tgrp"><span>${esc(tr('th.g.' + g))}</span>${D.types.filter(x => x.group === g).map(x => `<button class="${x.id === TH.type ? 'on' : ''}" data-t="${esc(x.id)}" title="${esc(tr('ct.' + (x.id || 'maj')))}">${esc(x.id || 'maj')}</button>`).join('')}</div>`).join('');
}
function renderScaleSel() {
  $('thScale').innerHTML = SCALE_GROUPS.map(g => `<optgroup label="${esc(tr('th.sg.' + g))}">${D.scales.filter(x => x.group === g).map(x => `<option value="${x.id}"${x.id === TH.scale ? ' selected' : ''}>${esc(tr('sc.' + x.id))}</option>`).join('')}</optgroup>`).join('');
}
function head(name, notes, formula, sub) {
  $('thName').textContent = name;
  $('thNotes').textContent = notes;
  $('thFormula').textContent = formula;
  $('thSub').textContent = sub || '';
}
function renderChords() {
  const t = typeOf(TH.type), ri = TH.root;
  const names = D.notes[rootName(ri)][t.id];
  const spell = speller(names);
  const lab = {}; t.iv.forEach((iv, i) => { lab[(ri + iv) % 12] = t.lab[i]; });
  head(chordSym(ri, t.id), names.join(' '), t.lab.join(' '), tr('ct.' + (t.id || 'maj')));
  const vs = voicingsFor(ri, t.id);
  if (TH.sel >= vs.length) TH.sel = 0;
  $('thVoicings').innerHTML = vs.length ? vs.map((v, i) => chordBox(v, i, chordSym(ri, t.id))).join('') : `<div class="empty">${esc(tr('th.noShapes'))}</div>`;
  const pcs = new Set(t.iv.map(iv => (ri + iv) % 12));
  const label = pc => (TH.labels === 'iv' ? lab[pc] : spell(pc));
  const kind = pc => (pc === ri ? 'root' : 'tone');
  let marks = allMarks(ri, pcs, label, kind);
  const v = vs[TH.sel];
  if (v) {
    // the chosen shape stands out; the rest of the chord's notes stay faintly visible
    marks = marks.map(m => ({ ...m, dim: v.frets[m.s] !== m.f }));
    v.frets.forEach((f, s) => { if (f < 0) marks.push({ s, f: -1 }); });
  }
  drawBoard(marks);
}
function scalePositions(rootPc, pcs) {
  // a position starts at each scale note on the low string (frets 0-11) and spans 5 frets
  const tun = TUNINGS[TH.tuning], starts = [];
  for (let f = 0; f < 12; f++) if (pcs.has((tun[0] + f) % 12)) starts.push(f);
  return starts.slice(0, 7);
}
function renderScales() {
  const sc = scaleOf(TH.scale), ri = TH.root;
  const names = D.scaleNotes[rootName(ri)][sc.id];
  const spell = speller(names);
  const pcs = new Set(sc.iv.map(iv => (ri + iv) % 12));
  const lab = {}; sc.iv.forEach((iv, i) => { lab[(ri + iv) % 12] = sc.lab[i]; });
  head(rootName(ri) + ' ' + tr('sc.' + sc.id), names.join(' '), sc.lab.join(' '), '');
  const starts = scalePositions(ri, pcs);
  if (TH.pos > starts.length) TH.pos = 0;
  $('thPos').innerHTML = `<span>${esc(tr('th.position'))}</span><button data-p="0" class="${TH.pos === 0 ? 'on' : ''}">${esc(tr('th.all'))}</button>` + starts.map((f, i) => `<button data-p="${i + 1}" class="${TH.pos === i + 1 ? 'on' : ''}" title="${esc(tr('th.fret', { n: f }))}">${i + 1}</button>`).join('');
  let marks = allMarks(ri, pcs, pc => (TH.labels === 'iv' ? lab[pc] : spell(pc)), pc => (pc === ri ? 'root' : 'scale'));
  if (TH.pos) {
    const s0 = starts[TH.pos - 1], a = s0 === 0 ? 0 : s0 - 1, b = s0 + 4;
    marks = marks.map(m => ({ ...m, dim: m.f < a || m.f > b }));
  }
  drawBoard(marks);
}
const PROG = {
  major: [['I', 'V', 'vi', 'IV'], ['I', 'vi', 'IV', 'V'], ['ii', 'V', 'I'], ['I', 'IV', 'V'], ['vi', 'IV', 'I', 'V'], ['I', 'IV', 'vi', 'V'], ['I', 'I', 'I', 'I', 'IV', 'IV', 'I', 'I', 'V', 'IV', 'I', 'V']],
  minor: [['i', 'VI', 'III', 'VII'], ['i', 'iv', 'v'], ['i', 'VII', 'VI', 'VII'], ['ii°', 'v', 'i'], ['i', 'iv', 'VII', 'III'], ['i', 'VI', 'VII', 'i']]
};
function renderKey() {
  const K = D.keys[TH.key.m][TH.key.r];
  const chords = TH.sev ? K.sev : K.triads;
  const scId = TH.key.m === 'major' ? 'major' : 'minor';
  const sc = scaleOf(scId), ri = TH.key.r;
  const names = D.scaleNotes[rootName(ri)][scId];
  const spell = speller(names);
  head(tr(TH.key.m === 'major' ? 'th.keyMaj' : 'th.keyMin', { k: K.tonic }), names.join(' '), '', tr(TH.key.m === 'major' ? 'th.relMin' : 'th.relMaj', { k: K.rel }));
  $('thMode').querySelectorAll('[data-m]').forEach(b => b.classList.toggle('on', b.dataset.m === TH.key.m));
  $('thSev').querySelectorAll('[data-sev]').forEach(b => b.classList.toggle('on', (b.dataset.sev === '1') === TH.sev));
  const chip = (c, rn) => `<button class="kc" data-r="${c.r}" data-t="${esc(c.t)}"><small>${esc(rn)}</small><b>${esc(c.s)}</b></button>`;
  const prog = PROG[TH.key.m].map(p => `<div class="kprog"><span class="rn">${p.map(esc).join(' – ')}</span><span class="pc">${p.map(rn => { const i = K.rn.indexOf(rn); return chip(chords[i], rn); }).join('')}</span></div>`).join('');
  const penta = TH.key.m === 'major' ? 'majpent' : 'minpent';
  $('thKey').innerHTML = `<div class="dia">${chords.map((c, i) => chip(c, K.rn[i])).join('')}</div>
    <h3>${esc(tr('th.progs'))}</h3>${prog}
    <h3>${esc(tr('th.solo'))}</h3><div class="solo"><button class="btn small" data-sc="${scId}">${esc(K.tonic + ' ' + tr('sc.' + scId))}</button><button class="btn small" data-sc="${penta}">${esc(K.tonic + ' ' + tr('sc.' + penta))}</button>${TH.key.m === 'minor' ? `<button class="btn small" data-sc="blues">${esc(K.tonic + ' ' + tr('sc.blues'))}</button>` : ''}</div>
    <p class="hint" id="keySaved">${esc(tr(!S.vid ? 'th.keyNoSong' : TH.keySaved ? 'th.keySaved' : 'th.keyPick'))}</p>`;
  const pcs = new Set(sc.iv.map(iv => (ri + iv) % 12));
  const lab = {}; sc.iv.forEach((iv, i) => { lab[(ri + iv) % 12] = sc.lab[i]; });
  drawBoard(allMarks(ri, pcs, pc => (TH.labels === 'iv' ? lab[pc] : spell(pc)), pc => (pc === ri ? 'root' : 'scale')));
}
export function render() {
  if (!D) return;
  for (const b of document.querySelectorAll('#theory [data-th]')) { const on = b.dataset.th === TH.tab; b.classList.toggle('on', on); b.setAttribute('aria-selected', on); }
  $('thChordsCtl').hidden = TH.tab !== 'chords'; $('thVoicings').hidden = TH.tab !== 'chords';
  $('thScalesCtl').hidden = TH.tab !== 'scales';
  $('thKeyCtl').hidden = TH.tab !== 'key'; $('thKey').hidden = TH.tab !== 'key';
  $('thLabels').querySelectorAll('[data-l]').forEach(b => b.classList.toggle('on', b.dataset.l === TH.labels));
  $('thTuning').value = TH.tuning; $('thLefty').checked = TH.lefty;
  renderRoots();
  if (TH.tab === 'chords') { renderTypes(); renderChords(); }
  else if (TH.tab === 'scales') { renderScaleSel(); renderScales(); }
  else renderKey();
}

/* ---------- sound ---------- */
function hear() {
  if (!D) return;
  const tun = TUNINGS[TH.tuning];
  if (TH.tab === 'chords') {
    const v = voicingsFor(TH.root, TH.type)[TH.sel]; if (!v) return;
    Pluck.play(v.frets.map((f, s) => (f >= 0 ? tun[s] + f : -1)).filter(m => m >= 0), 0.04);
  } else {
    const ri = TH.tab === 'key' ? TH.key.r : TH.root;
    const sc = scaleOf(TH.tab === 'key' ? (TH.key.m === 'major' ? 'major' : 'minor') : TH.scale);
    let base = 48; while (base % 12 !== ri) base++;
    const ms = sc.iv.map(i => base + i).concat([base + 12]);
    Pluck.play(ms, 0.26);
  }
}

/* ---------- init ---------- */
export async function init(a) {
  app = a; S = a.S;
  const g = store.get('woodshed:theory', {});
  if (g && typeof g === 'object') {
    if (['chords', 'scales', 'key'].includes(g.tab)) TH.tab = g.tab;
    if (Number.isInteger(g.root) && g.root >= 0 && g.root < 12) TH.root = g.root;
    if (typeof g.type === 'string') TH.type = g.type;
    if (typeof g.scale === 'string') TH.scale = g.scale;
    if (g.labels === 'iv' || g.labels === 'notes') TH.labels = g.labels;
    if (TUNINGS[g.tuning]) TH.tuning = g.tuning;
    TH.lefty = g.lefty === true; TH.sev = g.sev === true;
  }
  const loadSongKey = () => {
    const k = S.vid ? store.get(keyKey(), null) : null;
    if (k && Number.isInteger(k.r) && k.r >= 0 && k.r < 12 && (k.m === 'major' || k.m === 'minor')) { TH.key = { r: k.r, m: k.m }; TH.keySaved = true; }
    else TH.keySaved = false;
  };
  const saveSongKey = () => { if (S.vid) { store.set(keyKey(), { r: TH.key.r, m: TH.key.m }); TH.keySaved = true; } };

  $('theory').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b || !D) return;
    if (b.dataset.th) { TH.tab = b.dataset.th; TH.sel = 0; }
    else if (b.closest('#thRoots')) { if (TH.tab === 'key') { TH.key.r = +b.dataset.r; saveSongKey(); } else { TH.root = +b.dataset.r; TH.sel = 0; } }
    else if (b.closest('#thTypes') && b.dataset.t !== undefined) { TH.type = b.dataset.t; TH.sel = 0; }
    else if (b.closest('#thVoicings') && b.dataset.v !== undefined) { TH.sel = +b.dataset.v; saveG(); renderChords(); hear(); return; }
    else if (b.closest('#thPos')) TH.pos = +b.dataset.p;
    else if (b.closest('#thLabels')) TH.labels = b.dataset.l;
    else if (b.closest('#thMode')) { TH.key.m = b.dataset.m; saveSongKey(); }
    else if (b.closest('#thSev')) TH.sev = b.dataset.sev === '1';
    else if (b.classList.contains('kc')) { TH.tab = 'chords'; TH.root = +b.dataset.r; TH.type = b.dataset.t; TH.sel = 0; $('theory').scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
    else if (b.dataset.sc) { TH.tab = 'scales'; TH.root = TH.key.r; TH.scale = b.dataset.sc; TH.pos = 0; }
    else if (b.id === 'thPlay') { hear(); return; }
    else return;
    saveG(); render();
  });
  $('thScale').addEventListener('change', e => { TH.scale = e.target.value; TH.pos = 0; saveG(); render(); e.target.blur(); });
  $('thTuning').addEventListener('change', e => { TH.tuning = TUNINGS[e.target.value] ? e.target.value : 'std'; TH.sel = 0; saveG(); render(); e.target.blur(); });
  $('thLefty').addEventListener('change', e => { TH.lefty = e.target.checked; saveG(); render(); });
  app.on('video', () => { loadSongKey(); if (TH.tab === 'key') render(); });
  app.on('lang', render);
  loadSongKey();
  if (await load()) render();
}
export const _state = TH;
