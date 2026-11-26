// Backup & restore: every "woodshed:" key in localStorage, as one JSON file.
import { $, esc, toast } from './util.js?v=202611261520';
import { t as tr } from './i18n.js?v=202611261520';

const PREFIX = 'woodshed:';
const MAX_BYTES = 5 * 1024 * 1024;
let pending = null;      // parsed backup waiting for the user's merge / replace choice

function readAll() {
  const out = {};
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(PREFIX)) out[k] = localStorage.getItem(k);
    }
  } catch (e) {}
  return out;
}
const parse = s => { try { return JSON.parse(s); } catch (e) { return undefined; } };

// what a set of raw key/values contains, for the preview
export function summarize(data) {
  let songs = 0, parts = 0, markers = 0, notes = 0, keys = 0;
  const pl = parse(data[PREFIX + 'playlist']);
  if (Array.isArray(pl)) songs = pl.length;
  for (const [k, v] of Object.entries(data)) {
    const val = parse(v);
    if (k.startsWith(PREFIX + 'sections:') && Array.isArray(val)) { parts += val.length; notes += val.filter(x => x && typeof x.note === 'string' && x.note.trim()).length; }
    else if (k.startsWith(PREFIX + 'markers:') && Array.isArray(val)) markers += val.length;
    else if (k.startsWith(PREFIX + 'key:') || k.startsWith(PREFIX + 'bpm:')) keys++;
  }
  return { songs, parts, markers, notes, keys };
}

/* ---------- export ---------- */
export function exportFile() {
  const data = readAll();
  const body = JSON.stringify({ app: 'woodshed', schema: 1, exportedAt: new Date().toISOString(), data }, null, 1);
  const d = new Date(), pad = n => String(n).padStart(2, '0');
  const name = `woodshed-backup-${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}.json`;
  const blob = new Blob([body], { type: 'application/json' });
  const s = summarize(data);
  // iPhone / iPad: the share sheet is the reliable way to save a file
  const ios = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  if (ios && navigator.canShare && typeof File === 'function') {
    try {
      const f = new File([blob], name, { type: 'application/json' });
      if (navigator.canShare({ files: [f] })) { navigator.share({ files: [f], title: name }).catch(() => {}); return { name, ...s }; }
    } catch (e) {}
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.style.display = 'none';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  toast(tr('bk.exported', { n: name }));
  return { name, ...s };
}

/* ---------- import ---------- */
export function validate(text) {
  if (typeof text !== 'string' || !text.trim()) return { err: 'bk.errEmpty' };
  if (text.length > MAX_BYTES) return { err: 'bk.errBig' };
  const j = parse(text);
  if (!j || typeof j !== 'object' || j.app !== 'woodshed' || typeof j.data !== 'object' || !j.data || Array.isArray(j.data)) return { err: 'bk.errNotOurs' };
  if (!(j.schema >= 1) || j.schema > 1) return { err: 'bk.errVersion' };
  const data = {}; let skipped = 0;
  for (const [k, v] of Object.entries(j.data)) {
    if (typeof k === 'string' && k.startsWith(PREFIX) && k.length < 200 && typeof v === 'string' && parse(v) !== undefined) data[k] = v;
    else skipped++;
  }
  if (!Object.keys(data).length) return { err: 'bk.errEmpty' };
  return { data, skipped, exportedAt: typeof j.exportedAt === 'string' ? j.exportedAt : '' };
}

const byId = (arr, key = 'id') => new Map((Array.isArray(arr) ? arr : []).filter(x => x && x[key] !== undefined).map(x => [x[key], x]));
// combine one key: local stays as it is, the backup only fills in what's missing
function mergeValue(k, localRaw, backupRaw) {
  if (localRaw == null) return backupRaw;
  const L = parse(localRaw), B = parse(backupRaw);
  if (L === undefined) return backupRaw;
  if (B === undefined) return localRaw;
  const name = k.slice(PREFIX.length);
  if (name === 'playlist' && Array.isArray(L) && Array.isArray(B)) {
    const have = new Set(L.map(x => x && x.id));
    return JSON.stringify(L.concat(B.filter(x => x && !have.has(x.id))).slice(0, 300));
  }
  if (name === 'recent' && Array.isArray(L) && Array.isArray(B)) {
    const m = byId(B); for (const x of L) if (x && x.id) { const b = m.get(x.id); m.set(x.id, b && b.at > x.at ? b : x); }
    return JSON.stringify([...m.values()].sort((a, b) => (b.at || 0) - (a.at || 0)).slice(0, 40));
  }
  if (name === 'titles' && L && B && typeof L === 'object' && typeof B === 'object') return JSON.stringify(Object.assign({}, B, L));
  if (/^(sections|markers|chords):/.test(name) && Array.isArray(L) && Array.isArray(B)) {
    const m = byId(L); for (const x of B) if (x && x.id !== undefined && !m.has(x.id)) m.set(x.id, x);
    const list = [...m.values()];
    list.sort((a, b) => (a.a ?? a.t ?? 0) - (b.a ?? b.t ?? 0));
    return JSON.stringify(list);
  }
  return localRaw;        // settings, bpm, key, last song …: keep what this browser has
}

export function apply(data, mode) {
  const before = readAll();
  const next = {};
  if (mode === 'replace') Object.assign(next, data);
  else { Object.assign(next, before); for (const [k, v] of Object.entries(data)) next[k] = mergeValue(k, before[k], v); }
  try {
    if (mode === 'replace') for (const k of Object.keys(before)) if (!(k in next)) localStorage.removeItem(k);
    for (const [k, v] of Object.entries(next)) if (before[k] !== v) localStorage.setItem(k, v);
    return { ok: true };
  } catch (e) {
    // put everything back exactly as it was
    try {
      for (const k of Object.keys(readAll())) if (!(k in before)) localStorage.removeItem(k);
      for (const [k, v] of Object.entries(before)) localStorage.setItem(k, v);
    } catch (e2) {}
    return { ok: false };
  }
}

/* ---------- UI ---------- */
function showPreview(v, fileName) {
  pending = v; v.fileName = fileName;
  const s = summarize(v.data), cur = summarize(readAll());
  const when = v.exportedAt ? new Date(v.exportedAt) : null;
  $('bkPreview').hidden = false;
  $('bkPreview').innerHTML = `
    <p class="bkfile">${esc(fileName)}${when && !isNaN(when) ? ' · ' + esc(when.toLocaleString()) : ''}</p>
    <table class="bkt"><thead><tr><th></th><th>${esc(tr('bk.inFile'))}</th><th>${esc(tr('bk.here'))}</th></tr></thead><tbody>
      ${[['bk.songs', 'songs'], ['bk.parts', 'parts'], ['bk.notes', 'notes'], ['bk.markers', 'markers'], ['bk.tempos', 'keys']].map(([lk, f]) => `<tr><td>${esc(tr(lk))}</td><td class="num">${s[f]}</td><td class="num">${cur[f]}</td></tr>`).join('')}
    </tbody></table>
    ${v.skipped ? `<p class="hint">${esc(tr('bk.skipped', { n: v.skipped }))}</p>` : ''}
    <div class="row"><button class="btn pri" data-bk="merge">${esc(tr('bk.merge'))}</button><button class="btn" data-bk="replace">${esc(tr('bk.replace'))}</button><button class="btn ghost" data-bk="cancel">${esc(tr('cancel'))}</button></div>
    <p class="hint">${esc(tr('bk.mergeHint'))}</p>`;
}
function hidePreview() { pending = null; $('bkPreview').hidden = true; $('bkPreview').innerHTML = ''; $('bkFile').value = ''; }

let deferredInstall = null;
export function init(app) {
  // after an import we reload; say so once the page is back
  try { const m = sessionStorage.getItem('woodshed-toast'); if (m) { sessionStorage.removeItem('woodshed-toast'); setTimeout(() => toast(m), 400); } } catch (e) {}
  $('bkExport').onclick = () => exportFile();
  $('bkImport').onclick = () => $('bkFile').click();
  $('bkFile').addEventListener('change', async () => {
    const f = $('bkFile').files && $('bkFile').files[0];
    if (!f) return;
    if (f.size > MAX_BYTES) { toast(tr('bk.errBig')); $('bkFile').value = ''; return; }
    let text = '';
    try { text = await f.text(); } catch (e) { toast(tr('bk.errRead')); return; }
    const v = validate(text);
    if (v.err) { toast(tr(v.err)); $('bkFile').value = ''; return; }
    showPreview(v, f.name);
  });
  let sureReplace = 0;
  $('bkPreview').addEventListener('click', e => {
    const b = e.target.closest('[data-bk]'); if (!b || !pending) return;
    const what = b.dataset.bk;
    if (what === 'cancel') { hidePreview(); return; }
    if (what === 'replace' && Date.now() - sureReplace > 4000) { sureReplace = Date.now(); b.textContent = tr('bk.replace?'); return; }
    const r = apply(pending.data, what);
    if (!r.ok) { toast(tr('bk.errSpace')); return; }
    try { sessionStorage.setItem('woodshed-toast', tr(what === 'replace' ? 'bk.replaced' : 'bk.merged')); } catch (e) {}
    location.reload();
  });
  // install as an app
  window.addEventListener('beforeinstallprompt', e => { e.preventDefault(); deferredInstall = e; renderInstall(); });
  window.addEventListener('appinstalled', () => { deferredInstall = null; renderInstall(); toast(tr('app.installed')); });
  $('appInstall').onclick = async () => {
    if (!deferredInstall) return;
    deferredInstall.prompt();
    try { await deferredInstall.userChoice; } catch (e) {}
    deferredInstall = null; renderInstall();
  };
  app.on('lang', () => { renderInstall(); if (pending) showPreview(pending, pending.fileName); });
  renderInstall();
}
function renderInstall() {
  const standalone = (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) || navigator.standalone === true;
  const ios = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  $('appInstall').hidden = standalone || !deferredInstall;
  $('appHint').textContent = standalone ? tr('app.running') : deferredInstall ? '' : ios ? tr('app.ios') : tr('app.other');
}
