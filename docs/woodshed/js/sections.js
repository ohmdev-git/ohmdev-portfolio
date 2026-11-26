// Saved sections (A-B loops) per video, each with optional notes for chords / tabs.
import { $, esc, store, fmt, toast, uid, debounce } from './util.js?v=202611261235';
import { t as tr } from './i18n.js?v=202611261235';

let app, S;
let lastSig = '';
const openNotes = new Set();
const sureT = {};
// notes autosave (module level so every re-render can flush first)
const pending = {};
let pendingVid = '';
const saveNote = debounce(() => {
  if (!pendingVid) return;
  const k = 'woodshed:sections:' + pendingVid;
  const v = store.get(k, []); const L = Array.isArray(v) ? v : [];
  let changed = false;
  for (const id in pending) { const x = L.find(s => s && s.id === id); if (x) { x.note = pending[id]; changed = true; } delete pending[id]; }
  if (changed) store.set(k, L);
}, 400);

const key = () => 'woodshed:sections:' + S.vid;
export function getSecs(vid) {
  const v = store.get('woodshed:sections:' + (vid || S.vid), []);
  return Array.isArray(v) ? v.filter(x => x && typeof x === 'object' && isFinite(x.a) && isFinite(x.b) && x.id) : [];
}
export function count(vid) { return getSecs(vid).length; }
function save(L) { store.set(key(), L); app.emit('sections'); }

export function init(a) {
  app = a; S = a.S;
  app.on('video', () => { openNotes.clear(); render(true); });
  app.on('ab', () => render(false));
  app.on('lang', () => render(true));

  $('secs').addEventListener('click', e => {
    const go = e.target.closest('[data-go]'), del = e.target.closest('[data-del]'), nt = e.target.closest('[data-notes]');
    if (go) {
      const x = getSecs().find(s => s.id === go.dataset.go); if (!x || !app.ok()) return;
      app.setAB(x.a, x.b); S.active = x.id; app.setRate(x.s * 100); app.setLoop(true, true); app.seek(x.a); app.play(); render(false);
    } else if (nt) {
      const id = nt.dataset.notes;
      if (openNotes.has(id)) openNotes.delete(id); else openNotes.add(id);
      render(true);
      if (openNotes.has(id)) { const ta = $('secs').querySelector(`textarea[data-note="${CSS.escape(id)}"]`); if (ta) ta.focus(); }
    } else if (del) {
      const id = del.dataset.del;
      if (!del.classList.contains('sure')) {
        del.classList.add('sure'); del.textContent = tr('delete?');
        clearTimeout(sureT[id]); sureT[id] = setTimeout(() => { if (del.isConnected) render(true); }, 3000);
        return;
      }
      clearTimeout(sureT[id]);
      saveNote.flush();
      save(getSecs().filter(s => s.id !== id));
      if (S.active === id) S.active = null;
      openNotes.delete(id);
      render(true);
    }
  });
  $('secs').addEventListener('input', e => {
    const ta = e.target.closest('textarea[data-note]'); if (!ta) return;
    if (pendingVid !== S.vid) { saveNote.flush(); pendingVid = S.vid; }
    pending[ta.dataset.note] = ta.value.slice(0, 4000);
    const btn = $('secs').querySelector(`[data-notes="${CSS.escape(ta.dataset.note)}"]`);
    if (btn) btn.classList.toggle('has', ta.value.trim() !== '');
    saveNote();
  });
  $('secs').addEventListener('focusout', e => { if (e.target.matches && e.target.matches('textarea[data-note]')) saveNote.flush(); });
  window.addEventListener('pagehide', () => saveNote.flush());

  $('saveForm').addEventListener('submit', e => {
    e.preventDefault();
    if (!S.vid || S.A == null || S.B == null) return;
    const L = getSecs();
    const name = $('secName').value.trim() || tr('sectionN', { n: L.length + 1 });
    const x = { id: uid(), n: name.slice(0, 40), a: +S.A.toFixed(2), b: +S.B.toFixed(2), s: S.rate, note: '' };
    L.push(x); L.sort((p, q) => p.a - q.a);
    save(L); S.active = x.id; $('secName').value = ''; render(true); toast(tr('saved', { n: x.n }));
  });
  render(true);
}

function render(force) {
  if (force) saveNote.flush();
  const L = S.vid ? getSecs() : [];
  const sig = S.vid + '#' + L.map(x => [x.id, x.a, x.b, x.s, x.n].join(':')).join('|') + '#' + [...openNotes].join(',');
  if (!force && sig === lastSig) {
    $('secs').querySelectorAll('.sec').forEach(li => li.classList.toggle('act', li.dataset.id === S.active));
    return;
  }
  lastSig = sig;
  $('secEmpty').style.display = L.length ? 'none' : 'block';
  $('secs').innerHTML = L.map(x => {
    const open = openNotes.has(x.id), has = (x.note || '').trim() !== '';
    const first = has ? x.note.trim().split('\n')[0].slice(0, 60) : '';
    return `<li class="sec${S.active === x.id ? ' act' : ''}" data-id="${esc(x.id)}">
      <button class="go" data-go="${esc(x.id)}"><span class="n">${esc(x.n)}</span><span class="m num">${fmt(x.a)} – ${fmt(x.b)} · ${Math.round(x.s * 100)}%</span>${has && !open ? `<span class="pv">${esc(first)}</span>` : ''}</button>
      <button class="btn sq ghost nt${has ? ' has' : ''}${open ? ' on' : ''}" data-notes="${esc(x.id)}" aria-label="${esc(tr('notesFor', { n: x.n }))}" aria-expanded="${open}" title="${esc(tr('notes.t'))}"><svg class="i" viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16v4zM14 6l4 4"/></svg></button>
      <button class="btn sq ghost del" data-del="${esc(x.id)}" aria-label="${esc(tr('deleteX', { n: x.n }))}"><svg class="i" viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg></button>
      ${open ? `<div class="notes"><textarea data-note="${esc(x.id)}" rows="4" maxlength="4000" spellcheck="false" placeholder="${esc(tr('notes.ph'))}"></textarea></div>` : ''}
    </li>`;
  }).join('');
  // set textarea values as text (never markup)
  $('secs').querySelectorAll('textarea[data-note]').forEach(ta => { const x = L.find(s => s.id === ta.dataset.note); ta.value = (x && x.note) || ''; });
}
