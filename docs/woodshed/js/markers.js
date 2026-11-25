// Song-structure markers (Intro / Verse / Solo …) per video: a lane above the timeline + an editable list.
import { $, esc, store, fmt, toast, uid, debounce, touchOnly } from './util.js?v=202611260628';
import { t as tr } from './i18n.js?v=202611260628';

let app, S;
let M = [];
let lastListSig = '';
const key = vid => 'woodshed:markers:' + (vid || S.vid);
function load() {
  const v = S.vid ? store.get(key(), []) : [];
  M = Array.isArray(v) ? v.filter(x => x && x.id && isFinite(x.t) && x.t >= 0).map(x => ({ id: String(x.id), t: +x.t, n: typeof x.n === 'string' ? x.n.slice(0, 24) : '' })) : [];
  M.sort((a, b) => a.t - b.t);
}
// every write names its own song, so a late save can never land on another song
const writeM = (vid, list) => { if (vid) store.set(key(vid), list.map(({ id, t, n }) => ({ id, t: +t.toFixed(2), n }))); };
const saveSoon = debounce(writeM, 300);
function save() { saveSoon.flush(); writeM(S.vid, M); }

export function addAtPlayhead() {
  if (!app.ok()) { toast(tr('loadSongFirst')); return; }
  const t = app.now();
  if (M.some(x => Math.abs(x.t - t) < 0.3)) { toast(tr('mk.exists')); return; }
  if (M.length >= 60) { toast(tr('mk.many')); return; }
  const x = { id: uid(), t, n: tr('markerN', { n: M.length + 1 }) };
  M.push(x); M.sort((a, b) => a.t - b.t); save();
  renderList(true); renderLane();
  const inp = $('mkList').querySelector(`input[data-name="${CSS.escape(x.id)}"]`);
  if (inp && !touchOnly()) { inp.focus(); inp.select(); }
  else toast(tr('mk.added', { t: fmt(t) }));
}

function loopFrom(id) {
  const i = M.findIndex(x => x.id === id); if (i < 0 || !app.ok()) return;
  const a = M[i].t, b = i + 1 < M.length ? M[i + 1].t : S.dur;
  if (!(b > a + app.MIN)) { toast(tr(S.dur ? 'mk.close' : 'waitLoad')); return; }
  app.setAB(a, b); app.setLoop(true, true); app.seek(a);
  toast(tr('loopingX', { n: M[i].n || tr('part') }));
}

export function init(a) {
  app = a; S = a.S;
  app.on('video', () => { saveSoon.flush(); load(); renderList(true); renderLane(); });
  app.on('timeline', renderLane);
  app.on('dur', () => softList());
  app.on('lang', () => { lastListSig = ''; softList(); renderLane(); });
  if (window.ResizeObserver) new ResizeObserver(() => renderLane()).observe($('lane'));
  $('mkAdd').onclick = addAtPlayhead;
  $('mkAddT').onclick = addAtPlayhead;
  $('mkList').addEventListener('click', e => {
    const sk = e.target.closest('[data-seek]'), lp = e.target.closest('[data-loop]'), dl = e.target.closest('[data-del]');
    if (sk) { const x = M.find(y => y.id === sk.dataset.seek); if (x) app.userSeek(x.t); }
    else if (lp) loopFrom(lp.dataset.loop);
    else if (dl) { saveSoon.flush(); M = M.filter(y => y.id !== dl.dataset.del); save(); renderList(true); renderLane(); }
  });
  $('mkList').addEventListener('input', e => {
    const inp = e.target.closest('input[data-name]'); if (!inp) return;
    const x = M.find(y => y.id === inp.dataset.name); if (!x) return;
    x.n = inp.value.slice(0, 24); saveSoon(S.vid, M); renderLane();
  });
  $('mkList').addEventListener('keydown', e => { if (e.key === 'Enter' && e.target.matches('input[data-name]')) { e.preventDefault(); e.target.blur(); } });
  $('mkList').addEventListener('focusout', e => { if (e.target.matches && e.target.matches('input[data-name]')) { saveSoon.flush(); if (deferred) { deferred = false; setTimeout(() => softList(), 0); } } });
  window.addEventListener('pagehide', () => saveSoon.flush());
  $('lane').addEventListener('click', e => {
    const c = e.target.closest('[data-mk]'); if (!c) return;
    const x = M.find(y => y.id === c.dataset.mk); if (x) app.userSeek(x.t);
  });
  load(); renderList(true); renderLane();
}

let deferred = false;
// don't rebuild the list under the user's cursor while they're renaming
function softList() {
  const ae = document.activeElement;
  if (ae && $('mkList').contains(ae) && ae.matches('input')) { deferred = true; return; }
  renderList(false);
}
function renderList(force) {
  const sig = S.vid + '#' + S.dur + '#' + M.map(x => x.id + ':' + x.t).join('|');
  if (!force && sig === lastListSig) return;
  lastListSig = sig;
  $('mkEmpty').hidden = !!M.length;
  $('mkAdd').disabled = !S.vid;
  $('mkList').innerHTML = M.map((x, i) => {
    const end = i + 1 < M.length ? M[i + 1].t : S.dur;
    return `<li class="mk">
      <button class="mk-t num" data-seek="${esc(x.id)}" title="${esc(tr('jumpHere'))}">${fmt(x.t)}</button>
      <input class="mk-n" data-name="${esc(x.id)}" list="mkNames" maxlength="24" aria-label="${esc(tr('mk.name'))}" spellcheck="false">
      <button class="btn sq ghost" data-loop="${esc(x.id)}" title="${esc(tr(i + 1 < M.length ? 'mk.loopNext' : 'mk.loopEnd'))}" aria-label="${esc(tr('mk.loopAria'))}" ${end > x.t + app.MIN ? '' : 'disabled'}><svg class="i" viewBox="0 0 24 24"><path d="M17 2l4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14M7 22l-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/></svg></button>
      <button class="btn sq ghost" data-del="${esc(x.id)}" aria-label="${esc(tr('mk.del'))}" title="${esc(tr('del'))}"><svg class="i" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
    </li>`;
  }).join('');
  $('mkList').querySelectorAll('input[data-name]').forEach(inp => { const x = M.find(y => y.id === inp.dataset.name); inp.value = x ? x.n : ''; });
}

function renderLane() {
  const lane = $('lane');
  if (!S.dur || !M.length) { lane.innerHTML = ''; lane.classList.remove('has'); return; }
  lane.classList.add('has');
  const W = lane.clientWidth || 1;
  const vis = M.filter(x => app.inView(x.t));
  lane.innerHTML = vis.map((x, i) => {
    const p = app.pct(x.t);
    const nx = i + 1 < vis.length ? app.pct(vis[i + 1].t) : 100;
    const room = Math.max(0, (nx - p) / 100 * W - 6);
    const label = room >= 26 ? `<span style="max-width:${Math.floor(room - 12)}px">${esc(x.n || '•')}</span>` : '';
    return `<button class="flag" data-mk="${esc(x.id)}" style="left:${p}%" title="${esc((x.n || tr('marker')) + ' · ' + fmt(x.t))}">${label}</button>`;
  }).join('');
}
