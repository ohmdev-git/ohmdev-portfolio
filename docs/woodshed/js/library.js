// Playlist (songs to learn today) + Recent history + title cache (YouTube oEmbed).
import { $, esc, store, toast, ago, parseMany, isId, clamp } from './util.js?v=202611261235';
import { t as tr } from './i18n.js?v=202611261235';
import { count as secCount } from './sections.js?v=202611261235';

const K_PL = 'woodshed:playlist', K_RC = 'woodshed:recent', K_TI = 'woodshed:titles';
const MAX_PL = 300, MAX_RC = 40, MAX_TI = 600;
let app, S;
let tab = 'pl';

/* ---------- storage ---------- */
function getPL() { const v = store.get(K_PL, []); return Array.isArray(v) ? v.filter(x => x && isId(x.id)) : []; }
function setPL(L) { store.set(K_PL, L.slice(0, MAX_PL)); }
function getRC() { const v = store.get(K_RC, []); return Array.isArray(v) ? v.filter(x => x && isId(x.id)) : []; }
function setRC(L) { store.set(K_RC, L.slice(0, MAX_RC)); }
let titles = (() => { const v = store.get(K_TI, {}); return v && typeof v === 'object' && !Array.isArray(v) ? v : {}; })();
function saveTitles() {
  const ids = Object.keys(titles);
  if (ids.length > MAX_TI) { ids.sort((a, b) => (titles[a].at || 0) - (titles[b].at || 0)).slice(0, ids.length - MAX_TI).forEach(id => delete titles[id]); }
  store.set(K_TI, titles);
}

/* ---------- titles via oEmbed ---------- */
export function titleOf(id) { const t = titles[id]; return t && typeof t.t === 'string' ? t.t : ''; }
function authorOf(id) { const t = titles[id]; return t && typeof t.a === 'string' ? t.a : ''; }
const inflight = new Set(), failedAt = {};
let queue = [], active = 0;
function wantTitle(id) {
  if (!isId(id) || titleOf(id) || inflight.has(id)) return;
  if (failedAt[id] && Date.now() - failedAt[id] < 60000) return;
  inflight.add(id); queue.push(id); pump();
}
function pump() {
  while (active < 4 && queue.length) {
    const id = queue.shift(); active++;
    fetchTitle(id).then(r => {
      if (r) { titles[id] = { t: r.t, a: r.a, at: Date.now() }; saveTitles(); scheduleRender(); app.emit('titles'); }
      else { failedAt[id] = Date.now(); scheduleRender(); }
    }).finally(() => { inflight.delete(id); active--; pump(); });
  }
}
async function fetchTitle(id) {
  const watch = encodeURIComponent('https://www.youtube.com/watch?v=' + id);
  const urls = ['https://www.youtube.com/oembed?format=json&url=' + watch, 'https://noembed.com/embed?url=' + watch];
  for (const u of urls) {
    try {
      const ctl = new AbortController(); const tm = setTimeout(() => ctl.abort(), 8000);
      const r = await fetch(u, { signal: ctl.signal }); clearTimeout(tm);
      if (!r.ok) continue;
      const j = await r.json();
      if (j && typeof j.title === 'string' && j.title.trim()) return { t: j.title.trim().slice(0, 200), a: typeof j.author_name === 'string' ? j.author_name.slice(0, 100) : '' };
    } catch (e) {}
  }
  return null;
}

/* ---------- playlist ops ---------- */
function addIds(ids) {
  const L = getPL(); let added = 0, dup = 0;
  for (const id of ids) {
    if (L.some(x => x.id === id)) { dup++; continue; }
    if (L.length >= MAX_PL) break;
    L.push({ id, at: Date.now(), done: false, blocked: false }); added++; wantTitle(id);
  }
  setPL(L); render();
  return { added, dup };
}
function sayAdded(r, extra) {
  const parts = [];
  if (r.added) parts.push(tr('added', { n: r.added }));
  if (r.dup) parts.push(tr('dups', { n: r.dup }));
  if (extra) parts.push(extra);
  toast(parts.join(' · ') || tr('nothingAdded'));
}
// read a YouTube playlist through a hidden player (no API key needed)
function importList(list) {
  return new Promise(resolve => {
    if (!window.YT || !window.YT.Player) { resolve(null); return; }
    const host = document.createElement('div');
    host.style.cssText = 'position:fixed;left:-10000px;top:0;width:320px;height:180px;opacity:0;pointer-events:none';
    const inner = document.createElement('div'); host.appendChild(inner); document.body.appendChild(host);
    let p, done = false, last = -1, stable = 0, timer;
    const finish = ids => {
      if (done) return; done = true; clearInterval(timer);
      try { p && p.destroy(); } catch (e) {}
      host.remove(); resolve(ids);
    };
    const kill = setTimeout(() => finish(null), 12000);
    try {
      p = new YT.Player(inner, {
        playerVars: /^https?:$/.test(location.protocol) ? { origin: location.origin } : {},
        events: {
          onReady: () => {
            try { p.cuePlaylist({ listType: 'playlist', list }); } catch (e) { clearTimeout(kill); finish(null); return; }
            timer = setInterval(() => {
              let ids = null; try { ids = p.getPlaylist(); } catch (e) {}
              if (Array.isArray(ids) && ids.length) {
                if (ids.length === last) stable++; else { stable = 0; last = ids.length; }
                if (stable >= 2) { clearTimeout(kill); finish(ids.filter(isId)); }
              }
            }, 400);
          },
          onError: () => { clearTimeout(kill); finish(null); }
        }
      });
    } catch (e) { clearTimeout(kill); finish(null); }
  });
}
function waitApi() {
  return new Promise(res => {
    if (window.YT && window.YT.Player && S.api) return res(true);
    let n = 0; const t = setInterval(() => { if ((window.YT && window.YT.Player && S.api) || ++n > 50) { clearInterval(t); res(!!(window.YT && window.YT.Player)); } }, 200);
  });
}
async function addFromText(text) {
  const { videos, lists, bad } = parseMany(text);
  if (!videos.length && !lists.length) { toast(tr(bad ? 'noLinks' : 'pasteLinks')); return false; }
  const r = addIds(videos.map(v => v.id));
  let extra = bad ? tr('notRecognised', { n: bad }) : '';
  if (lists.length) {
    $('plAddBtn').disabled = true; $('plAddBtn').textContent = tr('importing');
    await waitApi();
    let got = 0, failed = 0;
    for (const l of lists) {
      const ids = await importList(l);
      if (ids && ids.length) { const q = addIds(ids); r.added += q.added; r.dup += q.dup; got++; } else failed++;
    }
    $('plAddBtn').disabled = false; $('plAddBtn').textContent = tr('add');
    if (failed) extra = (extra ? extra + ' · ' : '') + (failed === 1 ? tr('plReadFail') : tr('plsReadFail', { n: failed }));
  }
  sayAdded(r, extra);
  return true;
}
export function step(dir) {
  const L = getPL();
  if (!L.length) { toast(tr('plEmptyToast')); return; }
  const i = L.findIndex(x => x.id === S.vid);
  const j = i < 0 ? (dir > 0 ? 0 : L.length - 1) : i + dir;
  if (j < 0 || j >= L.length) { toast(tr(dir > 0 ? 'lastSong' : 'firstSong')); return; }
  app.openVideo(L[j].id);
}

/* ---------- rendering ---------- */
let rT = 0;
// timer, not requestAnimationFrame: rAF pauses in background tabs, which left titles stuck on “Loading…”
function scheduleRender() { clearTimeout(rT); rT = setTimeout(render, 60); }
const shown = id => titleOf(id) || (failedAt[id] ? tr('videoId', { id }) : tr('loadingTitle'));
const thumb = id => `https://i.ytimg.com/vi/${id}/mqdefault.jpg`;
let dragging = false;
function render() {
  if (dragging) return;          // never rebuild the list under a drag; end of drag re-renders
  const L = getPL();
  const done = L.filter(x => x.done).length;
  $('plTabCount').textContent = L.length ? `(${L.length})` : '';
  $('plCount').textContent = L.length ? tr('pl.count', { n: L.length, d: done }) : '';
  $('plTools').hidden = !L.length;
  $('plEmpty').hidden = !!L.length;
  $('plAddCur').disabled = !S.vid || L.some(x => x.id === S.vid);
  $('plList').innerHTML = L.map((x, i) => {
    const t = titleOf(x.id), a = authorOf(x.id), n = secCount(x.id);
    const meta = [a, n ? tr('savedParts', { n }) : ''].filter(Boolean).join(' · ');
    return `<li class="pl${x.id === S.vid ? ' cur' : ''}${x.done ? ' done' : ''}" data-id="${x.id}">
      <span class="grip" data-grip aria-label="${esc(tr('drag'))}" title="${esc(tr('drag'))}"><svg class="i" viewBox="0 0 24 24"><path d="M9 6h.01M15 6h.01M9 12h.01M15 12h.01M9 18h.01M15 18h.01" stroke-width="3"/></svg></span>
      <button class="pl-go" data-go="${x.id}"><span class="idx num">${i + 1}</span><span class="thumb"><img src="${thumb(x.id)}" alt="" loading="lazy" decoding="async"></span><span class="pl-t"><b>${esc(shown(x.id))}</b><small>${x.blocked ? '<em>' + esc(tr('blockedItem')) + '</em>' : esc(meta)}</small></span></button>
      <button class="btn sq ghost chk${x.done ? ' on' : ''}" data-done="${x.id}" aria-pressed="${!!x.done}" aria-label="${esc(tr(x.done ? 'unmarkLearned' : 'markLearned'))}" title="${esc(tr(x.done ? 'learned' : 'markLearned'))}"><svg class="i" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg></button>
      <button class="btn sq ghost" data-rm="${x.id}" aria-label="${esc(tr('removeFromPl'))}" title="${esc(tr('remove'))}"><svg class="i" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
    </li>`;
  }).join('');
  L.forEach(x => wantTitle(x.id));
  const R = getRC();
  $('rcEmpty').hidden = !!R.length; $('rcClear').hidden = !R.length;
  $('rcList').innerHTML = R.map(x => {
    const t = titleOf(x.id), inPl = L.some(p => p.id === x.id);
    return `<li class="pl${x.id === S.vid ? ' cur' : ''}" data-id="${x.id}">
      <button class="pl-go" data-go="${x.id}"><span class="thumb"><img src="${thumb(x.id)}" alt="" loading="lazy" decoding="async"></span><span class="pl-t"><b>${esc(shown(x.id))}</b><small>${esc(ago(x.at, tr))}</small></span></button>
      <button class="btn ghost addpl" data-addpl="${x.id}" ${inPl ? 'disabled' : ''} title="${esc(tr(inPl ? 'inPl.t' : 'addPl.t'))}">${esc(tr(inPl ? 'inPl' : 'addPl'))}</button>
    </li>`;
  }).join('');
  R.forEach(x => wantTitle(x.id));
}
function setTab(t) {
  tab = t === 'rc' ? 'rc' : 'pl';
  $('plPane').hidden = tab !== 'pl'; $('rcPane').hidden = tab !== 'rc';
  $('tabPl').classList.toggle('on', tab === 'pl'); $('tabRc').classList.toggle('on', tab === 'rc');
  $('tabPl').setAttribute('aria-selected', tab === 'pl'); $('tabRc').setAttribute('aria-selected', tab === 'rc');
  store.set('woodshed:libtab', tab);
}

/* ---------- drag to reorder ---------- */
function enableReorder() {
  const list = $('plList');
  let dragLi = null, pid = null, startOrder = '';
  list.addEventListener('pointerdown', e => {
    const g = e.target.closest('[data-grip]'); if (!g || e.button > 0) return;
    dragLi = g.closest('li'); pid = e.pointerId; startOrder = [...list.children].map(li => li.dataset.id).join();
    dragLi.classList.add('dragging'); dragging = true;
    // capture on the list itself: the row moves around the DOM while dragging, which would drop a capture held by the row
    try { list.setPointerCapture(pid); } catch (err) {}
    e.preventDefault();
  });
  const placeAt = y => {
    const items = [...list.children].filter(li => li !== dragLi);
    let before = null;
    for (const li of items) { const r = li.getBoundingClientRect(); if (y < r.top + r.height / 2) { before = li; break; } }
    if (before) { if (dragLi.nextElementSibling !== before) list.insertBefore(dragLi, before); }
    else if (list.lastElementChild !== dragLi) list.appendChild(dragLi);
  };
  // auto-scroll the list (or the page) when dragging near an edge, so long playlists can be reordered
  let lastY = 0, raf = 0;
  const autoScroll = () => {
    if (!dragLi) { raf = 0; return; }
    const r = list.getBoundingClientRect(), H = window.innerHeight;
    const maxTop = list.scrollHeight - list.clientHeight;
    let moved = false;
    // 1) the list's own scroll area (desktop): only right at its edges
    if (maxTop > 1) {
      if (lastY < r.top + 18 && list.scrollTop > 0) { list.scrollTop -= Math.ceil((r.top + 18 - lastY) / 3); moved = true; }
      else if (lastY > r.bottom - 18 && list.scrollTop < maxTop) { list.scrollTop += Math.ceil((lastY - r.bottom + 18) / 3); moved = true; }
    }
    // 2) the page (phones): only near the screen edges
    if (!moved) {
      if (lastY < 48 && window.scrollY > 0) { window.scrollBy(0, -Math.ceil((48 - lastY) / 3)); moved = true; }
      else if (lastY > H - 48) { window.scrollBy(0, Math.ceil((lastY - H + 48) / 3)); moved = true; }
    }
    if (moved) placeAt(lastY);
    raf = requestAnimationFrame(autoScroll);
  };
  list.addEventListener('pointermove', e => {
    if (!dragLi || e.pointerId !== pid) return;
    lastY = e.clientY; placeAt(lastY);
    if (!raf) raf = requestAnimationFrame(autoScroll);
  });
  const end = e => {
    if (!dragLi || (e && e.pointerId !== pid)) return;
    if (e && e.type === 'pointerup') placeAt(e.clientY);   // the last move can arrive after the finger lifts
    dragLi.classList.remove('dragging'); dragLi = null; dragging = false;
    const order = [...list.children].map(li => li.dataset.id);
    if (order.join() !== startOrder) {
      const L = getPL(); const by = Object.fromEntries(L.map(x => [x.id, x]));
      setPL(order.map(id => by[id]).filter(Boolean).concat(L.filter(x => !order.includes(x.id))));
    }
    render();
  };
  list.addEventListener('pointerup', end);
  list.addEventListener('pointercancel', end);
  list.addEventListener('lostpointercapture', end);
}

/* ---------- init ---------- */
export function init(a) {
  app = a; S = a.S;
  setTab(store.get('woodshed:libtab', 'pl'));
  $('tabPl').onclick = () => setTab('pl');
  $('tabRc').onclick = () => setTab('rc');
  $('plForm').addEventListener('submit', async e => {
    e.preventDefault();
    const v = $('plIn').value;
    if (await addFromText(v)) $('plIn').value = '';
  });
  $('plAddCur').onclick = () => { if (S.vid) sayAdded(addIds([S.vid])); };
  $('plNext').onclick = () => step(1);
  $('plPrev').onclick = () => step(-1);
  let sureClr = 0;
  $('plClear').onclick = () => {
    if (Date.now() - sureClr > 3000) { sureClr = Date.now(); $('plClear').textContent = tr('clearAll?'); setTimeout(() => { $('plClear').textContent = tr('clear'); }, 3000); return; }
    sureClr = 0; $('plClear').textContent = tr('clear'); setPL([]); render(); toast(tr('plCleared'));
  };
  $('plClearDone').onclick = () => {
    const L = getPL(), keep = L.filter(x => !x.done);
    if (keep.length === L.length) { toast(tr('noLearned')); return; }
    setPL(keep); render(); toast(tr('learnedRemoved', { n: L.length - keep.length }));
  };
  $('plList').addEventListener('click', e => {
    const go = e.target.closest('[data-go]'), dn = e.target.closest('[data-done]'), rm = e.target.closest('[data-rm]');
    if (go) app.openVideo(go.dataset.go);
    else if (dn) { const L = getPL(); const x = L.find(y => y.id === dn.dataset.done); if (x) { x.done = !x.done; setPL(L); render(); } }
    else if (rm) { setPL(getPL().filter(y => y.id !== rm.dataset.rm)); render(); }
  });
  $('rcList').addEventListener('click', e => {
    const go = e.target.closest('[data-go]'), ad = e.target.closest('[data-addpl]');
    if (go) app.openVideo(go.dataset.go);
    else if (ad) sayAdded(addIds([ad.dataset.addpl]));
  });
  let sureRc = 0;
  $('rcClear').onclick = () => {
    if (Date.now() - sureRc > 3000) { sureRc = Date.now(); $('rcClear').textContent = tr('rc.clear?'); setTimeout(() => { $('rcClear').textContent = tr('rc.clear'); }, 3000); return; }
    sureRc = 0; $('rcClear').textContent = tr('rc.clear'); setRC([]); render();
  };
  // broken thumbnails: hide the image, keep the row
  for (const l of [$('plList'), $('rcList')]) l.addEventListener('error', e => { if (e.target.tagName === 'IMG') e.target.classList.add('broken'); }, true);
  enableReorder();

  app.on('video', id => {
    const R = getRC().filter(x => x.id !== id); R.unshift({ id, at: Date.now() }); setRC(R);
    const L = getPL(); const x = L.find(y => y.id === id);
    if (x && x.blocked) { x.blocked = false; setPL(L); }
    wantTitle(id); render();
  });
  app.on('error', (code, id) => {
    if (code !== 101 && code !== 150) return;
    const L = getPL(); const x = L.find(y => y.id === id); if (x && !x.blocked) { x.blocked = true; setPL(L); render(); }
  });
  app.on('sections', render);
  app.on('lang', render);
  render();
}
