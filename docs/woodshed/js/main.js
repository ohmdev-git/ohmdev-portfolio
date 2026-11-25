// Woodshed — player, loop, timeline, speed, keyboard. Feature modules plug in through `app`.
import { $, clamp, esc, store, fmt, fmt2, parseYT, toast, isId } from './util.js?v=202611260618';
import * as Library from './library.js?v=202611260618';
import * as Markers from './markers.js?v=202611260618';
import * as Sections from './sections.js?v=202611260618';
import * as Met from './metronome.js?v=202611260618';
import * as Trainer from './trainer.js?v=202611260618';
import { t as tr, applyStatic, setLang, getLang } from './i18n.js?v=202611260618';

const MIN = 0.2;               // shortest loop, seconds
const PRESETS = [50, 60, 70, 75, 80, 85, 90, 100];
const STEPS = [0.01, 0.05, 0.1, 0.5];

const S = {
  api: false, P: null, ready: false, pending: null,
  vid: '', A: null, B: null, loop: false, gap: 0, reps: 0,
  rate: 1, rcT: null, dur: 0, state: -1, gapT: null, loadAt: 0, rateAt: 0, reapplied: false, active: null,
  zoom: false, zv: null, step: 0.1, err: null, ci: 0
};

/* ---------- tiny event bus for feature modules ---------- */
const listeners = {};
const app = {
  S, MIN,
  on(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
  emit(ev, ...a) { (listeners[ev] || []).forEach(fn => { try { fn(...a); } catch (e) { console.error(e); } }); }
};

/* ---------- overlay ---------- */
function overlay(title, text, isErr) {
  if (!title) { $('over').hidden = true; return; }
  $('over').hidden = false;
  $('overTitle').textContent = title;
  $('overText').textContent = text;
  $('overText').className = isErr ? 'err' : '';
  $('steps').hidden = !!isErr;
  if (!isErr) { $('overActs').hidden = true; $('overCode').textContent = ''; }
}

/* ---------- YouTube player ---------- */
function onApi() {
  S.api = true;
  app.emit('api');
  if (S.pending) { const p = S.pending; S.pending = null; createPlayer(p); }
}
window.onYouTubeIframeAPIReady = onApi;
(function loadApi() {
  if (window.YT && window.YT.Player) { onApi(); return; }
  const s = document.createElement('script');
  s.src = 'https://www.youtube.com/iframe_api';
  s.onerror = () => toast(tr('noYT'));
  document.head.appendChild(s);
})();

function createPlayer(p) {
  S.P = new YT.Player('player', {
    videoId: p.id,
    playerVars: Object.assign({ playsinline: 1, rel: 0, start: Math.floor(p.t || 0) }, /^https?:$/.test(location.protocol) ? { origin: location.origin } : {}),
    events: {
      onReady: () => {
        S.ready = true;
        applyRate(true);
        if (S.pending) { const q = S.pending; S.pending = null; cue(q); }
      },
      onStateChange: e => onState(e.data),
      onError: e => onErr(e.data),
      onPlaybackRateChange: () => onRate()
    }
  });
}
function cue(p) {
  S.loadAt = Date.now(); S.reapplied = false; S.dur = 0;
  if (S.ready) S.P.cueVideoById({ videoId: p.id, startSeconds: p.t || 0 });
  else S.pending = p;
}
function loadVideo(p, keep) {
  clearGap();
  S.vid = p.id; S.dur = 0; S.reps = 0; S.active = null; S.err = null;
  S.zoom = false; S.zv = null;
  if (!keep) { S.A = null; S.B = null; S.loop = false; }
  overlay(null);
  S.loadAt = Date.now(); S.reapplied = false;
  if (!S.api) S.pending = p;
  else if (!S.P) createPlayer(p);
  else cue(p);
  store.set('woodshed:last', p.id);
  renderAB(); renderStat(); renderTitle();
  app.emit('video', p.id);
}
// open a song from playlist / history / anywhere in the app
function openVideo(id, t) {
  if (!isId(id)) return;
  if (location.hash) history.replaceState(null, '', location.href.split('#')[0]);
  $('url').value = 'https://youtu.be/' + id;
  loadVideo({ id, t: t || 0 }, false);
}
function onState(st) {
  S.state = st;
  const playing = st === 1 || st === 3;
  $('playIco').innerHTML = playing ? '<path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/>' : '<path d="M8 5.5v13l11-6.5z"/>';
  $('play').setAttribute('aria-label', tr(playing ? 'pause' : 'play'));
  if (st === 1 || st === 5) { applyRate(false); renderTitle(); }
  if (st === 0 && loopOn()) wrap(true);
}
function onErr(code) {
  S.err = code;
  showErr();
  app.emit('error', code, S.vid);
}
function showErr() {
  const code = S.err; if (code == null) return;
  const key = { 2: 'err.2', 5: 'err.5', 100: 'err.100', 101: 'err.150', 150: 'err.150', 153: 'err.153' }[code];
  overlay(tr('err.title'), key ? tr(key) : tr('err.other', { c: code }), true);
  let title = '';
  try { title = (S.P && S.P.getVideoData && S.P.getVideoData().title) || Library.titleOf(S.vid) || ''; } catch (e) {}
  const q = encodeURIComponent((title.replace(/[(\[【][^)\]】]*[)\]】]/g, '').replace(/\s+/g, ' ').trim() || 'song') + ' cover');
  const box = $('overActs');
  box.innerHTML = (code === 101 || code === 150 ? `<a class="btn pri" target="_blank" rel="noopener" href="https://www.youtube.com/results?search_query=${q}">${esc(tr('findPlayable'))}</a>` : '') +
    (S.vid ? `<a class="btn" target="_blank" rel="noopener" href="https://www.youtube.com/watch?v=${encodeURIComponent(S.vid)}">${esc(tr('openYT'))}</a>` : '');
  box.hidden = false;
  $('overCode').textContent = tr('err.code', { c: code });
}
function applyRate(force) {
  if (!S.ready) return;
  const cur = S.P.getPlaybackRate ? S.P.getPlaybackRate() : 1;
  if (force || Math.abs(cur - S.rate) > 0.001) { noteReq(); S.P.setPlaybackRate(S.rate); }
}
function onRate() {
  // events can arrive late and out of order while the user drags the slider,
  // so wait until requests settle, then compare what YouTube is actually doing
  clearTimeout(S.rcT); S.rcT = setTimeout(reconcileRate, 350);
}
function reconcileRate() {
  if (!S.ready) return;
  if (Date.now() - S.rateAt < 300) { S.rcT = setTimeout(reconcileRate, 300); return; }
  const actual = S.P.getPlaybackRate();
  if (Math.abs(actual - S.rate) < 0.001) return;
  if (Date.now() - S.loadAt < 3000 && !S.reapplied) { S.reapplied = true; applyRate(true); return; }
  if (Date.now() - S.rateAt < 2500) toast(tr('ytOnly', { p: Math.round(actual * 100) }));
  S.rate = actual; renderRate(); app.emit('rate');
}
function noteReq() { S.rateAt = Date.now(); onRate(); }

/* ---------- transport ---------- */
const ok = () => S.ready && !!S.vid;
const now = () => ok() ? (S.P.getCurrentTime() || 0) : 0;
const loopOn = () => S.loop && S.A != null && S.B != null;
function seek(t) { if (ok()) S.P.seekTo(clamp(t, 0, S.dur || 1e9), true); }
function play() { if (ok()) S.P.playVideo(); }
function clearGap() { if (S.gapT) { clearTimeout(S.gapT); S.gapT = null; Met.cancelCountIn(); } }
// silence (gap) and/or count-in clicks, then start the song; returns false if nothing to wait for
function waitThenPlay(gapSec) {
  const beats = Met.countBeats(S.ci);
  if (!(gapSec > 0) && !beats) return false;
  clearGap();
  const go = () => { S.gapT = null; if (ok()) S.P.playVideo(); };
  const afterGap = () => {
    if (beats) { const ms = Met.countIn(beats); if (ms < 0) { go(); return; } S.gapT = setTimeout(go, ms); }
    else go();
  };
  if (gapSec > 0) S.gapT = setTimeout(() => { S.gapT = null; afterGap(); }, gapSec * 1000);
  else afterGap();
  return true;
}
function togglePlay() {
  if (!ok()) { if (!S.vid) $('url').focus(); return; }
  if (S.gapT) { clearGap(); S.P.playVideo(); return; }
  if (S.state === 1 || S.state === 3) S.P.pauseVideo();
  else {
    if (loopOn()) {
      const tt = now();
      if (tt < S.A || tt >= S.B - 0.05) seek(S.A);
      // count-in before the first rep too, when starting from the loop start
      if (S.ci && Math.abs(now() - S.A) < 0.3 && waitThenPlay(0)) return;
    }
    S.P.playVideo();
  }
}
function jump(d) {
  if (!ok()) return;
  let t = now() + d;
  if (loopOn()) t = clamp(t, S.A, S.B - 0.05);
  seek(t);
}
function userSeek(to) {
  if (loopOn() && (to < S.A - 0.05 || to > S.B)) { setLoop(false); toast(tr('loopOffOutside')); }
  seek(to);
}
function wrap(ended) {
  S.reps++; renderStat();
  app.emit('rep', S.reps);
  seek(S.A);
  const waits = S.gap > 0 || Met.countBeats(S.ci) > 0;
  if (waits) {
    if (!ended) S.P.pauseVideo();
    waitThenPlay(S.gap);
  } else {
    if (ended) S.P.playVideo();
    Met.onWrapped();
  }
}

/* ---------- A / B ---------- */
function setAB(a, b) {
  const D = S.dur > 0 ? S.dur : 1e9;
  if (a != null) a = clamp(a, 0, D);
  if (b != null) b = clamp(b, 0, D);
  if (a != null && b != null && b < a + MIN) {
    if (a + MIN <= D) b = a + MIN; else { b = D; a = Math.max(0, D - MIN); }
  }
  S.A = a; S.B = b; S.active = null;
  if (S.A == null || S.B == null) { setLoop(false, true); if (S.zoom) setZoom(false); }
  renderAB();
  app.emit('ab');
}
function setLoop(on, quiet) {
  on = !!on && S.A != null && S.B != null;
  if (!on) clearGap();
  if (on && !S.loop) S.reps = 0;
  S.loop = on;
  renderStat(); renderAB();
  if (on && !quiet && ok()) { const t = now(); if (t < S.A || t > S.B) seek(S.A); }
}
function markA() {
  if (!ok()) return;
  const t = now(); let b = S.B;
  if (b != null && b < t + MIN) { b = null; toast(tr('bCleared')); }
  setAB(t, b);
}
function markB() {
  if (!ok()) return;
  const t = now(); const a = S.A == null ? 0 : S.A;
  if (t < a + MIN) { toast(tr('bAfterA')); return; }
  setAB(a, t); setLoop(true);
}
function nudge(which, dir) {
  const d = Math.sign(dir) * S.step;
  if (which === 'a' && S.A != null) setAB(S.B != null ? Math.min(S.A + d, S.B - MIN) : S.A + d, S.B);
  if (which === 'b' && S.B != null) setAB(S.A, Math.max(S.B + d, (S.A || 0) + MIN));
}

/* ---------- zoom ---------- */
let drag = null;
function fitZoom() {
  const L = S.B - S.A, pad = Math.max(0.5, L * 0.15);
  S.zv = { s: Math.max(0, S.A - pad), e: Math.min(S.dur, S.B + pad) };
}
function setZoom(on) {
  on = !!on && S.A != null && S.B != null && S.dur > 0;
  S.zoom = on; S.zv = null;
  if (on) fitZoom();
  $('zoomBtn').classList.toggle('on', on);
  $('zoomBtn').setAttribute('aria-pressed', on);
  $('track').classList.toggle('zoomed', on);
  renderAB();
}
function view() {
  if (!S.zoom || S.A == null || S.B == null || !S.dur) return { s: 0, e: S.dur || 1 };
  if (!S.zv || (!drag && (S.A < S.zv.s || S.B > S.zv.e))) fitZoom();
  return S.zv;
}
function pct(t) { const v = view(); return clamp((t - v.s) / (v.e - v.s) * 100, 0, 100); }
function inView(t) { const v = view(); return t >= v.s - 1e-6 && t <= v.e + 1e-6; }

/* ---------- rendering ---------- */
function renderAB() {
  for (const [k, el] of [['A', $('at')], ['B', $('bt')]]) {
    const v = S[k]; el.textContent = v == null ? tr('notSet') : (S.step < 0.1 ? fmt2(v) : fmt(v)); el.classList.toggle('none', v == null);
  }
  const show = S.dur > 0;
  $('hA').style.display = show && S.A != null && inView(S.A) ? 'block' : 'none';
  $('hB').style.display = show && S.B != null && inView(S.B) ? 'block' : 'none';
  if (show && S.A != null) $('hA').style.left = pct(S.A) + '%';
  if (show && S.B != null) $('hB').style.left = pct(S.B) + '%';
  const r = $('region');
  if (show && S.A != null && S.B != null) {
    r.style.display = 'block'; r.style.left = pct(S.A) + '%'; r.style.width = (pct(S.B) - pct(S.A)) + '%';
    r.style.opacity = S.loop ? 1 : .45;
  } else r.style.display = 'none';
  $('loopBtn').classList.toggle('on', S.loop);
  $('loopBtn').setAttribute('aria-pressed', S.loop);
  $('loopBtn').querySelector('span').textContent = tr(S.loop ? 'looping' : 'loop');
  $('loopBtn').disabled = S.A == null || S.B == null;
  $('toA').disabled = S.A == null;
  $('saveBtn').disabled = S.A == null || S.B == null;
  $('zoomBtn').disabled = S.A == null || S.B == null || !show;
  // zoom extras: overview strip + range labels
  const v = view();
  $('mini').hidden = !S.zoom;
  $('vs').textContent = S.zoom ? fmt2(v.s) : ''; $('ve').textContent = S.zoom ? fmt2(v.e) : '';
  if (S.zoom && S.dur) {
    $('miniWin').style.left = (v.s / S.dur * 100) + '%';
    $('miniWin').style.width = ((v.e - v.s) / S.dur * 100) + '%';
  }
  renderStat();
  app.emit('timeline');
}
function renderStat() {
  $('stat').classList.toggle('live', S.loop);
  $('loopState').textContent = S.loop ? tr('loopingLen', { len: fmt(S.B - S.A) }) : tr('loopOff');
  $('loopCount').textContent = tr('reps', { n: S.reps });
}
function renderRate() {
  const p = Math.round(S.rate * 100);
  $('spv').textContent = p + '%';
  $('spx').textContent = p === 100 ? tr('origSpeed') : (p < 100 ? tr('xSlower', { x: (100 / p).toFixed(2).replace(/\.?0+$/, '') }) : tr('xFaster', { x: (p / 100).toFixed(2).replace(/\.?0+$/, '') }));
  $('sp').value = p;
  $('sp').style.setProperty('--p', ((p - 25) / 175 * 100) + '%');
  $('chips').querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.p === p));
}
function setRate(p) {
  S.rate = Math.round(clamp(p, 25, 200)) / 100;
  renderRate();
  if (S.ready) { noteReq(); S.P.setPlaybackRate(S.rate); }
  app.emit('rate');
}
function renderTitle() {
  let t = S.vid ? Library.titleOf(S.vid) : '';
  if (!t && S.ready && S.P.getVideoData) { try { t = S.P.getVideoData().title || ''; } catch (e) {} }
  $('songTitle').textContent = t || tr(S.vid ? 'ytVideo' : 'noSong');
  $('songTitle').title = t || '';
}

/* ---------- share ---------- */
function shareLink() {
  let h = '#v=' + S.vid;
  if (S.A != null && S.B != null) h += '&a=' + S.A.toFixed(2) + '&b=' + S.B.toFixed(2);
  if (S.rate !== 1) h += '&s=' + S.rate;
  return location.href.split('#')[0] + h;
}
$('share').addEventListener('click', () => {
  if (!S.vid) { toast(tr('loadFirst')); return; }
  const link = shareLink();
  const fallback = () => {
    const ta = document.createElement('textarea'); ta.value = link; ta.style.cssText = 'position:fixed;opacity:0';
    document.body.appendChild(ta); ta.select();
    let done = false; try { done = document.execCommand('copy'); } catch (e) {}
    ta.remove(); toast(tr(done ? 'copied' : 'copyFail'));
    if (!done) history.replaceState(null, '', link);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(link).then(() => toast(tr('linkCopied')), fallback);
  else fallback();
});
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  const id = q.get('v'); if (!isId(id)) return null;
  const n = k => { const v = parseFloat(q.get(k)); return isFinite(v) && v >= 0 ? v : null; };
  return { id, a: n('a'), b: n('b'), s: n('s') };
}
function openFromHash() {
  const h = readHash(); if (!h) return false;
  $('url').value = 'https://youtu.be/' + h.id;
  if (h.s != null) setRate(h.s * 100);
  const hasAB = h.a != null && h.b != null && h.b > h.a;
  loadVideo({ id: h.id, t: hasAB ? h.a : 0 }, false);
  if (hasAB) { setAB(h.a, h.b); setLoop(true, true); }
  return true;
}
window.addEventListener('hashchange', openFromHash);

/* ---------- timeline ---------- */
const track = $('track');
const tAt = x => { const r = track.getBoundingClientRect(), v = view(); return v.s + clamp((x - r.left) / r.width, 0, 1) * (v.e - v.s); };
track.addEventListener('pointerdown', e => {
  if (!S.dur || e.button > 0) return;
  const h = e.target.closest('.hdl');
  const ae = document.activeElement; if (ae && ae !== document.body && /^(INPUT|SELECT|TEXTAREA)$/.test(ae.tagName)) ae.blur();
  drag = { mode: h ? h.dataset.h : 'pend', x0: e.clientX, t0: tAt(e.clientX), moved: false };
  try { track.setPointerCapture(e.pointerId); } catch (err) {}
  e.preventDefault();
});
track.addEventListener('pointermove', e => {
  if (S.dur) { const hv = $('hover'); hv.textContent = (S.zoom ? fmt2 : fmt)(tAt(e.clientX)); hv.style.left = clamp((e.clientX - track.getBoundingClientRect().left), 24, track.clientWidth - 24) + 'px'; }
  if (!drag) return;
  const t = tAt(e.clientX);
  if (Math.abs(e.clientX - drag.x0) > 6) drag.moved = true;
  if (drag.mode === 'pend' && drag.moved) drag.mode = 'sel';
  if (!drag.moved) return;
  if (drag.mode === 'sel') setAB(Math.min(drag.t0, t), Math.max(drag.t0, t));
  else if (drag.mode === 'a') setAB(S.B != null ? Math.min(t, S.B - MIN) : t, S.B);
  else if (drag.mode === 'b') setAB(S.A, Math.max(t, (S.A || 0) + MIN));
});
function endDrag() {
  if (!drag) return;
  const m = drag.mode, t0 = drag.t0, moved = drag.moved; drag = null;
  if (m === 'pend') userSeek(t0);
  else if (!moved) seek(m === 'a' ? S.A : S.B);   // tapped a marker: jump to it
  else if (m === 'sel') { setLoop(true, true); seek(S.A); }
  else if (S.loop) { const t = now(); if (t < S.A || t > S.B) seek(S.A); }
  if (S.zoom && S.A != null && S.B != null) {
    const v = view();
    if (S.B - S.A < (v.e - v.s) * 0.25) fitZoom();   // loop got much shorter: zoom in again
  }
  renderAB();
}
track.addEventListener('pointerup', endDrag);
track.addEventListener('pointercancel', () => { drag = null; renderAB(); });
track.addEventListener('lostpointercapture', endDrag);

/* ---------- controls ---------- */
$('urlForm').addEventListener('submit', e => {
  e.preventDefault();
  const p = parseYT($('url').value);
  if (!p) { toast(tr('notYTLink')); $('url').focus(); return; }
  if (p.list) { toast(tr('isPlaylist')); return; }
  $('url').blur();
  if (location.hash) history.replaceState(null, '', location.href.split('#')[0]);
  loadVideo(p, false);
});
$('play').onclick = togglePlay;
$('b5').onclick = () => jump(-5); $('b2').onclick = () => jump(-2);
$('f2').onclick = () => jump(2);  $('f5').onclick = () => jump(5);
$('setA').onclick = markA; $('setB').onclick = markB;
document.querySelectorAll('[data-n]').forEach(b => b.onclick = () => { const [k, d] = b.dataset.n.split(','); nudge(k, +d); });
$('loopBtn').onclick = () => setLoop(!S.loop);
$('toA').onclick = () => { if (S.A != null) seek(S.A); };
$('clr').onclick = () => { setAB(null, null); };
$('zoomBtn').onclick = () => setZoom(!S.zoom);
$('chips').innerHTML = PRESETS.map(p => `<button class="btn" data-p="${p}">${p}%</button>`).join('');
$('chips').addEventListener('click', e => { const b = e.target.closest('[data-p]'); if (b) setRate(+b.dataset.p); });
$('sp').addEventListener('input', e => setRate(+e.target.value));
$('spDown').onclick = () => setRate(S.rate * 100 - 5);
$('spUp').onclick = () => setRate(S.rate * 100 + 5);
$('gap').addEventListener('click', e => {
  const b = e.target.closest('[data-g]'); if (!b) return;
  S.gap = +b.dataset.g; store.set('woodshed:gap', S.gap);
  $('gap').querySelectorAll('[data-g]').forEach(x => x.classList.toggle('on', x === b));
});
$('ciSeg').addEventListener('click', e => {
  const b = e.target.closest('[data-ci]'); if (!b) return;
  setCountIn(+b.dataset.ci);
  if (S.ci && !Met.hasBpm()) toast(tr('needBpm'));
});
function setCountIn(v) {
  S.ci = [0, 1, 2].includes(v) ? v : 0; store.set('woodshed:countin', S.ci);
  $('ciSeg').querySelectorAll('[data-ci]').forEach(x => x.classList.toggle('on', +x.dataset.ci === S.ci));
}
document.querySelector('.langs').addEventListener('click', e => {
  const b = e.target.closest('[data-lang]'); if (!b) return;
  setLang(b.dataset.lang); relang();
});
function relang() {
  applyStatic();
  document.querySelectorAll('.langs [data-lang]').forEach(x => x.classList.toggle('on', x.dataset.lang === getLang()));
  $('ci1').textContent = tr('countIn.bars', { n: 1 }); $('ci2').textContent = tr('countIn.bars', { n: 2 });
  $('mkNames').innerHTML = tr('mk.names').map(n => `<option value="${esc(n)}">`).join('');
  setStep(S.step);
  renderRate(); renderAB(); renderStat(); renderTitle();
  if (S.state >= 0) onState(S.state); else $('play').setAttribute('aria-label', tr('play'));
  if (S.err != null && !$('over').hidden) showErr();
  app.emit('lang');
}
$('stepSeg').addEventListener('click', e => {
  const b = e.target.closest('[data-step]'); if (!b) return;
  setStep(+b.dataset.step);
});
function setStep(v) {
  S.step = STEPS.includes(v) ? v : 0.1; store.set('woodshed:step', S.step);
  $('stepSeg').querySelectorAll('[data-step]').forEach(x => x.classList.toggle('on', +x.dataset.step === S.step));
  document.querySelectorAll('[data-n]').forEach(b => { const [k, d] = b.dataset.n.split(','); b.title = tr(d < 0 ? 'earlierBy' : 'laterBy', { s: S.step }); });
  renderAB();
}

/* ---------- keyboard ---------- */
document.addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const tg = e.target, tag = tg.tagName;
  const typing = tag === 'INPUT' && !['range', 'checkbox', 'radio', 'button', 'submit'].includes(tg.type);
  if (tag === 'TEXTAREA' || tag === 'SELECT' || tg.isContentEditable || typing) return;
  const k = e.key;
  if (tag === 'INPUT' && (k.startsWith('Arrow') || k === 'Home' || k === 'End' || k === 'PageUp' || k === 'PageDown')) return;
  let handled = true;
  if (k === ' ' || k === 'Spacebar') togglePlay();
  else if (k === 'ArrowLeft') jump(e.shiftKey ? -5 : -2);
  else if (k === 'ArrowRight') jump(e.shiftKey ? 5 : 2);
  else if (k === 'a' || k === 'A') markA();
  else if (k === 'b' || k === 'B') markB();
  else if (k === 'l' || k === 'L') { if (S.A == null || S.B == null) toast(tr('setABFirst')); else setLoop(!S.loop); }
  else if (k === 'q' || k === 'Q') { if (S.A != null) seek(S.A); }
  else if (k === 'z' || k === 'Z') { if (S.A == null || S.B == null) toast(tr('setABFirst')); else setZoom(!S.zoom); }
  else if (k === 'm' || k === 'M') Markers.addAtPlayhead();
  else if (k === 'n' || k === 'N') Library.step(1);
  else if (k === 'p' || k === 'P') Library.step(-1);
  else if (k === 'k' || k === 'K') Met.toggle();
  else if (k === 't' || k === 'T') { if (!e.repeat) Met.tap(); }
  else if (k === '[') setRate(S.rate * 100 - 5);
  else if (k === ']') setRate(S.rate * 100 + 5);
  else if (/^[03-9]$/.test(k)) setRate(k === '0' ? 100 : +k * 10);
  else handled = false;
  if (handled) e.preventDefault();
});
// stop Space from also "clicking" a focused button on keyup
document.addEventListener('keyup', e => { if ((e.key === ' ' || e.key === 'Spacebar') && (e.target.tagName === 'BUTTON' || (e.target.tagName === 'INPUT' && /^(checkbox|radio)$/.test(e.target.type)))) e.preventDefault(); });
// mouse clicks shouldn't leave focus on buttons (keeps Space for play/pause)
document.addEventListener('pointerup', e => { const b = e.target.closest && e.target.closest('button, label.tog'); if (b) setTimeout(() => { b.blur(); const c = b.querySelector && b.querySelector('input'); if (c) c.blur(); }, 0); });

/* ---------- clock ---------- */
setInterval(() => {
  if (!ok()) return;
  const t = S.P.getCurrentTime() || 0, d = S.P.getDuration() || 0;
  if (d > 0 && Math.abs(d - S.dur) > 0.01) {
    S.dur = d;
    if (S.B != null && S.B > d) setAB(S.A, d); else renderAB();
    app.emit('dur', d);
  }
  // wrap a hair early so the overshoot past B stays tiny at any speed
  if (loopOn() && !S.gapT && S.state === 1 && t + 0.02 * S.rate >= S.B) wrap(false);
  $('now').textContent = fmt(t);
  $('dur').textContent = '/ ' + fmt(S.dur);
  if (S.dur) {
    const p = pct(t) + '%', vis = inView(t);
    $('head').style.left = p; $('head').style.display = vis ? 'block' : 'none';
    $('prog').style.width = p;
    if (S.zoom) $('miniHead').style.left = (t / S.dur * 100) + '%';
  }
}, 30);

/* ---------- app API for modules ---------- */
Object.assign(app, { ok, now, seek, userSeek, play, setAB, setLoop, setRate, loadVideo, openVideo, pct, inView, view, renderTitle, fmt, tr });

/* ---------- boot ---------- */
if (location.protocol === 'file:') $('fileWarn').hidden = false;
S.gap = [0, 1, 2, 3].includes(store.get('woodshed:gap', 0)) ? store.get('woodshed:gap', 0) : 0;
$('gap').querySelectorAll('[data-g]').forEach(x => x.classList.toggle('on', +x.dataset.g === S.gap));
Library.init(app); Markers.init(app); Sections.init(app); Met.init(app); Trainer.init(app);
setCountIn(+store.get('woodshed:countin', 0));
relang();
app.on('titles', renderTitle);
setStep(+store.get('woodshed:step', 0.1));
renderRate(); renderAB(); renderStat(); renderTitle();
if (!openFromHash()) {
  const last = store.get('woodshed:last', null);
  if (isId(last)) { $('url').value = 'https://youtu.be/' + last; loadVideo({ id: last, t: 0 }, false); }
}
