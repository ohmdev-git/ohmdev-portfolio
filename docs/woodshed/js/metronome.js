// Metronome (Web Audio look-ahead scheduler), tap tempo, and count-in clicks.
import { $, store, clamp, toast } from './util.js?v=202611261520';
import { t } from './i18n.js?v=202611261520';

let app, S;
let ctx = null, master = null;
const M = {
  on: false, bpm: 120, beats: 4, hasBpm: false,
  follow: true, realign: true, vol: 0.7,
  nextTime: 0, beat: 0, timer: null
};
const LOOKAHEAD = 0.12, TICK = 25;
let live = [];          // scheduled oscillators, so they can be cancelled
let visT = [];          // timers for the beat dots / count badge

/* ---------- audio ---------- */
export function audio() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    try { ctx = new AC(); } catch (e) { return null; }
    master = ctx.createGain(); master.gain.value = 1; master.connect(ctx.destination);
  }
  if (ctx.state === 'suspended') { try { ctx.resume(); } catch (e) {} }
  return ctx;
}
function click(time, accent, loud = 1) {
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.type = 'triangle';
  o.frequency.value = accent ? 1975 : 1318;
  const peak = Math.max(0.0001, M.vol * (accent ? 1 : 0.62) * loud);
  g.gain.setValueAtTime(0.0001, time);
  g.gain.exponentialRampToValueAtTime(peak, time + 0.002);
  g.gain.exponentialRampToValueAtTime(0.0001, time + 0.06);
  o.connect(g); g.connect(master);
  o.start(time); o.stop(time + 0.07);
  const rec = { o, time };
  live.push(rec);
  o.onended = () => { live = live.filter(x => x !== rec); };
}
const effBpm = () => clamp(M.follow ? M.bpm * S.rate : M.bpm, 10, 600);
const interval = () => 60 / effBpm();

/* ---------- running metronome ---------- */
function schedule() {
  if (!ctx) return;
  while (M.nextTime < ctx.currentTime + LOOKAHEAD) {
    const b = M.beat % M.beats;
    click(M.nextTime, b === 0);
    showBeat(M.nextTime, b);
    M.nextTime += interval();
    M.beat++;
  }
}
function showBeat(time, b) {
  const ms = Math.max(0, (time - ctx.currentTime) * 1000);
  visT.push(setTimeout(() => {
    const dots = $('beatDots').children;
    for (let i = 0; i < dots.length; i++) dots[i].classList.toggle('on', i === b);
  }, ms));
  if (visT.length > 64) visT = visT.slice(-32);
}
export function start() {
  if (!audio()) { toast(t('met.audioFail')); return false; }
  M.on = true; M.beat = 0; M.nextTime = ctx.currentTime + 0.06;
  clearInterval(M.timer); M.timer = setInterval(schedule, TICK); schedule();
  render(); return true;
}
export function stop() {
  M.on = false; clearInterval(M.timer); M.timer = null;
  for (const r of live) { try { if (r.time > ctx.currentTime) r.o.stop(); } catch (e) {} }
  live = [];
  visT.forEach(clearTimeout); visT = [];
  [...$('beatDots').children].forEach(d => d.classList.remove('on'));
  render();
}
export const toggle = () => (M.on ? stop() : start());
export const isOn = () => M.on;

// put beat 1 at `when` (audio-clock seconds); drop clicks already queued after it
function realignAt(when) {
  if (!M.on || !ctx) return;
  for (const r of live) { if (r.time >= when - 0.005) { try { r.o.stop(); } catch (e) {} } }
  live = live.filter(r => r.time < when - 0.005);
  M.beat = 0; M.nextTime = Math.max(when, ctx.currentTime + 0.01);
}
const YT_LATENCY = 0.12;   // YouTube needs a moment to start sound after seek/play
export function onWrapped() { if (M.on && M.realign && ctx) realignAt(ctx.currentTime + YT_LATENCY); }

/* ---------- count-in ---------- */
export const countBeats = bars => (bars > 0 ? bars * M.beats : 0);
// Plays n clicks; returns milliseconds until the song should start (or -1 if no sound possible)
export function countIn(n) {
  if (!audio()) { toast(t('met.audioFail')); return -1; }
  cancelCountIn();
  const iv = interval(), t0 = ctx.currentTime + 0.08;
  if (M.on) realignAt(t0);           // the running click becomes the count-in
  else for (let i = 0; i < n; i++) click(t0 + i * iv, i % M.beats === 0, 1.1);
  const badge = $('cbadge');
  for (let i = 0; i < n; i++) {
    const ms = Math.max(0, (t0 + i * iv - ctx.currentTime) * 1000);
    visT.push(setTimeout(() => { badge.hidden = false; $('cbNum').textContent = (i % M.beats) + 1; $('cbText').textContent = t('countIn'); }, ms));
  }
  const startAt = t0 + n * iv;
  visT.push(setTimeout(() => { badge.hidden = true; }, Math.max(0, (startAt - ctx.currentTime) * 1000)));
  // when the click is running it simply continues: n is whole bars, so beat 1 lands on the song's start
  return Math.max(0, (startAt - YT_LATENCY - ctx.currentTime) * 1000);
}
export function cancelCountIn() {
  if (!ctx) return;
  if (!M.on) { for (const r of live) { try { r.o.stop(); } catch (e) {} } live = []; }
  visT.forEach(clearTimeout); visT = [];
  $('cbadge').hidden = true;
}

/* ---------- tap tempo ---------- */
let taps = [];
export function tap() {
  const now = performance.now();
  if (taps.length && now - taps[taps.length - 1] > 2000) taps = [];
  taps.push(now); if (taps.length > 9) taps.shift();
  audio();
  if (taps.length < 3) { $('tapTxt').textContent = t('tap.again'); return; }
  const iv = [];
  for (let i = 1; i < taps.length; i++) iv.push(taps[i] - taps[i - 1]);
  iv.sort((a, b) => a - b);
  const med = iv[Math.floor(iv.length / 2)];
  let played = 60000 / med;
  const song = M.follow ? played / S.rate : played;
  setBpm(Math.round(song), true);
  $('tapTxt').textContent = t('tap.set', { b: M.bpm });
  // put the click in phase with the taps
  if (M.on && ctx) { realignAt(ctx.currentTime + interval()); }
  clearTimeout(tap.t); tap.t = setTimeout(() => { $('tapTxt').textContent = t('tap'); }, 2200);
}

/* ---------- settings ---------- */
const songKey = () => 'woodshed:bpm:' + S.vid;
function setBpm(v, userSet) {
  v = Math.round(clamp(+v || 120, 30, 300));
  M.bpm = v;
  if (userSet) { M.hasBpm = true; if (S.vid) store.set(songKey(), { bpm: M.bpm, beats: M.beats }); }
  render();
}
function setBeats(b) {
  M.beats = [2, 3, 4, 6].includes(b) ? b : 4;
  if (S.vid && M.hasBpm) store.set(songKey(), { bpm: M.bpm, beats: M.beats });
  renderDots(); render();
}
export const hasBpm = () => M.hasBpm;
export const bpm = () => M.bpm;

function renderDots() {
  $('beatDots').innerHTML = Array.from({ length: M.beats }, (_, i) => `<i class="${i === 0 ? 'acc' : ''}"></i>`).join('');
}
export function render() {
  if (document.activeElement !== $('bpmIn')) $('bpmIn').value = M.bpm;
  $('metBtn').classList.toggle('on', M.on);
  $('metBtn').setAttribute('aria-pressed', M.on);
  $('metBtn').querySelector('span').textContent = t(M.on ? 'met.stop' : 'met.start');
  $('meterSeg').querySelectorAll('[data-b]').forEach(x => x.classList.toggle('on', +x.dataset.b === M.beats));
  const p = Math.round(S.rate * 100);
  $('metEff').textContent = M.follow && p !== 100 ? t('met.eff', { b: Math.round(effBpm()), p }) : '';
  $('metFollow').checked = M.follow; $('metRealign').checked = M.realign;
  $('metVol').value = Math.round(M.vol * 100);
}

export function init(a) {
  app = a; S = a.S;
  const g = store.get('woodshed:met', {});
  if (g && typeof g === 'object') {
    if (typeof g.follow === 'boolean') M.follow = g.follow;
    if (typeof g.realign === 'boolean') M.realign = g.realign;
    if (isFinite(g.vol)) M.vol = clamp(+g.vol, 0, 1);
  }
  const saveG = () => store.set('woodshed:met', { follow: M.follow, realign: M.realign, vol: M.vol });
  // unlock audio on the first touch / key (iOS & Chrome autoplay rules)
  const unlock = () => { audio(); window.removeEventListener('pointerdown', unlock, true); window.removeEventListener('keydown', unlock, true); };
  window.addEventListener('pointerdown', unlock, true); window.addEventListener('keydown', unlock, true);

  $('metBtn').onclick = toggle;
  $('tapBtn').addEventListener('pointerdown', e => { e.preventDefault(); tap(); });   // pointerdown = tighter timing than click
  $('tapBtn').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); tap(); } });
  $('bpmDown').onclick = () => setBpm(M.bpm - 1, true);
  $('bpmUp').onclick = () => setBpm(M.bpm + 1, true);
  $('bpmIn').addEventListener('change', () => setBpm($('bpmIn').value, true));
  $('bpmIn').addEventListener('keydown', e => { if (e.key === 'Enter') $('bpmIn').blur(); });
  $('meterSeg').addEventListener('click', e => { const b = e.target.closest('[data-b]'); if (b) setBeats(+b.dataset.b); });
  $('metFollow').onchange = () => { M.follow = $('metFollow').checked; saveG(); render(); };
  $('metRealign').onchange = () => { M.realign = $('metRealign').checked; saveG(); };
  $('metVol').addEventListener('input', () => { M.vol = clamp(+$('metVol').value / 100, 0, 1); saveG(); });

  app.on('video', () => {
    const v = S.vid ? store.get(songKey(), null) : null;
    if (v && isFinite(v.bpm)) { M.bpm = Math.round(clamp(+v.bpm, 30, 300)); M.beats = [2, 3, 4, 6].includes(v.beats) ? v.beats : 4; M.hasBpm = true; }
    else { M.hasBpm = false; }
    renderDots(); render();
  });
  app.on('rate', render);
  app.on('lang', render);
  renderDots(); render();
}
