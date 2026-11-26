// Plucked-string sound (Karplus–Strong), rendered once per note and cached.
import { audio } from './metronome.js?v=202611261446';

const cache = new Map();
function buffer(ctx, midi) {
  const key = ctx.sampleRate + ':' + midi;
  if (cache.has(key)) return cache.get(key);
  const sr = ctx.sampleRate, freq = 440 * Math.pow(2, (midi - 69) / 12);
  const len = Math.floor(sr * 1.8), N = Math.max(2, Math.round(sr / freq));
  const buf = ctx.createBuffer(1, len, sr), out = buf.getChannelData(0);
  const ring = new Float32Array(N);
  let seed = midi * 9301 + 49297;
  const rnd = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280 * 2 - 1; };
  for (let i = 0; i < N; i++) ring[i] = rnd();
  const damp = 0.996 - Math.max(0, midi - 60) * 0.0004;
  let idx = 0, prev = 0;
  for (let i = 0; i < len; i++) {
    const cur = ring[idx];
    const next = damp * 0.5 * (cur + prev);
    prev = cur; ring[idx] = next; idx = (idx + 1) % N;
    out[i] = cur * Math.min(1, (len - i) / (sr * 0.05));
  }
  cache.set(key, buf);
  return buf;
}
let bus = null, live = [];
function out(ctx) {
  if (!bus) { bus = ctx.createGain(); bus.gain.value = 0.5; bus.connect(ctx.destination); }
  return bus;
}
export function stopAll() { for (const s of live) { try { s.stop(); } catch (e) {} } live = []; }
// play midi notes; gap = seconds between notes (0.03 strum, 0.25 scale)
export function play(midis, gap = 0.035) {
  const ctx = audio(); if (!ctx) return false;
  stopAll();
  const t0 = ctx.currentTime + 0.03;
  midis.forEach((m, i) => {
    const s = ctx.createBufferSource(); s.buffer = buffer(ctx, m);
    s.connect(out(ctx)); s.start(t0 + i * gap);
    live.push(s); s.onended = () => { live = live.filter(x => x !== s); };
  });
  return true;
}
