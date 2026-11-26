// Neck diagram shared by the theory panel and the practice games. Returns SVG markup.
import { esc } from './util.js?v=202611261520';
import { TUNINGS } from './voicings.js?v=202611261520';

const FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
export const H = 210;
const L = 52, RGT = 14, TOP = 26, GAP = 30;

// opts: tuning id, lefty, frets (last fret drawn), marks [{s,f,label,kind,dim}], strName(pc),
//       hits (true = add tap targets for every string/fret cell), off (Set of strings drawn faded)
export function geometry(frets, lefty) {
  const W = frets >= 15 ? 1000 : frets >= 12 ? 840 : 520;
  const fx = n => L + (W - L - RGT) * (1 - Math.pow(2, -n / 12)) / (1 - Math.pow(2, -frets / 12));
  const mirror = x => (lefty ? W - x : x);
  const noteX = f => (f === 0 ? L - 20 : (fx(f - 1) + fx(f)) / 2);
  const sy = s => TOP + (5 - s) * GAP;            // high e on top
  return { W, fx, mirror, noteX, sy };
}
export function boardSVG(o) {
  const frets = o.frets || 15, lefty = !!o.lefty;
  const tun = TUNINGS[o.tuning] || TUNINGS.std;
  const { W, fx, mirror, noteX, sy } = geometry(frets, lefty);
  const name = o.strName || (pc => FLAT[pc]);
  let g = '';
  const x0 = Math.min(mirror(L), mirror(fx(frets))), x1 = Math.max(mirror(L), mirror(fx(frets)));
  g += `<rect class="wood" x="${x0}" y="${TOP - 12}" width="${x1 - x0}" height="${GAP * 5 + 24}" rx="4"/>`;
  for (const f of [3, 5, 7, 9, 15]) if (f <= frets) g += `<circle class="inlay" cx="${mirror(noteX(f))}" cy="${TOP + GAP * 2.5}" r="6"/>`;
  if (frets >= 12) g += `<circle class="inlay" cx="${mirror(noteX(12))}" cy="${TOP + GAP * 1.5}" r="6"/><circle class="inlay" cx="${mirror(noteX(12))}" cy="${TOP + GAP * 3.5}" r="6"/>`;
  for (let f = 1; f <= frets; f++) g += `<line class="fret" x1="${mirror(fx(f))}" y1="${TOP - 12}" x2="${mirror(fx(f))}" y2="${TOP + GAP * 5 + 12}"/>`;
  g += `<line class="nut" x1="${mirror(L)}" y1="${TOP - 12}" x2="${mirror(L)}" y2="${TOP + GAP * 5 + 12}"/>`;
  for (let s = 0; s < 6; s++) {
    const off = o.off && o.off.has(s);
    g += `<line class="str${off ? ' off' : ''}" style="stroke-width:${0.8 + (5 - s) * 0.32}px" x1="${x0}" y1="${sy(s)}" x2="${x1}" y2="${sy(s)}"/>`;
    g += `<text class="slab${off ? ' off' : ''}" x="${lefty ? W - 12 : 12}" y="${sy(s) + 4}">${esc(name(tun[s] % 12))}</text>`;
  }
  for (const f of [1, 3, 5, 7, 9, 12, 15]) if (f <= frets) g += `<text class="fnum" x="${mirror(noteX(f))}" y="${H - 6}">${f}</text>`;
  for (const m of o.marks || []) {
    const x = mirror(m.f < 0 ? L - 20 : noteX(m.f)), y = sy(m.s);
    if (m.f < 0) { g += `<text class="mute" x="${x}" y="${y + 4}">×</text>`; continue; }
    g += `<g class="n ${m.kind || ''}${m.dim ? ' dim' : ''}"><circle cx="${x}" cy="${y}" r="11"/><text x="${x}" y="${y + 3.8}">${esc(m.label == null ? '' : m.label)}</text></g>`;
  }
  if (o.hits) {
    for (let s = 0; s < 6; s++) {
      if (o.off && o.off.has(s)) continue;
      for (let f = 0; f <= frets; f++) {
        const a = f === 0 ? L - 40 : fx(f - 1), b = f === 0 ? L : fx(f);
        const xa = Math.min(mirror(a), mirror(b)), xb = Math.max(mirror(a), mirror(b));
        g += `<rect class="hit" data-s="${s}" data-f="${f}" x="${xa}" y="${sy(s) - GAP / 2}" width="${xb - xa}" height="${GAP}"/>`;
      }
    }
  }
  return { svg: g, W, H };
}
