import { generate, required, TUNINGS, sounds, boxBase } from '../../docs/woodshed/js/voicings.js';
import fs from 'fs';
const D = JSON.parse(fs.readFileSync('docs/woodshed/data/theory.json'));
const V = JSON.parse(fs.readFileSync('docs/woodshed/data/voicings.json'));
let pass = 0, fail = 0; const bad = [];
const ck = (c, n, x = '') => { if (c) pass++; else { fail++; bad.push(n + ' ' + JSON.stringify(x).slice(0, 200)); } };
const fingersNeeded = v => { const f = v.frets.filter(x => x > 0); if (!f.length) return 0; const mn = Math.min(...f); return v.barre ? 1 + f.filter(x => x > mn).length : f.length; };
let total = 0, tmax = 0, few = [];
for (const [tn, tun] of Object.entries(TUNINGS)) {
  for (const [ri, r] of D.roots.entries()) {
    for (const t of D.types) {
      const t0 = performance.now();
      const vs = generate(tun, ri, t.iv);
      tmax = Math.max(tmax, performance.now() - t0);
      total += vs.length;
      if (vs.length < 3) few.push(`${tn} ${r}${t.id}:${vs.length}`);
      const pcs = new Set(t.iv.map(i => (ri + i) % 12));
      const req = [...required(t.iv)].map(i => (ri + i) % 12);
      for (const v of vs) {
        const s = sounds(tun, v.frets);
        const name = `${tn} ${r}${t.id} ${v.frets.join(' ')}`;
        ck(s.every(p => pcs.has(p)), 'only chord tones ' + name);
        ck(req.every(p => s.includes(p)), 'required tones ' + name);
        const first = v.frets.findIndex(f => f >= 0);
        ck((tun[first] + v.frets[first]) % 12 === ri, 'root in bass ' + name);
        const fr = v.frets.filter(f => f > 0);
        ck(!fr.length || Math.max(...fr) - Math.min(...fr) <= 3, 'span <= 4 frets ' + name);
        ck(fingersNeeded(v) <= 4, 'max 4 fingers ' + name);
        ck(v.frets.every(f => f <= 15), 'within 15 frets ' + name);
        // finger numbers: every fretted note has a finger 1-4, open/muted have none
        ck(v.frets.every((f, i) => (f > 0 ? v.fingers[i] >= 1 && v.fingers[i] <= 4 : v.fingers[i] === 0)), 'fingers ' + name, v.fingers);
        const bb = boxBase(v.frets); ck(fr.every(f => f >= bb && f <= bb + 4), 'fits chord box ' + name);
      }
      // positions increase
      for (let i = 1; i < vs.length; i++) ck(vs[i].pos >= vs[i - 1].pos, `sorted ${tn} ${r}${t.id}`);
    }
  }
}
// well-known shapes must be found (standard tuning)
const known = { 'C:': '-1,3,2,0,1,0', 'G:': '3,2,0,0,0,3', 'E:': '0,2,2,1,0,0', 'A:': '-1,0,2,2,2,0', 'D:': '-1,-1,0,2,3,2', 'F:': '1,3,3,2,1,1',
  'A:m': '-1,0,2,2,1,0', 'E:m': '0,2,2,0,0,0', 'D:m': '-1,-1,0,2,3,1', 'E:5': '0,2,-1,-1,-1,-1', 'A:': '-1,0,2,2,2,0', 'G:7': '3,2,0,0,0,1',
  'A:7': '-1,0,2,0,2,0', 'B:7': '-1,2,1,2,0,2', 'C:maj7': '-1,3,2,0,0,0', 'A:m7': '-1,0,2,0,1,0', 'B:m7b5': '-1,2,3,2,3,-1' };
for (const [k, f] of Object.entries(known)) {
  const [r, tid] = k.split(':'); const t = D.types.find(x => x.id === tid);
  const vs = generate(TUNINGS.std, D.roots.indexOf(r), t.iv);
  const keys = vs.map(v => v.frets.join(','));
  const wanted = f.startsWith('0,2,-1') ? '0,2,2,-1,-1,-1' : f;   // power chord: allow either 2 or 3 strings
  ck(keys.includes(f) || keys.includes(wanted), 'finds ' + r + tid + ' ' + f, keys);
}
// curated data checks
for (const r of Object.keys(V)) for (const id of Object.keys(V[r])) {
  const t = D.types.find(x => x.id === id), ri = D.roots.indexOf(r);
  for (const [frets, fingers, barre] of V[r][id]) {
    ck(frets.length === 6 && fingers.length === 6, 'db shape ' + r + id);
    const s = sounds(TUNINGS.std, frets);
    const extra = id === '11' ? [(ri + 4) % 12] : [];
    ck(s.every(p => t.iv.some(i => (ri + i) % 12 === p) || extra.includes(p)), 'db tones ' + r + id + ' ' + frets);
  }
}
// spelling sanity
ck(D.notes['F#']['m7b5'].join() === 'F#,A,C,E', 'F#m7b5 spelling');
ck(D.notes['Eb']['7#9'].join() === 'Eb,G,Bb,Db,F#', 'Eb7#9 spelling');
ck(D.scaleNotes['A']['minpent'].join() === 'A,C,D,E,G', 'A minor pentatonic');
ck(D.keys.major[7].triads.map(x => x.s).join() === 'G,Am,Bm,C,D,Em,F#dim', 'G major key chords', D.keys.major[7].triads.map(x => x.s));
ck(D.keys.minor[9].triads.map(x => x.s).join() === 'Am,Bdim,C,Dm,Em,F,G', 'A minor key chords', D.keys.minor[9].triads.map(x => x.s));
console.log('voicings generated', total, 'slowest generate ms', tmax.toFixed(1));
console.log('chords with < 3 voicings', few.length, few.slice(0, 40).join(' '));
console.log(`${pass} passed, ${fail} failed`); if (fail) console.log(bad.slice(0, 30).join('\n'));
// Eb tuning reuses chords-db shapes from a half step up: they must sound the right chord
{
  let ok = 0, badN = 0;
  for (const [ri, r] of D.roots.entries()) for (const t of D.types) {
    const up = D.roots[(ri + 1) % 12]; const L = (V[up] || {})[t.id] || [];
    const pcs = new Set(t.iv.map(i => (ri + i) % 12)); const extra = t.id === '11' ? [(ri + 4) % 12] : [];
    for (const [frets] of L) { const s = sounds(TUNINGS.eb, frets); if (s.every(p => pcs.has(p) || extra.includes(p))) ok++; else badN++; }
  }
  console.log('Eb-tuning reused shapes ok', ok, 'bad', badN);
}
