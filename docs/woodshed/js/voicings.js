// Guitar chord voicing generator (any tuning). Pure functions — used in the browser and in node tests.
// A voicing: { frets: [6 numbers, low E → high e, -1 = muted, 0 = open], fingers: [6], barre: fret|0, gen: true }

export const TUNINGS = {
  std: [40, 45, 50, 55, 59, 64],   // E A D G B E
  eb: [39, 44, 49, 54, 58, 63],    // Eb Ab Db Gb Bb Eb
  dropd: [38, 45, 50, 55, 59, 64]  // D A D G B E
};

// which chord tones must be present (pitch-class intervals 0-11)
export function required(intervals) {
  const set = new Set(intervals.map(i => ((i % 12) + 12) % 12));
  const base = new Set(intervals.filter(i => i < 12));
  const req = new Set([0]);
  const has = n => set.has(n);
  // third or suspension (only tones written inside the octave count as a suspension)
  if (has(3) || has(4)) { if (has(3)) req.add(3); if (has(4)) req.add(4); }
  else { if (base.has(2)) req.add(2); if (base.has(5)) req.add(5); }
  // fifth: only when altered, or when it's the only other tone (power chord)
  if (has(6) && !has(7)) req.add(6);
  if (has(8) && !has(7) && !has(9)) req.add(8);
  if (set.size === 2 && has(7)) req.add(7);
  // sevenths / sixths
  if (has(10)) req.add(10);
  if (has(11)) req.add(11);
  if (has(9)) req.add(9);
  // extensions: the top one names the chord and must sound; altered ones (b9 #9 #11 b13) always must;
  // natural 9 / 11 under a higher extension may be left out, as guitarists usually do
  const ext = intervals.filter(i => i >= 12);
  if (ext.length) {
    req.add(Math.max(...ext) % 12);
    for (const i of ext) if ([13, 15, 18, 20].includes(i)) req.add(i % 12);
  }
  // dim7: double-flat 7 (9) already covered; dim triad: b5 required
  if (has(3) && has(6) && !has(7)) req.add(6);
  return req;
}

function fingering(frets) {
  const notes = frets.map((f, s) => ({ f, s })).filter(n => n.f > 0);
  const fingers = [0, 0, 0, 0, 0, 0];
  if (!notes.length) return { fingers, barre: 0, count: 0 };
  const min = Math.min(...notes.map(n => n.f));
  const atMin = notes.filter(n => n.f === min);
  let barre = 0;
  if (atMin.length >= 2) {
    const lo = atMin[0].s, hi = atMin[atMin.length - 1].s;
    // every string the barre crosses must be fretted at >= min (not open, not muted)
    let ok = true;
    for (let s = lo; s <= hi; s++) if (frets[s] < min) { ok = false; break; }
    if (ok) barre = min;
  }
  const rest = barre ? notes.filter(n => n.f > min) : notes;
  const count = (barre ? 1 : 0) + rest.length;
  if (barre) atMin.forEach(n => { fingers[n.s] = 1; });
  // lowest fret first; each finger roughly matches its fret (1 fret = 1 finger), never above 4,
  // always leaving enough fingers for the notes still to come
  const sorted = rest.slice().sort((a, b) => a.f - b.f || a.s - b.s);
  let prev = barre ? 1 : 0;
  sorted.forEach((n, i) => {
    const want = n.f - min + 1;
    const f = Math.min(Math.max(prev + 1, want), 4 - (sorted.length - 1 - i));
    fingers[n.s] = f; prev = f;
  });
  return { fingers, barre, count };
}

// all reasonable root-position voicings, low to high on the neck
export function generate(tuning, rootPc, intervals, opts = {}) {
  const maxFret = opts.maxFret || 15, maxPer = opts.maxPerPosition || 3, limit = opts.limit || 18;
  const pcs = new Set(intervals.map(i => (rootPc + i) % 12));
  const reqRel = required(intervals);
  const req = [...reqRel].map(i => (rootPc + i) % 12);
  const isPower = pcs.size === 2;
  const minStrings = isPower ? 2 : 4;
  const seen = new Set(), out = [];
  for (let start = 0; start <= maxFret - 3; start++) {
    const lo = Math.max(1, start), hi = start + 3;
    const cand = tuning.map(open => {
      const c = [-1];
      if (pcs.has(open % 12)) c.push(0);
      for (let f = lo; f <= hi; f++) if (pcs.has((open + f) % 12)) c.push(f);
      return c;
    });
    const fr = [0, 0, 0, 0, 0, 0];
    const rec = s => {
      if (s === 6) { consider(fr.slice()); return; }
      for (const f of cand[s]) { fr[s] = f; rec(s + 1); }
    };
    const consider = frets => {
      const sounding = frets.map((f, s) => (f >= 0 ? s : -1)).filter(s => s >= 0);
      if (sounding.length < minStrings) return;
      const first = sounding[0], last = sounding[sounding.length - 1];
      // muted strings: any number below the bass, at most one gap inside, at most two above the top note
      let gaps = 0; for (let s = first; s <= last; s++) if (frets[s] < 0) gaps++;
      if (gaps > 1 || (!isPower && 5 - last > 2)) return;
      if (isPower && (sounding.length > 3 || gaps)) return;
      // root in the bass
      if ((tuning[first] + frets[first]) % 12 !== rootPc) return;
      const got = new Set(sounding.map(s => (tuning[s] + frets[s]) % 12));
      for (const p of req) if (!got.has(p)) return;
      const fretted = frets.filter(f => f > 0);
      if (fretted.length) {
        const mn = Math.min(...fretted), mx = Math.max(...fretted);
        if (mx - mn > 3) return;
        if (frets.includes(0) && mx > 4) return;          // open strings only with nut-position shapes
      }
      const fg = fingering(frets);
      if (fg.count > 4) return;
      const key = frets.join(',');
      if (seen.has(key)) return;
      seen.add(key);
      const pos = fretted.length ? Math.min(...fretted) : 0;
      // score: more strings, fewer fingers, smaller stretch, fewer doubled notes
      const span = fretted.length ? Math.max(...fretted) - Math.min(...fretted) : 0;
      const score = sounding.length * 10 - fg.count * 3 - span * 2 - gaps * 6 - (frets.includes(0) && pos > 0 ? 2 : 0);
      out.push({ frets, fingers: fg.fingers, barre: fg.barre, pos, score, gen: true });
    };
    rec(0);
  }
  // keep the best few per position, then order up the neck
  const byPos = {};
  for (const v of out) (byPos[v.pos] = byPos[v.pos] || []).push(v);
  const keep = [];
  Object.keys(byPos).map(Number).sort((a, b) => a - b).forEach(p => {
    byPos[p].sort((a, b) => b.score - a.score).slice(0, maxPer).forEach(v => keep.push(v));
  });
  return keep.slice(0, limit);
}

// pitch classes a voicing actually sounds
export const sounds = (tuning, frets) => frets.map((f, s) => (f >= 0 ? (tuning[s] + f) % 12 : -1)).filter(p => p >= 0);

// first fret to draw in a 5-fret chord box
export function boxBase(frets) {
  const fr = frets.filter(f => f > 0);
  if (!fr.length) return 1;
  const mx = Math.max(...fr), mn = Math.min(...fr);
  return mx <= 4 ? 1 : mn;
}
