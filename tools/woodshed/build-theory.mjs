// Build Woodshed's theory data from the reference libraries (run with node, not shipped).
//   node tools/build-theory.mjs <node_modules dir> <out dir>
// tonal 6.5.1 (MIT) -> chord formulas, spelled notes, scales, keys
// @tombatossals/chords-db 0.5.1 (MIT) -> curated guitar voicings, checked note-by-note against tonal
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { TUNINGS, required } from '../../docs/woodshed/js/voicings.js';

const NM = process.argv[2], OUT = process.argv[3];
const require = createRequire(path.join(NM, 'x.js'));
const T = require('tonal');
const db = JSON.parse(fs.readFileSync(path.join(NM, '@tombatossals/chords-db/lib/guitar.json')));

export const ROOTS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
const DBKEY = { 'C#': 'Csharp', 'F#': 'Fsharp' };

// [id, tonal symbol, chords-db suffix, group]
const TYPES = [
  ['', '', 'major', 'basic'], ['m', 'm', 'minor', 'basic'], ['5', '5', null, 'basic'],
  ['7', '7', '7', 'seventh'], ['maj7', 'maj7', 'maj7', 'seventh'], ['m7', 'm7', 'm7', 'seventh'], ['mMaj7', 'mMaj7', 'mmaj7', 'seventh'],
  ['6', '6', '6', 'sixth'], ['m6', 'm6', 'm6', 'sixth'], ['69', '69', '69', 'sixth'], ['m69', 'm69', 'm69', 'sixth'],
  ['sus2', 'sus2', 'sus2', 'sus'], ['sus4', 'sus4', 'sus4', 'sus'], ['7sus4', '7sus4', '7sus4', 'sus'],
  ['add9', 'add9', 'add9', 'ext'], ['madd9', 'madd9', 'madd9', 'ext'],
  ['9', '9', '9', 'ext'], ['maj9', 'maj9', 'maj9', 'ext'], ['m9', 'm9', 'm9', 'ext'],
  ['11', '11', '11', 'ext'], ['m11', 'm11', 'm11', 'ext'],
  ['13', '13', '13', 'ext'], ['maj13', 'maj13', 'maj13', 'ext'], ['m13', 'm13', null, 'ext'],
  ['dim', 'dim', 'dim', 'dimaug'], ['dim7', 'dim7', 'dim7', 'dimaug'], ['m7b5', 'm7b5', 'm7b5', 'dimaug'],
  ['aug', 'aug', 'aug', 'dimaug'], ['7#5', '7#5', 'aug7', 'dimaug'], ['maj7#5', 'maj7#5', 'maj7#5', 'dimaug'],
  ['7b5', '7b5', '7b5', 'alt'], ['7b9', '7b9', '7b9', 'alt'], ['7#9', '7#9', '7#9', 'alt'], ['9#5', '9#5', 'aug9', 'alt'], ['9#11', '9#11', '9#11', 'alt']
];
// conventional extra tones that are fine in a voicing (relative semitones)
const ALLOWED_EXTRA = { '11': [4], 'm11': [], '9#11': [] };

const LABEL = { '1P': 'R', '2m': 'b2', '2M': '2', '2A': '#2', '3m': 'b3', '3M': '3', '4P': '4', '4A': '#4', '5d': 'b5', '5P': '5', '5A': '#5',
  '6m': 'b6', '6M': '6', '7d': 'bb7', '7m': 'b7', '7M': '7', '8P': 'R', '9m': 'b9', '9M': '9', '9A': '#9', '11P': '11', '11A': '#11', '13m': 'b13', '13M': '13' };
const semis = iv => T.Interval.semitones(iv);

const types = TYPES.map(([id, sym, , group]) => {
  const c = T.Chord.get('C' + sym);
  if (c.empty) throw new Error('tonal lacks ' + sym);
  return { id, group, name: c.name || id, iv: c.intervals.map(semis), lab: c.intervals.map(i => LABEL[i] || i) };
});

// spelled notes for every root x type
const notes = {};
for (const r of ROOTS) {
  notes[r] = {};
  for (const [id, sym] of TYPES) notes[r][id] = T.Chord.get(r + sym).notes.map(n => T.Note.simplify(n));
}

// scales: [id, tonal name, group]
const SCALES = [
  ['major', 'major', 'common'], ['minor', 'minor', 'common'], ['majpent', 'major pentatonic', 'common'], ['minpent', 'minor pentatonic', 'common'],
  ['blues', 'minor blues', 'common'], ['majblues', 'major blues', 'common'],
  ['dorian', 'dorian', 'mode'], ['phrygian', 'phrygian', 'mode'], ['lydian', 'lydian', 'mode'], ['mixolydian', 'mixolydian', 'mode'], ['locrian', 'locrian', 'mode'],
  ['harmminor', 'harmonic minor', 'minor'], ['melminor', 'melodic minor', 'minor'], ['phrydom', 'phrygian dominant', 'minor'],
  ['dimwh', 'diminished', 'sym'], ['dimhw', 'half-whole diminished', 'sym'], ['whole', 'whole tone', 'sym'],
  ['altered', 'altered', 'jazz'], ['lyddom', 'lydian dominant', 'jazz'], ['bebop', 'bebop', 'jazz']
];
const scales = SCALES.map(([id, name, group]) => {
  const s = T.Scale.get('C ' + name);
  if (s.empty) throw new Error('tonal lacks scale ' + name);
  return { id, group, iv: s.intervals.map(semis), lab: s.intervals.map(i => LABEL[i] || i) };
});
const scaleNotes = {};
for (const r of ROOTS) { scaleNotes[r] = {}; for (const [id, name] of SCALES) scaleNotes[r][id] = T.Scale.get(r + ' ' + name).notes.map(n => T.Note.simplify(n)); }

// keys: diatonic triads + sevenths, linked to (root index, type id)
const typeByIv = new Map(types.map(t => [t.iv.map(x => x % 12).sort((a, b) => a - b).join(','), t.id]));
function link(sym) {
  const c = T.Chord.get(sym);
  const tid = typeByIv.get(c.intervals.map(semis).map(x => x % 12).sort((a, b) => a - b).join(','));
  if (tid === undefined) throw new Error('no type for ' + sym);
  return { s: sym.replace(/^([A-G][#b]?)M$/, '$1'), r: T.Note.chroma(c.tonic), t: tid };
}
const KEY_MAJ = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
const KEY_MIN = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B'];
const keys = { major: [], minor: [] };
for (const k of KEY_MAJ) {
  const K = T.Key.majorKey(k);
  keys.major.push({ tonic: k, rel: K.minorRelative, triads: K.triads.map(link), sev: K.chords.map(link), rn: ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°'] });
}
for (const k of KEY_MIN) {
  const K = T.Key.minorKey(k).natural;
  keys.minor.push({ tonic: k, rel: T.Key.minorKey(k).relativeMajor, triads: K.triads.map(link), sev: K.chords.map(link), rn: ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII'] });
}

// curated voicings from chords-db, validated against the formula
const std = TUNINGS.std;
const voicings = {}; const report = { kept: 0, dropped: 0, droppedList: [] };
for (const [ri, r] of ROOTS.entries()) {
  voicings[r] = {};
  for (const [id, , suf] of TYPES) {
    if (!suf) continue;
    const entry = (db.chords[DBKEY[r] || r] || []).find(c => c.suffix === suf);
    if (!entry) continue;
    const t = types.find(x => x.id === id);
    const pcs = new Set(t.iv.map(i => (ri + i) % 12));
    const extra = new Set((ALLOWED_EXTRA[id] || []).map(i => (ri + i) % 12));
    const req = [...required(t.iv)].map(i => (ri + i) % 12);
    const list = [];
    for (const p of entry.positions) {
      const frets = p.frets.map(f => (f <= 0 ? f : f + p.baseFret - 1));
      const got = frets.map((f, s) => (f >= 0 ? (std[s] + f) % 12 : -1)).filter(x => x >= 0);
      const bad = got.filter(x => !pcs.has(x) && !extra.has(x));
      const miss = req.filter(x => !got.includes(x));
      if (bad.length || miss.length) { report.dropped++; report.droppedList.push(r + id + ' ' + frets.join(' ') + (bad.length ? ' extra' : ' missing')); continue; }
      report.kept++;
      list.push([frets, p.fingers, (p.barres || [])[0] ? p.barres[0] + p.baseFret - 1 : 0]);
    }
    if (list.length) voicings[r][id] = list;
  }
}

fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'theory.json'), JSON.stringify({ roots: ROOTS, types, notes, scales, scaleNotes, keys }));
fs.writeFileSync(path.join(OUT, 'voicings.json'), JSON.stringify(voicings));
console.log('types', types.length, 'scales', scales.length, 'voicings kept', report.kept, 'dropped', report.dropped);
fs.writeFileSync(path.join(OUT, '..', 'tools', 'dropped-voicings.txt'), report.droppedList.join('\n'));
