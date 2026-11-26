// Speed trainer: start slow, add a few % every N loop reps until the goal speed.
import { $, store, clamp, toast } from './util.js?v=202611261339';
import { t } from './i18n.js?v=202611261339';

let app, S;
const T = { on: false, from: 60, to: 100, step: 5, every: 2, since: 0 };
const FROM = Array.from({ length: 16 }, (_, i) => 25 + i * 5);      // 25..100
const TO = Array.from({ length: 25 }, (_, i) => 50 + i * 5);        // 50..170
const STEP = [1, 2, 5, 10];
const EVERY = [1, 2, 3, 4, 5, 6, 8, 10];
let bumping = false;

const save = () => store.set('woodshed:trainer', { on: T.on, from: T.from, to: T.to, step: T.step, every: T.every });
const cur = () => Math.round(S.rate * 100);

function fill(sel, vals, fmt, v) {
  sel.innerHTML = vals.map(x => `<option value="${x}"${x === v ? ' selected' : ''}>${fmt(x)}</option>`).join('');
}
function renderOptions() {
  fill($('trFrom'), FROM, x => x + '%', T.from);
  fill($('trTo'), TO, x => x + '%', T.to);
  fill($('trStep'), STEP, x => '+' + x + '%', T.step);
  fill($('trEvery'), EVERY, x => t('reps', { n: x }), T.every);
}
function render() {
  $('trOn').setAttribute('aria-checked', T.on);
  $('trOn').classList.toggle('on', T.on);
  $('trOn').closest('.sub').classList.toggle('active', T.on);
  const c = cur();
  let msg;
  if (!T.on) msg = t('trainer.idle');
  else if (S.A == null || S.B == null) msg = t('trainer.needLoop');
  else if (c >= T.to) msg = t('trainer.done', { to: T.to });
  else msg = t('trainer.status', { cur: c, to: T.to, step: Math.min(T.step, T.to - c), left: Math.max(1, T.every - T.since) });
  $('trStatus').textContent = msg;
  const span = Math.max(1, T.to - T.from);
  $('trFill').style.width = (T.on ? clamp((c - T.from) / span, 0, 1) * 100 : 0) + '%';
}
function setOn(on) {
  T.on = !!on; T.since = 0; save();
  if (T.on) {
    if (T.to <= T.from) T.to = Math.min(TO[TO.length - 1], T.from + 20);
    renderOptions();
    bumping = true; app.setRate(T.from); bumping = false;
  }
  render();
}

export function init(a) {
  app = a; S = a.S;
  const g = store.get('woodshed:trainer', {});
  if (g && typeof g === 'object') {
    if (FROM.includes(g.from)) T.from = g.from;
    if (TO.includes(g.to)) T.to = g.to;
    if (STEP.includes(g.step)) T.step = g.step;
    if (EVERY.includes(g.every)) T.every = g.every;
    T.on = false;   // never start silently changing speed on page load
  }
  renderOptions();
  $('trOn').onclick = () => setOn(!T.on);
  const onSel = () => {
    T.from = +$('trFrom').value; T.to = +$('trTo').value; T.step = +$('trStep').value; T.every = +$('trEvery').value;
    if (T.to <= T.from) { T.to = Math.min(TO[TO.length - 1], T.from + 5 * Math.ceil((T.from - T.to + 5) / 5)); renderOptions(); }
    T.since = 0; save();
    if (T.on && cur() < T.from) { bumping = true; app.setRate(T.from); bumping = false; }
    render();
  };
  // blur after picking so keyboard shortcuts work again straight away
  ['trFrom', 'trTo', 'trStep', 'trEvery'].forEach(id => $(id).addEventListener('change', e => { onSel(); e.target.blur(); }));

  app.on('rep', () => {
    if (!T.on) return;
    T.since++;
    const c = cur();
    if (c < T.to && T.since >= T.every) {
      const n = Math.min(T.to, c + T.step);
      T.since = 0;
      bumping = true; app.setRate(n); bumping = false;
      toast(n >= T.to ? t('trainer.reached', { p: n }) : t('trainer.up', { p: n }));
    }
    render();
  });
  app.on('rate', () => { if (!bumping) T.since = 0; render(); });
  app.on('ab', () => { T.since = 0; render(); });
  app.on('lang', () => { renderOptions(); render(); });
  render();
}
