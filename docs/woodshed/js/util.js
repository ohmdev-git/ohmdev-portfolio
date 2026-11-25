// Small shared helpers. No app state lives here.

export const $ = id => document.getElementById(id);
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
export const isId = v => typeof v === 'string' && /^[\w-]{11}$/.test(v);
export const isFiniteNum = v => typeof v === 'number' && isFinite(v);

export const store = {
  get(k, d) { try { const v = JSON.parse(localStorage.getItem(k)); return v == null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch (e) { return false; } },
  del(k) { try { localStorage.removeItem(k); } catch (e) {} }
};

export function fmt(t) {
  if (t == null || !isFinite(t)) return '–';
  const T = Math.floor(Math.max(0, t) * 10 + 1e-6);
  const h = Math.floor(T / 36000), m = Math.floor(T % 36000 / 600), s = Math.floor(T % 600 / 10), f = T % 10;
  return (h ? h + ':' + String(m).padStart(2, '0') : m) + ':' + String(s).padStart(2, '0') + '.' + f;
}
// precise format for zoomed work: 1:02.35
export function fmt2(t) {
  if (t == null || !isFinite(t)) return '–';
  const T = Math.round(Math.max(0, t) * 100);
  const m = Math.floor(T / 6000), s = Math.floor(T % 6000 / 100), f = T % 100;
  return m + ':' + String(s).padStart(2, '0') + '.' + String(f).padStart(2, '0');
}

export function parseTime(v) {
  if (!v) return 0;
  if (/^\d+(\.\d+)?$/.test(v)) return +v;
  const m = String(v).match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$/);
  return m ? (+m[1] || 0) * 3600 + (+m[2] || 0) * 60 + (+m[3] || 0) : 0;
}

// Returns {id, t} for a video, {list} for a playlist-only link, or null.
export function parseYT(input, allowBareId = true) {
  const raw = String(input || '').trim();
  if (allowBareId && isId(raw)) return { id: raw, t: 0 };
  let u;
  try { u = new URL(/^https?:\/\//i.test(raw) ? raw : 'https://' + raw); } catch (e) { return null; }
  const host = u.hostname.toLowerCase().replace(/^(www|m|music)\./, '');
  let id = null;
  if (host === 'youtu.be') id = u.pathname.split('/')[1];
  else if (host === 'youtube.com' || host === 'youtube-nocookie.com') {
    id = u.searchParams.get('v');
    if (!id) { const m = u.pathname.match(/^\/(?:shorts|embed|live|v)\/([\w-]{11})/); if (m) id = m[1]; }
    if (!id) { const list = u.searchParams.get('list'); if (list && /^[\w-]{10,64}$/.test(list)) return { list }; }
  }
  if (!isId(id)) return null;
  return { id, t: parseTime(u.searchParams.get('t') || u.searchParams.get('start')) };
}

// Many links pasted at once (spaces, commas or new lines between them)
export function parseMany(text) {
  const tokens = String(text || '').split(/[\s,]+/).filter(Boolean);
  const single = tokens.length === 1;
  const videos = [], lists = []; let bad = 0;
  for (const tok of tokens) {
    const p = parseYT(tok, single);
    if (!p) bad++;
    else if (p.list) { if (!lists.includes(p.list)) lists.push(p.list); }
    else if (!videos.some(v => v.id === p.id)) videos.push(p);
  }
  return { videos, lists, bad };
}

let toastT;
export function toast(msg) {
  const el = $('toast'); if (!el) return;
  el.textContent = msg; el.classList.add('show');
  clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove('show'), 2600);
}

export function ago(ts, tr) {
  const d = (Date.now() - ts) / 1000;
  if (!isFinite(d) || d < 60) return tr('ago.now');
  if (d < 3600) return tr('ago.m', { n: Math.floor(d / 60) });
  if (d < 86400) return tr('ago.h', { n: Math.floor(d / 3600) });
  if (d < 86400 * 30) return tr('ago.d', { n: Math.floor(d / 86400) });
  return new Date(ts).toLocaleDateString();
}

// flush() runs the call now, but only if one is actually waiting
export function debounce(fn, ms) {
  let t, waiting = false, args = [];
  const f = (...a) => { args = a; waiting = true; clearTimeout(t); t = setTimeout(() => { waiting = false; fn(...args); }, ms); };
  f.flush = () => { if (!waiting) return; waiting = false; clearTimeout(t); fn(...args); };
  return f;
}

export const touchOnly = () => window.matchMedia && window.matchMedia('(hover: none)').matches;
