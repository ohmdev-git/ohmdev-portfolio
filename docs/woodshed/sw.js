// Woodshed service worker: makes the site installable and lets the practice tools open offline.
// Only this site's own files are cached — YouTube, titles and thumbnails always go to the network.
const VERSION = '202611261520';
const CACHE = 'woodshed-' + VERSION;
const FILES = ["./", "data/theory.json?v=202611261520", "data/voicings.json?v=202611261520", "icons/apple-touch-icon.png", "icons/icon-192.png", "icons/icon-512.png", "icons/maskable-512.png", "icons/apple-touch-icon.png?v=202611261520", "css/app.css?v=202611261520", "js/backup.js?v=202611261520", "js/fretboard.js?v=202611261520", "js/i18n.js?v=202611261520", "js/library.js?v=202611261520", "js/main.js?v=202611261520", "js/markers.js?v=202611261520", "js/metronome.js?v=202611261520", "js/pedal.js?v=202611261520", "js/pluck.js?v=202611261520", "js/practice.js?v=202611261520", "js/sections.js?v=202611261520", "js/theory.js?v=202611261520", "js/trainer.js?v=202611261520", "js/util.js?v=202611261520", "js/voicings.js?v=202611261520", "manifest.webmanifest?v=202611261520"];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k.startsWith('woodshed-') && k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== location.origin || !url.pathname.startsWith(new URL('./', self.registration.scope).pathname)) return;
  if (req.mode === 'navigate') {
    // page: always try the network first so a new version shows up straight away
    e.respondWith(fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put('./', copy)); }
      return res;
    }).catch(() => caches.match('./', { cacheName: CACHE }).then(r => r || caches.match('./'))));
    return;
  }
  // versioned files never change: cache first
  e.respondWith(caches.match(req).then(hit => hit || fetch(req).then(res => {
    if (res.ok && url.search.includes('v=')) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
    return res;
  })));
});
