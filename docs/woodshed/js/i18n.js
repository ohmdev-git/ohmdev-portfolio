// Language: English / Thai. Static text uses data-i18n* attributes; code uses t(key, vars).
const pl = (n, one, many) => n + ' ' + (n === 1 ? one : many);

const D = {
  en: {
    // header & page
    'page.title': 'Woodshed — Learn any song by ear',
    'brand.tag': 'Learn any song by ear',
    'url.ph': 'Paste a YouTube link…', 'url.aria': 'YouTube link', 'load': 'Load',
    'fileWarn': 'You opened this page as a local file. YouTube blocks playback from local files. Host it (e.g. GitHub Pages) or run a local web server.',
    'lang.aria': 'Language',
    // intro overlay
    'intro.title': 'Hit the shed.',
    'intro.text': "Paste a YouTube link above, mark the part you want to learn, slow it down, and loop it until it's in your hands.",
    'step1': 'Paste link', 'step2': 'Set A–B', 'step3': 'Slow it down',
    // transport
    'noSong': 'No song loaded', 'ytVideo': 'YouTube video',
    'back5': 'Back 5 seconds', 'back2': 'Back 2 seconds', 'fwd2': 'Forward 2 seconds', 'fwd5': 'Forward 5 seconds',
    'back5.t': 'Back 5s', 'back2.t': 'Back 2s (←)', 'fwd2.t': 'Forward 2s (→)', 'fwd5.t': 'Forward 5s',
    'play': 'Play', 'pause': 'Pause', 'play.t': 'Play / Pause (Space)',
    'loopOff': 'Loop off', 'loopingLen': 'Looping {len}', 'reps': v => pl(v.n, 'rep', 'reps'),
    'timeline.aria': 'Timeline: click to seek, drag to select a loop', 'markers.aria': 'Markers',
    'timeline.hint': 'Click to jump · drag to select a loop · drag the A / B tabs to fine-tune',
    'marker': 'Marker', 'marker.t': 'Add a marker at the playhead (M)',
    'zoom': 'Zoom', 'zoom.t': 'Zoom the timeline to the loop (Z)',
    // library
    'lib.aria': 'Playlist and history', 'tab.pl': 'Playlist', 'tab.rc': 'Recent',
    'prevSong': 'Previous song', 'nextSong': 'Next song', 'prevSong.t': 'Previous song (P)', 'nextSong.t': 'Next song (N)',
    'pl.ph': 'Paste song or playlist links (many at once)', 'pl.aria': 'Add songs to playlist', 'add': 'Add', 'importing': 'Importing…',
    'pl.addCur': '+ Current song', 'pl.clearDone': 'Remove learned', 'clear': 'Clear', 'clearAll?': 'Clear all?',
    'pl.empty': 'Planning a practice session? Paste all the links here — you can copy several at once — then tap a song to start.',
    'rc.empty': 'Songs you open will show up here.', 'rc.clear': 'Clear history', 'rc.clear?': 'Clear history?',
    'pl.count': v => `${pl(v.n, 'song', 'songs')} · ${v.d} learned`,
    'savedParts': v => pl(v.n, 'saved part', 'saved parts'),
    'blockedItem': 'Can’t be played outside YouTube',
    'loadingTitle': 'Loading title…', 'videoId': 'YouTube video · {id}',
    'drag': 'Drag to reorder', 'markLearned': 'Mark as learned', 'unmarkLearned': 'Mark as not learned', 'learned': 'Learned',
    'removeFromPl': 'Remove from playlist', 'remove': 'Remove',
    'inPl': 'In playlist', 'addPl': '+ Playlist', 'inPl.t': 'Already in playlist', 'addPl.t': 'Add to playlist',
    'added': v => pl(v.n, 'song added', 'songs added'), 'dups': v => `${v.n} already in playlist`, 'nothingAdded': 'Nothing added',
    'notRecognised': v => `${v.n} not recognised`, 'plReadFail': 'couldn’t read the playlist (private or empty?)', 'plsReadFail': v => `${v.n} playlists couldn’t be read`,
    'noLinks': 'No YouTube links found in that text', 'pasteLinks': 'Paste one or more YouTube links',
    'plEmptyToast': 'Your playlist is empty', 'lastSong': 'That was the last song', 'firstSong': 'That’s the first song',
    'plCleared': 'Playlist cleared', 'noLearned': 'No learned songs to remove', 'learnedRemoved': v => `${v.n} learned removed`,
    'ago.now': 'just now', 'ago.m': '{n}m ago', 'ago.h': '{n}h ago', 'ago.d': '{n}d ago',
    // speed
    'speed': 'Speed', 'pitchSame': 'Pitch stays the same', 'slower5': 'Slower 5%', 'faster5': 'Faster 5%', 'speed.aria': 'Playback speed',
    'origSpeed': 'original speed', 'xSlower': '{x}× slower', 'xFaster': '{x}× faster',
    'ytOnly': 'YouTube only allows {p}% here',
    'trainer': 'Speed trainer', 'trainer.from': 'Start', 'trainer.to': 'Goal', 'trainer.step': 'Step', 'trainer.every': 'Every',
    'trainer.repsUnit': 'reps', 'trainer.on': 'On', 'trainer.off': 'Off',
    'trainer.status': v => `${v.cur}% → ${v.to}% · next +${v.step}% in ${pl(v.left, 'rep', 'reps')}`,
    'trainer.done': 'Goal reached — {to}%',
    'trainer.idle': 'Speeds up a little every few loop reps',
    'trainer.needLoop': 'Set an A–B loop first — the trainer counts loop reps',
    'trainer.up': 'Speed up → {p}%', 'trainer.reached': 'Goal speed reached: {p}%',
    // loop
    'loop': 'Loop', 'looping': 'Looping', 'notSet': 'not set', 'set': 'Set',
    'aEarlier': 'Move A earlier', 'aLater': 'Move A later', 'bEarlier': 'Move B earlier', 'bLater': 'Move B later',
    'setA.t': 'Set A at playhead (A)', 'setB.t': 'Set B at playhead (B)',
    'fineStep': 'Fine-tune step', 'earlierBy': 'Earlier {s}s', 'laterBy': 'Later {s}s',
    'jumpA': 'Jump to A', 'jumpA.t': 'Jump to A (Q)', 'gap': 'Pause between reps', 'none': 'None',
    'countIn': 'Count-in', 'countIn.bars': v => pl(v.n, 'bar', 'bars'), 'off': 'Off',
    'needBpm': 'Set the song’s BPM in the metronome first (tap it)',
    'loopOffOutside': 'Loop off — you jumped outside it', 'bCleared': 'B was before A — B cleared', 'bAfterA': 'B has to come after A',
    'setABFirst': 'Set A and B first',
    // sections
    'sections': 'Saved sections', 'copyLink': 'Copy loop link', 'sec.ph': 'Name it — e.g. Solo, bar 1–4', 'sec.aria': 'Section name', 'save': 'Save',
    'sec.empty': 'Set A and B, then save the part so you can come back to it. Tap the pencil to keep chords or tab with it.',
    'sectionN': 'Section {n}', 'saved': 'Saved “{n}”', 'notesFor': 'Notes for {n}', 'notes.t': 'Chords / tab notes', 'deleteX': 'Delete {n}', 'delete?': 'Delete?',
    'notes.ph': 'Chords, tab, fingering…  e.g.  Am  F  C  G',
    'loadFirst': 'Load a video first', 'linkCopied': 'Link copied — send it to a friend', 'copied': 'Link copied', 'copyFail': 'Copy failed — link is in the address bar',
    // markers
    'markers': 'Markers', 'mk.add': '+ Add at playhead', 'mk.add.t': 'Add at playhead (M)',
    'mk.empty': 'Mark where the intro, verse, chorus and solo start. Tap the loop icon to practise one part.',
    'markerN': 'Marker {n}', 'mk.added': 'Marker added at {t}', 'mk.exists': 'There’s already a marker here', 'mk.many': 'That’s a lot of markers — remove some first',
    'loadSongFirst': 'Load a song first', 'mk.close': 'This marker is too close to the next one', 'waitLoad': 'Wait for the video to load',
    'loopingX': 'Looping “{n}”', 'part': 'part', 'jumpHere': 'Jump here', 'mk.loopNext': 'Loop from here to the next marker', 'mk.loopEnd': 'Loop from here to the end',
    'mk.loopAria': 'Loop this part', 'mk.del': 'Delete marker', 'del': 'Delete', 'mk.name': 'Marker name',
    'mk.names': ['Intro', 'Verse', 'Pre-chorus', 'Chorus', 'Bridge', 'Solo', 'Riff', 'Interlude', 'Outro'],
    // metronome
    'metronome': 'Metronome', 'bpm': 'BPM', 'bpmDown': 'Slower 1 BPM', 'bpmUp': 'Faster 1 BPM', 'bpm.aria': 'Tempo in BPM',
    'met.start': 'Start click', 'met.stop': 'Stop click', 'tap': 'Tap tempo', 'tap.again': 'Keep tapping…',
    'beats': 'Beats per bar', 'met.follow': 'Match video speed', 'met.realign': 'Beat 1 at loop start each rep', 'volume': 'Volume',
    'met.eff': 'Clicking at {b} BPM ({p}% speed)', 'met.audioFail': 'Your browser blocked sound — tap the page and try again',
    'tap.set': 'Tempo set: {b} BPM',
    // errors
    'err.title': 'Can’t play this one', 'err.2': 'That link doesn’t look like a valid video.', 'err.5': 'This video can’t be played in the browser player.',
    'err.100': 'Video not found — it may be private or removed.',
    'err.150': 'YouTube won’t play this video here — usually because the uploader turned off playback outside YouTube. Try another upload of the song (covers, lyric videos and “audio” uploads usually work). If every video fails, turn off your VPN or ad-blocker and reload.',
    'err.153': 'YouTube refused to play here. If you opened this page as a local file, host it on a website instead.',
    'err.other': 'YouTube couldn’t play this video (error {c}).', 'err.code': 'YouTube error {c}',
    'findPlayable': 'Find a playable version', 'openYT': 'Open on YouTube',
    'noYT': "Couldn't reach YouTube. Check your connection.",
    'notYTLink': 'That doesn’t look like a YouTube link', 'isPlaylist': 'That’s a playlist — paste it into the Playlist box below to add all its songs',
    'tapToGo': 'Tap play to continue',
    // shortcuts & footer
    'shortcuts': 'Shortcuts', 'k.play': 'Play / pause', 'k.seek': 'Back / forward 2s (Shift: 5s)', 'k.a': 'Set A at playhead', 'k.b': 'Set B and start looping',
    'k.l': 'Loop on / off', 'k.q': 'Jump to A', 'k.z': 'Zoom timeline to the loop', 'k.m': 'Add marker', 'k.n': 'Next song', 'k.p': 'previous',
    'k.speed': 'Slower / faster 5%', 'k.num': 'Speed 30% – 90% · 100%', 'k.k': 'Metronome on / off', 'k.t': 'Tap tempo',
    'footer': "Woodshed plays videos through YouTube's official embedded player. Your playlist, history and saved parts stay in this browser only. Some videos (often official music videos) don't allow embedding."
  },
  th: {
    'page.title': 'Woodshed — แกะเพลงจาก YouTube',
    'brand.tag': 'แกะเพลงด้วยหูได้ทุกเพลง',
    'url.ph': 'วางลิงก์ YouTube…', 'url.aria': 'ลิงก์ YouTube', 'load': 'โหลด',
    'fileWarn': 'คุณเปิดหน้านี้จากไฟล์ในเครื่อง ซึ่ง YouTube ไม่ยอมเล่น ให้เปิดผ่านเว็บไซต์ (เช่น GitHub Pages) หรือรันเว็บเซิร์ฟเวอร์ในเครื่องแทน',
    'lang.aria': 'ภาษา',
    'intro.title': 'มาเริ่มซ้อมกัน',
    'intro.text': 'วางลิงก์ YouTube ด้านบน เลือกท่อนที่อยากแกะ ลดความเร็ว แล้ววนซ้ำจนเล่นได้',
    'step1': 'วางลิงก์', 'step2': 'ตั้งจุด A–B', 'step3': 'ลดความเร็ว',
    'noSong': 'ยังไม่ได้โหลดเพลง', 'ytVideo': 'วิดีโอ YouTube',
    'back5': 'ถอยหลัง 5 วินาที', 'back2': 'ถอยหลัง 2 วินาที', 'fwd2': 'เดินหน้า 2 วินาที', 'fwd5': 'เดินหน้า 5 วินาที',
    'back5.t': 'ถอย 5 วิ', 'back2.t': 'ถอย 2 วิ (←)', 'fwd2.t': 'ไปหน้า 2 วิ (→)', 'fwd5.t': 'ไปหน้า 5 วิ',
    'play': 'เล่น', 'pause': 'หยุด', 'play.t': 'เล่น / หยุด (Space)',
    'loopOff': 'ปิดลูป', 'loopingLen': 'วนลูป {len}', 'reps': v => `${v.n} รอบ`,
    'timeline.aria': 'แถบเวลา: คลิกเพื่อข้าม ลากเพื่อเลือกช่วงวนซ้ำ', 'markers.aria': 'มาร์กเกอร์',
    'timeline.hint': 'คลิกเพื่อข้ามไป · ลากเพื่อเลือกช่วงวน · ลากแท็บ A / B เพื่อปรับละเอียด',
    'marker': 'มาร์กเกอร์', 'marker.t': 'ปักมาร์กเกอร์ตรงจุดที่เล่นอยู่ (M)',
    'zoom': 'ซูม', 'zoom.t': 'ซูมแถบเวลาให้เห็นเฉพาะช่วงลูป (Z)',
    'lib.aria': 'เพลย์ลิสต์และประวัติ', 'tab.pl': 'เพลย์ลิสต์', 'tab.rc': 'ล่าสุด',
    'prevSong': 'เพลงก่อนหน้า', 'nextSong': 'เพลงถัดไป', 'prevSong.t': 'เพลงก่อนหน้า (P)', 'nextSong.t': 'เพลงถัดไป (N)',
    'pl.ph': 'วางลิงก์เพลงหรือเพลย์ลิสต์ (หลายลิงก์พร้อมกันได้)', 'pl.aria': 'เพิ่มเพลงลงเพลย์ลิสต์', 'add': 'เพิ่ม', 'importing': 'กำลังนำเข้า…',
    'pl.addCur': '+ เพลงที่เปิดอยู่', 'pl.clearDone': 'ลบเพลงที่แกะได้แล้ว', 'clear': 'ล้าง', 'clearAll?': 'ล้างทั้งหมด?',
    'pl.empty': 'วางแผนซ้อมวันนี้? วางลิงก์ทุกเพลงที่นี่ (ก๊อปมาหลายลิงก์ทีเดียวได้) แล้วแตะเพลงเพื่อเริ่ม',
    'rc.empty': 'เพลงที่เคยเปิดจะขึ้นที่นี่', 'rc.clear': 'ล้างประวัติ', 'rc.clear?': 'ล้างประวัติ?',
    'pl.count': v => `${v.n} เพลง · แกะได้แล้ว ${v.d}`,
    'savedParts': v => `เซฟไว้ ${v.n} ท่อน`,
    'blockedItem': 'เล่นนอก YouTube ไม่ได้',
    'loadingTitle': 'กำลังโหลดชื่อเพลง…', 'videoId': 'วิดีโอ YouTube · {id}',
    'drag': 'ลากเพื่อเรียงลำดับ', 'markLearned': 'ติ๊กว่าแกะได้แล้ว', 'unmarkLearned': 'ยกเลิกว่าแกะได้แล้ว', 'learned': 'แกะได้แล้ว',
    'removeFromPl': 'เอาออกจากเพลย์ลิสต์', 'remove': 'เอาออก',
    'inPl': 'อยู่ในเพลย์ลิสต์', 'addPl': '+ เพลย์ลิสต์', 'inPl.t': 'มีในเพลย์ลิสต์แล้ว', 'addPl.t': 'เพิ่มลงเพลย์ลิสต์',
    'added': v => `เพิ่ม ${v.n} เพลง`, 'dups': v => `${v.n} เพลงมีอยู่แล้ว`, 'nothingAdded': 'ไม่ได้เพิ่มอะไร',
    'notRecognised': v => `อ่านไม่ออก ${v.n} ลิงก์`, 'plReadFail': 'อ่านเพลย์ลิสต์ไม่ได้ (เป็นส่วนตัวหรือว่างเปล่า?)', 'plsReadFail': v => `อ่านไม่ได้ ${v.n} เพลย์ลิสต์`,
    'noLinks': 'ไม่เจอลิงก์ YouTube ในข้อความนี้', 'pasteLinks': 'วางลิงก์ YouTube อย่างน้อยหนึ่งลิงก์',
    'plEmptyToast': 'เพลย์ลิสต์ยังว่าง', 'lastSong': 'นี่คือเพลงสุดท้ายแล้ว', 'firstSong': 'นี่คือเพลงแรกแล้ว',
    'plCleared': 'ล้างเพลย์ลิสต์แล้ว', 'noLearned': 'ยังไม่มีเพลงที่ติ๊กว่าแกะได้', 'learnedRemoved': v => `ลบเพลงที่แกะได้แล้ว ${v.n} เพลง`,
    'ago.now': 'เมื่อกี้', 'ago.m': '{n} นาทีที่แล้ว', 'ago.h': '{n} ชม.ที่แล้ว', 'ago.d': '{n} วันที่แล้ว',
    'speed': 'ความเร็ว', 'pitchSame': 'คีย์เพลงไม่เพี้ยน', 'slower5': 'ช้าลง 5%', 'faster5': 'เร็วขึ้น 5%', 'speed.aria': 'ความเร็วในการเล่น',
    'origSpeed': 'ความเร็วจริง', 'xSlower': 'ช้าลง {x} เท่า', 'xFaster': 'เร็วขึ้น {x} เท่า',
    'ytOnly': 'YouTube ใช้ได้แค่ {p}% ตรงนี้',
    'trainer': 'ไต่สปีดอัตโนมัติ', 'trainer.from': 'เริ่ม', 'trainer.to': 'เป้าหมาย', 'trainer.step': 'เพิ่มทีละ', 'trainer.every': 'ทุก',
    'trainer.repsUnit': 'รอบ', 'trainer.on': 'เปิด', 'trainer.off': 'ปิด',
    'trainer.status': v => `${v.cur}% → ${v.to}% · อีก ${v.left} รอบเพิ่ม +${v.step}%`,
    'trainer.done': 'ถึงเป้าหมายแล้ว — {to}%',
    'trainer.idle': 'เร่งความเร็วขึ้นทีละนิดทุกๆ ไม่กี่รอบที่วนลูป',
    'trainer.needLoop': 'ตั้งลูป A–B ก่อน ระบบนับรอบจากลูป',
    'trainer.up': 'เร่งเป็น {p}%', 'trainer.reached': 'ถึงความเร็วเป้าหมายแล้ว: {p}%',
    'loop': 'วนลูป', 'looping': 'กำลังวน', 'notSet': 'ยังไม่ตั้ง', 'set': 'ตั้ง',
    'aEarlier': 'ขยับ A ให้เร็วขึ้น', 'aLater': 'ขยับ A ให้ช้าลง', 'bEarlier': 'ขยับ B ให้เร็วขึ้น', 'bLater': 'ขยับ B ให้ช้าลง',
    'setA.t': 'ตั้ง A ตรงจุดที่เล่นอยู่ (A)', 'setB.t': 'ตั้ง B ตรงจุดที่เล่นอยู่ (B)',
    'fineStep': 'ขยับทีละ', 'earlierBy': 'เร็วขึ้น {s} วิ', 'laterBy': 'ช้าลง {s} วิ',
    'jumpA': 'กลับไปจุด A', 'jumpA.t': 'กลับไปจุด A (Q)', 'gap': 'พักก่อนวนรอบใหม่', 'none': 'ไม่พัก',
    'countIn': 'นับจังหวะก่อนเข้า', 'countIn.bars': v => `${v.n} ห้อง`, 'off': 'ปิด',
    'needBpm': 'ตั้ง BPM ของเพลงในเมโทรนอมก่อน (กดเคาะตามเพลงได้)',
    'loopOffOutside': 'ปิดลูปแล้ว เพราะข้ามไปนอกช่วง', 'bCleared': 'B อยู่ก่อน A เลยล้าง B ออก', 'bAfterA': 'B ต้องอยู่หลัง A',
    'setABFirst': 'ตั้งจุด A และ B ก่อน',
    'sections': 'ท่อนที่เซฟไว้', 'copyLink': 'ก๊อปลิงก์ลูปนี้', 'sec.ph': 'ตั้งชื่อ เช่น โซโล่ ห้อง 1–4', 'sec.aria': 'ชื่อท่อน', 'save': 'เซฟ',
    'sec.empty': 'ตั้ง A และ B แล้วเซฟท่อนไว้กลับมาซ้อมทีหลัง แตะรูปดินสอเพื่อจดคอร์ดหรือแท็บไว้คู่กัน',
    'sectionN': 'ท่อน {n}', 'saved': 'เซฟ “{n}” แล้ว', 'notesFor': 'โน้ตของ {n}', 'notes.t': 'จดคอร์ด / แท็บ', 'deleteX': 'ลบ {n}', 'delete?': 'ลบเลย?',
    'notes.ph': 'คอร์ด แท็บ การวางนิ้ว…  เช่น  Am  F  C  G',
    'loadFirst': 'โหลดวิดีโอก่อน', 'linkCopied': 'ก๊อปลิงก์แล้ว ส่งให้เพื่อนได้เลย', 'copied': 'ก๊อปลิงก์แล้ว', 'copyFail': 'ก๊อปไม่ได้ ลิงก์อยู่ในช่องที่อยู่เว็บแล้ว',
    'markers': 'มาร์กเกอร์', 'mk.add': '+ ปักตรงจุดที่เล่นอยู่', 'mk.add.t': 'ปักตรงจุดที่เล่นอยู่ (M)',
    'mk.empty': 'ปักจุดเริ่มของอินโทร เวิร์ส คอรัส และโซโล่ แล้วแตะไอคอนวนเพื่อซ้อมทีละท่อน',
    'markerN': 'มาร์กเกอร์ {n}', 'mk.added': 'ปักมาร์กเกอร์ที่ {t}', 'mk.exists': 'มีมาร์กเกอร์ตรงนี้แล้ว', 'mk.many': 'มาร์กเกอร์เยอะมากแล้ว ลบบางอันก่อน',
    'loadSongFirst': 'โหลดเพลงก่อน', 'mk.close': 'มาร์กเกอร์นี้ใกล้กับอันถัดไปเกินไป', 'waitLoad': 'รอวิดีโอโหลดก่อน',
    'loopingX': 'กำลังวน “{n}”', 'part': 'ท่อนนี้', 'jumpHere': 'ไปที่จุดนี้', 'mk.loopNext': 'วนจากตรงนี้ถึงมาร์กเกอร์ถัดไป', 'mk.loopEnd': 'วนจากตรงนี้ถึงท้ายเพลง',
    'mk.loopAria': 'วนท่อนนี้', 'mk.del': 'ลบมาร์กเกอร์', 'del': 'ลบ', 'mk.name': 'ชื่อมาร์กเกอร์',
    'mk.names': ['Intro', 'Verse', 'Pre-chorus', 'Chorus', 'Bridge', 'Solo', 'Riff', 'Interlude', 'Outro', 'อินโทร', 'ท่อนเวิร์ส', 'ท่อนฮุก', 'โซโล่', 'ท่อนจบ'],
    'metronome': 'เมโทรนอม', 'bpm': 'BPM', 'bpmDown': 'ช้าลง 1 BPM', 'bpmUp': 'เร็วขึ้น 1 BPM', 'bpm.aria': 'จังหวะ (BPM)',
    'met.start': 'เปิดเสียงเคาะ', 'met.stop': 'ปิดเสียงเคาะ', 'tap': 'เคาะหา BPM', 'tap.again': 'เคาะต่ออีก…',
    'beats': 'จังหวะต่อห้อง', 'met.follow': 'ปรับตามความเร็ววิดีโอ', 'met.realign': 'เริ่มจังหวะ 1 ใหม่ทุกครั้งที่วนกลับ A', 'volume': 'ความดัง',
    'met.eff': 'เคาะจริง {b} BPM (ที่ความเร็ว {p}%)', 'met.audioFail': 'เบราว์เซอร์บล็อกเสียง แตะหน้าจอแล้วลองใหม่',
    'tap.set': 'ตั้งจังหวะแล้ว: {b} BPM',
    'err.title': 'เล่นคลิปนี้ไม่ได้', 'err.2': 'ลิงก์นี้ดูไม่ใช่วิดีโอที่ถูกต้อง', 'err.5': 'วิดีโอนี้เล่นในเบราว์เซอร์ไม่ได้',
    'err.100': 'ไม่เจอวิดีโอ อาจเป็นแบบส่วนตัวหรือถูกลบไปแล้ว',
    'err.150': 'YouTube ไม่ยอมให้เล่นคลิปนี้ที่นี่ ส่วนใหญ่เพราะเจ้าของคลิปปิดการเล่นนอก YouTube ลองหาคลิปอื่นของเพลงเดียวกัน (คลิป cover, lyrics หรือ audio มักเล่นได้) ถ้าทุกคลิปเล่นไม่ได้ ลองปิด VPN หรือตัวบล็อกโฆษณาแล้วรีเฟรช',
    'err.153': 'YouTube ไม่ยอมเล่นที่นี่ ถ้าเปิดหน้านี้จากไฟล์ในเครื่อง ให้เปิดผ่านเว็บไซต์แทน',
    'err.other': 'YouTube เล่นวิดีโอนี้ไม่ได้ (error {c})', 'err.code': 'YouTube error {c}',
    'findPlayable': 'หาคลิปที่เล่นได้', 'openYT': 'เปิดใน YouTube',
    'noYT': 'เชื่อมต่อ YouTube ไม่ได้ ลองเช็คอินเทอร์เน็ต',
    'notYTLink': 'นี่ดูไม่ใช่ลิงก์ YouTube', 'isPlaylist': 'นี่คือลิงก์เพลย์ลิสต์ วางในช่องเพลย์ลิสต์ด้านล่างเพื่อเพิ่มทุกเพลง',
    'tapToGo': 'แตะปุ่มเล่นเพื่อไปต่อ',
    'shortcuts': 'คีย์ลัด', 'k.play': 'เล่น / หยุด', 'k.seek': 'ถอย / ไปหน้า 2 วิ (Shift: 5 วิ)', 'k.a': 'ตั้ง A ตรงจุดที่เล่นอยู่', 'k.b': 'ตั้ง B แล้วเริ่มวนลูป',
    'k.l': 'เปิด / ปิดลูป', 'k.q': 'กลับไปจุด A', 'k.z': 'ซูมแถบเวลาไปที่ลูป', 'k.m': 'ปักมาร์กเกอร์', 'k.n': 'เพลงถัดไป', 'k.p': 'เพลงก่อนหน้า',
    'k.speed': 'ช้าลง / เร็วขึ้น 5%', 'k.num': 'ความเร็ว 30% – 90% · 100%', 'k.k': 'เปิด / ปิดเมโทรนอม', 'k.t': 'เคาะหา BPM',
    'footer': 'Woodshed เล่นวิดีโอผ่านตัวเล่นทางการของ YouTube เพลย์ลิสต์ ประวัติ และท่อนที่เซฟไว้ เก็บอยู่ในเบราว์เซอร์นี้เท่านั้น คลิปบางอัน (มักเป็น MV ทางการ) ไม่อนุญาตให้เล่นนอก YouTube'
  }
};

const KEY = 'woodshed:lang';
function detect() {
  try { const v = JSON.parse(localStorage.getItem(KEY)); if (v === 'en' || v === 'th') return v; } catch (e) {}
  const n = (navigator.languages && navigator.languages[0]) || navigator.language || '';
  return /^th\b/i.test(n) ? 'th' : 'en';
}
let lang = detect();
export const getLang = () => lang;

export function t(k, v = {}) {
  let e = D[lang][k];
  if (e === undefined) e = D.en[k];
  if (e === undefined) { console.warn('i18n missing', k); return k; }
  if (typeof e === 'function') return e(v);
  if (Array.isArray(e)) return e;
  return e.replace(/\{(\w+)\}/g, (_, x) => (v[x] !== undefined ? v[x] : ''));
}

export function applyStatic(root = document) {
  root.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  root.querySelectorAll('[data-i18n-ph]').forEach(el => { el.placeholder = t(el.dataset.i18nPh); });
  root.querySelectorAll('[data-i18n-title]').forEach(el => { el.title = t(el.dataset.i18nTitle); });
  root.querySelectorAll('[data-i18n-aria]').forEach(el => { el.setAttribute('aria-label', t(el.dataset.i18nAria)); });
  document.documentElement.lang = lang;
  document.title = t('page.title');
}

export function setLang(l) {
  if (l !== 'en' && l !== 'th') return;
  lang = l;
  try { localStorage.setItem(KEY, JSON.stringify(l)); } catch (e) {}
  applyStatic();
}

// for tests: every key present in both languages
export const _keys = () => ({ en: Object.keys(D.en), th: Object.keys(D.th) });
