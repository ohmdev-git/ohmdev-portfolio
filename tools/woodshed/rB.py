import asyncio, subprocess, sys, time, json
from playwright.async_api import async_playwright
from common import wire, SITE

PORT = 8781
BASE = f'http://127.0.0.1:{PORT}/index.html'
R = []
def ck(c, n, x=''):
    R.append((bool(c), n)); print('PASS' if c else 'FAIL', n, '' if c else x)

MIDI_MOCK = """
(() => {
  const inputs = new Map();
  const inp = { id: 'i1', name: 'FS-1 Pedal', state: 'connected', onmidimessage: null };
  inputs.set('i1', inp);
  const access = { inputs, outputs: new Map(), onstatechange: null };
  window.__midiReq = 0;
  navigator.requestMIDIAccess = () => { window.__midiReq++; return window.__midiDeny ? Promise.reject(new Error('no')) : Promise.resolve(access); };
  window.__midiSend = (...d) => { if (inp.onmidimessage) inp.onmidimessage({ data: new Uint8Array(d) }); };
  window.__midiAdd = () => { inputs.set('i2', { id: 'i2', name: 'Second', state: 'connected', onmidimessage: null }); access.onstatechange && access.onstatechange({}); };
  const q = navigator.permissions && navigator.permissions.query ? navigator.permissions.query.bind(navigator.permissions) : null;
  if (navigator.permissions) navigator.permissions.query = d => d && d.name === 'midi' ? Promise.resolve({ state: window.__midiPerm || 'prompt' }) : q(d);
})();
"""
NO_MIDI = "try { delete Navigator.prototype.requestMIDIAccess; } catch (e) {} navigator.requestMIDIAccess = undefined;"

async def newpage(b, vp=(1400, 950), init=None, **kw):
    ctx = await b.new_context(viewport={'width': vp[0], 'height': vp[1]}, **kw)
    await wire(ctx)
    if init: await ctx.add_init_script(init)
    p = await ctx.new_page(); errs = []
    p.on('pageerror', lambda e: errs.append(str(e)))
    p.on('console', lambda m: errs.append(m.text) if m.type == 'error' and 'net::' not in m.text and 'Failed to load resource' not in m.text else None)
    return ctx, p, errs

async def key(p, k, code=None, repeat=False, **mods):
    await p.evaluate("([k,c,r,m]) => (document.activeElement||document.body).dispatchEvent(new KeyboardEvent('keydown',{key:k,code:c||'',repeat:r,bubbles:true,cancelable:true,...m}))", [k, code, repeat, mods])

async def main():
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1', '-d', SITE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1)
    async with async_playwright() as pw:
        b = await pw.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--no-sandbox', '--autoplay-policy=no-user-gesture-required'])
        ctx, p, errs = await newpage(b, init=MIDI_MOCK)
        M = lambda e: p.evaluate('window.__mock&&window.__mock.' + e)
        T = lambda: p.text_content('#toast')
        ST = lambda: p.evaluate("JSON.parse(localStorage.getItem('woodshed:pedal')||'null')")
        async def blur(): await p.evaluate("document.activeElement && document.activeElement.blur()")

        # ---------- no video yet ----------
        await p.goto(BASE); await p.wait_for_timeout(500)
        ck(await p.is_visible('#pdCard'), 'pedal card visible')
        ck(not await p.is_checked('#pdOn') and await p.is_hidden('#pdBody'), 'off by default, body hidden')
        await key(p, 'ArrowDown', 'ArrowDown'); await p.wait_for_timeout(100)
        ck(await p.is_hidden('#pdHud'), 'pedal off: ArrowDown does nothing')
        await p.click('#pdOn'); await p.wait_for_timeout(100)
        ck(await p.locator('#pdList .pd').count() == 14, '14 actions listed', await p.locator('#pdList .pd').count())
        chips = await p.locator('#pdList [data-a="play"] .pkey').all_text_contents()
        ck(any('Page Down' in c for c in chips) and any('↓' in c for c in chips), 'defaults on play', chips)
        chips = await p.locator('#pdList [data-a="restartA"] .pkey').all_text_contents()
        ck(any('Page Up' in c for c in chips) and any('↑' in c for c in chips), 'defaults on restartA', chips)
        ck((await ST())['on'] is True, 'on saved')
        await blur(); await key(p, 'PageDown', 'PageDown'); await p.wait_for_timeout(150)
        ck(not errs, 'pedal with no video: no errors', errs)
        ck(await p.evaluate("document.activeElement.id") == 'url', 'pedal with no video focuses link box')

        # ---------- with a video ----------
        await p.goto(BASE + '#v=aaaaaaaaaa1&a=10&b=12'); await p.wait_for_timeout(600)
        await blur()
        await key(p, 'PageDown', 'PageDown'); await p.wait_for_timeout(400)
        ck(await M('st') == 1, 'PageDown plays', await M('st'))
        ck(await p.is_visible('#pdHud') and 'Play' in await p.text_content('#pdHud'), 'HUD shows action', await p.text_content('#pdHud'))
        await p.wait_for_timeout(1100)
        ck(await p.is_hidden('#pdHud'), 'HUD hides')
        await key(p, 'ArrowDown', 'ArrowDown'); await p.wait_for_timeout(300)
        ck(await M('st') == 2, 'ArrowDown pauses', await M('st'))
        await key(p, 'ArrowDown', 'ArrowDown', repeat=True); await p.wait_for_timeout(300)
        ck(await M('st') == 2, 'auto-repeat ignored', await M('st'))
        ck('ArrowDown' not in (await p.text_content('#pdStatus') or '') and '↓' in await p.text_content('#pdStatus'), 'last press shown', await p.text_content('#pdStatus'))
        # restart from A
        await p.evaluate("window.__mock && 0"); await key(p, 'ArrowRight', 'ArrowRight'); await p.wait_for_timeout(200)
        await key(p, 'PageUp', 'PageUp'); await p.wait_for_timeout(400)
        t = await M('t')
        ck(await M('st') == 1 and 10 <= t < 10.6, 'PageUp = back to A and play', (await M('st'), t))
        await key(p, ' ', 'Space'); await p.wait_for_timeout(200)
        ck(await M('st') == 2, 'Space still works with pedal on')
        await key(p, 'ArrowLeft', 'ArrowLeft'); await p.wait_for_timeout(100)
        ck(not errs, 'arrow-left seek still fine', errs)
        # restart with count-in waits
        await p.fill('#bpmIn', '120'); await p.press('#bpmIn', 'Enter'); await p.click('#ciSeg [data-ci="1"]'); await blur()
        await p.click('#loopBtn') if await p.get_attribute('#loopBtn', 'aria-pressed') != 'true' else None
        await blur()
        t0 = time.time(); await key(p, 'PageUp', 'PageUp')
        while await M('st') != 1 and time.time() - t0 < 5: await p.wait_for_timeout(20)
        w = time.time() - t0
        ck(1.7 < w < 2.4, 'back-to-A uses count-in when looping', w)
        await key(p, 'PageDown', 'PageDown'); await p.wait_for_timeout(200)
        await p.click('#ciSeg [data-ci="0"]'); await blur()
        # range focused: pedal wins, slider unchanged
        rate0 = await M('rate')
        await p.focus('#rate') if await p.locator('#rate').count() else None
        await key(p, 'ArrowDown', 'ArrowDown'); await p.wait_for_timeout(300)
        ck(await M('rate') == rate0 and await M('st') == 1, 'pedal key on focused slider runs pedal action', (await M('rate'), rate0, await M('st')))
        await blur(); await key(p, 'ArrowDown', 'ArrowDown'); await p.wait_for_timeout(200)
        # typing: never
        await p.focus('#url'); await key(p, 'ArrowDown', 'ArrowDown'); await p.wait_for_timeout(200)
        ck(await M('st') == 2, 'not while typing in link box')
        await p.focus('#secName'); await key(p, 'PageDown', 'PageDown'); await p.wait_for_timeout(200)
        ck(await M('st') == 2, 'not while typing a section name')
        await blur(); await key(p, 'ArrowDown', 'ArrowDown', ctrlKey=True); await p.wait_for_timeout(200)
        ck(await M('st') == 2, 'Ctrl+key not hijacked')

        # ---------- learn ----------
        await p.click('#pdList [data-learn="loop"]'); await p.wait_for_timeout(100)
        ck(await p.locator('#pdList [data-a="loop"].learn').count() == 1, 'learn mode shown')
        await key(p, 'Shift', 'ShiftLeft'); await p.wait_for_timeout(50)
        ck(await p.locator('#pdList [data-a="loop"].learn').count() == 1, 'modifier alone ignored while learning')
        await key(p, 'Enter', 'Enter'); await p.wait_for_timeout(150)
        ck(await p.locator('#pdList [data-a="loop"] .pkey').all_text_contents() == ['Enter×'], 'Enter bound to loop', await p.locator('#pdList [data-a="loop"] .pkey').all_text_contents())
        ck('Enter' in await T(), 'assign toast', await T())
        lp0 = await p.get_attribute('#loopBtn', 'aria-pressed')
        await key(p, 'Enter', 'Enter'); await p.wait_for_timeout(150)
        ck(await p.get_attribute('#loopBtn', 'aria-pressed') != lp0, 'Enter toggles loop', lp0)
        await key(p, 'Enter', 'Enter'); await p.wait_for_timeout(150)
        # move to met
        await p.click('#pdList [data-learn="met"]'); await key(p, 'Enter', 'Enter'); await p.wait_for_timeout(150)
        ck(await p.locator('#pdList [data-a="loop"] .pkey').count() == 0 and await p.locator('#pdList [data-a="met"] .pkey').count() == 1, 'rebinding moves the key')
        ck('was' in await T(), 'moved toast', await T())
        # escape cancels, nothing bound
        await p.click('#pdList [data-learn="tap"]'); await key(p, 'Escape', 'Escape'); await p.wait_for_timeout(100)
        ck(await p.locator('#pdList .learn').count() == 0 and await p.locator('#pdList [data-a="tap"] .pkey').count() == 0, 'Escape cancels learn')
        # cancel by clicking again
        await p.click('#pdList [data-learn="tap"]'); await p.click('#pdList [data-learn="tap"]'); await p.wait_for_timeout(100)
        ck(await p.locator('#pdList .learn').count() == 0, 'second click cancels learn')
        # printable key binds by physical key: works in Thai layout too
        await p.click('#pdList [data-learn="nextSec"]'); await key(p, 'x', 'KeyX'); await p.wait_for_timeout(100)
        ck(await p.locator('#pdList [data-a="nextSec"] .pkey').all_text_contents() == ['X×'], 'letter key name', await p.locator('#pdList [data-a="nextSec"] .pkey').all_text_contents())
        ck((await ST())['map'].get('code:KeyX') == 'nextSec', 'stored by physical key', await ST())
        await key(p, 'ป', 'KeyX'); await p.wait_for_timeout(200)
        ck('No saved sections' in await T(), 'Thai-layout X fires nextSec (no sections toast)', await T())
        # Space binding
        await p.click('#pdList [data-learn="fwd"]'); await key(p, ' ', 'Space'); await p.wait_for_timeout(100)
        ck((await ST())['map'].get('Space') == 'fwd', 'Space can be learned', await ST())
        t0 = await M('t'); await key(p, ' ', 'Space'); await p.wait_for_timeout(200)
        ck(await M('t') > t0 + 1 and await M('st') == 2, 'Space now = forward (pedal overrides shortcut)', (t0, await M('t')))
        # unbind chip
        await p.click('#pdList [data-a="fwd"] .pkey'); await p.wait_for_timeout(100)
        ck('Space' not in (await ST())['map'], 'chip click unbinds')
        await key(p, ' ', 'Space'); await p.wait_for_timeout(300)
        ck(await M('st') == 1, 'Space back to play/pause after unbind')
        await key(p, ' ', 'Space'); await p.wait_for_timeout(200)
        # learn timeout
        await p.click('#pdList [data-learn="slower"]'); await p.wait_for_timeout(12400)
        ck(await p.locator('#pdList .learn').count() == 0 and 'Nothing pressed' in await T(), 'learn times out', await T())

        # ---------- saved sections next / prev ----------
        await p.evaluate("localStorage.setItem('woodshed:sections:aaaaaaaaaa1', JSON.stringify([{id:'s3',n:'Solo',a:60,b:70,s:0.8},{id:'s1',n:'Intro',a:5,b:9,s:1},{id:'s2',n:'Verse',a:20,b:30,s:0.9}]))")
        await p.goto(BASE + '#v=aaaaaaaaaa1'); await p.wait_for_timeout(600); await blur()
        await key(p, 'x', 'KeyX'); await p.wait_for_timeout(400)
        ck((await p.text_content('#at')).strip() in ('0:05.0', '0:05.00') or abs(await M('t') - 5) < 0.7, 'next section from start = Intro', (await p.text_content('#at'), await M('t')))
        ck(await p.locator('.sec.act .n').text_content() == 'Intro', 'Intro active')
        await key(p, 'x', 'KeyX'); await p.wait_for_timeout(300)
        ck(await p.locator('.sec.act .n').text_content() == 'Verse' and abs(await M('rate') - 0.9) < 0.01, 'next = Verse at its speed', await M('rate'))
        await key(p, 'x', 'KeyX'); await p.wait_for_timeout(300)
        ck(await p.locator('.sec.act .n').text_content() == 'Solo', 'next = Solo')
        await key(p, 'x', 'KeyX'); await p.wait_for_timeout(200)
        ck('last saved section' in await T() and await p.locator('.sec.act .n').text_content() == 'Solo', 'end of list toast', await T())
        await p.click('#pdList [data-learn="prevSec"]'); await key(p, 'z', 'KeyZ'); await p.wait_for_timeout(100)
        await key(p, 'z', 'KeyZ'); await p.wait_for_timeout(300)
        ck(await p.locator('.sec.act .n').text_content() == 'Verse', 'prev = Verse')
        ck(await p.locator('#zoomBtn.on').count() == 0, 'Z bound to pedal no longer zooms')
        # prev from no active: picks section before playhead
        await p.click('#clr'); await p.evaluate("window.__mock && 0"); await blur()
        await p.goto(BASE + '#v=aaaaaaaaaa1&a=40&b=45'); await p.wait_for_timeout(600); await blur()
        await p.keyboard.press('q'); await p.wait_for_timeout(300)
        await key(p, 'z', 'KeyZ'); await p.wait_for_timeout(300)
        ck(await p.locator('.sec.act .n').text_content() == 'Verse', 'prev from 0:40 = Verse')

        # ---------- other actions ----------
        for act, k in (('slower', 'F1'), ('faster', 'F2'), ('tap', 'F3'), ('met', 'F4'), ('nextSong', 'F5'), ('toA', 'F6'), ('back', 'F7')):
            await p.click(f'#pdList [data-learn="{act}"]'); await key(p, k, k); await p.wait_for_timeout(60)
        r0 = await M('rate'); await key(p, 'F1', 'F1'); await p.wait_for_timeout(200)
        ck(abs(await M('rate') - (r0 - 0.05)) < 0.011, 'slower −5%', (r0, await M('rate')))
        await key(p, 'F2', 'F2'); await p.wait_for_timeout(200)
        ck(abs(await M('rate') - r0) < 0.011, 'faster +5%')
        await key(p, 'F4', 'F4'); await p.wait_for_timeout(150)
        ck(await p.locator('#metBtn.on, #metBtn[aria-pressed="true"]').count() == 1, 'metronome toggled on')
        await key(p, 'F4', 'F4'); await p.wait_for_timeout(150)
        ck(await p.locator('#metBtn.on, #metBtn[aria-pressed="true"]').count() == 0, 'metronome toggled off')
        await key(p, 'F5', 'F5'); await p.wait_for_timeout(200)
        ck('empty' in (await T()).lower() or 'playlist' in (await T()).lower(), 'next song with empty playlist toast', await T())
        await p.click('#clr'); await blur(); await key(p, 'F6', 'F6'); await p.wait_for_timeout(150)
        ck('Set A' in await T() or 'A and B' in await T() or 'first' in await T(), 'jump to A without A explains', await T())
        ck(await p.is_hidden('#pdHud') or 'Jump' not in await p.text_content('#pdHud'), 'no HUD for an action that did nothing')
        # restartA without A goes to start and plays
        await key(p, 'PageUp', 'PageUp'); await p.wait_for_timeout(400)
        ck(await M('st') == 1 and await M('t') < 0.8, 'back-to-A without A = start + play', (await M('st'), await M('t')))
        await key(p, 'PageDown', 'PageDown'); await p.wait_for_timeout(150)

        # ---------- persistence + reset ----------
        await p.reload(); await p.wait_for_timeout(600)
        ck(await p.is_checked('#pdOn') and await p.locator('#pdList [data-a="nextSec"] .pkey').count() == 1, 'bindings survive reload')
        await p.click('#pdReset'); await p.wait_for_timeout(100)
        ck((await ST())['map'] == {'PageDown': 'play', 'ArrowDown': 'play', 'PageUp': 'restartA', 'ArrowUp': 'restartA'}, 'reset to defaults', await ST())
        await p.click('#pdOn'); await p.wait_for_timeout(100)
        ck(await p.is_hidden('#pdBody') and (await ST())['on'] is False, 'turn off hides + saves')
        await blur(); await key(p, 'ArrowDown', 'ArrowDown'); await p.wait_for_timeout(200)
        ck(await M('st') != 1, 'off: pedal keys inert')
        # learn while off turns it on
        await p.click('#pdOn'); await p.click('#pdList [data-learn="loop"]'); await p.click('#pdOn'); await p.wait_for_timeout(100)
        ck(await p.locator('#pdList .learn').count() == 0, 'turning off cancels learning')
        await p.click('#pdOn')

        # ---------- MIDI ----------
        ck(await p.is_visible('#pdMidi') and await p.is_hidden('#pdMidiNo'), 'MIDI button shown when supported')
        await p.click('#pdMidi'); await p.wait_for_timeout(200)
        ck('1' in await T() and 'FS-1 Pedal' in await p.text_content('#pdStatus'), 'MIDI connected + device listed', (await T(), await p.text_content('#pdStatus')))
        ck(await p.is_disabled('#pdMidi'), 'MIDI button disabled after connect')
        await p.click('#pdList [data-learn="play"]'); await p.evaluate("__midiSend(0xB0,64,127)"); await p.wait_for_timeout(100)
        ck((await ST())['map'].get('midi:cc64') == 'play', 'MIDI CC learned', await ST())
        ck(any('MIDI CC 64' in c for c in await p.locator('#pdList [data-a="play"] .pkey').all_text_contents()), 'MIDI chip name')
        await p.evaluate("__midiSend(0xB0,64,0)")
        st0 = await M('st'); await p.evaluate("__midiSend(0xB0,64,127)"); await p.wait_for_timeout(300)
        ck(await M('st') == 1, 'MIDI CC press plays', (st0, await M('st')))
        await p.evaluate("__midiSend(0xB0,64,120)"); await p.wait_for_timeout(200)
        ck(await M('st') == 1, 'CC held (no release) does not re-fire')
        await p.evaluate("__midiSend(0xB0,64,0)"); await p.evaluate("__midiSend(0xB0,64,127)"); await p.wait_for_timeout(200)
        ck(await M('st') == 2, 'CC release + press fires again')
        await p.click('#pdList [data-learn="restartA"]'); await p.evaluate("__midiSend(0x90,60,0)"); await p.wait_for_timeout(50)
        ck(await p.locator('#pdList .learn').count() == 1, 'note-on velocity 0 ignored')
        await p.evaluate("__midiSend(0x91,62,100)"); await p.wait_for_timeout(100)
        ck((await ST())['map'].get('midi:n62') == 'restartA', 'MIDI note learned (any channel)')
        await p.click('#pdList [data-learn="tap"]'); await p.evaluate("__midiSend(0xC0,3)"); await p.wait_for_timeout(100)
        ck((await ST())['map'].get('midi:pc3') == 'tap', 'program change learned')
        await p.evaluate("__midiSend(0xF8)"); await p.evaluate("__midiSend(0x80,62,0)"); await p.wait_for_timeout(100)
        ck(not errs, 'clock / note-off ignored, no errors', errs)
        await p.evaluate("__midiAdd()"); await p.wait_for_timeout(100)
        ck('Second' in await p.text_content('#pdStatus'), 'hot-plugged device listed')
        ck((await ST())['midi'] is True, 'MIDI flag saved')
        # reload: no prompt unless already granted
        await p.reload(); await p.wait_for_timeout(500)
        ck(await p.evaluate("__midiReq") == 0, 'no MIDI request on load when not granted')
        await p.evaluate("window.__midiPerm='granted'");
        await ctx.add_init_script("window.__midiPerm='granted'")
        await p.reload(); await p.wait_for_timeout(500)
        ck(await p.evaluate("__midiReq") == 1 and 'FS-1' in await p.text_content('#pdStatus'), 'auto-reconnect when granted')
        st0 = await M('st'); await p.evaluate("__midiSend(0x90,62,90)"); await p.wait_for_timeout(400)
        ck(await M('st') == 1, 'MIDI note works after reload', (st0, await M('st')))

        # ---------- Thai ----------
        await p.click('.langs [data-lang="th"]') if await p.locator('.langs [data-lang="th"]').count() else await p.click('.langs button:nth-child(2)')
        await p.wait_for_timeout(200)
        names = await p.locator('#pdList .pn').all_text_contents()
        ck(all(not any(c.isascii() and c.isalpha() for c in n.replace('A','')) for n in names), 'Thai action names', names)
        ck('ฟุตสวิตช์' in await p.text_content('#pdCard h2'), 'Thai title')
        txt = await p.text_content('#pdCard')
        ck('Reset' not in txt and 'Set' not in txt.replace('Set ', ''), 'no English UI words in Thai card', txt[:200])

        # ---------- corrupted storage ----------
        for bad in ['"garbage"', '{"on":true,"map":{"PageDown":"evil","k":"play"}}', '{"on":"yes","map":[1,2]}', 'not json']:
            await p.evaluate("v => localStorage.setItem('woodshed:pedal', v)", bad)
            await p.reload(); await p.wait_for_timeout(400)
            n = await p.locator('#pdList .pd').count()
            ck(n == 14, f'corrupt storage survives {bad[:20]}', n)
        await p.evaluate("localStorage.setItem('woodshed:pedal', JSON.stringify({on:true,map:{PageDown:'evil',k:'play'}}))"); await p.reload(); await p.wait_for_timeout(400)
        await p.click('#pdList [data-learn="loop"]'); await key(p, 'q', 'KeyQ'); await p.wait_for_timeout(100)
        st = await ST()
        ck('PageDown' not in st['map'] and st['map'].get('k') == 'play', 'bad action dropped on save', st)

        # ---------- keyboard layout fallbacks (pedal off) ----------
        await p.evaluate("localStorage.removeItem('woodshed:pedal')"); await p.goto(BASE + '#v=aaaaaaaaaa1'); await p.wait_for_timeout(600); await blur()
        await key(p, 'ArrowRight', 'ArrowRight'); await key(p, 'ArrowRight', 'ArrowRight'); await p.wait_for_timeout(150)
        await key(p, 'ฟ', 'KeyA'); await p.wait_for_timeout(150)
        ck('ยัง' not in await p.text_content('#at') and 'not' not in await p.text_content('#at'), 'Thai layout ฟ = A', await p.text_content('#at'))
        await key(p, '-', 'Digit3'); await p.wait_for_timeout(300)
        ck(abs(await M('rate') - 0.3) < 0.01, 'Thai layout 3-key = 30%', await M('rate'))
        await key(p, 'ๅ', 'Digit1'); await key(p, 'บ', 'BracketLeft'); await p.wait_for_timeout(300)
        ck(abs(await M('rate') - 0.25) < 0.01, 'Thai layout [ = slower', await M('rate'))
        await key(p, '0', 'Digit0'); await p.wait_for_timeout(200)
        await key(p, 'a', 'KeyQ'); await p.wait_for_timeout(150)
        at = await p.text_content('#at')
        ck(abs(await M('rate') - 1) < 0.01, '0 = 100%')
        # AZERTY: key 'a' on KeyQ must act as A, not Q
        await key(p, 'é', 'Digit2'); await p.wait_for_timeout(100)
        ck(abs(await M('rate') - 1) < 0.01, 'unmapped 2-key does nothing')
        ck(not errs, 'no errors', errs)

        # backup includes pedal
        await p.evaluate("localStorage.setItem('woodshed:pedal', JSON.stringify({on:true,map:{PageDown:'play'},midi:false}))")
        async with p.expect_download() as dl:
            await p.click('#bkExport')
        d = await dl.value; data = json.loads(open(await d.path()).read())
        ck('woodshed:pedal' in data['data'], 'backup includes foot switch settings')
        await ctx.close()

        # ---------- no MIDI browser + mobile ----------
        ctx2, p2, errs2 = await newpage(b, vp=(390, 844), init=NO_MIDI, is_mobile=True, has_touch=True)
        await p2.goto(BASE); await p2.wait_for_timeout(500)
        await p2.locator('#pdOn').scroll_into_view_if_needed(); await p2.tap('#pdCard .tog')
        await p2.wait_for_timeout(150)
        ck(await p2.is_checked('#pdOn'), 'mobile toggle on')
        ck(await p2.is_hidden('#pdMidi') and await p2.is_visible('#pdMidiNo'), 'no MIDI: button hidden + note')
        ow = await p2.evaluate("document.documentElement.scrollWidth - innerWidth")
        ck(ow <= 0, 'no horizontal overflow on phone', ow)
        rows = await p2.evaluate("[...document.querySelectorAll('#pdList .pd')].map(r=>r.getBoundingClientRect().right<=innerWidth+0.5).every(Boolean)")
        ck(rows, 'rows fit phone width')
        await p2.locator('#pdCard').screenshot(path='/home/claude/tests/rB_mobile.png')
        ck(not errs2, 'mobile no errors', errs2)
        await ctx2.close()
        ctx3, p3, e3 = await newpage(b)
        await p3.goto(BASE); await p3.wait_for_timeout(400); await p3.click('#pdOn')
        await p3.locator('#pdCard').screenshot(path='/home/claude/tests/rB_desktop.png')
        await b.close()
    srv.terminate()
    f = [n for c, n in R if not c]; print(f'\n{len(R)-len(f)} passed, {len(f)} failed: {f}')
asyncio.run(main())
