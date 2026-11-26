import asyncio, subprocess, sys, time, json, re
from playwright.async_api import async_playwright
from common import wire, SITE

PORT = 8783
BASE = f'http://127.0.0.1:{PORT}/index.html'
VER = sys.argv[1] if len(sys.argv) > 1 else 'D1'
R = []
def ck(c, n, x=''):
    R.append((bool(c), n)); print('PASS' if c else 'FAIL', n, '' if c else x)

async def newpage(b, vp=(1400, 950), **kw):
    ctx = await b.new_context(viewport={'width': vp[0], 'height': vp[1]}, permissions=['clipboard-read', 'clipboard-write'], **kw)
    await wire(ctx)
    p = await ctx.new_page(); errs = []
    p.on('pageerror', lambda e: errs.append(str(e)))
    p.on('console', lambda m: errs.append(m.text) if m.type == 'error' and 'net::' not in m.text and 'Failed to load resource' not in m.text else None)
    return ctx, p, errs

async def main():
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1', '-d', SITE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1)
    async with async_playwright() as pw:
        b = await pw.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--no-sandbox', '--autoplay-policy=no-user-gesture-required'])
        ctx, p, errs = await newpage(b)
        CH = f"(await import(new URL('js/chords.js?v={VER}', location)))"
        TH = f"(await import(new URL('js/theory.js?v={VER}', location)))"
        st = lambda: p.evaluate(f"(async()=>{CH}._state())()")
        M = lambda e: p.evaluate('window.__mock.' + e)
        T = lambda: p.text_content('#toast')

        # ---------- parser ----------
        await p.goto(BASE); await p.wait_for_timeout(500)
        cases = {'Am': ('Am', 9, 'm', -1), 'F#m7b5': ('F#m7b5', 6, 'm7b5', -1), 'Bb/D': ('Bb/D', 10, '', 2), 'C6/9': ('C69', 0, '69', -1),
                 'CM7': ('Cmaj7', 0, 'maj7', -1), 'CΔ7': ('Cmaj7', 0, 'maj7', -1), 'C-7': ('Cm7', 0, 'm7', -1), 'Cø': ('Cm7b5', 0, 'm7b5', -1),
                 'C°7': ('Cdim7', 0, 'dim7', -1), 'C+': ('Caug', 0, 'aug', -1), 'Csus': ('Csus4', 0, 'sus4', -1), 'G7(b9)': ('G7b9', 7, '7b9', -1),
                 'am': ('Am', 9, 'm', -1), 'C/C': ('C', 0, '', -1), 'Db': ('Db', 1, '', -1), 'Cb': ('Cb', 11, '', -1), 'Cmin7': ('Cm7', 0, 'm7', -1),
                 'D2': ('Dsus2', 2, 'sus2', -1), 'Emadd9': ('Emadd9', 4, 'madd9', -1), 'Gmaj': ('G', 7, '', -1), 'Am/G': ('Am/G', 9, 'm', 7),
                 'F♯m': ('F#m', 6, 'm', -1), 'Bbm6/9': ('Bbm69', 10, 'm69', -1), 'N.C.': ('N.C.', -1, '', -1), 'nc': ('N.C.', -1, '', -1), 'C5': ('C5', 0, '5', -1),
                 'CmM7': ('CmMaj7', 0, 'mMaj7', -1), 'C7sus': ('C7sus4', 0, '7sus4', -1)}
        res = await p.evaluate(f"(async(cs)=>{{const M={CH};const o={{}};for(const k of cs){{const r=M.parseChord(k);o[k]=r&&[r.n,r.r,r.ty,r.b]}}return o}})", list(cases))
        bad = {k: (res[k], v) for k, v in cases.items() if res[k] != list(v)}
        ck(not bad, f'parser {len(cases)} symbols', bad)
        rej = await p.evaluate(f"(async()=>{{const M={CH};return ['H','Cxyz','','X7','C7#','1','Am F'].map(k=>M.parseChord(k))}})()")
        ck(all(r is None for r in rej), 'parser rejects junk', rej)
        pl = await p.evaluate(f"(async()=>{{const M={CH};const a=M.parseLine('Am F % G | C, Dm'),b=M.parseLine('Am Xq F');return [a.chords.map(c=>c.n),a.bad,b.bad]}})()")
        ck(pl[0] == ['Am', 'F', 'F', 'G', 'C', 'Dm'] and pl[1] == [] and pl[2] == ['Xq'], 'parse line (+repeat, bars, commas)', pl)
        ids = await p.evaluate(f"(async()=>{{const M={CH};const T={TH};return [M.IDS, T.data().types.map(x=>x.id)]}})()")
        ck(sorted(ids[0]) == sorted(ids[1]), 'chord types match the chart data')

        # ---------- empty song ----------
        await p.goto(BASE + '#v=aaaaaaaaaa1'); await p.wait_for_timeout(600)
        ck(await p.is_hidden('#csNow') and await p.is_hidden('#csEd') and await p.locator('#csLane .cs').count() == 0, 'nothing shown without chords')
        ck(await p.get_attribute('#csEdBtn', 'aria-pressed') == 'false', 'editor closed by default')
        await p.click('#csEdBtn'); await p.wait_for_timeout(150)
        ck(await p.is_visible('#csEd'), 'editor opens')
        pal = [x.strip() for x in await p.locator('#csPal .kc b').all_text_contents()]
        ck(pal == ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim', 'N.C.'], 'palette = key of C + break', pal)
        ck('C major' in await p.text_content('#csKey'), 'key label', await p.text_content('#csKey'))
        await p.click('#csSev [data-sev="1"]'); await p.wait_for_timeout(80)
        ck('Cmaj7' in await p.text_content('#csPal'), '7ths palette')
        await p.click('#csSev [data-sev="0"]')

        async def seek(t):
            await p.evaluate("document.getElementById('track').scrollIntoView({block:'center'})"); bx = await p.locator('#track').bounding_box()
            await p.mouse.click(bx['x'] + bx['width'] * t / 200, bx['y'] + bx['height'] / 2); await p.wait_for_timeout(120)
        async def lane_at(t, dx=3):
            await p.evaluate("document.getElementById('track').scrollIntoView({block:'center'})")
            bx = await p.locator('#csLane').bounding_box()
            await p.mouse.click(bx['x'] + bx['width'] * t / 200 + dx, bx['y'] + bx['height'] / 2); await p.wait_for_timeout(120)
        async def pal_tap(n):
            await p.locator(f'#csPal [data-n="{n}"]').click(); await p.wait_for_timeout(60)

        # ---------- tap in while paused ----------
        await seek(5)
        t0 = await M('t')
        await pal_tap('Am')
        s = await st()
        ck(len(s['C']) == 1 and abs(s['C'][0]['t'] - t0) < 0.05 and s['C'][0]['n'] == 'Am', 'paused: chord at the playhead', (s['C'], t0))
        ck(await p.is_visible('#csNow') and await p.text_content('#csCur') == 'Am', 'now display shows Am')
        ck(await p.locator('#csLane .cs').count() == 1, 'chip on the lane')
        await pal_tap('C')
        s = await st()
        ck(len(s['C']) == 1 and s['C'][0]['n'] == 'C', 'tapping at the same spot replaces it', s['C'])
        await pal_tap('Am')
        # ---------- tap in while playing (reaction-time offset) ----------
        await seek(9); await p.click('#play'); await p.wait_for_timeout(900)
        tn = await M('t'); await pal_tap('F'); s = await st()
        f = [c for c in s['C'] if c['n'] == 'F'][0]
        ck(tn - 0.3 < f['t'] < tn, 'playing: chord placed a bit before the tap (reaction time)', (tn, f['t']))
        await p.wait_for_timeout(800); await pal_tap('G'); await p.wait_for_timeout(300)
        ck(await p.text_content('#csCur') == 'G', 'now follows playhead', await p.text_content('#csCur'))
        await p.click('#play'); await p.wait_for_timeout(100)
        s = await st(); ts = [c['t'] for c in s['C']]
        ck([c['n'] for c in s['C']] == ['Am', 'F', 'G'] and ts == sorted(ts), 'three chords in order', s['C'])
        # now / next / progress
        await seek(2); await p.wait_for_timeout(100)
        ck(await p.text_content('#csCur') == '–' and await p.text_content('#csNext') == 'Am', 'before first chord: – / Am')
        ck(await p.is_disabled('#csCur'), 'no chart button before first chord')
        mid = (ts[0] + ts[1]) / 2
        await seek(mid); await p.wait_for_timeout(100)
        fill = await p.evaluate("parseFloat(document.getElementById('csFill').style.width)")
        ck(await p.text_content('#csCur') == 'Am' and await p.text_content('#csNext') == 'F' and 30 < fill < 70, 'between chords: Am → F, bar half way', fill)
        ck(await p.locator('#csLane .cs.cur').count() == 1 and 'Am' in await p.locator('#csLane .cs.cur').text_content(), 'lane highlights current chord')
        # chart link
        await p.click('#csCur'); await p.wait_for_timeout(250)
        th = await p.evaluate(f"(async()=>{{const T={TH};return [T._state.tab,T._state.root,T._state.type]}})()")
        ck(th == ['chords', 9, 'm'], 'tap current chord opens its chart', th)

        # ---------- select / edit on the lane ----------
        s = await st(); await lane_at(s['C'][1]['t'], 1)
        s = await st()
        ck(s['sel'] == s['C'][1]['id'] and await p.is_visible('#csSel') and await p.text_content('#csSelName') == 'F', 'tap chip selects it')
        ck(abs(await M('t') - s['C'][1]['t']) < 0.1, 'and jumps there')
        ck('change the selected' in await p.text_content('#csPalHint'), 'palette hint switches to change mode')
        await pal_tap('Dm'); s = await st()
        ck(s['C'][1]['n'] == 'Dm' and len(s['C']) == 3, 'palette replaces selected chord')
        t1 = s['C'][1]['t']
        await p.click('#csSelNext'); await p.wait_for_timeout(80); s = await st()
        ck(abs(s['C'][1]['t'] - (t1 + 0.1)) < 0.01, '+0.1 s nudge', s['C'][1]['t'])
        await p.click('#csSelPrev'); await p.click('#csSelPrev'); await p.wait_for_timeout(80); s = await st()
        ck(abs(s['C'][1]['t'] - (t1 - 0.1)) < 0.01, '−0.1 s nudge')
        # other chord picker
        await p.select_option('#csRoot', '6'); await p.select_option('#csType', 'm7b5'); await p.click('#csAddOther'); await p.wait_for_timeout(80)
        s = await st()
        ck(s['C'][1]['n'] == 'F#m7b5', 'other-chord picker replaces selected', s['C'][1])
        ck('F#m7b5' in await p.text_content('#csPal'), 'used chord appears in palette')
        # drag
        chip = p.locator('#csLane .cs').nth(1); bb = await chip.bounding_box()
        lane = await p.locator('#csLane').bounding_box()
        s0 = await st()
        await p.mouse.move(bb['x'] + 1, bb['y'] + bb['height'] / 2); await p.mouse.down()
        await p.mouse.move(bb['x'] + 1 - lane['width'] * 2 / 200, bb['y'] + 10, steps=5); await p.mouse.up(); await p.wait_for_timeout(120)
        s = await st()
        ck(abs(s['C'][1]['t'] - (s0['C'][1]['t'] - 2)) < 0.3, 'drag moves chord', (s0['C'][1]['t'], s['C'][1]['t']))
        bb = await p.locator('#csLane .cs').nth(1).bounding_box()
        await p.mouse.move(bb['x'] + 1, bb['y'] + 10); await p.mouse.down()
        await p.mouse.move(bb['x'] + lane['width'] * 0.5, bb['y'] + 10, steps=6); await p.mouse.up(); await p.wait_for_timeout(120)
        s = await st()
        ck(s['C'][1]['t'] < s['C'][2]['t'] and [c['n'] for c in s['C']] == ['Am', 'F#m7b5', 'G'], 'drag cannot pass the next chord', [(c['n'], c['t']) for c in s['C']])
        # move to playhead / delete / undo
        await seek(1)
        sid = s['C'][2]['id']
        if await p.is_visible('#csSel'): await p.click('#csSelX')
        await lane_at(s['C'][2]['t']); await p.wait_for_timeout(80)
        await seek(30); await p.click('#csSelHere'); await p.wait_for_timeout(80); s = await st()
        g = [c for c in s['C'] if c['id'] == sid][0]
        ck(abs(g['t'] - 30) < 0.3, 'move to playhead', g['t'])
        await p.click('#csSelDel'); await p.wait_for_timeout(80); s = await st()
        ck(len(s['C']) == 2 and s['sel'] is None and await p.is_hidden('#csSel'), 'delete chord')
        await p.click('#csUndo'); await p.wait_for_timeout(80); s = await st()
        ck(len(s['C']) == 3 and any(c['id'] == sid for c in s['C']), 'undo brings it back')
        await p.click('#csUndo'); await p.wait_for_timeout(80); s = await st()
        g = [c for c in s['C'] if c['id'] == sid][0]
        ck(abs(g['t'] - 30) > 1, 'undo again: back to before the move', g['t'])
        # deselect
        await lane_at([c for c in s['C'] if c['id'] == sid][0]['t']); await p.click('#csSelX'); await p.wait_for_timeout(80)
        ck((await st())['sel'] is None, 'deselect')
        # break chord
        await seek(40); await pal_tap('N.C.'); await p.wait_for_timeout(150)
        ck(await p.text_content('#csCur') == 'N.C.' and await p.is_disabled('#csCur'), 'N.C. shown, no chart')

        # ---------- persistence ----------
        await p.wait_for_timeout(400)
        saved = await p.evaluate("JSON.parse(localStorage.getItem('woodshed:chords:aaaaaaaaaa1'))")
        ck(len(saved) == 4 and all(set(x) == {'id', 't', 'n'} for x in saved), 'saved compactly', saved)
        await p.reload(); await p.wait_for_timeout(700)
        s = await st()
        ck([c['n'] for c in s['C']] == ['Am', 'F#m7b5', 'G', 'N.C.'] and s['open'], 'chords + open editor survive reload', s)
        # editor closed: tap seeks only, no drag
        await p.click('#csDone'); await p.wait_for_timeout(80)
        ck(await p.is_hidden('#csEd'), 'Done closes editor')
        await p.evaluate("document.getElementById('track').scrollIntoView({block:'center'})")
        bb = await p.locator('#csLane .cs').nth(0).bounding_box(); s0 = await st()
        await p.mouse.move(bb['x'] + 4, bb['y'] + 10); await p.mouse.down(); await p.mouse.move(bb['x'] + 80, bb['y'] + 10, steps=4); await p.mouse.up(); await p.wait_for_timeout(100)
        s = await st()
        ck(s['C'][0]['t'] == s0['C'][0]['t'] and s['sel'] is None, 'editor closed: chips cannot be dragged or selected')
        await lane_at(s['C'][2]['t'] + 1)
        ck(abs(await M('t') - s['C'][2]['t']) < 0.1, 'editor closed: tap a chip jumps there')

        # ---------- typed chords into the loop ----------
        await p.click('#csEdBtn')
        await p.click('#clr'); await p.wait_for_timeout(60)
        await p.fill('#csText', 'C G Am F'); await p.click('#csForm button'); await p.wait_for_timeout(100)
        ck('A–B' in await T() and len((await st())['C']) == 4, 'needs an A–B loop', await T())
        await p.goto(BASE + '#v=aaaaaaaaaa1&a=50&b=66'); await p.wait_for_timeout(700)
        await p.fill('#csText', 'C G Am Xq'); await p.click('#csForm button'); await p.wait_for_timeout(100)
        ck('Xq' in await T() and len((await st())['C']) == 4, 'bad token rejected, nothing changed', await T())
        await p.fill('#csText', 'C G Am F'); await p.click('#csForm button'); await p.wait_for_timeout(100)
        s = await st(); L = [(c['n'], round(c['t'], 2)) for c in s['C'] if 50 <= c['t'] < 66]
        ck(L == [('C', 50), ('G', 54), ('Am', 58), ('F', 62)], 'no BPM: spread evenly', L)
        ck(await p.input_value('#csText') == '' and 'spread' in await T(), 'field cleared + toast')
        ck(len(s['C']) == 8, 'chords outside the loop kept', len(s['C']))
        await p.fill('#bpmIn', '120'); await p.press('#bpmIn', 'Enter'); await p.wait_for_timeout(100)
        await p.fill('#csText', 'Dm G % C'); await p.click('#csForm button'); await p.wait_for_timeout(100)
        s = await st(); L = [(c['n'], round(c['t'], 2)) for c in s['C'] if 50 <= c['t'] < 66]
        ck(L == [('Dm', 50), ('G', 52), ('G', 54), ('C', 56), ('Dm', 58), ('G', 60), ('G', 62), ('C', 64)], '120 BPM 4/4: one per bar, repeating', L)
        ck('bar' in await T(), 'bar toast')
        # loop "next" wraps to A
        await seek(65); await p.wait_for_timeout(120)
        ck(await p.text_content('#csCur') == 'C' and await p.text_content('#csNext') == 'Dm', 'inside loop: next after last = chord at A', (await p.text_content('#csCur'), await p.text_content('#csNext')))
        await p.click('#csUndo'); await p.wait_for_timeout(80)
        s = await st(); L = [c['n'] for c in s['C'] if 50 <= c['t'] < 66]
        ck(L == ['C', 'G', 'Am', 'F'], 'undo the fill', L)

        # ---------- zoom ----------
        await p.click('#zoomBtn'); await p.wait_for_timeout(150)
        n = await p.locator('#csLane .cs').count()
        ck(n == 4 or n == 5, 'zoomed: only chords near the loop', n)
        lefts = await p.evaluate("[...document.querySelectorAll('#csLane .cs')].map(e=>parseFloat(e.style.left))")
        ck(all(0 <= x <= 100 for x in lefts), 'zoomed positions in range', lefts)
        await p.click('#zoomBtn')

        # ---------- share link ----------
        await p.click('#share'); await p.wait_for_timeout(200)
        link = await p.evaluate('navigator.clipboard.readText()')
        ck('&c=' in link, 'share link carries chords', link[:120])
        ctx2, p2, errs2 = await newpage(b)
        await p2.goto(link); await p2.wait_for_timeout(700)
        s2 = await p2.evaluate(f"(async()=>{CH.replace('location', 'location')}._state())()")
        s1 = await st()
        ck([(c['n'], round(c['t'], 1)) for c in s2['C']] == [(c['n'], round(c['t'], 1)) for c in s1['C']], 'fresh browser: chords loaded from link', (len(s2['C']), len(s1['C'])))
        ck('loaded from the link' in await p2.text_content('#toast'), 'loaded toast')
        # same browser, different local chords → banner
        await p2.evaluate("localStorage.setItem('woodshed:chords:aaaaaaaaaa1', JSON.stringify([{id:'x1',t:3,n:'E'}]))")
        await p2.goto('about:blank'); await p2.goto(link); await p2.wait_for_timeout(700)
        s2 = await p2.evaluate(f"(async()=>{CH}._state())()")
        ck(len(s2['C']) == 1 and s2['link'] == len(s1['C']) and await p2.is_visible('#csLink'), 'different local chords: offer, keep mine for now', s2)
        await p2.click('#csLinkUse'); await p2.wait_for_timeout(100)
        s2 = await p2.evaluate(f"(async()=>{CH}._state())()")
        ck(len(s2['C']) == len(s1['C']) and await p2.is_hidden('#csLink'), 'use link chords')
        await p2.click('#csUndo'); await p2.wait_for_timeout(100)
        s2 = await p2.evaluate(f"(async()=>{CH}._state())()")
        ck([c['n'] for c in s2['C']] == ['E'], 'undo back to mine')
        await p2.goto('about:blank'); await p2.goto(link); await p2.wait_for_timeout(600)
        await p2.click('#csLinkKeep'); await p2.wait_for_timeout(80)
        s2 = await p2.evaluate(f"(async()=>{CH}._state())()")
        ck([c['n'] for c in s2['C']] == ['E'] and await p2.is_hidden('#csLink'), 'keep mine')
        ck(not errs2, 'link page no errors', errs2)
        await ctx2.close()
        # too many chords for a link
        await p.evaluate("localStorage.setItem('woodshed:chords:aaaaaaaaaa1', JSON.stringify(Array.from({length:300},(_, i)=>({id:'k'+i,t:i*0.6,n:'C#m7b5/G#'}))))")
        await p.reload(); await p.wait_for_timeout(700)
        await p.click('#share'); await p.wait_for_timeout(200)
        link = await p.evaluate('navigator.clipboard.readText()')
        ck('&c=' not in link and 'too many' in await T(), 'too many chords: link without them + note', await T())

        # ---------- clear all ----------
        await p.click('#csClear'); await p.wait_for_timeout(60)
        ck(len((await st())['C']) == 300 and 'again' in await T(), 'clear needs a second tap')
        await p.click('#csClear'); await p.wait_for_timeout(100)
        ck(len((await st())['C']) == 0 and await p.is_hidden('#csNow'), 'cleared')
        await p.wait_for_timeout(300)
        ck(await p.evaluate("localStorage.getItem('woodshed:chords:aaaaaaaaaa1')") is None, 'storage key removed when empty')
        await p.click('#csUndo'); await p.wait_for_timeout(100)
        ck(len((await st())['C']) == 300, 'undo clear')

        # ---------- other song ----------
        await p.fill('#url', 'https://youtu.be/bbbbbbbbbb2'); await p.press('#url', 'Enter'); await p.wait_for_timeout(600)
        ck(len((await st())['C']) == 0 and await p.locator('#csLane .cs').count() == 0 and await p.is_hidden('#csNow'), 'other song has no chords')
        pal = [x.strip() for x in await p.locator('#csPal .kc b').all_text_contents()]
        ck('F#m7b5' not in pal and 'C#m7b5/G#' not in pal, 'palette does not carry other song chords')
        # song key drives palette
        await p.click('#theory [data-th="key"]'); await p.click('#thRoots [data-r="9"]'); await p.click('#thMode [data-m="minor"]'); await p.wait_for_timeout(150)
        pal = [x.strip() for x in await p.locator('#csPal .kc b').all_text_contents()]
        ck(pal[:7] == ['Am', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G'], 'palette follows song key (A minor)', pal)
        await seek(20); await pal_tap('Em'); await p.wait_for_timeout(50)
        await p.fill('#url', 'https://youtu.be/aaaaaaaaaa1'); await p.press('#url', 'Enter'); await p.wait_for_timeout(600)
        ck(len((await st())['C']) == 300, 'first song chords back')
        await p.fill('#url', 'https://youtu.be/bbbbbbbbbb2'); await p.press('#url', 'Enter'); await p.wait_for_timeout(600)
        ck([c['n'] for c in (await st())['C']] == ['Em'], 'second song kept its own chord (no mix-up)')

        # ---------- corrupt storage ----------
        await p.evaluate("localStorage.setItem('woodshed:chords:bbbbbbbbbb2', JSON.stringify([{t:-1,n:'C'},{t:'x',n:'D'},{t:4,n:'Hzz'},{t:6,n:'G7'},null,5,{t:2,n:'Am',id:7}]))")
        await p.reload(); await p.wait_for_timeout(600)
        ck([c['n'] for c in (await st())['C']] == ['Am', 'G7'], 'corrupt entries dropped, rest sorted', (await st())['C'])
        await p.evaluate("localStorage.setItem('woodshed:chords:bbbbbbbbbb2', '\"junk\"'); localStorage.setItem('woodshed:chordEd','[1]')")
        await p.reload(); await p.wait_for_timeout(600)
        ck(len((await st())['C']) == 0, 'junk storage ignored')

        # ---------- backup counts chords ----------
        await p.evaluate("localStorage.setItem('woodshed:chords:bbbbbbbbbb2', JSON.stringify([{id:'a',t:1,n:'C'},{id:'b',t:2,n:'G'}]))")
        async with p.expect_download() as dl:
            await p.click('#bkExport')
        d = await dl.value; path = await d.path()
        await p.set_input_files('#bkFile', path); await p.wait_for_timeout(300)
        ck('Chords on timeline' in await p.text_content('#bkPreview'), 'backup preview counts chords')
        await p.click('#bkPreview [data-bk="cancel"]') if await p.locator('#bkPreview [data-bk="cancel"]').count() else None

        # ---------- Thai ----------
        await p.click('.langs [data-lang="th"]'); await p.wait_for_timeout(200)
        await p.click('#csEdBtn') if await p.is_hidden('#csEd') else None
        t = await p.evaluate("(()=>{const e=document.getElementById('csEd').cloneNode(true);e.querySelectorAll('#csPal,select').forEach(x=>x.remove());return e.textContent})()") + (await p.get_attribute('#csText', 'placeholder'))
        eng = [w for w in re.findall(r'[A-Za-z]{3,}', t) if not re.fullmatch(r'[A-G][#b]?(m|dim|maj|aug|sus)?\d*', w) and w not in ('BPM', 'dim', 'maj', 'aug', 'sus', 'madd', 'add')]
        ck(not eng, 'Thai editor', eng[:12])
        ck('ถัดไป' in await p.text_content('#csNow'), 'Thai next label')
        await p.click('.langs [data-lang="en"]')
        ck(not errs, 'no errors', errs)
        await p.evaluate("document.getElementById('track').scrollIntoView({block:'start'})")
        await p.locator('.stage').screenshot(path='/home/claude/tests/rD_desktop.png')
        await ctx.close()

        # ---------- mobile ----------
        ctx3, p3, errs3 = await newpage(b, vp=(390, 844), is_mobile=True, has_touch=True)
        await p3.goto(BASE + '#v=aaaaaaaaaa1&a=20&b=36'); await p3.wait_for_timeout(600)
        await p3.locator('#csEdBtn').scroll_into_view_if_needed(); await p3.tap('#csEdBtn'); await p3.wait_for_timeout(100)
        await p3.fill('#csText', 'Am F C G'); await p3.tap('#csForm button'); await p3.wait_for_timeout(200)
        ck(await p3.locator('#csLane .cs').count() == 4, 'mobile: fill works')
        ow = await p3.evaluate("document.documentElement.scrollWidth - innerWidth")
        ck(ow <= 0, 'mobile: no overflow', ow)
        h = await p3.evaluate("Math.min(...[...document.querySelectorAll('#csPal .kc')].map(b=>b.getBoundingClientRect().height))")
        ck(h >= 44, 'mobile: palette buttons big enough', h)
        ta = await p3.evaluate("getComputedStyle(document.querySelector('#csLane .cs')).touchAction")
        ck(ta == 'none', 'editing: chips own the touch (drag)', ta)
        await p3.tap('#csDone')
        ta = await p3.evaluate("getComputedStyle(document.querySelector('#csLane .cs')).touchAction")
        ck(ta != 'none', 'not editing: page scrolls normally over chips', ta)
        await p3.locator('.stage').screenshot(path='/home/claude/tests/rD_mobile.png')
        ck(not errs3, 'mobile no errors', errs3)
        await b.close()
    srv.terminate()
    f = [n for c, n in R if not c]; print(f'\n{len(R)-len(f)} passed, {len(f)} failed: {f}')
asyncio.run(main())
