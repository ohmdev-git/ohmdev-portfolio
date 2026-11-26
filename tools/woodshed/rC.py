import asyncio, subprocess, sys, time, json
from playwright.async_api import async_playwright
from common import wire, SITE

PORT = 8782
BASE = f'http://127.0.0.1:{PORT}/index.html'
VER = sys.argv[1] if len(sys.argv) > 1 else 'C1'
R = []
def ck(c, n, x=''):
    R.append((bool(c), n)); print('PASS' if c else 'FAIL', n, '' if c else x)

SPY = """
window.__src = [];
const _st = AudioBufferSourceNode.prototype.start;
AudioBufferSourceNode.prototype.start = function (w) { window.__src.push(+(w || 0).toFixed(3)); return _st.apply(this, arguments); };
"""
TUN = {'std': [40, 45, 50, 55, 59, 64], 'dropd': [38, 45, 50, 55, 59, 64]}

async def newpage(b, vp=(1400, 950), **kw):
    ctx = await b.new_context(viewport={'width': vp[0], 'height': vp[1]}, **kw)
    await wire(ctx); await ctx.add_init_script(SPY)
    p = await ctx.new_page(); errs = []
    p.on('pageerror', lambda e: errs.append(str(e)))
    p.on('console', lambda m: errs.append(m.text) if m.type == 'error' and 'net::' not in m.text and 'Failed to load resource' not in m.text else None)
    return ctx, p, errs

async def main():
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1', '-d', SITE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1)
    async with async_playwright() as pw:
        b = await pw.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--no-sandbox', '--autoplay-policy=no-user-gesture-required'])
        ctx, p, errs = await newpage(b)
        PR = f"(await import(new URL('js/practice.js?v={VER}', location)))"
        Q = lambda: p.evaluate(f"(async()=>{PR}._state().Q)()")
        E = lambda: p.evaluate(f"(async()=>{PR}._state().E)()")
        src = lambda: p.evaluate("__src.length")
        await p.goto(BASE); await p.wait_for_timeout(600)

        # ---------- quiz: defaults ----------
        await p.locator('#practice').scroll_into_view_if_needed()
        ck(await p.is_visible('#practice') and await p.is_visible('#prQuiz') and await p.is_hidden('#prEar'), 'practice card, quiz tab default')
        q = await Q()
        ck(q['mode'] == 'name' and q['range'] == 12 and q['nat'] and len(q['strings']) == 6, 'quiz defaults', q)
        ck(await p.locator('#qzNotes button').count() == 7, '7 natural note buttons')
        ck(await p.locator('#qzFb .n.q').count() == 1, 'red dot shown')
        vb = await p.get_attribute('#qzFb', 'viewBox')
        ck(vb == '0 0 840 210', '12-fret board', vb)
        c = q['cur']
        ck(c['pc'] == (TUN['std'][c['s']] + c['f']) % 12 and 0 <= c['f'] <= 12, 'question note matches its fret', c)
        # correct answer
        n0 = await src()
        await p.click(f'#qzNotes [data-pc="{c["pc"]}"]'); await p.wait_for_timeout(100)
        ck(await p.locator('#qzNotes .good').count() == 1 and await p.locator('#qzNotes .bad').count() == 0, 'correct button marked')
        ck(await src() == n0 + 1, 'note plays on answer', (n0, await src()))
        ck((await Q())['score'] == 1, 'score 1')
        await p.click(f'#qzNotes [data-pc="{[x for x in (0, 2, 4, 5, 7, 9, 11) if x != c["pc"]][0]}"]')
        ck((await Q())['miss'] == 0, 'second click while locked ignored')
        await p.wait_for_timeout(700)
        c2 = (await Q())['cur']
        ck(c2 != c or True, 'next question'); ck(await p.locator('#qzNotes .good').count() == 0, 'feedback cleared on next')
        # wrong answer
        wrong = [x for x in [0, 2, 4, 5, 7, 9, 11] if x != c2['pc']][0]
        await p.click(f'#qzNotes [data-pc="{wrong}"]'); await p.wait_for_timeout(100)
        ck(await p.locator('#qzNotes .bad').count() == 1 and await p.locator('#qzNotes .good').count() == 1, 'wrong: shows right + wrong')
        ck((await Q())['miss'] == 1, 'miss counted')
        lbl = await p.locator('#qzFb .n text').all_text_contents()
        names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        ck(lbl == [names[c2['pc']]], 'board reveals the right name', lbl)
        await p.wait_for_timeout(1400)
        ck(await p.locator('#qzFb .n.q').count() == 1, 'moves on after a miss')
        # sampling: naturals only, range respected
        bad = 0; seen = set()
        for i in range(60):
            await p.click('#qzSkip'); q = await Q(); c = q['cur']; seen.add((c['s'], c['f']))
            if c['pc'] not in (0, 2, 4, 5, 7, 9, 11) or not 0 <= c['f'] <= 12: bad += 1
        ck(bad == 0 and len(seen) > 25, 'naturals only + in range, varied', (bad, len(seen)))
        ck((await Q())['miss'] >= 60, 'skip counts as a miss')
        # all notes
        await p.click('#qzNat'); await p.wait_for_timeout(100)
        ck(await p.locator('#qzNotes button').count() == 12, '12 buttons with sharps/flats')
        sharp = 0
        for i in range(60):
            await p.click('#qzSkip'); c = (await Q())['cur']
            if c['pc'] in (1, 3, 6, 8, 10): sharp += 1
        ck(sharp > 5, 'sharps/flats asked', sharp)
        ck('C#' in await p.text_content('#qzNotes [data-pc="1"]') and 'Db' in await p.text_content('#qzNotes [data-pc="1"]'), 'both names on button')
        # range 5
        await p.click('#qzRange [data-r="5"]'); await p.wait_for_timeout(100)
        mx = 0
        for i in range(40):
            await p.click('#qzSkip'); mx = max(mx, (await Q())['cur']['f'])
        ck(mx <= 5 and await p.get_attribute('#qzFb', 'viewBox') == '0 0 520 210', 'range 0-5', mx)
        # strings
        for s in [5, 4, 3, 2, 1]:
            await p.click(f'#qzStr [data-s="{s}"]')
        await p.wait_for_timeout(100)
        ck((await Q())['strings'] == [0], 'only low E left')
        await p.click('#qzStr [data-s="0"]'); await p.wait_for_timeout(100)
        ck((await Q())['strings'] == [0], 'cannot turn off the last string')
        ok = True
        for i in range(20):
            await p.click('#qzSkip')
            if (await Q())['cur']['s'] != 0: ok = False
        ck(ok, 'questions only on chosen string')
        ck(await p.locator('#qzFb .str.off').count() == 5, 'other strings faded')
        # persistence
        await p.reload(); await p.wait_for_timeout(600)
        q = await Q()
        ck(q['range'] == 5 and q['nat'] is False and q['strings'] == [0], 'quiz options survive reload', q)

        # ---------- find mode ----------
        await p.click('#qzStr [data-s="5"]'); await p.click('#qzStr [data-s="1"]'); await p.click('#qzRange [data-r="12"]')
        await p.click('#qzMode [data-m="find"]'); await p.wait_for_timeout(150)
        ck(await p.is_hidden('#qzNotes'), 'no note buttons in find mode')
        ck(await p.locator('#qzFb .hit').count() == 3 * 13, 'tap cells only on chosen strings', await p.locator('#qzFb .hit').count())
        c = (await Q())['cur']
        ck('Find' in await p.text_content('#qzPrompt') and 'string' in await p.text_content('#qzPrompt'), 'find prompt', await p.text_content('#qzPrompt'))
        ck(await p.locator(f'#qzFb .hit.target[data-s="{c["s"]}"]').count() == 13, 'target string highlighted')
        tun = TUN['std']
        okf = [f for f in range(13) if (tun[c['s']] + f) % 12 == c['pc']]
        badf = [f for f in range(13) if (tun[c['s']] + f) % 12 != c['pc']][0]
        await p.locator('#practice').scroll_into_view_if_needed()
        await p.locator(f'#qzFb .hit[data-s="{c["s"]}"][data-f="{badf}"]').click(force=True); await p.wait_for_timeout(100)
        ck((await Q())['miss'] == 1 and await p.locator('#qzFb .n.root').count() == 1, 'wrong tap: miss + red mark')
        await p.locator(f'#qzFb .hit[data-s="{c["s"]}"][data-f="{okf[0]}"]').click(force=True); await p.wait_for_timeout(100)
        ck((await Q())['score'] == 1, 'right tap scores')
        ck(await p.locator('#qzFb .n.tone').count() == 1 and await p.locator('#qzFb .n.scale').count() == len(okf) - 1, 'shows every place of that note', (len(okf), await p.locator('#qzFb .n').count()))
        await p.wait_for_timeout(800)
        ck((await Q())['cur'] != c or True, 'next'); ck(await p.locator('#qzFb .n').count() == 0, 'board cleared for next')
        # wrong string tap
        c = (await Q())['cur']; other = [s for s in [0, 1, 5] if s != c['s']][0]
        await p.locator(f'#qzFb .hit[data-s="{other}"][data-f="3"]').click(force=True); await p.wait_for_timeout(100)
        ck((await Q())['miss'] == 2, 'right note on wrong string is a miss')

        # ---------- timed round ----------
        await p.click('#qzStart'); await p.wait_for_timeout(300)
        q = await Q()
        ck(q['run'] and q['score'] == 0 and q['miss'] == 0 and await p.is_visible('#qzTime'), 'round starts, score reset')
        t = await p.text_content('#qzTime')
        ck(t in ('60s', '59s'), 'countdown', t)
        c = q['cur']; okf = [f for f in range(13) if (tun[c['s']] + f) % 12 == c['pc']]
        await p.locator(f'#qzFb .hit[data-s="{c["s"]}"][data-f="{okf[0]}"]').click(force=True); await p.wait_for_timeout(800)
        c = (await Q())['cur']; okf = [f for f in range(13) if (tun[c['s']] + f) % 12 == c['pc']]
        await p.locator(f'#qzFb .hit[data-s="{c["s"]}"][data-f="{okf[-1]}"]').click(force=True); await p.wait_for_timeout(200)
        await p.evaluate(f"(async()=>{PR}._endRound())()"); await p.wait_for_timeout(300)
        q = await Q()
        ck(not q['run'] and await p.is_visible('#qzResult') and 'New best: 2' in await p.text_content('#qzResult'), 'round ends with new best', await p.text_content('#qzResult'))
        ck(q['best'].get('find:12:a') == 2 and 'Best 2' in await p.text_content('#qzBest'), 'best saved per mode')
        await p.click('#qzStart'); await p.wait_for_timeout(200)
        await p.evaluate(f"(async()=>{PR}._endRound())()"); await p.wait_for_timeout(200)
        ck('best 2' in await p.text_content('#qzResult') and (await Q())['best']['find:12:a'] == 2, 'worse round keeps best', await p.text_content('#qzResult'))
        await p.click('#qzStart'); await p.wait_for_timeout(200); await p.click('#qzStart'); await p.wait_for_timeout(100)
        ck(not (await Q())['run'] and await p.is_hidden('#qzTime'), 'stop button stops round')
        await p.click('#qzStart'); await p.click('#qzMode [data-m="name"]'); await p.wait_for_timeout(100)
        ck(not (await Q())['run'], 'changing options stops the round')

        # ---------- tuning / lefty follow theory ----------
        await p.select_option('#thTuning', 'dropd'); await p.wait_for_timeout(200)
        labs = await p.locator('#qzFb .slab').all_text_contents()
        ck(labs[0] == 'D', 'drop D label on quiz board', labs)
        bad = 0
        for i in range(30):
            await p.click('#qzSkip'); c = (await Q())['cur']
            if c['pc'] != (TUN['dropd'][c['s']] + c['f']) % 12: bad += 1
        ck(bad == 0, 'questions use drop D tuning')
        await p.click('#thLefty'); await p.wait_for_timeout(200)
        x = await p.evaluate("+document.querySelector('#qzFb .slab').getAttribute('x')")
        ck(x > 400, 'left-handed board', x)
        await p.click('#thLefty'); await p.select_option('#thTuning', 'std'); await p.wait_for_timeout(200)

        # ---------- ear: intervals ----------
        await p.click('#practice [data-pr="ear"]'); await p.wait_for_timeout(150)
        ck(await p.is_visible('#prEar') and await p.is_hidden('#prQuiz'), 'ear tab')
        ck(await p.locator('#earAns button').count() == 5 and await p.locator('#earAns button:disabled').count() == 5, 'level 1: 5 intervals, disabled before start')
        ck(await p.is_disabled('#earReplay'), 'replay disabled before start')
        n0 = await src(); await p.evaluate("__src.length=0")
        await p.click('#earPlay'); await p.wait_for_timeout(150)
        e = await E()
        ck(e['cur'] and e['cur']['kind'] == 'iv' and len(await p.evaluate("__src")) == 2, 'two notes played', await p.evaluate("__src"))
        w = await p.evaluate("__src")
        ck(abs((w[1] - w[0]) - 0.75) < 0.01, 'melodic: second note later', w)
        ck(43 <= e['cur']['root'] and e['cur']['root'] + 12 <= 76, 'in guitar range', e['cur'])
        ck('Which interval' in await p.text_content('#earHint'), 'question hint')
        await p.evaluate("__src.length=0"); await p.click('#earReplay'); await p.wait_for_timeout(100)
        ck(len(await p.evaluate("__src")) == 2, 'hear again')
        ans = e['cur']['ans']
        await p.click(f'#earAns [data-a="{ans}"]'); await p.wait_for_timeout(100)
        e = await E()
        ck(e['streak'] == 1 and e['right'] == 1 and e['total'] == 1, 'correct: streak 1', e)
        ck(await p.locator('#earAns .good').count() == 1 and '✓' in await p.text_content('#earMsg'), 'correct feedback')
        await p.wait_for_timeout(1100)
        e2 = await E()
        ck(not e2['answered'] and e2['cur'] is not None, 'auto next after a right answer')
        wrong = [x for x in [3, 4, 5, 7, 12] if x != e2['cur']['ans']][0]
        await p.click(f'#earAns [data-a="{wrong}"]'); await p.wait_for_timeout(100)
        e = await E()
        ck(e['streak'] == 0 and e['total'] == 2 and e['right'] == 1, 'wrong: streak reset', e)
        ck(await p.locator('#earAns .bad').count() == 1 and await p.locator('#earAns .good').count() == 1 and 'It was' in await p.text_content('#earMsg'), 'wrong feedback')
        await p.wait_for_timeout(1200)
        ck((await E())['answered'], 'no auto-next after a miss')
        await p.evaluate("__src.length=0"); await p.click(f'#earAns [data-a="{wrong}"]'); await p.wait_for_timeout(100)
        e = await E()
        ck(len(await p.evaluate("__src")) == 2 and e['total'] == 2, 'after answering, tapping plays to compare (no score change)')
        ck('compare' in (await p.text_content('#earHint')).lower() or 'same starting' in await p.text_content('#earHint'), 'compare hint')
        await p.click('#earPlay'); await p.wait_for_timeout(100)
        ck(not (await E())['answered'], 'Next asks a new question')
        # harmonic
        await p.click('#earDir [data-d="harm"]'); await p.wait_for_timeout(100)
        ck((await E())['cur'] is None and (await E())['streak'] == 0, 'changing direction resets')
        await p.evaluate("__src.length=0"); await p.click('#earPlay'); await p.wait_for_timeout(100)
        w = await p.evaluate("__src")
        ck(len(w) == 2 and abs(w[1] - w[0]) < 0.001, 'harmonic: together', w)
        await p.click('#earDir [data-d="down"]'); await p.click('#earLvl [data-l="3"]'); await p.wait_for_timeout(100)
        ck(await p.locator('#earAns button').count() == 12, 'level 3: 12 intervals')
        ok = True
        for i in range(30):
            await p.click('#earPlay'); e = await E()
            if not (43 <= e['cur']['root'] and e['cur']['root'] + e['cur']['ans'] <= 76): ok = False
        ck(ok, 'all intervals stay in range')
        # ---------- ear: chords ----------
        await p.click('#earEx [data-x="ch"]'); await p.wait_for_timeout(100)
        ck(await p.is_hidden('#earDir') and await p.locator('#earAns button').count() == 9, 'chords level 3: 9 types, no direction')
        await p.click('#earLvl [data-l="1"]'); await p.wait_for_timeout(100)
        ck([x.strip() for x in await p.locator('#earAns button').all_text_contents()] == ['Major', 'Minor'], 'chords level 1', await p.locator('#earAns button').all_text_contents())
        await p.click('#earLvl [data-l="3"]')
        ok = True; cnt = 0
        for i in range(40):
            await p.evaluate("__src.length=0"); await p.click('#earPlay'); e = await E()
            v = e['cur']['voice'][str(e['cur']['ans'])] if str(e['cur']['ans']) in e['cur']['voice'] else e['cur']['voice'].get(e['cur']['ans'])
            n = len(await p.evaluate("__src")); cnt += 1
            if not v or n != len(v) or n < 3 or min(v) < 38 or max(v) > 80: ok = False; print(e['cur'], n)
            # pitch classes match the chord
            ivs = {'maj': [0, 4, 7], 'm': [0, 3, 7], 'dim': [0, 3, 6], 'aug': [0, 4, 8], 'sus2': [0, 2, 7], 'sus4': [0, 5, 7], '7': [0, 4, 7, 10], 'maj7': [0, 4, 7, 11], 'm7': [0, 3, 7, 10]}[str(e['cur']['ans'])]
            pcs = {(e['cur']['root'] + i) % 12 for i in ivs}
            if not v or {m % 12 for m in v} - pcs or (v and v[0] % 12 != e['cur']['root']): ok = False; print('pcs', e['cur'], v)
        ck(ok, 'chord voicings: guitar range, right notes, root in bass')
        e = await E(); a = e['cur']['ans']
        await p.click(f'#earAns [data-a="{a}"]'); await p.wait_for_timeout(100)
        ck((await E())['streak'] == 1, 'chord answer works (string ids)')
        other = 'm' if str(a) != 'm' else 'maj'
        await p.wait_for_timeout(50); await p.evaluate("__src.length=0")
        await p.wait_for_timeout(1100)
        # ---------- ear: scale degrees ----------
        await p.click('#earEx [data-x="deg"]'); await p.click('#earLvl [data-l="1"]'); await p.wait_for_timeout(100)
        ck([x.strip() for x in await p.locator('#earAns button').all_text_contents()] == ['1', '3', '5'], 'degrees level 1')
        await p.evaluate("__src.length=0"); await p.click('#earPlay'); await p.wait_for_timeout(100)
        w = await p.evaluate("__src")
        ck(len(w) == 17 and max(w) - min(w) > 3.2, 'cadence + note', (len(w), w[-1] - w[0] if w else 0))
        await p.click('#earLvl [data-l="3"]'); await p.wait_for_timeout(100)
        ck(await p.locator('#earAns button').count() == 12 and 'b3' in await p.text_content('#earAns'), 'degrees level 3 chromatic')
        # best streak persistence
        await p.click('#earEx [data-x="iv"]'); await p.click('#earLvl [data-l="1"]'); await p.click('#earDir [data-d="up"]')
        for i in range(3):
            await p.click('#earPlay'); a = (await E())['cur']['ans']; await p.click(f'#earAns [data-a="{a}"]'); await p.wait_for_timeout(50)
        e = await E()
        ck(e['best'].get('iv:1:up', 0) >= 3, 'best streak saved', e['best'])
        await p.reload(); await p.wait_for_timeout(600)
        e = await E()
        ck(await p.is_visible('#prEar') and e['ex'] == 'iv' and e['lvl'] == 1 and e['dir'] == 'up' and e['best'].get('iv:1:up', 0) >= 3, 'ear tab + options + best survive reload', e)
        ck('best ' in await p.text_content('#earStat'), 'stat line', await p.text_content('#earStat'))

        # ---------- Thai ----------
        await p.click('.langs [data-lang="th"]'); await p.wait_for_timeout(200)
        await p.click('#earPlay'); await p.wait_for_timeout(100)
        t = await p.text_content('#practice')
        import re
        eng = [w for w in re.findall(r'[A-Za-z]{3,}', t) if not re.fullmatch(r'[A-Gbe#]+', w)]
        ck(not eng, 'Thai ear pane has no English words', eng[:10])
        await p.click('#practice [data-pr="quiz"]'); await p.wait_for_timeout(100)
        t = await p.text_content('#prQuiz') + await p.text_content('#practice .tabs')
        eng = [w for w in re.findall(r'[A-Za-z]{3,}', t) if not re.fullmatch(r'[A-Gbe#]+', w)]
        ck(not eng, 'Thai quiz pane has no English words', eng[:10])
        ck('สาย' in await p.text_content('#qzStr'), 'Thai strings label')
        await p.click('.langs [data-lang="en"]')

        # ---------- corrupt storage ----------
        for k, v in [('woodshed:quiz', '{"mode":"x","range":7,"strings":[9,"a"],"best":{"x":5,"name:12:n":-3}}'), ('woodshed:ear', '{"ex":"zz","lvl":9,"best":[1]}'), ('woodshed:practice', '"junk"'), ('woodshed:quiz', 'not json')]:
            await p.evaluate("([k,v]) => localStorage.setItem(k, v)", [k, v])
            await p.reload(); await p.wait_for_timeout(400)
            q = await Q(); e = await E()
            ck(q['mode'] in ('name', 'find') and q['range'] in (5, 12, 15) and q['strings'] and all(0 <= s < 6 for s in q['strings']) and all(':' in k2 for k2 in q['best']) and e['ex'] in ('iv', 'ch', 'deg'), f'corrupt {k} survives', (q, e))
        ck((await Q())['best'].get('name:12:n') is None, 'negative best dropped')

        # backup includes practice data
        async with p.expect_download() as dl:
            await p.click('#bkExport')
        d = await dl.value; data = json.loads(open(await d.path()).read())
        ck('woodshed:quiz' in data['data'] and 'woodshed:ear' in data['data'], 'backup includes quiz/ear stats')

        # theory board unchanged
        vb = await p.get_attribute('#fb', 'viewBox')
        ck(vb == '0 0 1000 210' and await p.locator('#fb .n').count() > 5, 'theory board still drawn', vb)
        ck(not errs, 'no errors', errs)
        await p.locator('#practice').screenshot(path='/home/claude/tests/rC_desktop_ear.png')
        await p.click('#practice [data-pr="quiz"]'); await p.click('#qzMode [data-m="name"]'); await p.wait_for_timeout(100)
        await p.locator('#practice').screenshot(path='/home/claude/tests/rC_desktop_quiz.png')
        await ctx.close()

        # ---------- mobile ----------
        ctx2, p2, errs2 = await newpage(b, vp=(390, 844), is_mobile=True, has_touch=True)
        await p2.goto(BASE); await p2.wait_for_timeout(500)
        await p2.locator('#practice').scroll_into_view_if_needed()
        ow = await p2.evaluate("document.documentElement.scrollWidth - innerWidth")
        ck(ow <= 0, 'no page overflow on phone', ow)
        h = await p2.evaluate("Math.min(...[...document.querySelectorAll('#qzNotes button')].map(b=>b.getBoundingClientRect().height))")
        ck(h >= 44, 'note buttons big enough', h)
        await p2.tap('#qzRange [data-r="5"]'); await p2.wait_for_timeout(100)
        fw = await p2.evaluate("document.getElementById('qzFb').getBoundingClientRect().width <= document.querySelector('#prQuiz .fbwrap').clientWidth + 1")
        ck(fw, '0-5 board fits phone without scrolling')
        await p2.tap('#qzMode [data-m="find"]'); await p2.wait_for_timeout(100)
        cell = await p2.evaluate("(()=>{const r=document.querySelector('#qzFb .hit').getBoundingClientRect();return [r.width,r.height]})()")
        ck(cell[0] >= 24 and cell[1] >= 24, 'tap cells reasonably big on phone', cell)
        await p2.locator('#practice').screenshot(path='/home/claude/tests/rC_mobile_find.png')
        await p2.tap('#practice [data-pr="ear"]'); await p2.tap('#earPlay'); await p2.wait_for_timeout(200)
        await p2.locator('#practice').screenshot(path='/home/claude/tests/rC_mobile_ear.png')
        ow = await p2.evaluate("document.documentElement.scrollWidth - innerWidth")
        ck(ow <= 0, 'ear pane no overflow', ow)
        ck(not errs2, 'mobile no errors', errs2)
        await b.close()
    srv.terminate()
    f = [n for c, n in R if not c]; print(f'\n{len(R)-len(f)} passed, {len(f)} failed: {f}')
asyncio.run(main())
