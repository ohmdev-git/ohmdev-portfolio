import asyncio, subprocess, sys, time, json, os
from playwright.async_api import async_playwright
from common import wire
SITE = '/home/claude/buildA'
BASE = 'http://127.0.0.1:8775/'
R = []
def ck(c, n, x=''):
    R.append((bool(c), n)); print('PASS' if c else 'FAIL', n, '' if c else x)
def build(v): subprocess.run([sys.executable, '/home/claude/ws2/build.py', SITE, v], check=True, capture_output=True)

async def newpage(b, vp=(1400, 950), **kw):
    ctx = await b.new_context(viewport={'width': vp[0], 'height': vp[1]}, accept_downloads=True, **kw)
    await wire(ctx)
    p = await ctx.new_page(); errs = []
    p.on('pageerror', lambda e: errs.append(str(e)))
    return ctx, p, errs

SEED = """() => { localStorage.clear();
  const S = (k, v) => localStorage.setItem('woodshed:' + k, JSON.stringify(v));
  S('playlist', [{id:'aaaaaaaaaa1',at:1,done:false},{id:'bbbbbbbbbb2',at:2,done:true}]);
  S('recent', [{id:'aaaaaaaaaa1',at:Date.now()}]);
  S('titles', {aaaaaaaaaa1:{t:'Song A',a:'X',at:1}});
  S('sections:aaaaaaaaaa1', [{id:'s1',n:'Solo',a:10,b:14,s:0.75,note:'Am F C G'},{id:'s2',n:'Riff',a:20,b:22,s:1,note:''}]);
  S('markers:aaaaaaaaaa1', [{id:'m1',t:5,n:'Intro'}]);
  S('bpm:aaaaaaaaaa1', {bpm:98,beats:4}); S('key:aaaaaaaaaa1', {r:9,m:'minor'});
  S('lang', 'en'); S('last', 'aaaaaaaaaa1'); }"""

async def main():
    build('A1')
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', '8775', '--bind', '127.0.0.1', '-d', SITE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1)
    async with async_playwright() as pw:
        b = await pw.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--no-sandbox'])
        ctx, p, errs = await newpage(b)
        T = lambda: p.text_content('#toast')
        await p.goto(BASE); await p.evaluate(SEED); await p.reload(); await p.wait_for_timeout(800)
        # ---------- focus fix ----------
        await p.wait_for_selector('.screen iframe')
        ib = await p.locator('.screen iframe').bounding_box()
        await p.mouse.click(ib['x'] + ib['width'] / 2, ib['y'] + ib['height'] / 2); await p.wait_for_timeout(50)
        await p.wait_for_timeout(150)
        ae = await p.evaluate('document.activeElement.tagName')
        await p.keyboard.press('k'); await p.wait_for_timeout(100)
        ck(ae != 'IFRAME' and await p.locator('#metBtn.on').count() == 1, 'shortcuts work after clicking the video', ae)
        await p.keyboard.press('k')
        # ---------- export ----------
        async with p.expect_download() as dl:
            await p.click('#bkExport')
        d = await dl.value
        path = await d.path(); fname = d.suggested_filename
        j = json.load(open(path))
        ck(fname.startswith('woodshed-backup-') and fname.endswith('.json'), 'backup file name', fname)
        ck(j['app'] == 'woodshed' and j['schema'] == 1 and all(k.startswith('woodshed:') and isinstance(v, str) for k, v in j['data'].items()), 'backup format')
        ck(json.loads(j['data']['woodshed:sections:aaaaaaaaaa1'])[0]['note'] == 'Am F C G' and 'woodshed:key:aaaaaaaaaa1' in j['data'], 'backup has notes + key')
        open('/tmp/bk.json', 'w').write(json.dumps(j))
        # ---------- restore into an empty browser ----------
        c2, q, e2 = await newpage(b)
        await q.goto(BASE); await q.evaluate('localStorage.clear()'); await q.reload(); await q.wait_for_timeout(500)
        await q.set_input_files('#bkFile', '/tmp/bk.json'); await q.wait_for_timeout(200)
        ck(await q.is_visible('#bkPreview'), 'preview shown')
        cells = await q.eval_on_selector_all('#bkPreview tbody tr', 'r=>r.map(x=>[...x.cells].map(c=>c.textContent))')
        ck(cells[0][1:] == ['2', '0'] and cells[1][1:] == ['2', '0'] and cells[2][1:] == ['1', '0'] and cells[3][1:] == ['1', '0'] and cells[4][1:] == ['0', '0'] and cells[5][1:] == ['2', '0'], 'preview counts', cells)
        await q.click('#bkPreview [data-bk="cancel"]'); ck(await q.is_hidden('#bkPreview'), 'cancel hides preview')
        await q.set_input_files('#bkFile', '/tmp/bk.json'); await q.wait_for_timeout(200)
        async with q.expect_navigation():
            await q.click('#bkPreview [data-bk="merge"]')
        await q.wait_for_timeout(800)
        ck('merged' in await q.text_content('#toast'), 'merged toast after reload', await q.text_content('#toast'))
        ck(await q.locator('#plList .pl').count() == 2 and await q.locator('.sec').count() == 2 and await q.locator('#mkList .mk').count() == 1, 'data restored and shown')
        ck(await q.input_value('#bpmIn') == '98', 'bpm restored')
        same = await q.evaluate("(()=>{const o={};for(let i=0;i<localStorage.length;i++){const k=localStorage.key(i);if(k.startsWith('woodshed:'))o[k]=localStorage.getItem(k)};return o})()")
        bkd = j['data']
        ck(all(same.get(k) == v for k, v in bkd.items() if not k.endswith(':recent') and not k.endswith(':last')), 'restored values identical', [k for k, v in bkd.items() if same.get(k) != v])
        # ---------- merge with conflicts ----------
        await q.evaluate("""()=>{const S=(k,v)=>localStorage.setItem('woodshed:'+k,JSON.stringify(v));
          S('playlist',[{id:'ccccccccc_3',at:9,done:false}]); S('lang','th');
          S('sections:aaaaaaaaaa1',[{id:'s1',n:'Solo EDITED',a:11,b:15,s:0.8,note:'mine'}]); localStorage.removeItem('woodshed:markers:aaaaaaaaaa1'); }""")
        await q.reload(); await q.wait_for_timeout(500)
        await q.set_input_files('#bkFile', '/tmp/bk.json'); await q.wait_for_timeout(200)
        async with q.expect_navigation():
            await q.click('#bkPreview [data-bk="merge"]')
        await q.wait_for_timeout(600)
        st = await q.evaluate("(()=>{const g=k=>JSON.parse(localStorage.getItem('woodshed:'+k));return {pl:g('playlist').map(x=>x.id),sec:g('sections:aaaaaaaaaa1'),lang:g('lang'),mk:g('markers:aaaaaaaaaa1')}})()")
        ck(st['pl'] == ['ccccccccc_3', 'aaaaaaaaaa1', 'bbbbbbbbbb2'], 'merge: playlist keeps local order + adds', st['pl'])
        ck(st['sec'][0]['n'] == 'Solo EDITED' and len(st['sec']) == 2 and st['sec'][1]['id'] == 's2', 'merge: local edits win, missing parts added', st['sec'])
        ck(st['lang'] == 'th' and st['mk'] and st['mk'][0]['n'] == 'Intro', 'merge: settings kept, markers filled', st)
        ck(await q.evaluate('document.documentElement.lang') == 'th' and 'รวมข้อมูล' in await q.text_content('#toast'), 'merge toast in Thai')
        # ---------- replace ----------
        await q.evaluate("localStorage.setItem('woodshed:markers:zzzzzzzzzz9', JSON.stringify([{id:'x',t:1,n:'only here'}]))")
        await q.set_input_files('#bkFile', '/tmp/bk.json'); await q.wait_for_timeout(200)
        await q.click('#bkPreview [data-bk="replace"]'); await q.wait_for_timeout(100)
        ck(await q.locator('#bkPreview').is_visible() and 'แน่ใจ' in await q.text_content('#bkPreview [data-bk="replace"]'), 'replace asks twice')
        async with q.expect_navigation():
            await q.click('#bkPreview [data-bk="replace"]')
        await q.wait_for_timeout(600)
        same = await q.evaluate("(()=>{const o={};for(let i=0;i<localStorage.length;i++){const k=localStorage.key(i);if(k.startsWith('woodshed:'))o[k]=localStorage.getItem(k)};return o})()")
        extra = [k for k in same if k not in bkd and not k.endswith(':recent') and not k.endswith(':last') and not k.endswith(':titles')]
        ck(not extra and all(same.get(k) == v for k, v in bkd.items() if not k.endswith(':recent') and not k.endswith(':last') and not k.endswith(':titles')), 'replace makes an exact copy', extra)
        ck(await q.evaluate('document.documentElement.lang') == 'en', 'replace brought back backup language')
        # ---------- bad files ----------
        await q.evaluate("localStorage.setItem('woodshed:keep','\"me\"')")
        bad = {'notjson.json': 'hello', 'other.json': json.dumps({'app': 'x', 'data': {}}), 'newer.json': json.dumps({'app': 'woodshed', 'schema': 2, 'data': {'woodshed:a': '1'}}),
               'empty.json': json.dumps({'app': 'woodshed', 'schema': 1, 'data': {}}), 'wrongkeys.json': json.dumps({'app': 'woodshed', 'schema': 1, 'data': {'evil': '1', 'woodshed:x': 'not json{'}})}
        want = {'notjson.json': 'isn’t a Woodshed', 'other.json': 'isn’t a Woodshed', 'newer.json': 'newer version', 'empty.json': 'no Woodshed data', 'wrongkeys.json': 'no Woodshed data'}
        for n, body in bad.items():
            open('/tmp/' + n, 'w').write(body)
            await q.set_input_files('#bkFile', '/tmp/' + n); await q.wait_for_timeout(150)
            ck(want[n] in await q.text_content('#toast') and await q.is_hidden('#bkPreview'), 'rejects ' + n, await q.text_content('#toast'))
        mixed = dict(j); mixed['data'] = dict(j['data']); mixed['data']['evil'] = 'x'; mixed['data']['woodshed:broken'] = '{nope'
        open('/tmp/mixed.json', 'w').write(json.dumps(mixed))
        await q.set_input_files('#bkFile', '/tmp/mixed.json'); await q.wait_for_timeout(150)
        ck('2 unreadable' in await q.text_content('#bkPreview'), 'skips bad entries, warns', await q.text_content('#bkPreview'))
        await q.click('#bkPreview [data-bk="cancel"]')
        big = '/tmp/big.json'; open(big, 'w').write(' ' * (5 * 1024 * 1024 + 10))
        await q.set_input_files('#bkFile', big); await q.wait_for_timeout(150)
        ck('too big' in await q.text_content('#toast'), 'rejects huge file')
        ck(await q.evaluate("localStorage.getItem('woodshed:keep')") == '"me"', 'bad files changed nothing')
        # ---------- storage full during restore -> rolled back ----------
        before = await q.evaluate("(()=>{const o={};for(let i=0;i<localStorage.length;i++){const k=localStorage.key(i);o[k]=localStorage.getItem(k)};return JSON.stringify(o)})()")
        full = dict(j); full['data'] = dict(j['data']); full['data']['woodshed:markers:qqqqqqqqqq7'] = json.dumps([{'id': 'q', 't': 1, 'n': 'boom'}])
        full['data']['woodshed:sections:qqqqqqqqqq7'] = json.dumps([{'id': 'q1', 'n': 'x', 'a': 1, 'b': 2, 's': 1}])
        open('/tmp/full.json', 'w').write(json.dumps(full))
        await q.evaluate("""()=>{const o=Storage.prototype.setItem; Storage.prototype.setItem=function(k,v){ if(k==='woodshed:markers:qqqqqqqqqq7') throw new DOMException('full','QuotaExceededError'); return o.call(this,k,v);} }""")
        await q.set_input_files('#bkFile', '/tmp/full.json'); await q.wait_for_timeout(150)
        await q.click('#bkPreview [data-bk="merge"]'); await q.wait_for_timeout(300)
        after = await q.evaluate("(()=>{const o={};for(let i=0;i<localStorage.length;i++){const k=localStorage.key(i);o[k]=localStorage.getItem(k)};return JSON.stringify(o)})()")
        ck('storage' in await q.text_content('#toast') and json.loads(before) == json.loads(after), 'storage full -> nothing changed', await q.text_content('#toast'))
        ck(not e2, 'no errors in restore page', e2)
        # ---------- app: manifest, icons, service worker ----------
        r = await p.request.get(BASE + 'manifest.webmanifest')
        man = await r.json()
        ck(man['start_url'] == './' and man['display'] == 'standalone' and len(man['icons']) == 3, 'manifest ok')
        for ic in man['icons'] + [{'src': 'icons/apple-touch-icon.png'}]:
            rr = await p.request.get(BASE + ic['src']); body = await rr.body()
            ck(rr.status == 200 and body[:4] == b'\x89PNG', 'icon ' + ic['src'])
        ctl = await p.evaluate("(async()=>{await navigator.serviceWorker.ready; return !!navigator.serviceWorker.controller})()")
        if not ctl: await p.reload(); await p.wait_for_timeout(800)
        ck(await p.evaluate("!!navigator.serviceWorker.controller"), 'service worker controls the page')
        keys = await p.evaluate("caches.keys()")
        n = await p.evaluate("(async()=>{const c=await caches.open('woodshed-A1');return (await c.keys()).length})()")
        ck(keys == ['woodshed-A1'] and n >= 20, 'everything precached', (keys, n))
        # offline
        await ctx.set_offline(True)
        await p.reload(); await p.wait_for_timeout(1200)
        ck(await p.title() and await p.locator('#theory .vbox').count() > 0 and await p.text_content('#thName') != '', 'works offline (theory loads from cache)')
        await p.click('#theory [data-th="scales"]'); await p.wait_for_timeout(100)
        ck(await p.locator('#fb g.n').count() > 20, 'scales offline')
        ck(not [e for e in errs if 'TypeError' in e or 'ReferenceError' in e], 'no crashes offline', errs)
        await ctx.set_offline(False)
        # new version replaces old cache
        build('A2')
        await p.reload(); await p.wait_for_timeout(1500)
        await p.reload(); await p.wait_for_timeout(1000)
        src = await p.evaluate("[...document.scripts].map(s=>s.src).find(s=>s.includes('main.js'))")
        keys = await p.evaluate("caches.keys()")
        ck('v=A2' in src and keys == ['woodshed-A2'], 'new version served, old cache deleted', (src, keys))
        # install button
        await p.evaluate("""()=>{const e=new Event('beforeinstallprompt'); e.prompt=()=>{window.__prompted=true}; e.userChoice=Promise.resolve({outcome:'accepted'}); window.dispatchEvent(e);}""")
        await p.wait_for_timeout(50)
        ck(await p.is_visible('#appInstall'), 'install button appears when the browser allows')
        await p.click('#appInstall'); await p.wait_for_timeout(100)
        ck(await p.evaluate('window.__prompted') and await p.is_hidden('#appInstall'), 'install prompt shown')
        c3 = await b.new_context(user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1', viewport={'width': 390, 'height': 844})
        await wire(c3); i = await c3.new_page(); await i.goto(BASE); await i.wait_for_timeout(500)
        ck('Add to Home Screen' in await i.text_content('#appHint'), 'iPhone install hint')
        for w in (320, 390, 768):
            await i.set_viewport_size({'width': w, 'height': 800}); await i.wait_for_timeout(100)
            ck(await i.evaluate('document.documentElement.scrollWidth') <= w, f'no overflow {w}')
        await i.evaluate("document.getElementById('dataCard').scrollIntoView()")
        await i.locator('#dataCard').screenshot(path='rA_data.png')
        ck(not errs, 'no page errors', errs)
        await b.close()
    srv.terminate()
    f = [n for c, n in R if not c]
    print(f'\n{len(R)-len(f)} passed, {len(f)} failed: {f}')
asyncio.run(main())
