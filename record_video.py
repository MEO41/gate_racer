"""Render viewer.html frame by frame into an .mp4: chase camera (left) + onboard drone camera (right).

Needs Playwright + Chromium and ffmpeg. Frames are rendered at fixed sim-time steps, so the
video is smooth regardless of how fast the machine renders.

    python record_video.py --html viewer.html --drone 0 --out race.mp4
    python record_video.py --out race.mp4 --slow_out race_slowmo.mp4   # also a 0.5x copy
"""
import argparse, asyncio, json, os, re, shutil, subprocess, tempfile
from playwright.async_api import async_playwright

W, H = 1920, 1080

LAYOUT_CSS = """
body { grid-template-columns: 1080px 840px !important; grid-template-rows: 1080px !important;
       grid-template-areas: "stage col" !important; }
.bar, #secInput, .note { display: none !important; }
#col { padding: 22px 48px !important; gap: 18px !important; align-content: center !important; }
.camwrap { width: 560px !important; margin: 0 auto; }
#errPlot { height: 92px !important; }
.tag { position: absolute; left: 16px; bottom: 16px; font: 600 13px "IBM Plex Sans", sans-serif;
       letter-spacing: .08em; text-transform: uppercase; color: #e6e8ee;
       background: rgba(25,27,32,.86); border: 1px solid #343944; border-radius: 6px; padding: 8px 12px; }
.tag b { text-transform: none; font-family: "IBM Plex Mono", monospace; font-weight: 500; letter-spacing: 0; margin-left: 10px; color: #959cab; }
"""


async def main(a):
    data = json.loads(re.search(r'<script id="traj" type="application/json">(.*?)</script>',
                                open(a.html).read(), re.S).group(1))
    n_steps = len(data['paths'][a.drone])
    dt = data['dt']
    fps = a.fps
    step = (1.0 / fps) / dt * a.slowmo           # sim steps per video frame
    frames = int((n_steps - 1) / step)
    if a.max_frames:
        frames = min(frames, a.max_frames)
    tmp = tempfile.mkdtemp(prefix='frames_')
    print(f'{frames} frames for {n_steps * dt:.1f} s of flight')

    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'])
        pg = await b.new_page(viewport={'width': W, 'height': H})
        def serve(path):
            return lambda route: route.fulfill(path=path)
        for pat, path in a.routes:
            await pg.route(pat, serve(path))
        if a.font_css:
            await pg.route('**/fonts.googleapis.com/**',
                           lambda r: r.fulfill(body=a.font_css, content_type='text/css'))
        html = open(a.html).read()
        await pg.set_content('<!doctype html><meta charset=utf-8>' + html, wait_until='load')
        await pg.add_style_tag(content=LAYOUT_CSS)
        await pg.evaluate("""() => {
            const tag = document.createElement('div'); tag.className = 'tag';
            tag.innerHTML = 'Chase view<b id="tagClock">0.00 s</b>';
            document.getElementById('stage').appendChild(tag);
        }""")
        await pg.evaluate(f"""() => {{
            document.getElementById('play').click();             // pause the live loop
            const f = document.getElementById('focus'); f.value = '{a.drone}'; f.onchange();
            document.getElementById('cam').value = '{a.camera}';
        }}""")
        await pg.evaluate("document.fonts.ready")
        # let the chase camera settle behind the drone before the first frame
        await pg.evaluate(f"() => {{ t = {a.start}; }}")
        await pg.wait_for_timeout(2500)
        for i in range(frames):
            tt = a.start + i * step
            await pg.evaluate(f"""() => new Promise(res => {{
                t = {tt};
                document.getElementById('tagClock').textContent = ({tt} * {dt}).toFixed(2) + ' s';
                requestAnimationFrame(() => requestAnimationFrame(res));
            }})""")
            await pg.screenshot(path=os.path.join(tmp, f'{i:05d}.png'))
            if i % 100 == 0:
                print(f'  frame {i}/{frames}', flush=True)
        await b.close()

    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps),
                    '-i', os.path.join(tmp, '%05d.png'), '-c:v', 'libx264', '-preset', 'slow',
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.out], check=True)
    if a.slow_out:   # same frames played at half the rate = 0.5x slow motion
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps // 2),
                        '-i', os.path.join(tmp, '%05d.png'), '-c:v', 'libx264', '-preset', 'slow',
                        '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.slow_out], check=True)
        print('wrote', a.slow_out)
    if a.keep_frames:
        print('frames kept in', tmp)
    else:
        shutil.rmtree(tmp)
    print('wrote', a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--html', default='viewer.html')
    ap.add_argument('--drone', type=int, default=0)
    ap.add_argument('--camera', default='chase', choices=['chase', 'orbit', 'fpv'])
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--slowmo', type=float, default=1.0, help='0.5 = half speed')
    ap.add_argument('--start', type=float, default=0.0, help='first sim step')
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--keep_frames', action='store_true')
    ap.add_argument('--out', default='race.mp4')
    ap.add_argument('--slow_out', default=None, help='also write a 0.5x slow-motion copy')
    # offline rendering: serve CDN scripts / fonts from local files (pattern=path)
    ap.add_argument('--route', action='append', default=[])
    ap.add_argument('--font_css_file', default=None)
    a = ap.parse_args()
    a.routes = [r.split('=', 1) for r in a.route]
    a.font_css = open(a.font_css_file).read() if a.font_css_file else None
    asyncio.run(main(a))
