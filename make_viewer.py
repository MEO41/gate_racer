"""Embed a rollout (traj.json) into the 3D viewer -> a single self-contained HTML file.

    python make_viewer.py traj.json viewer.html
"""
import json, sys

src = sys.argv[1] if len(sys.argv) > 1 else 'traj.json'
out = sys.argv[2] if len(sys.argv) > 2 else 'viewer.html'
data = json.load(open(src))
html = open('viewer_template.html').read().replace(
    '/*__TRAJ__*/', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
open(out, 'w').write(html)
print('wrote', out)
