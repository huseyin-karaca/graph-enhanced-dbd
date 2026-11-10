#!/usr/bin/env python3
"""Typeset every formula of the figure with LaTeX and convert to path-only SVG (dvisvgm --no-fonts)."""
import json, os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'art')
Y = r'\hat{\boldsymbol{y}}'
F = {
  'pij': r'(p_i - p_j)',
  'nij': r'(n_i - p_j)',
  'tok_p': r'([t_i^p],\,[t_j^p])',
  'tok_n': r'([t_i^n],\,[t_j^p])',
  'seq_i': r'[t_{i0}^p,\, t_{i1}^p,\, \ldots,\, t_{in}^p]',
  'seq_j': r'[t_{j0}^p,\, t_{j1}^p,\, \ldots,\, t_{jn}^p]',
  'cls': r'[\mathit{CLS}]',
  'hi': r'h_i^p',
  'hj': r'h_j^p',
  'pos1': Y + r' = \mathrm{sim}(h_i^p,\, h_j^p)',
  'pos2': r'\mathcal{L}_{\mathrm{BCE}}(1,\, ' + Y + r') = -\left[\log(' + Y + r')\right]',
  'neg1': Y + r' = \mathrm{sim}(h_i^n,\, h_j^p)',
  'neg2': r'\mathcal{L}_{\mathrm{BCE}}(0,\, ' + Y + r') = -\left[\log(1 - ' + Y + r')\right]',
}
sizes = {}
for k, f in F.items():
    d = tempfile.mkdtemp()
    open(f'{d}/m.tex', 'w').write(r'\documentclass[10pt]{article}\usepackage{amsmath,amssymb,bm}\pagestyle{empty}\begin{document}$' + f + r'$\end{document}')
    subprocess.run(['latex', '-interaction=batchmode', '-output-directory', d, f'{d}/m.tex'], check=True, stdout=subprocess.DEVNULL)
    out = f'{OUT}/m_{k}.svg'
    subprocess.run(['dvisvgm', '--no-fonts', '--exact-bbox', '-o', out, f'{d}/m.dvi'], check=True, stderr=subprocess.DEVNULL)
    s = open(out).read()
    m = re.search(r'viewBox=[\'"]([\d.\- ]+)[\'"]', s)
    _, _, w, h = map(float, m.group(1).split())
    sizes[k] = {'w': w, 'h': h}
json.dump(sizes, open(f'{OUT}/math.json', 'w'), indent=1)
print(sizes)
