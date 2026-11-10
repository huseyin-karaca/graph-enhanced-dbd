import * as D from '../engine/cs.js';
// dbd-overview, restyled to the HIT-ASR / HEFS hero design language (slate ink, saturated hue strokes on pale tints,
// thin 0.6 pt rules, rounded white or tinted cards, small filled arrowheads, bold slate headers). Same content and layout as figure.js.
export const figure = { width: 149, height: 98, name: 'dbd-overview-v2', format: 'journal' };

// Formulas are typeset by LaTeX (gen_math.py -> art/m_*.svg, path-only, filled slate) and placed as vector art.
export default async function draw(F, L) {
  const R = F.root;
  const INK = '#1c2230', INK2 = '#434a5a', MUTED = '#6b7180', LINE = '#c3c7cf';
  const BLUE = '#3355d6', BLUE_T = '#f0f3fd', BLUE_S = '#dbe3fb';
  const TEAL = '#0a9a8a', TEAL_T = '#e6f5f3', TEAL_S = '#c9ebe6', TEAL_D = '#067568';
  const ORG = '#e08a00', ORG_T = '#fdf3e1', ORG_S = '#fbe0b3', ORG_D = '#a35f00';
  const PUR = '#8e63c7', PUR_T = '#f3edfa', PUR_S = '#e1d4f3';
  const SZ = await (await fetch('/figure/art/math.json')).json();
  const S = 0.82;
  const T = (x, y, s, o = {}) => L.text(R, x, y, s, { size: 6, anchor: 'middle', valign: 'middle', color: INK2, ...o });
  const M = (k, x, y, o = {}) => {
    const s = (o.scale ?? 1) * S, w = SZ[k].w * s, h = SZ[k].h * s;
    return L.svgFile(R, `/figure/art/m_${k}.svg`, x - w / 2, y - h / 2, w, { h });
  };
  const arrow = (a, b, o = {}) => D.route(R, [a, b], { head: 'fill', stroke: MUTED, width: 0.6, headLength: 3.2, headWidth: 3.2, ...o });
  const box = (x, y, w, h, fill, stroke, r = 4, width = 0.6, o = {}) => L.rect(R, x, y, w, h, { fill, stroke, width, r, ...o });
  const W = 422;

  // ---- section headers: bold slate, unnumbered ----------------------------------------------------
  const header = (y, s) => T(6, y, s, { size: 7.5, weight: 'bold', anchor: 'start', color: INK2 });
  header(10, 'Data preprocessing');
  L.line(R, [6, 118], [W - 6, 118], { stroke: LINE, width: 0.5 });
  header(130, 'Embedding construction and training');

  // ---- 1) preprocessing ------------------------------------------------------------------------
  const cy = 58;
  const gray = L.linearGradient(R, [[0, '#f3f4f7'], [1, '#dcdfe6']], 90);
  // file icon with the GitHub mark
  L.path(R, `M8 ${cy - 24} H25 L36 ${cy - 13} V${cy + 12} Q36 ${cy + 14} 34 ${cy + 14} H10 Q8 ${cy + 14} 8 ${cy + 12} Z`,
    { fill: gray, stroke: '#878c98', width: 0.6, join: 'round' });
  L.path(R, `M25 ${cy - 24} V${cy - 15} Q25 ${cy - 13} 27 ${cy - 13} H36`, { fill: '#dcdfe6', stroke: '#878c98', width: 0.6, join: 'round' });
  const gh = L.group(R, { x: 14.5, y: cy - 4.5, scale: 0.6 });
  L.path(gh.node, 'M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.921.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12', { fill: INK2 });
  T(22, cy + 25, 'Issues');
  arrow([40, cy - 2], [54, cy - 2]);

  // table (teal: the labelled issue data)
  const tx = 58, tw = 79;
  box(tx, cy - 26, tw, 52, TEAL_T, TEAL, 3);
  L.rect(R, tx, cy - 26, tw, 11, { fill: TEAL_S, stroke: 'none', r: 3 });
  L.rect(R, tx, cy - 19, tw, 4, { fill: TEAL_S, stroke: 'none' });
  const cols = [[tx + 12, 'Issue ID', 3], [tx + 39.5, 'Duplicate ID', 3], [tx + 66, 'Issue', 4]];
  for (const [x, s, n] of cols) {
    T(x, cy - 20, s, { size: 5, weight: 'bold', color: TEAL_D });
    for (let r = 0; r < 4; r++) for (let k = 0; k < n; k++) L.circle(R, x - (n - 1) * 3 + k * 6, cy - 8 + r * 8, 0.9, { fill: TEAL });
  }
  T(tx + tw / 2, cy + 36, 'Duplicate / Non-duplicate');
  arrow([141, cy - 2], [153, cy - 2]);

  // pair preparation (blue frame, as the router frame in HIT-ASR; positive = teal, negative = orange)
  const px = 156, pw = 80;
  box(px, cy - 30, pw, 60, BLUE_T, BLUE, 4, 0.8);
  box(px + 7, cy - 23, pw - 14, 22, '#fff', TEAL, 3);
  T(px + pw / 2, cy - 12, 'Positive pair matching');
  box(px + 7, cy + 4, pw - 14, 22, '#fff', ORG, 3);
  T(px + pw / 2, cy + 15, 'Negative sampling');
  T(px + pw / 2, cy + 40, 'Pair preparation');

  // pairs -> tokenizer
  arrow([px + pw, cy - 14], [274, cy - 14], { stroke: TEAL });
  arrow([px + pw, cy + 14], [274, cy + 14], { stroke: ORG });
  await M('pij', 255, cy - 24);
  await M('nij', 255, cy + 24);

  // tokenizer (Hugging Face emoji kept; frozen-encoder grey card)
  L.rect(R, 276, cy - 22, 64, 44, { fill: gray, stroke: '#878c98', width: 0.6, r: 6 });
  T(291, cy + 0.5, '🤗', { size: 10 });
  T(299, cy, 'Tokenizers', { weight: 'bold', anchor: 'start', color: INK });
  T(308, cy + 36, 'AutoTokenizer');
  arrow([342, cy], [356, cy]);

  // token ids
  box(360, cy - 22, 56, 44, BLUE_T, BLUE, 4);
  await M('tok_p', 388, cy - 9);
  await M('tok_n', 388, cy + 9);
  T(388, cy + 36, 'Token IDs');

  // ---- 2) embedding + training ----------------------------------------------------------------
  function encoder(x0, y0, key, hue, tint) {
    const xs = [x0, x0 + 17, x0 + 34, x0 + 51], ys = [y0, y0 + 15, y0 + 30];
    for (let c = 0; c < 3; c++) for (const a of ys) for (const b of ys)
      L.line(R, [xs[c], a], [xs[c + 1], b], { stroke: LINE, width: 0.4 });
    xs.forEach((x, c) => ys.forEach((y) => {
      if (c === 0) L.rect(R, x - 3.2, y - 4.4, 6.4, 8.8, { fill: '#e9ebf0', stroke: MUTED, width: 0.6, r: 1.2 });
      else if (c === 3) L.rect(R, x - 3.2, y - 4.4, 6.4, 8.8, { fill: tint, stroke: hue, width: 0.6, r: 1.2 });
      else L.ellipse(R, x, y, 3.6, 4.8, { fill: '#fff', stroke: hue, width: 0.6 });
    }));
    M(key, x0 + 25.5, y0 - 12);
    return { out: xs[3] + 3.2 };
  }
  const x0 = 26;
  const e1 = encoder(x0, 168, 'seq_i', BLUE, BLUE_S);
  const e2 = encoder(x0, 226, 'seq_j', PUR, PUR_S);
  arrow([e1.out + 5, 183], [e1.out + 19, 183]);
  arrow([e2.out + 5, 241], [e2.out + 19, 241]);

  const cx = e1.out + 25;                           // CLS bars
  const cls = (y, fill, stroke) => {
    for (let i = 0; i < 3; i++) L.rect(R, cx, y + i * 9, 9, 9, { fill, stroke, width: 0.6 });
    M('cls', cx + 4.5, y - 7);
  };
  cls(170, BLUE_S, BLUE);
  cls(228, PUR_S, PUR);

  // graph of embeddings
  const N = { hi: [180, 172], A: [236, 184], B: [148, 214], C: [248, 238], hj: [180, 250] };
  const E = [['hi', 'A'], ['hi', 'B'], ['hi', 'C'], ['A', 'B'], ['A', 'C'], ['A', 'hj'], ['B', 'C'], ['B', 'hj'], ['C', 'hj'], ['hi', 'hj']];
  for (const [a, b] of E) L.line(R, N[a], N[b], { stroke: '#a3a8b4', width: 0.6 });
  for (const k of ['A', 'B', 'C']) L.circle(R, N[k][0], N[k][1], 5, { fill: '#e9ebf0', stroke: MUTED, width: 0.6 });
  L.circle(R, N.hi[0], N.hi[1], 6, { fill: BLUE, stroke: BLUE, width: 0.6 });
  L.circle(R, N.hj[0], N.hj[1], 6, { fill: PUR, stroke: PUR, width: 0.6 });
  await M('hi', N.hi[0], N.hi[1] - 14, { scale: 1.1 });
  await M('hj', N.hj[0], N.hj[1] + 14, { scale: 1.1 });
  L.arrow(R, [cx + 14, 184], [N.hi[0] - 10, 175], { both: true, bend: 0.28, stroke: BLUE, width: 0.6, head: 2.6 });
  L.arrow(R, [cx + 14, 244], [N.hj[0] - 10, 249], { both: true, bend: -0.28, stroke: PUR, width: 0.6, head: 2.6 });

  // divider
  L.line(R, [270, 142], [270, 258], { stroke: LINE, width: 0.6, dash: '1.6 2.2', cap: 'round' });

  // training panels (positive = teal, negative = orange, as in pair preparation)
  const panel = async (y, fill, stroke, dark, title, k1, k2) => {
    box(282, y, 134, 55, fill, stroke, 5, 0.6);
    T(349, y + 11, title, { size: 6.5, weight: 'bold', color: dark });
    await M(k1, 349, y + 28);
    await M(k2, 349, y + 43);
  };
  await panel(144, TEAL_T, TEAL, TEAL_D, 'Positive pair training', 'pos1', 'pos2');
  await panel(205, ORG_T, ORG, ORG_D, 'Negative pair training', 'neg1', 'neg2');
}
