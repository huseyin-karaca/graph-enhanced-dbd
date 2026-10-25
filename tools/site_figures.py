"""Draw the figures and tables of the project page.

Every figure is written as inline SVG so that it follows the site's palette: each mark carries a fallback hex
from the light theme plus one of the ``f-*``/``s-*`` classes defined in docs/stylesheets/extra.css, which the
dark theme redefines. Figures come in a wide and a tall variant; the stylesheet shows one or the other.

All numbers live in the DATA section below and are taken from the round-2 report, so the page and the paper
cannot drift apart.

Writes:
    overrides/partials/hero_figure.html   the hero: corpus, graph, decision
    docs/snippets/fig_method.md           the pipeline
    docs/snippets/fig_performance.md      precision/recall dumbbells
    docs/snippets/fig_convergence.md      loss and F1 over five epochs
    docs/snippets/fig_confusion.md        the two confusion matrices
    docs/snippets/tbl_dataset.md          dataset statistics
    docs/snippets/tbl_performance.md      precision, recall, F1
    docs/snippets/tbl_time.md             training time
    docs/snippets/tbl_similarity.md       example similarities
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNIP = ROOT / "docs" / "snippets"
PARTIALS = ROOT / "overrides" / "partials"

# ---------------------------------------------------------------- palette (light fallbacks; see extra.css)

INK, INK2, MUTED, FAINT = "#1c2230", "#434a5a", "#6b7180", "#9ba1ad"
LINE, GRID, WASH, BG = "#c9ccd3", "#eceef1", "#f6f7f9", "#ffffff"
ACCENT, SOFT, TINT = "#3355d6", "#a3b3ee", "#eef1fc"
AMBER, SINGLE = "#b67b0c", "#8b919c"

FONT = "Schibsted Grotesk, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"

# ---------------------------------------------------------------- DATA (round-2 report)

DATASETS = ("Eclipse", "Thunderbird")

STATS = [
    ("Reports", "68,124", "26,040"),
    ("Non-duplicate", "50,606", "15,994"),
    ("Duplicate", "17,512", "10,046"),
    ("Share duplicate", "25.7%", "38.6%"),
    ("Duplicate groups", "6,282", "2,772"),
    ("Evaluated pairs", "87,224", "53,156"),
]

# model -> (precision, recall, F1) per dataset
PERF = {
    "Eclipse": [
        ("BERT", 0.921, 0.933, 0.927),
        ("RoBERTa", 0.918, 0.930, 0.924),
        ("DistilBERT", 0.936, 0.921, 0.929),
        ("CodeBERT", 0.920, 0.928, 0.924),
        ("Ours", 0.913, 0.935, 0.924),
    ],
    "Thunderbird": [
        ("BERT", 0.953, 0.961, 0.957),
        ("RoBERTa", 0.946, 0.968, 0.957),
        ("DistilBERT", 0.950, 0.965, 0.957),
        ("CodeBERT", 0.902, 0.930, 0.916),
        ("Ours", 0.949, 0.966, 0.957),
    ],
}

TIME = {  # minutes per epoch
    "Eclipse": [("BERT", 16.2), ("RoBERTa", 16.3), ("DistilBERT", 15.9), ("CodeBERT", 16.5), ("Ours", 18.1)],
    "Thunderbird": [("BERT", 10.1), ("RoBERTa", 10.2), ("DistilBERT", 9.8), ("CodeBERT", 10.3), ("Ours", 11.1)],
}

CONV = {
    "Eclipse": {
        "loss": {"Train": [0.125, 0.075, 0.048, 0.038, 0.032], "Val": [0.085, 0.058, 0.048, 0.042, 0.038]},
        "f1": {"Train": [0.795, 0.872, 0.905, 0.920, 0.933], "Val": [0.890, 0.908, 0.918, 0.925, 0.930]},
    },
    "Thunderbird": {
        "loss": {"Train": [0.135, 0.078, 0.042, 0.032, 0.025], "Val": [0.082, 0.052, 0.042, 0.035, 0.032]},
        "f1": {"Train": [0.815, 0.895, 0.925, 0.933, 0.933], "Val": [0.905, 0.928, 0.933, 0.935, 0.933]},
    },
}

CONF = {  # true/predicted counts over 10,000 test pairs
    "Eclipse": {"tn": 4132, "fp": 478, "fn": 351, "tp": 5039},
    "Thunderbird": {"tn": 2965, "fp": 348, "fn": 229, "tp": 6458},
}

SIM_DUP = [(8470, 35054, 0.923), (70983, 83204, 0.993), (62405, 104926, 0.979)]
SIM_NON = [(158896, 47204, -0.033), (124062, 21003, -0.107), (73654, 28863, 0.138)]

THETA = 0.65  # the operating point the reference run selected on validation

# ---------------------------------------------------------------- SVG helpers


def _attrs(d):
    out = []
    for k, v in d.items():
        if v is None:
            continue
        out.append(f'{k.replace("_", "-")}="{v}"')
    return " ".join(out)


def tag(name, inner=None, **kw):
    cls = kw.pop("cls", None)
    if cls:
        kw["class"] = cls
    a = _attrs(kw)
    a = (" " + a) if a else ""
    if inner is None:
        return f"<{name}{a}/>"
    return f"<{name}{a}>{inner}</{name}>"


def rect(x, y, w, h, fill=None, stroke=None, cls=None, rx=None, sw=None, op=None):
    return tag("rect", x=r(x), y=r(y), width=r(w), height=r(h), rx=rx, fill=fill, stroke=stroke,
               stroke_width=sw, fill_opacity=op, cls=cls)


def line(x1, y1, x2, y2, stroke=LINE, sw=1, cls="s-line", dash=None):
    return tag("line", x1=r(x1), y1=r(y1), x2=r(x2), y2=r(y2), stroke=stroke, stroke_width=sw,
               stroke_dasharray=dash, cls=cls)


def circle(cx, cy, rad, fill=None, stroke=None, cls=None, sw=None, title=None, op=None):
    return tag("circle", tag("title", esc(title)) if title else None,
               cx=r(cx), cy=r(cy), r=r(rad), fill=fill, stroke=stroke, stroke_width=sw,
               fill_opacity=op, cls=cls)


def text(x, y, s, size=11, fill=MUTED, cls="f-muted", anchor="middle", weight=None, style=None):
    return tag("text", s, x=r(x), y=r(y), font_size=size, text_anchor=anchor, fill=fill,
               font_weight=weight, font_style=style, cls=cls)


def poly(points, fill, cls):
    pts = " ".join(f"{r(x)},{r(y)}" for x, y in points)
    return tag("polygon", points=pts, fill=fill, cls=cls)


def path(d, stroke=LINE, sw=1.3, cls="s-line", fill="none", dash=None):
    return tag("path", d=d, stroke=stroke, stroke_width=sw, fill=fill, stroke_dasharray=dash, cls=cls)


def polyline(points, stroke, sw, cls, dash=None):
    pts = " ".join(f"{r(x)},{r(y)}" for x, y in points)
    return tag("polyline", points=pts, fill="none", stroke=stroke, stroke_width=sw,
               stroke_linejoin="round", stroke_linecap="round", stroke_dasharray=dash, cls=cls)


def r(v):
    if isinstance(v, str) or v is None:
        return v
    return round(float(v), 2)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def msub(*runs):
    """Build text with sub- and superscripts out of tspans.

    Unicode subscript letters are missing from the site's face, so ``cos(z_i, z_j)`` has to be drawn rather
    than spelled. Each run is ``(text, kind)`` with kind "", "sub" or "sup"; ``dy`` is relative, so every run
    shifts from wherever the previous one left off.
    """
    shift = {"": 0.0, "sub": 2.6, "sup": -3.6}
    out, cur = [], 0.0
    for t, kind in runs:
        want = shift[kind]
        size = ' font-size="0.72em"' if kind else ""
        out.append(f'<tspan dy="{round(want - cur, 2)}"{size}>{esc(t)}</tspan>')
        cur = want
    return "".join(out)


COS_IJ = msub(("cos(z", ""), ("i", "sub"), (", z", ""), ("j", "sub"), (")", ""))
S_IJ = msub(("s", ""), ("ij", "sub"))
Z_TR = msub(("z", ""), ("tr", "sup"))
Z_GNN = msub(("z", ""), ("gnn", "sup"))


def arrow_r(x1, x2, y, color=LINE, scls="s-line", fcls="f-line"):
    """A left-to-right arrow ending at x2."""
    return line(x1, y, x2 - 5, y, color, 1.3, scls) + poly(
        [(x2, y), (x2 - 5.5, y - 3.2), (x2 - 5.5, y + 3.2)], color, fcls)


def arrow_d(x, y1, y2, color=LINE, scls="s-line", fcls="f-line"):
    return line(x, y1, x, y2 - 5, color, 1.3, scls) + poly(
        [(x, y2), (x - 3.2, y2 - 5.5), (x + 3.2, y2 - 5.5)], color, fcls)


def box(x, y, w, h, label, sub=None, accent=False, size=11.5):
    """A rounded stage box with a centred label and an optional second line."""
    fill, stroke = (BG, ACCENT if accent else LINE)
    cls = "f-bg " + ("s-accent" if accent else "s-line")
    out = rect(x, y, w, h, fill=fill, stroke=stroke, cls=cls, rx=6, sw=1.5 if accent else 1.1)
    if sub:
        out += text(x + w / 2, y + h / 2 - 1, label, size, INK, "f-ink", weight="600")
        out += text(x + w / 2, y + h / 2 + 12, sub, 9.5, MUTED, "f-muted")
    else:
        out += text(x + w / 2, y + h / 2 + 4, label, size, INK, "f-ink", weight="600")
    return out


def svg(w, h, body, label, extra_cls=""):
    cls = ("pg-svg " + extra_cls).strip()
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="{cls}" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{esc(label)}" font-family="{FONT}">'
            f"{body}</svg>")


def figure(wide, tall, caption, fid=None):
    i = f' id="{fid}"' if fid else ""
    return (f'<figure class="pg-fig"{i}>\n<div class="pg-wide">{wide}</div>\n'
            f'<div class="pg-tall">{tall}</div>\n<figcaption>{caption}</figcaption>\n</figure>\n')


# ---------------------------------------------------------------- hero: corpus, graph, decision

# positions of the graph glyph, in a 1x1 box; the first six are the supervised nodes
GRAPH_NODES = [
    (0.14, 0.26), (0.30, 0.13), (0.52, 0.22), (0.72, 0.12), (0.86, 0.34), (0.66, 0.42),
    (0.08, 0.58), (0.24, 0.46), (0.40, 0.62), (0.56, 0.78), (0.78, 0.68), (0.92, 0.82),
    (0.20, 0.86), (0.36, 0.30), (0.46, 0.44), (0.62, 0.60), (0.88, 0.56), (0.70, 0.88),
    (0.12, 0.74), (0.44, 0.92),
]
GRAPH_EDGES = [
    (0, 13), (13, 1), (1, 2), (2, 14), (14, 5), (5, 3), (3, 4), (4, 16), (16, 10), (10, 15),
    (15, 8), (8, 7), (7, 6), (6, 18), (18, 12), (12, 19), (19, 9), (9, 17), (17, 11), (11, 16),
    (13, 14), (14, 8), (5, 15), (2, 5), (7, 0), (9, 15), (10, 17), (12, 19),
]
PROPAGATED = {13, 14, 15, 16, 10, 7}  # reached in one or two hops from a supervised node


def hero_repo(ox, oy, w, cols=10, rows=5):
    """A field of reports; a few are linked by a confirmed duplicate label."""
    pitch = w / (cols - 1 + 1.4)
    rad = min(3.2, pitch * 0.22)
    pairs = {(1, 0): (2, 0), (4, 2): (5, 2), (7, 3): (8, 3)}
    marked = set()
    for a, b in pairs.items():
        marked |= {a, b}
    out = ""
    for (c1, r1), (c2, r2) in pairs.items():
        out += line(ox + c1 * pitch, oy + r1 * pitch, ox + c2 * pitch, oy + r2 * pitch, ACCENT, 1.6, "s-accent")
    for row in range(rows):
        for col in range(cols):
            on = (col, row) in marked
            out += circle(ox + col * pitch, oy + row * pitch, rad + (0.5 if on else 0),
                          fill=ACCENT if on else FAINT, cls="f-accent" if on else "f-faint")
    return out, pitch * (rows - 1)


def hero_graph(ox, oy, w, h):
    out = ""
    for i, j in GRAPH_EDGES:
        x1, y1 = GRAPH_NODES[i]
        x2, y2 = GRAPH_NODES[j]
        out += line(ox + x1 * w, oy + y1 * h, ox + x2 * w, oy + y2 * h, LINE, 0.8, "s-line")
    for k, (x, y) in enumerate(GRAPH_NODES):
        # a half-strength accent reads as "reached by propagation" in both themes, which the fixed soft
        # tint does not: it is a dark navy on the dark surface
        if k < 6:
            fill, cls, rad, op = ACCENT, "f-accent", 4.4, None
        elif k in PROPAGATED:
            fill, cls, rad, op = ACCENT, "f-accent", 3.8, 0.45
        else:
            fill, cls, rad, op = FAINT, "f-faint", 3.2, None
        out += circle(ox + x * w, oy + y * h, rad, fill=fill, cls=cls, op=op)
    return out


def hero_decision(ox, oy, h, w):
    """A similarity axis carrying the report's own validation examples."""
    top, bot = oy, oy + h
    def ypos(s):
        return top + (1.0 - s) / 1.2 * h

    out = line(ox, top, ox, bot, LINE, 1.1, "s-line")
    for v in (1.0, 0.5, 0.0):
        y = ypos(v)
        out += line(ox - 3, y, ox, y, LINE, 1.1, "s-line")
        out += text(ox - 7, y + 3.5, f"{v:.1f}", 9.5, MUTED, "f-muted", anchor="end")
    ty = ypos(THETA)
    out += line(ox, ty, ox + w, ty, ACCENT, 1.2, "s-accent", dash="3 3")
    out += text(ox + w, ty - 5, "θ", 11, ACCENT, "f-accent", anchor="end", weight="600", style="italic")
    for k, (a, b, s) in enumerate(SIM_DUP):
        out += circle(ox + 16 + k * 17, ypos(s), 4.2, fill=ACCENT, cls="f-accent",
                      title=f"bugs {a} and {b}: duplicates, cosine {s:+.3f}")
    for k, (a, b, s) in enumerate(SIM_NON):
        out += circle(ox + 16 + k * 17, ypos(s), 4.2, fill=SINGLE, cls="f-single",
                      title=f"bugs {a} and {b}: not duplicates, cosine {s:+.3f}")
    out += text(ox + 70, ypos(0.97) + 3, "duplicates", 9.5, ACCENT, "f-accent", anchor="start")
    out += text(ox + 70, ypos(0.02) + 3, "unrelated", 9.5, MUTED, "f-muted", anchor="start")
    return out


HERO_ALT = ("Three steps: a repository of 68,124 bug reports of which only a few pairs carry a duplicate label; "
            "a title-similarity graph in which every report is a node, so supervision spreads from the labelled "
            "nodes to the rest; and a cosine-similarity axis on which duplicate pairs land near 1 and unrelated "
            "pairs near 0, split by a threshold chosen on validation.")

PILLS = [("Eclipse", "F1 0.924"), ("Thunderbird", "F1 0.957"), ("vs. BERT", "+10–12% time")]


def hero_pills(ox, oy, total_w, gap=12, stacked=False):
    """Three result pills, side by side when there is room and stacked when there is not."""
    n = len(PILLS)
    out = ""
    for k, (lab, val) in enumerate(PILLS):
        if stacked:
            x, y, w = ox, oy + k * (28 + gap / 2), total_w
        else:
            w = (total_w - gap * (n - 1)) / n
            x, y = ox + k * (w + gap), oy
        h = 28 if stacked else 30
        out += rect(x, y, w, h, fill=TINT, cls="f-tint", rx=h / 2)
        out += text(x + w / 2, y + h / 2 + 4, f"{lab}  {val}", 11.5, INK, "f-ink", weight="600")
    return out


def hero_wide():
    b = ""
    b += text(4, 20, "1 · the repository", 10.5, MUTED, "f-muted", anchor="start", weight="600")
    dots, _ = hero_repo(10, 46, 126)
    b += dots
    b += text(73, 122, "68,124 reports", 10, INK2, "f-ink2")
    b += text(73, 135, "few carry a duplicate label", 10, MUTED, "f-muted")
    b += arrow_r(158, 178, 72)

    b += text(186, 20, "2 · the graph", 10.5, MUTED, "f-muted", anchor="start", weight="600")
    b += hero_graph(196, 38, 200, 70)
    b += text(296, 122, "every report is a node", 10, INK2, "f-ink2")
    b += text(296, 135, "BERT titles → PCA(10) → 2-layer GCN", 10, MUTED, "f-muted")
    b += arrow_r(414, 434, 72)

    b += text(442, 20, "3 · the decision", 10.5, MUTED, "f-muted", anchor="start", weight="600")
    b += hero_decision(470, 42, 76, 126)
    b += text(528, 135, COS_IJ + " on validation pairs", 10, MUTED, "f-muted")

    b += hero_pills(9, 160, 582)
    b += text(300, 212, "Bilkent CS 588 · Eclipse Platform and Mozilla Thunderbird", 10, FAINT, "f-faint")
    return svg(600, 224, b, HERO_ALT)


def hero_tall():
    b = ""
    b += text(10, 18, "1 · the repository", 10.5, MUTED, "f-muted", anchor="start", weight="600")
    dots, grid_h = hero_repo(76, 40, 208, cols=10, rows=5)
    b += dots
    b += text(180, 40 + grid_h + 26, "68,124 reports", 11, INK2, "f-ink2")
    b += text(180, 40 + grid_h + 41, "few carry a duplicate label", 10, MUTED, "f-muted")
    b += arrow_d(180, 40 + grid_h + 52, 40 + grid_h + 76)

    b += text(10, 206, "2 · the graph", 10.5, MUTED, "f-muted", anchor="start", weight="600")
    b += hero_graph(40, 222, 280, 104)
    b += text(180, 356, "every report is a node", 11, INK2, "f-ink2")
    b += text(180, 371, "BERT titles → PCA(10) → 2-layer GCN", 10, MUTED, "f-muted")
    b += arrow_d(180, 382, 406)

    b += text(10, 434, "3 · the decision", 10.5, MUTED, "f-muted", anchor="start", weight="600")
    b += hero_decision(116, 452, 92, 140)
    b += text(180, 578, COS_IJ + " on validation pairs", 10, MUTED, "f-muted")

    b += hero_pills(20, 600, 320, stacked=True)
    b += text(180, 726, "Bilkent CS 588 · Eclipse and Thunderbird", 10, FAINT, "f-faint")
    return svg(360, 740, b, HERO_ALT)


# ---------------------------------------------------------------- method: the pipeline

METHOD_ALT = ("The pipeline. Top: labelled report pairs run through a weight-tied BERT encoder, whose [CLS] "
              "vector is projected to 128 dimensions. Bottom: the titles of every report, labelled or not, are "
              "encoded, reduced with PCA to ten dimensions and turned into a similarity graph over which a "
              "two-layer GCN passes messages. The two embeddings are fused, and a pair is called duplicate when "
              "their cosine similarity clears a threshold chosen on validation.")


def mini_graph(ox, oy, w, h):
    """A small graph glyph: two supervised nodes among unlabelled ones."""
    pts = [(0.10, 0.22), (0.46, 0.08), (0.84, 0.30), (0.28, 0.62), (0.64, 0.52), (0.92, 0.80), (0.14, 0.92),
           (0.50, 0.88)]
    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (1, 4), (2, 4), (4, 5), (3, 6), (6, 7), (7, 4), (2, 5)]
    out = ""
    for i, j in edges:
        out += line(ox + pts[i][0] * w, oy + pts[i][1] * h, ox + pts[j][0] * w, oy + pts[j][1] * h,
                    LINE, 0.8, "s-line")
    for k, (x, y) in enumerate(pts):
        on = k in (1, 4)
        out += circle(ox + x * w, oy + y * h, 3.6 if on else 2.9,
                      fill=ACCENT if on else FAINT, cls="f-accent" if on else "f-faint")
    return out


def pair_cards(ox, oy, w, labels):
    out = ""
    for k, lab in enumerate(labels):
        y = oy + k * 26
        out += rect(ox, y, w, 20, fill=WASH, stroke=LINE, cls="f-wash s-line", rx=4, sw=1)
        out += text(ox + w / 2, y + 14, lab, 10, INK2, "f-ink2")
    return out


def method_decision(x, y, w, h):
    out = rect(x, y, w, h, fill=WASH, cls="f-wash", rx=8)
    cx = x + w / 2
    out += text(cx, y + 24, "fuse", 10, MUTED, "f-muted", weight="600")
    out += text(cx, y + 43, f"z = λ {Z_TR} + (1−λ) {Z_GNN}", 11, INK, "f-ink", weight="600")
    out += line(x + 12, y + 58, x + w - 12, y + 58, GRID, 1, "s-grid")
    out += text(cx, y + 78, "score", 10, MUTED, "f-muted", weight="600")
    out += text(cx, y + 97, f"{S_IJ} = {COS_IJ}", 11, INK, "f-ink", weight="600")
    out += line(x + 12, y + 112, x + w - 12, y + 112, GRID, 1, "s-grid")
    out += text(cx, y + 132, "decide", 10, MUTED, "f-muted", weight="600")
    out += text(cx, y + 151, f"{S_IJ} ≥ θ ⇒ duplicate", 11, ACCENT, "f-accent", weight="600")
    return out


def method_wide():
    b = ""
    # band 1: supervision
    b += text(8, 28, "① labelled pairs — supervision", 10.5, MUTED, "f-muted", anchor="start", weight="600")
    b += pair_cards(8, 44, 76, ["report i", "report j"])
    b += arrow_r(90, 110, 67)
    b += box(110, 44, 76, 48, "BERT", "weight-tied")
    b += arrow_r(192, 212, 67)
    for k in range(4):
        b += rect(218, 47 + k * 11, 11, 9, fill=ACCENT, cls="f-accent", rx=2, op=0.85)
    b += text(223.5, 39, "[CLS]", 9.5, MUTED, "f-muted")
    b += arrow_r(235, 255, 67)
    b += box(255, 44, 76, 48, "project", "→ 128-d")
    b += arrow_r(337, 357, 67)
    b += text(375, 71, Z_TR, 13, ACCENT, "f-accent", weight="700")

    # band 2: propagation
    b += text(8, 144, "② every report — propagation", 10.5, MUTED, "f-muted", anchor="start", weight="600")
    b += pair_cards(8, 158, 76, ["all titles", "train+val+test"])
    b += arrow_r(90, 110, 181)
    b += box(110, 157, 92, 48, "BERT + PCA", "768 → 10 dims")
    b += arrow_r(208, 228, 181)
    b += mini_graph(232, 150, 76, 62)
    b += text(270, 224, "similarity graph", 9.5, MUTED, "f-muted")
    b += arrow_r(314, 334, 181)
    b += box(334, 157, 74, 48, "GCN", "2 layers")
    b += arrow_r(414, 434, 181)
    b += text(452, 185, Z_GNN, 13, ACCENT, "f-accent", weight="700")

    # the two paths meet and enter the decision panel
    b += path("M 392 71 L 486 71 L 486 120", LINE, 1.3, "s-line")
    b += path("M 470 185 L 486 185 L 486 120", LINE, 1.3, "s-line")
    b += arrow_r(486, 512, 120)
    b += method_decision(512, 44, 160, 172)
    b += text(592, 30, "③ decision", 10.5, MUTED, "f-muted", weight="600")
    b += text(592, 236, "θ by grid search on validation", 9.5, FAINT, "f-faint")
    return svg(680, 248, b, METHOD_ALT)


def method_tall():
    b = ""
    b += text(10, 18, "① labelled pairs", 10.5, MUTED, "f-muted", anchor="start", weight="600")
    b += text(190, 18, "② every report", 10.5, MUTED, "f-muted", anchor="start", weight="600")

    b += pair_cards(18, 30, 140, ["report i", "report j"])
    b += arrow_d(88, 82, 102)
    b += box(18, 102, 140, 44, "BERT", "weight-tied")
    b += arrow_d(88, 146, 166)
    b += box(18, 166, 140, 44, "[CLS] → project", "→ 128-d")
    b += arrow_d(88, 210, 232)
    b += text(88, 248, Z_TR, 14, ACCENT, "f-accent", weight="700")

    b += pair_cards(198, 30, 140, ["all titles", "train+val+test"])
    b += arrow_d(268, 82, 102)
    b += box(198, 102, 140, 44, "BERT + PCA", "768 → 10 dims")
    b += arrow_d(268, 146, 166)
    b += mini_graph(226, 164, 84, 48)
    b += text(268, 224, "similarity graph", 9.5, MUTED, "f-muted")
    b += arrow_d(268, 230, 252)
    b += box(198, 252, 140, 40, "GCN · 2 layers")
    b += arrow_d(268, 292, 314)
    b += text(268, 330, Z_GNN, 14, ACCENT, "f-accent", weight="700")

    b += path("M 88 258 L 88 352 L 178 352", LINE, 1.3, "s-line")
    b += path("M 268 340 L 268 352 L 178 352", LINE, 1.3, "s-line")
    b += arrow_d(178, 352, 382)
    b += text(10, 380, "③ decision", 10.5, MUTED, "f-muted", anchor="start", weight="600")
    b += method_decision(18, 390, 320, 172)
    b += text(180, 582, "θ by grid search on validation", 9.5, FAINT, "f-faint")
    return svg(360, 594, b, METHOD_ALT)


# ---------------------------------------------------------------- performance: precision/recall dumbbells

PERF_LO, PERF_HI = 0.89, 0.975
PERF_ALT = ("Precision and recall for four transformer baselines and the proposed model, on Eclipse and "
            "Thunderbird. Our model sits at the recall end of the spread on both datasets; its F1 of 0.924 on "
            "Eclipse and 0.957 on Thunderbird matches the baselines.")


def perf_panel(ox, oy, w, dataset, show_legend):
    rows = PERF[dataset]
    lab_w, f1_w = 62, 40
    px = ox + lab_w + 8
    pw = w - lab_w - f1_w - 16
    pitch = 25
    top = oy + 24

    def xpos(v):
        return px + (v - PERF_LO) / (PERF_HI - PERF_LO) * pw

    out = text(ox, oy + 4, dataset, 11.5, INK, "f-ink", anchor="start", weight="600")
    if show_legend:
        lx = ox + w - f1_w - 10
        out += circle(lx - 118, oy, 4.2, fill=ACCENT, cls="f-accent")
        out += text(lx - 110, oy + 4, "precision", 10, MUTED, "f-muted", anchor="start")
        out += circle(lx - 48, oy, 4.2, fill=AMBER, cls="f-mlp")
        out += text(lx - 40, oy + 4, "recall", 10, MUTED, "f-muted", anchor="start")

    for g in (0.90, 0.92, 0.94, 0.96):
        out += line(xpos(g), top - 6, xpos(g), top + pitch * len(rows) - 12, GRID, 1, "s-grid")

    for k, (name, p, rc, f1) in enumerate(rows):
        y = top + k * pitch
        ours = name == "Ours"
        if ours:
            out += rect(ox - 4, y - 11, w + 4, 22, fill=TINT, cls="f-tint", rx=4)
        out += text(ox, y + 4, name, 10.5, INK if ours else INK2, "f-ink" if ours else "f-ink2",
                    anchor="start", weight="600" if ours else None)
        x1, x2 = sorted((xpos(p), xpos(rc)))
        out += line(x1, y, x2, y, SINGLE, 1.6, "s-single")
        out += circle(xpos(p), y, 4.6, fill=ACCENT, cls="f-accent",
                      title=f"{name} on {dataset}: precision {p:.3f}")
        out += circle(xpos(rc), y, 4.6, fill=AMBER, cls="f-mlp",
                      title=f"{name} on {dataset}: recall {rc:.3f}")
        out += text(ox + w, y + 4, f"{f1:.3f}", 10.5, INK if ours else MUTED,
                    "f-ink" if ours else "f-muted", anchor="end", weight="600" if ours else None)

    ay = top + pitch * len(rows) - 6
    out += line(px, ay, px + pw, ay, LINE, 1, "s-line")
    for g in (0.90, 0.92, 0.94, 0.96):
        out += text(xpos(g), ay + 14, f"{g:.2f}", 9.5, MUTED, "f-muted")
    out += text(ox + w, oy + 4, "F1", 10, MUTED, "f-muted", anchor="end", weight="600")
    return out


def perf_wide():
    b = perf_panel(4, 20, 274, "Eclipse", True)
    b += perf_panel(322, 20, 274, "Thunderbird", False)
    return svg(600, 182, b, PERF_ALT)


def perf_tall():
    b = perf_panel(4, 20, 348, "Eclipse", True)
    b += perf_panel(4, 220, 348, "Thunderbird", False)
    return svg(360, 382, b, PERF_ALT)


# ---------------------------------------------------------------- convergence

CONV_ALT = ("Training and validation curves over five epochs on both datasets. Loss falls smoothly and the "
            "training and validation F1 curves meet near 0.93, with no sign of overfitting.")

CONV_SPEC = {
    "loss": {"label": "loss", "lo": 0.0, "hi": 0.15, "ticks": (0.00, 0.05, 0.10, 0.15), "fmt": "{:.2f}"},
    "f1": {"label": "F1", "lo": 0.75, "hi": 1.0, "ticks": (0.80, 0.90, 1.00), "fmt": "{:.2f}"},
}
SERIES = (("Train", AMBER, "f-mlp", "s-mlp", "4 3"), ("Val", ACCENT, "f-accent", "s-accent", None))


def conv_panel(ox, oy, w, h, dataset, metric):
    spec = CONV_SPEC[metric]
    data = CONV[dataset][metric]
    px, pw = ox + 34, w - 34 - 36   # 36 on the right for the end-of-run labels
    top, ph = oy + 16, h - 42

    def xp(e):
        return px + (e - 1) / 4 * pw

    def yp(v):
        return top + (spec["hi"] - v) / (spec["hi"] - spec["lo"]) * ph

    out = text(ox, oy + 4, f"{dataset} · {spec['label']}", 10.5, INK, "f-ink", anchor="start", weight="600")
    for t in spec["ticks"]:
        y = yp(t)
        out += line(px, y, px + pw, y, GRID, 1, "s-grid")
        out += text(px - 6, y + 3.5, spec["fmt"].format(t), 9, MUTED, "f-muted", anchor="end")
    out += line(px, top + ph, px + pw, top + ph, LINE, 1, "s-line")
    for e in (1, 2, 3, 4, 5):
        out += text(xp(e), top + ph + 15, str(e), 9, MUTED, "f-muted")
    out += text(px + pw / 2, top + ph + 29, "epoch", 9, FAINT, "f-faint")

    ends = []
    for name, color, fcls, scls, dash in SERIES:
        pts = [(xp(i + 1), yp(v)) for i, v in enumerate(data[name])]
        out += polyline(pts, color, 1.8, scls, dash)
        for i, ((x, y), v) in enumerate(zip(pts, data[name])):
            out += circle(x, y, 3.1, fill=color, stroke=BG, cls=f"{fcls} s-bg", sw=1.6,
                          title=f"{dataset} {name.lower()} {spec['label']}, epoch {i + 1}: {v:.3f}")
        ends.append([pts[-1][0], pts[-1][1], data[name][-1], color, fcls])

    # the two runs often finish within a hair of each other; nudge the labels apart so both stay readable
    ends.sort(key=lambda e: e[1])
    if ends[1][1] - ends[0][1] < 11:
        mid = (ends[0][1] + ends[1][1]) / 2
        ends[0][1], ends[1][1] = mid - 5.5, mid + 5.5
    for x, y, v, color, fcls in ends:
        out += text(x + 7, y + 3.5, f"{v:.3f}", 9.5, color, fcls, anchor="start", weight="600")
    return out


def conv_legend(x, y):
    out = ""
    for k, (name, color, fcls, scls, dash) in enumerate(SERIES):
        lx = x + k * 74
        out += line(lx, y, lx + 18, y, color, 1.8, scls, dash)
        out += circle(lx + 9, y, 3.1, fill=color, stroke=BG, cls=f"{fcls} s-bg", sw=1.6)
        out += text(lx + 24, y + 3.5, name, 10, MUTED, "f-muted", anchor="start")
    return out


def conv_wide():
    b = conv_legend(444, 12)
    for i, dataset in enumerate(DATASETS):
        for j, metric in enumerate(("loss", "f1")):
            b += conv_panel(6 + j * 300, 26 + i * 150, 282, 140, dataset, metric)
    return svg(600, 322, b, CONV_ALT)


def conv_tall():
    b = conv_legend(202, 12)
    k = 0
    for dataset in DATASETS:
        for metric in ("loss", "f1"):
            b += conv_panel(6, 26 + k * 150, 346, 140, dataset, metric)
            k += 1
    return svg(360, 622, b, CONV_ALT)


# ---------------------------------------------------------------- confusion matrices

CONF_ALT = ("Confusion matrices over 10,000 test pairs per dataset. Eclipse: 4,132 true negatives, 478 false "
            "positives, 351 false negatives, 5,039 true positives. Thunderbird: 2,965, 348, 229 and 6,458.")


def conf_panel(ox, oy, w, dataset):
    c = CONF[dataset]
    cells = [("tn", 0, 0), ("fp", 1, 0), ("fn", 0, 1), ("tp", 1, 1)]
    names = {"tn": "true negative", "fp": "false positive", "fn": "false negative", "tp": "true positive"}
    cw, ch = (w - 54) / 2, 48
    gx, gy = ox + 54, oy + 30
    hi = max(c.values())

    out = text(ox, oy + 4, dataset, 11.5, INK, "f-ink", anchor="start", weight="600")
    out += text(gx + cw, oy + 22, "non-dup.", 9.5, MUTED, "f-muted")
    out += text(gx + cw * 1.5 + 2, oy + 22, "duplicate", 9.5, MUTED, "f-muted")
    out += text(gx + cw, oy + 22, "", 9.5)
    out += text(gx - 50 + 24, oy + 22, "predicted →", 9, FAINT, "f-faint", anchor="start")
    for key, cx, cy in cells:
        x, y = gx + cx * (cw + 2), gy + cy * (ch + 2)
        v = c[key]
        out += rect(x, y, cw, ch, fill=ACCENT, cls="f-accent", rx=4, op=round(0.05 + 0.26 * v / hi, 3))
        out += tag("rect", tag("title", f"{dataset}: {v:,} {names[key]} pairs"),
                   x=r(x), y=r(y), width=r(cw), height=r(ch), fill="transparent")
        out += text(x + cw / 2, y + ch / 2 + 5, f"{v:,}", 14, INK, "f-ink", weight="700")
    for k, lab in enumerate(("non-dup.", "duplicate")):
        out += text(gx - 6, gy + k * (ch + 2) + ch / 2 + 3.5, lab, 9.5, MUTED, "f-muted", anchor="end")
    out += text(ox + 2, gy + ch + 2, "actual ↓", 9, FAINT, "f-faint", anchor="start")

    p = c["tp"] / (c["tp"] + c["fp"])
    rc = c["tp"] / (c["tp"] + c["fn"])
    acc = (c["tp"] + c["tn"]) / sum(c.values())
    f1 = 2 * p * rc / (p + rc)
    out += text(ox, gy + 2 * ch + 22, f"accuracy {acc:.1%}  ·  precision {p:.1%}  ·  recall {rc:.1%}  "
                                     f"·  F1 {f1:.1%}", 9.5, MUTED, "f-muted", anchor="start")
    return out


def conf_wide():
    b = conf_panel(6, 18, 280, "Eclipse") + conf_panel(314, 18, 280, "Thunderbird")
    return svg(600, 172, b, CONF_ALT)


def conf_tall():
    b = conf_panel(6, 18, 348, "Eclipse") + conf_panel(6, 182, 348, "Thunderbird")
    return svg(360, 336, b, CONF_ALT)


# ---------------------------------------------------------------- tables


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def pg_table(groups, subs, rows, note=None):
    """A model-comparison table with a spanning header, so the per-dataset columns stay narrow enough to
    fit the reading column instead of scrolling sideways. The classes come from docs/stylesheets/extra.css.
    """
    head = '<tr><th></th>' + "".join(
        f'<th class="grp" colspan="{n}">{g}</th>' for g, n in groups) + "</tr>"
    head += '<tr><th></th>' + "".join(f'<th class="num">{s}</th>' for s in subs) + "</tr>"
    body = ""
    for name, cells in rows:
        ours = name == "Ours"
        tr = ' class="pg-row pg-group"' if ours else ""
        body += f'<tr{tr}><td class="sys">{name}</td>' + "".join(
            f'<td class="num">{c}</td>' for c in cells) + "</tr>"
    out = (f'<div class="pg-table">\n<table>\n<thead>{head}</thead>\n<tbody>{body}</tbody>\n</table>\n')
    if note:
        out += f'<p class="pg-note">{note}</p>\n'
    return out + "</div>\n"


def tbl_dataset():
    out = [row(["", "Eclipse Platform", "Mozilla Thunderbird"]), "|:--|--:|--:|"]
    out += [row([k, a, b]) for k, a, b in STATS]
    return "\n".join(out) + "\n"


def tbl_performance():
    rows = []
    for k, (name, p, rc, f1) in enumerate(PERF["Eclipse"]):
        _, p2, r2, f2 = PERF["Thunderbird"][k]
        rows.append((name, [f"{v:.3f}" for v in (p, rc, f1, p2, r2, f2)]))
    return pg_table([("Eclipse", 3), ("Thunderbird", 3)], ["P", "R", "F1"] * 2, rows,
                    note="Precision, recall and F1 on the held-out test pairs.")


def tbl_time():
    base = {d: dict(TIME[d])["BERT"] for d in DATASETS}
    rows = []
    for k, (name, t) in enumerate(TIME["Eclipse"]):
        t2 = TIME["Thunderbird"][k][1]
        rows.append((name, [f"{t:.1f}", f"{t / base['Eclipse']:.2f}×",
                            f"{t2:.1f}", f"{t2 / base['Thunderbird']:.2f}×"]))
    return pg_table([("Eclipse", 2), ("Thunderbird", 2)], ["min/epoch", "rel."] * 2, rows,
                    note="One A100 (80 GB); relative to the BERT baseline on the same dataset.")


def tbl_similarity():
    out = [row(["duplicate pair", "cos", "unrelated pair", "cos"]), "|:--|--:|:--|--:|"]
    for (a, b, s), (c, d, t) in zip(SIM_DUP, SIM_NON):
        out.append(row([f"`{a}` · `{b}`", f"**{s:+.3f}**", f"`{c}` · `{d}`", f"{t:+.3f}"]))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- main


def main():
    SNIP.mkdir(parents=True, exist_ok=True)
    PARTIALS.mkdir(parents=True, exist_ok=True)

    (PARTIALS / "hero_figure.html").write_text(
        "{#- Generated by tools/site_figures.py; do not edit. -#}\n"
        f'<figure class="pg-hero__figure">\n<div class="pg-wide">{hero_wide()}</div>\n'
        f'<div class="pg-tall">{hero_tall()}</div>\n</figure>\n')

    figs = [
        ("fig_method.md", method_wide(), method_tall(),
         "The pipeline. Supervision runs over a feasible subset of labelled pairs (top); every report in the "
         "corpus, labelled or not, is a node of a title-similarity graph over which a two-layer GCN passes "
         "messages (bottom). The two embeddings are fused and compared by cosine similarity."),
        ("fig_performance.md", perf_wide(), perf_tall(),
         "Precision and recall on the test pairs, with F1 at the right. Our model sits at the recall end of the "
         "spread on both datasets — the highest recall of any model on Eclipse — and ties the best baselines on "
         "Thunderbird."),
        ("fig_convergence.md", conv_wide(), conv_tall(),
         "Five epochs on both datasets. Training and validation curves stay close throughout, so the extra "
         "capacity of the graph branch does not cost stability."),
        ("fig_confusion.md", conf_wide(), conf_tall(),
         "Test-set confusion over 10,000 pairs per dataset. Errors are split fairly evenly between the two "
         "kinds, so the threshold is not trading one class off against the other."),
    ]
    for name, wide, tall, cap in figs:
        (SNIP / name).write_text(figure(wide, tall, cap))

    for name, body in [("tbl_dataset.md", tbl_dataset()), ("tbl_performance.md", tbl_performance()),
                       ("tbl_time.md", tbl_time()), ("tbl_similarity.md", tbl_similarity())]:
        (SNIP / name).write_text(body)

    print(f"wrote {len(figs)} figures and 4 tables to {SNIP}, hero to {PARTIALS}")


if __name__ == "__main__":
    main()
