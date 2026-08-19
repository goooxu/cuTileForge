"""Generate the inline SVG charts used by index.html.

Charts are static SVG with no JavaScript and no external library: the report has
to render from a file:// URL on a laptop with no network. Numbers live in the
data tables at the bottom of this file, so a re-scored model means editing them
here and re-running:

    python3 make_charts.py && python3 refresh_charts.py

refresh_charts.py matches figures by number, so it only rewrites the SVG and
leaves the prose alone.
"""

import io
import os

GREEN = "#76b900"
BLUE = "#3b7dd8"
GREY = "#9aa4ae"
RED = "#d0563f"
AMBER = "#e0a33e"
INK = "#1b1f24"
MUTED = "#626b75"
LINE = "#e3e7ec"

W = 900
LEFT = 108         # model label column
RIGHT = 132        # value label column


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def frame(height, title, body):
    return (
        '<figure class="chart">\n'
        '<svg viewBox="0 0 %d %d" role="img" aria-label="%s" width="100%%">\n'
        '%s</svg>\n'
        '<figcaption>%s</figcaption>\n'
        '</figure>\n' % (W, height, esc(title), body, esc(title)))


def gridlines(top, height, plot_w, ticks, fmt):
    out = []
    for t in ticks:
        x = LEFT + plot_w * t
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" '
                   'stroke-width="1"/>' % (x, top, x, top + height, LINE))
        out.append('<text x="%.1f" y="%d" fill="%s" font-size="13" '
                   'text-anchor="middle">%s</text>'
                   % (x, top - 6, MUTED, fmt(t)))
    return "\n".join(out) + "\n"


def grouped_bars(title, models, series, tick_fmt=lambda t: "%d%%" % (t * 100),
                 ticks=(0, 0.25, 0.5, 0.75, 1.0), row_h=None, legend=True):
    """series: list of (name, colour, {model: (fraction, label)})."""
    n_ser = len(series)
    bar_h = 18 if n_ser > 1 else 22
    gap = 5
    row_h = row_h or (n_ser * (bar_h + gap) + 18)
    top = 36 if legend else 26
    plot_w = W - LEFT - RIGHT
    height = top + len(models) * row_h + 12

    parts = []
    if legend:
        x = LEFT
        for name, colour, _ in series:
            parts.append('<rect x="%d" y="5" width="13" height="13" rx="2" '
                         'fill="%s"/>' % (x, colour))
            parts.append('<text x="%d" y="17" fill="%s" font-size="14">%s</text>'
                         % (x + 19, MUTED, esc(name)))
            x += 24 + 10 * len(name) + 34
    parts.append(gridlines(top, len(models) * row_h, plot_w, ticks, tick_fmt))

    for i, m in enumerate(models):
        y0 = top + i * row_h
        parts.append('<text x="%d" y="%.1f" fill="%s" font-size="15" '
                     'text-anchor="end">%s</text>'
                     % (LEFT - 12, y0 + row_h / 2 + 3, INK, esc(m)))
        for j, (_, colour, data) in enumerate(series):
            if m not in data:
                continue
            frac, label = data[m]
            y = y0 + 6 + j * (bar_h + gap)
            w = max(1.0, plot_w * frac)
            parts.append('<rect x="%d" y="%.1f" width="%.1f" height="%d" '
                         'rx="2" fill="%s"/>' % (LEFT, y, w, bar_h, colour))
            parts.append('<text x="%.1f" y="%.1f" fill="%s" font-size="13.5">%s</text>'
                         % (LEFT + w + 7, y + bar_h - 3, MUTED, esc(label)))
    return frame(height, title, "\n".join(parts) + "\n")


def stacked_bars(title, rows, keys, total, row_h=52):
    """rows: list of (label, {key: value}); keys: list of (name, colour)."""
    top = 36
    plot_w = W - LEFT - 46
    height = top + len(rows) * row_h + 14
    parts = []
    x = LEFT
    for name, colour in keys:
        parts.append('<rect x="%d" y="5" width="13" height="13" rx="2" '
                     'fill="%s"/>' % (x, colour))
        parts.append('<text x="%d" y="17" fill="%s" font-size="14">%s</text>'
                     % (x + 19, MUTED, esc(name)))
        x += 24 + 9 * len(name) + 38
    for i, (label, vals) in enumerate(rows):
        y = top + i * row_h + 6
        parts.append('<text x="%d" y="%d" fill="%s" font-size="15" '
                     'text-anchor="end">%s</text>' % (LEFT - 12, y + 20, INK, esc(label)))
        cx = LEFT
        for name, colour in keys:
            v = vals.get(name, 0)
            if not v:
                continue
            w = plot_w * v / total
            parts.append('<rect x="%.1f" y="%d" width="%.1f" height="28" '
                         'fill="%s"/>' % (cx, y, w, colour))
            if w > 48:
                parts.append('<text x="%.1f" y="%d" fill="#ffffff" font-size="14" '
                             'text-anchor="middle">%s</text>'
                             % (cx + w / 2, y + 19, ('%g' % v)))
            cx += w
    return frame(height, title, "\n".join(parts) + "\n")


def chain_bars(title, rows, legend, vmax, row_h=40):
    """rows: (label, value, colour, note). One bar per checkpoint, coloured by
    what kind of intervention produced it."""
    top = 36
    left = 210
    plot_w = W - left - 150
    height = top + len(rows) * row_h + 14
    parts = []
    x = left
    for name, colour in legend:
        parts.append('<rect x="%d" y="5" width="13" height="13" rx="2" '
                     'fill="%s"/>' % (x, colour))
        parts.append('<text x="%d" y="17" fill="%s" font-size="14">%s</text>'
                     % (x + 19, MUTED, esc(name)))
        x += 24 + 10 * len(name) + 30
    prev = None
    for i, (label, value, colour, note) in enumerate(rows):
        y = top + i * row_h
        parts.append('<text x="%d" y="%.1f" fill="%s" font-size="15" '
                     'text-anchor="end">%s</text>'
                     % (left - 12, y + row_h / 2 + 5, INK, esc(label)))
        w = plot_w * value / float(vmax)
        parts.append('<rect x="%d" y="%.1f" width="%.1f" height="22" rx="2" '
                     'fill="%s"/>' % (left, y + 6, w, colour))
        delta = "" if prev is None else "  %+d" % (value - prev)
        parts.append('<text x="%.1f" y="%.1f" fill="%s" font-size="14">'
                     '%d%s</text>' % (left + w + 8, y + 23, INK, value, delta))
        if note:
            parts.append('<text x="%.1f" y="%.1f" fill="%s" font-size="12.5">'
                         '%s</text>' % (left + w + 78, y + 23, MUTED, esc(note)))
        prev = value
    return frame(height, title, "\n".join(parts) + "\n")


def heatmap(title, col_labels, rows, totals):
    """rows: list of (family, n_problems, [counts aligned with col_labels])."""
    cell_w = 96
    cell_h = 40
    left = 150
    top = 56
    width = left + cell_w * len(col_labels) + 10
    height = top + cell_h * len(rows) + 10
    parts = []
    for j, c in enumerate(col_labels):
        parts.append('<text x="%d" y="%d" fill="%s" font-size="15" '
                     'text-anchor="middle">%s</text>'
                     % (left + j * cell_w + cell_w / 2, top - 22, INK, esc(c)))
    parts.append('<text x="%d" y="%d" fill="%s" font-size="13" text-anchor="end">'
                 '族（题数）</text>' % (left - 10, top - 22, MUTED))
    for i, (fam, n, counts) in enumerate(rows):
        y = top + i * cell_h
        parts.append('<text x="%d" y="%.1f" fill="%s" font-size="15" '
                     'text-anchor="end">%s (%d)</text>'
                     % (left - 10, y + cell_h / 2 + 5, INK, esc(fam), n))
        for j, v in enumerate(counts):
            frac = v / float(n)
            # white -> accent green
            r = int(255 - (255 - 0x76) * frac)
            g = int(255 - (255 - 0xb9) * frac)
            b = int(255 - (255 - 0x00) * frac)
            fg = "#ffffff" if frac > 0.62 else INK
            parts.append('<rect x="%d" y="%d" width="%d" height="%d" fill="rgb(%d,%d,%d)" '
                         'stroke="#ffffff" stroke-width="1.5"/>'
                         % (left + j * cell_w, y, cell_w, cell_h, r, g, b))
            parts.append('<text x="%d" y="%.1f" fill="%s" font-size="15" '
                         'text-anchor="middle">%d</text>'
                         % (left + j * cell_w + cell_w / 2, y + cell_h / 2 + 5, fg, v))
    svg = ('<figure class="chart">\n'
           '<svg viewBox="0 0 %d %d" role="img" aria-label="%s" width="100%%">\n'
           '%s</svg>\n<figcaption>%s</figcaption>\n</figure>\n'
           % (width, height, esc(title), "\n".join(parts) + "\n", esc(title)))
    return svg


# ---------------------------------------------------------------- data

TABLE_A = [
    # model, latency solved, latency p@1, throughput solved, throughput p@1
    ("Next-Q", 634, 69.4, 131, 87.9),
    ("Next-M", 629, 66.4, 130, 85.4),
    ("GL-A",   619, 57.5, 126, 72.5),
    ("GL",     580, 51.5, 123, 68.9),
    ("G4t",    569, 49.4, 124, 71.0),
    ("Q38",    366, 27.1, 121, 54.3),
    ("Next",   166,  8.5,  59, 16.2),
]

TABLE_B = [
    ("Next-Q", 634, 69.3, 130, 87.4),
    ("Next-M", 632, 66.6, 130, 87.8),
    ("Q38nt",  398, 24.3, 110, 45.5),
    ("G4",     354, 29.2,  95, 49.6),
    ("Next",   182,  9.4,  63, 19.6),
]

FAMILY = [  # family, n, [Next, GL, G4t, Q38, Next-M, Next-Q, GL-A]
    ("conv",        209, [34, 146, 119, 23, 174, 174, 156]),
    ("activation",  150, [55, 141, 136, 130, 132, 135, 138]),
    ("elementwise", 105, [34, 89, 91, 86, 96, 98, 95]),
    ("matmul",      102, [13, 55, 73, 52, 72, 69, 63]),
    ("norm",        102, [23, 78, 85, 37, 71, 75, 87]),
    ("pool",         75, [2, 49, 40, 19, 64, 63, 57]),
    ("reduction",    22, [3, 18, 20, 14, 17, 17, 18]),
    ("loss",          5, [2, 4, 5, 5, 3, 3, 5]),
]
FAMILY_COLS = ["Next", "GL", "G4t", "Q38", "Next-M", "Next-Q", "GL-A"]

SHAPE = [  # model, common solved, awkward solved
    ("Next-Q", 490, 144), ("Next-M", 482, 147), ("GL-A", 475, 144),
    ("GL", 448, 132), ("G4t", 439, 130), ("Q38", 279, 87), ("Next", 129, 37),
]

LAT_SPEED = [  # model, own-set median, intersection median
    ("Next", 2.43, 4.42), ("GL", 2.33, 5.15), ("G4t", 2.22, 5.18),
    ("Q38", 3.21, 4.35), ("Next-M", 2.35, 4.75), ("Next-Q", 2.41, 4.71),
    ("GL-A", 2.30, 4.66),
]

BANDWIDTH = [  # model, cutile GB/s, torch.compile GB/s
    ("Next-Q", 5824, 4754), ("GL-A", 5889, 4754), ("GL", 5700, 4784),
]


def build():
    charts = {}

    charts["A_latency"] = grouped_bars(
        "图 2.1　表 A 延迟轨：pass@4 与 pass@1",
        [m for m, *_ in TABLE_A],
        [("pass@4", GREEN,
          {m: (s / 770.0, "%.1f%%（%d 题）" % (100.0 * s / 770, s))
           for m, s, _, _, _ in TABLE_A}),
         ("pass@1", BLUE,
          {m: (p / 100.0, "%.1f%%" % p) for m, _, p, _, _ in TABLE_A})])

    charts["A_throughput"] = grouped_bars(
        "图 2.2　表 A 吞吐轨：pass@4 与 pass@1",
        [m for m, *_ in TABLE_A],
        [("pass@4", GREEN,
          {m: (s / 139.0, "%.1f%%（%d 题）" % (100.0 * s / 139, s))
           for m, _, _, s, _ in TABLE_A}),
         ("pass@1", BLUE,
          {m: (p / 100.0, "%.1f%%" % p) for m, _, _, _, p in TABLE_A})])

    charts["family"] = heatmap(
        "图 2.3　按算子族的解出题数（颜色深浅 = 该族解出比例）",
        FAMILY_COLS, FAMILY, None)

    charts["failure"] = stacked_bars(
        "图 2.4　3636 个样本的去向",
        [("GL", {"通过": 1970, "纯度失败": 970, "跑不起来": 657, "超时": 39}),
         ("GL-A", {"通过": 2175, "纯度失败": 791, "跑不起来": 642, "超时": 28})],
        [("通过", GREEN), ("纯度失败", AMBER), ("跑不起来", RED), ("超时", GREY)],
        3636)

    charts["shape"] = grouped_bars(
        "图 2.5　常见形状与不规则形状的解出率",
        [m for m, *_ in SHAPE],
        [("常见（583 道）", GREEN,
          {m: (c / 583.0, "%.0f%%" % (100.0 * c / 583)) for m, c, _ in SHAPE}),
         ("不规则（187 道）", BLUE,
          {m: (a / 187.0, "%.0f%%" % (100.0 * a / 187)) for m, _, a in SHAPE})])

    mx = 5.5
    charts["lat_speed"] = grouped_bars(
        "图 2.6　延迟轨中位加速比：同一批 kernel，换个统计集合就差一倍",
        [m for m, *_ in LAT_SPEED],
        [("自己解出集", GREY,
          {m: (o / mx, "%.2fx" % o) for m, o, _ in LAT_SPEED}),
         ("七模型共同 101 题", GREEN,
          {m: (i / mx, "%.2fx" % i) for m, _, i in LAT_SPEED})],
        ticks=(0, 1 / mx, 2 / mx, 3 / mx, 4 / mx, 5 / mx),
        tick_fmt=lambda t: "%.0fx" % round(t * mx))

    bmax = 8000.0
    charts["bandwidth"] = grouped_bars(
        "图 2.8　吞吐轨达成带宽中位（GB200 单卡标称约 8000 GB/s）",
        [m for m, *_ in BANDWIDTH],
        [("生成的 cuTile kernel", GREEN,
          {m: (c / bmax, "%d GB/s" % c) for m, c, _ in BANDWIDTH}),
         ("torch.compile 参考", GREY,
          {m: (r / bmax, "%d GB/s" % r) for m, _, r in BANDWIDTH})],
        ticks=(0, 0.25, 0.5, 0.75, 1.0),
        tick_fmt=lambda t: "%d" % (t * bmax))

    # share of solved problems faster / within noise / slower than torch.compile
    lat_dist = [("Next", 86.1, 10.8), ("GL", 82.8, 14.0), ("G4t", 84.4, 13.0),
                ("Q38", 92.9, 6.3), ("Next-M", 85.5, 10.5), ("Next-Q", 87.2, 9.1),
                ("GL-A", 76.7, 19.7)]
    thr_dist = [("Next", 62.7, 25.4), ("GL", 48.8, 30.1), ("G4t", 63.7, 19.4),
                ("Q38", 65.3, 24.8), ("Next-M", 63.1, 20.0), ("Next-Q", 60.3, 19.8),
                ("GL-A", 63.5, 20.6)]
    for key, data, label in (("lat_dist", lat_dist, "延迟轨"),
                             ("thr_dist", thr_dist, "吞吐轨")):
        charts[key] = stacked_bars(
            "图 %s　%s：解出的题里，比 torch.compile 快 / 打平 / 慢的占比"
            % ("2.7" if key == "lat_dist" else "2.9", label),
            [(m, {"快过 compile": f, "±5% 以内": round(100 - f - s, 1),
                  "慢于 compile": s}) for m, f, s in data],
            [("快过 compile", GREEN), ("±5% 以内", GREY), ("慢于 compile", RED)],
            100, row_h=34)

    TEAL = "#2f8f7d"
    charts["chain"] = chain_bars(
        "图 3.1　配方链条：KernelBench 200 题上的 k=4 解出数（当时的尺子）",
        [("Next（基座）", 47, GREY, ""),
         ("B　拒绝采样 SFT ×2", 52, GREEN, "第二、四阶段"),
         ("E　第一次 GRPO", 51, BLUE, "6.7 GPU 小时，零提升"),
         ("F　训练时撤掉文档包", 84, GREEN, "prompt 14.9k → 2.4k token"),
         ("H　自蒸馏", 108, TEAL, "三个口径同时涨"),
         ("J　在自蒸馏之上叠 GRPO", 127, BLUE, "增益没被吃掉"),
         ("K　frontier 类别配额", 134, BLUE, "补上缺失的算子族"),
         ("L　数值错分级 reward", 144, BLUE, "策略漂移降到 1/54"),
         ("M　纯度分级 + 链尾任务", 150, BLUE, "pass@1 首次过半"),
         ("Q　第二次自蒸馏", 141, TEAL, "pass@1 最高，覆盖面变窄")],
        [("基座", GREY), ("SFT", GREEN), ("自蒸馏", TEAL), ("GRPO", BLUE)],
        160)

    charts["B_throughput"] = grouped_bars(
        "图 2.11　表 B 吞吐轨：pass@4 与 pass@1",
        [m for m, *_ in TABLE_B],
        [("pass@4", GREEN,
          {m: (s / 139.0, "%.1f%%（%d 题）" % (100.0 * s / 139, s))
           for m, _, _, s, _ in TABLE_B}),
         ("pass@1", BLUE,
          {m: (p / 100.0, "%.1f%%" % p) for m, _, _, _, p in TABLE_B})])

    charts["B_tracks"] = grouped_bars(
        "图 2.10　表 B 延迟轨（关 thinking，8K）：pass@4 与 pass@1",
        [m for m, *_ in TABLE_B],
        [("pass@4", GREEN,
          {m: (s / 770.0, "%.1f%%（%d 题）" % (100.0 * s / 770, s))
           for m, s, _, _, _ in TABLE_B}),
         ("pass@1", BLUE,
          {m: (p / 100.0, "%.1f%%" % p) for m, _, p, _, _ in TABLE_B})])

    return charts


def main():
    charts = build()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts.html")
    with io.open(out, "w", encoding="utf-8") as f:
        for k, v in charts.items():
            f.write("<!--CHART %s-->\n%s\n" % (k, v))
    print("wrote", out, "with", len(charts), "charts")


if __name__ == "__main__":
    main()
