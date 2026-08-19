"""Replace already-injected figures in the report with freshly generated ones.

Figures are matched by their number ("图 2.1"), so captions and data can change
without breaking the match, and running this twice is a no-op.
"""

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "index.html")
CHARTS = os.path.join(HERE, "charts.html")

FIG_NO = re.compile(r"<figcaption>(图 [\d.]+)")


def figures(text):
    """Map figure number -> full <figure> block."""
    out = {}
    for m in re.finditer(r'<figure class="chart">.*?</figure>', text, re.S):
        n = FIG_NO.search(m.group(0))
        if n:
            out[n.group(1)] = m.group(0)
    return out


def main():
    fresh = figures(io.open(CHARTS, encoding="utf-8").read())
    html = io.open(REPORT, encoding="utf-8").read()
    current = figures(html)

    missing = set(fresh) - set(current)
    if missing:
        sys.exit("figures not present in report: %s" % sorted(missing))

    n = 0
    for num, block in fresh.items():
        if current[num] != block:
            html = html.replace(current[num], block)
            n += 1
    io.open(REPORT, "w", encoding="utf-8").write(html)
    print("refreshed %d of %d figures" % (n, len(fresh)))


if __name__ == "__main__":
    main()
