"""
make_figure_robustness.py: Figure 5.

Boundary-invalid recall rate on the cross-boundary probes of the unsafe rungs,
pooled over the twelve conditions, (a) by the bridge category of the trap and
(b) by the number of distinct boundaries spanned by the trap's five source
fragments, with Wilson 95% intervals. Scoring uses the HARDENED predicate of
rescore_final_ci.py, the scorer behind Table 1. Bar opacity grows with the
number of responses n; bars at 100% are drawn darker, with bold labels.

Usage, from the repository root (needs matplotlib):
  python3 analysis/make_figure_robustness.py --out "$(mktemp -d)/fig_robustness.pdf" > analysis/make_figure_robustness.log

Writes the figure to --out, by default fig_robustness.pdf in a new temporary
folder, and prints the plotted counts. The released figure is
paper/fig_robustness.pdf. The PDF's bytes depend on the matplotlib version and
the fonts installed; the counts do not.
"""
import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from rescore_final_ci import boundary_invalid, load_store, wilson_ci

ROOT = Path(__file__).resolve().parent.parent
UNSAFE = ("shared_vector_index", "summary_merge")
CATEGORIES = {"ip": "Shared IP", "hash": "Malware hash", "cve": "CVE", "vendor": "Vendor", "actor": "Threat actor"}

BAR = "#8A8A8E"
BAR_EMPH = "#3A3A3C"
ERR = "#1C1C1E"
GRID = "#D9D9DE"
TEXT = "#1C1C1E"
FS_BODY = 7.0
FS_TICK = 7.0
FS_PANEL = 8.0
FS_VALUE = 6.5
FS_N = 7.0

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.edgecolor": TEXT,
})


def load_counts(data_dir: Path):
    """Boundary-invalid count k and number n of unsafe-rung trap responses, by category and by span."""
    by_cat = defaultdict(lambda: {"k": 0, "n": 0})
    by_span = defaultdict(lambda: {"k": 0, "n": 0})
    for cd in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        store = load_store(cd)
        with (cd / "probes.jsonl").open() as f:
            probes = {p["probe_id"]: p for p in map(json.loads, f)}
        with (cd / "responses.jsonl").open() as f:
            for r in map(json.loads, f):
                if r["probe_type"] != "cross_boundary" or r["rung"] not in UNSAFE:
                    continue
                probe = probes[r["probe_id"]]
                hit = boundary_invalid(r, store)
                if probe.get("bridge_context"):
                    cat = CATEGORIES[probe["bridge_context"][0].split(":", 1)[0]]
                    by_cat[cat]["n"] += 1
                    by_cat[cat]["k"] += hit
                span = len({store[pid]["boundary_id"] for pid in probe["fragment_provenance_ids"]})
                by_span[span]["n"] += 1
                by_span[span]["k"] += hit
    return by_cat, by_span


def opacity_for_n(n, n_max, lo=0.45, hi=1.0):
    return lo + (hi - lo) * math.sqrt(n / n_max) if n_max > 0 else hi


def fmt_pct(p):
    v = 100 * p
    return f"{int(round(v))}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def panel(ax, labels, ks, ns, xlab=""):
    n_max = max(ns) if ns else 1
    xs = list(range(len(labels)))
    for y in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(y, color=GRID, linewidth=0.5, zorder=1)
    for x, k, n in zip(xs, ks, ns):
        p = k / n
        lo, hi = wilson_ci(k, n)
        emph = k == n
        ax.bar(x, p, width=0.30, color=BAR_EMPH if emph else BAR, alpha=opacity_for_n(n, n_max),
               edgecolor="none", zorder=2)
        ax.plot([x, x], [lo, hi], color=ERR, linewidth=0.8, zorder=3, solid_capstyle="butt")
        ax.plot([x - 0.10, x + 0.10], [lo, lo], color=ERR, linewidth=0.8, zorder=3)
        ax.plot([x - 0.10, x + 0.10], [hi, hi], color=ERR, linewidth=0.8, zorder=3)
        ax.text(x, min(hi + 0.04, 1.08), f"{fmt_pct(p)}%", ha="center", va="bottom", fontsize=FS_VALUE,
                color=TEXT, fontweight="bold" if emph else "normal")
        ax.text(x, -0.07, f"n={n}", ha="center", va="top", fontsize=FS_N, color=TEXT)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FS_TICK, color=TEXT)
    ax.tick_params(axis="x", length=2.5, pad=13)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=FS_TICK, color=TEXT)
    ax.tick_params(axis="y", length=2.5, pad=3)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0, 1.18)
    if xlab:
        ax.set_xlabel(xlab, fontsize=FS_BODY, color=TEXT, labelpad=14)


def main():
    ap = argparse.ArgumentParser(description="Figure 5.")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, help="PDF to write (default: fig_robustness.pdf in a new temporary folder)")
    ap.add_argument("--width", type=float, default=3.30, help="figure width in inches")
    args = ap.parse_args()
    by_cat, by_span = load_counts(args.data)
    cats = list(CATEGORIES.values())
    spans = sorted(by_span)

    print("Figure 5: boundary-invalid recall on unsafe-rung trap responses, 12 conditions pooled")
    print("(a) by bridge category")
    for c in cats:
        k, n = by_cat[c]["k"], by_cat[c]["n"]
        print(f"  {c:13s} {k}/{n} = {100 * k / n:.2f}%")
    print("(b) by number of distinct boundaries spanned")
    for s in spans:
        k, n = by_span[s]["k"], by_span[s]["n"]
        print(f"  {s} boundaries  {k}/{n} = {100 * k / n:.2f}%")

    fig = plt.figure(figsize=(args.width, 1.78), dpi=300)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, left=0.115, right=0.99, top=0.84, bottom=0.30, wspace=0.08, width_ratios=[5.6, 4])
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    panel(ax_a, ["Shared\nIP", "Malware\nhash", "CVE", "Vendor", "Threat\nactor"],
          [by_cat[c]["k"] for c in cats], [by_cat[c]["n"] for c in cats])
    panel(ax_b, [str(s) for s in spans], [by_span[s]["k"] for s in spans], [by_span[s]["n"] for s in spans],
          xlab="Distinct boundaries spanned")
    ax_a.set_ylabel("Boundary-invalid recall rate (%)", fontsize=FS_BODY - 0.5, color=TEXT, labelpad=2)
    ax_b.set_yticklabels([])
    ax_a.tick_params(axis="x", pad=11)
    ax_b.tick_params(axis="x", pad=11)
    ax_b.xaxis.label.set_size(FS_BODY - 0.5)
    ax_b.xaxis.labelpad = 4
    for ax, tag, title, x in ((ax_a, "(a)", "Failure rate by bridge category", 0.13),
                              (ax_b, "(b)", "Failure rate by boundary span", 0.16)):
        ax.text(0.0, 1.13, tag, transform=ax.transAxes, fontsize=FS_PANEL - 1, fontweight="bold", color=TEXT,
                va="bottom")
        ax.text(x, 1.13, title, transform=ax.transAxes, fontsize=FS_BODY - 0.5, color=TEXT, va="bottom")
    out = (args.out or Path(tempfile.mkdtemp(prefix="fig_robustness_")) / "fig_robustness.pdf").resolve()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03, facecolor="white", metadata={"CreationDate": None})
    print("wrote", out.relative_to(ROOT) if out.is_relative_to(ROOT) else out, file=sys.stderr)


if __name__ == "__main__":
    main()
