"""
match_duration.py — "How long is a ranked match?" card (1080x1080).

Defines the numbers every duration chart uses (see chart_stats.duration_summary):
    median, Q1 / Q3 (IQR = middle 50% of matches, gold), 12.5th / 87.5th percentiles
    (middle 75%, orange), early-game window = ends before Q1 (rounded down to a 5-min edge),
    late-game window = runs past Q3 (rounded up to a 5-min edge)

Compare mode overlays a second period (e.g. the week before a patch) as a white outline,
switches the y-axis to "% of matches" so different-sized weeks compare fairly, and puts the
change next to every headline number.

Input, best first:
    match_duration_minutes.csv  (duration_min, games) — exact, 1-minute bins (new query)
    avg_duration.csv            (5-min buckets)       — fallback, quantiles estimated
Usage:
    python3 match_duration.py [week_dir]
    python3 match_duration.py week9_dir --compare week8_dir --compare-label "before patch"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

import theme as T
from chart_stats import LONG_GAME_MIN, SHORT_GAME_MIN, duration_summary

IQR_COLOR = T.GOLD           # middle 50%
MID75_COLOR = "#fb923c"      # middle 75% — orange, between the gold IQR and the teal tails
EARLY_COLOR = "#60a5fa"      # same blue family as "hidden gems"; not an archetype color
LATE_COLOR = "#c084fc"
BAR_COLOR = "#2f7d6d"
COMPARE_COLOR = T.TEXT


def fmt_min(m: float) -> str:
    secs = int(round(m * 60))            # round once, so 41.999 min shows 42:00, not 41:60
    return f"{secs // 60}:{secs % 60:02d}"


def fmt_delta_min(d: float) -> str:
    secs = int(round(d * 60))
    sign = "+" if secs > 0 else "−" if secs < 0 else "±"
    secs = abs(secs)
    return f"{sign}{secs // 60}:{secs % 60:02d}"


def _density(s: dict, as_pct: bool):
    """Per-minute height of every bin (so 1-min and 5-min inputs share one scale)."""
    hist = s["hist"]
    widths = (hist["end_min"] - hist["start_min"]).to_numpy(float)
    per_min = hist["games"].to_numpy(float) / widths
    return per_min / s["n"] * 100 if as_pct else per_min


def render(s: dict, meta: T.ChartMeta, progress: float = 1.0, cmp: dict | None = None,
           label: str = "this week", cmp_label: str = "before") -> "T.Figure":
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    p_bars = T.ease_out_cubic(T.phase(progress, 0.15, 0.55))
    p_75 = T.ease_out_cubic(T.phase(progress, 0.55, 0.72))
    p_iqr = T.ease_out_cubic(T.phase(progress, 0.68, 0.85))
    p_cmp = T.ease_out_cubic(T.phase(progress, 0.82, 1.0))
    hist = s["hist"]
    as_pct = cmp is not None

    fig, ax = T.new_card()
    est = "" if s["source"] == "exact" else "  ·  estimated from 5-min buckets"
    if cmp is None:
        T.draw_header(fig, f"HALF OF ALL GAMES LAST {s['q1']:.0f}–{s['q3']:.0f} MINUTES",
                      "Gold: middle 50% of matches  ·  orange: middle 75%" + est, alpha=p_frame)
        tiles = [
            (fmt_min(s["median"]), "median length", T.TEXT),
            (f"{s['q1']:.0f}–{s['q3']:.0f}", "middle 50% (min)", IQR_COLOR),
            (f"{s['p125']:.0f}–{s['p875']:.0f}", "middle 75% (min)", MID75_COLOR),
            (f"{s['short_share']:.0f}%", f"end before {SHORT_GAME_MIN} min", EARLY_COLOR),
            (f"{s['long_share']:.0f}%", f"go past {LONG_GAME_MIN} min", LATE_COLOR),
        ]
    else:
        d_med = s["median"] - cmp["median"]
        if abs(d_med) < 0.5:
            title = f"MATCH LENGTH BARELY MOVED {cmp_label.upper()}" if cmp_label.lower().startswith("since") \
                else "MATCH LENGTH BARELY MOVED"
        else:
            title = f"GAMES ARE {abs(d_med):.0f} MIN {'LONGER' if d_med > 0 else 'SHORTER'} THAN {cmp_label.upper()}"
        T.draw_header(fig, title, f"Filled bars: {label}  ·  white outline: {cmp_label}" + est, alpha=p_frame)
        pts = lambda d: f"{d:+.0f} pts".replace("-", "−")
        tiles = [
            (fmt_min(s["median"]), "median length", T.TEXT, fmt_delta_min(d_med)),
            (f"{s['q1']:.0f}–{s['q3']:.0f}", "middle 50% (min)", IQR_COLOR, f"was {cmp['q1']:.0f}–{cmp['q3']:.0f}"),
            (f"{s['p125']:.0f}–{s['p875']:.0f}", "middle 75% (min)", MID75_COLOR,
             f"was {cmp['p125']:.0f}–{cmp['p875']:.0f}"),
            (f"{s['short_share']:.0f}%", f"end before {SHORT_GAME_MIN} min", EARLY_COLOR,
             pts(s["short_share"] - cmp["short_share"])),
            (f"{s['long_share']:.0f}%", f"go past {LONG_GAME_MIN} min", LATE_COLOR,
             pts(s["long_share"] - cmp["long_share"])),
        ]
    T.draw_kpi_row(fig, tiles, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    heights = _density(s, as_pct)
    cmp_heights = _density(cmp, True) if cmp is not None else None
    lows = [hist["start_min"].min(), s["p10"] - 5] + ([cmp["p10"] - 5] if cmp else [])
    highs = [s["p90"] + 12] + ([cmp["p90"] + 12] if cmp else [])
    x_lo = float(np.floor(min(lows) / 5) * 5)
    x_hi = float(min(hist["end_min"].max(), np.ceil(max(highs) / 5) * 5))
    ax.set_xlim(x_lo, x_hi)
    peak = max(heights.max(), cmp_heights.max() if cmp_heights is not None else 0)
    y_max = peak * 1.24                                # headroom for the marker labels
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    if as_pct:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax.set_ylabel("% of matches (per minute of length)", labelpad=10)
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}"))
        ax.set_ylabel("Matches" if s["source"] == "exact" else "Matches per minute (bucket average)",
                      labelpad=10)
    ax.set_xlabel("Match length (minutes)", labelpad=10)
    ax.grid(axis="x", visible=False)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_alpha(p_frame)

    # zones: early / late tails, then the nested 75% and 50% bands
    for a, b, col, name in ((x_lo, s["early_cut"], EARLY_COLOR, "EARLY GAME"),
                            (s["late_cut"], x_hi, LATE_COLOR, "LATE GAME")):
        ax.axvspan(a, b, color=col, alpha=0.08 * p_75, lw=0, zorder=0.5)
        ax.text((a + b) / 2, y_max * 0.97, name, ha="center", va="top", color=col,
                fontsize=T.px_to_pt(20), fontweight="bold", alpha=p_75)
    ax.axvspan(s["p125"], s["p875"], color=MID75_COLOR, alpha=T.ZONE_TINT_ALPHA * 0.8 * p_75, lw=0, zorder=0.6)
    ax.axvspan(s["q1"], s["q3"], color=IQR_COLOR, alpha=T.ZONE_TINT_ALPHA * 1.1 * p_iqr, lw=0, zorder=0.7)

    # bars (grow up): gold inside the IQR, orange inside the 75% band, teal in the tails
    for (_, r), h in zip(hist.iterrows(), heights):
        if r["end_min"] <= x_lo or r["start_min"] >= x_hi:
            continue
        mid = (r["start_min"] + r["end_min"]) / 2
        if s["q1"] <= mid <= s["q3"]:          # bars recolor as each band animates in
            col, a = (IQR_COLOR if p_iqr > 0 else MID75_COLOR if p_75 > 0 else BAR_COLOR), 0.95
        elif s["p125"] <= mid <= s["p875"]:
            col, a = (MID75_COLOR if p_75 > 0 else BAR_COLOR), 0.92
        else:
            col, a = BAR_COLOR, 0.85
        ax.add_patch(Rectangle((r["start_min"] + 0.06, 0), (r["end_min"] - r["start_min"]) - 0.12,
                               h * p_bars, color=col, alpha=a, lw=0, zorder=2))

    # comparison period: white step outline + its median
    if cmp is not None:
        ch = cmp["hist"]
        xs = np.r_[ch["start_min"].to_numpy(float), ch["end_min"].iloc[-1]]
        ys = np.r_[cmp_heights, cmp_heights[-1]]
        ax.step(xs, ys, where="post", color=COMPARE_COLOR, lw=2.4, alpha=0.9 * p_cmp, zorder=3.5)
        ax.plot([cmp["median"]] * 2, [0, y_max * 0.78], color=COMPARE_COLOR, lw=1.8, ls=(0, (2, 3)),
                alpha=0.8 * p_cmp, zorder=3.6)
        ax.legend(handles=[Rectangle((0, 0), 1, 1, color=IQR_COLOR, alpha=0.95),
                           Line2D([0], [0], color=COMPARE_COLOR, lw=2.4)],
                  labels=[label, cmp_label], loc="upper right", bbox_to_anchor=(1.0, 0.9),
                  frameon=False, labelcolor=T.TEXT, fontsize=T.px_to_pt(17)).set_alpha(p_cmp)

    # quartile + median markers (top row) and middle-75% edges (lower row, so labels never collide)
    for key, lab, col, lw in (("q1", "Q1", IQR_COLOR, 2), ("median", "MEDIAN", T.TEXT, 3),
                              ("q3", "Q3", IQR_COLOR, 2)):
        x = s[key]
        ax.plot([x, x], [0, y_max * 0.84], color=col, lw=lw, ls="-" if key == "median" else (0, (6, 4)),
                alpha=p_iqr, zorder=4)
        ax.text(x, y_max * 0.855, f"{lab}\n{fmt_min(x)}", ha="center", va="bottom", color=col,
                fontsize=T.px_to_pt(18), fontweight="bold", alpha=p_iqr, zorder=5, linespacing=1.1,
                path_effects=T.STROKE)
    for key, ha in (("p125", "right"), ("p875", "left")):
        x = s[key]
        ax.plot([x, x], [0, y_max * 0.62], color=MID75_COLOR, lw=2, ls=(0, (3, 3)), alpha=p_75, zorder=4)
        ax.text(x + (-0.3 if ha == "right" else 0.3), y_max * 0.6, fmt_min(x), ha=ha, va="center",
                color=MID75_COLOR, fontsize=T.px_to_pt(17), fontweight="bold", alpha=p_75, zorder=5,
                path_effects=T.STROKE)

    # range arrows along the bottom: 75% (outer, orange) and IQR (inner, gold)
    for lo_k, hi_k, y, col, text, pk in (("p125", "p875", 0.035, MID75_COLOR, f"middle 75%: {s['mid75']:.1f} min", p_75),
                                         ("q1", "q3", 0.13, IQR_COLOR, f"middle 50%: {s['iqr']:.1f} min", p_iqr)):
        # dark arrow so it shows on top of the gold/orange bars it spans
        ax.annotate("", xy=(s[hi_k], y_max * y), xytext=(s[lo_k], y_max * y),
                    arrowprops=dict(arrowstyle="<->", color=T.BG, lw=2.8, alpha=pk), zorder=5)
        ax.text((s[lo_k] + s[hi_k]) / 2, y_max * (y + 0.018), text, ha="center", va="bottom",
                color=T.TEXT, fontsize=T.px_to_pt(17), fontweight="bold", alpha=pk, zorder=6,
                path_effects=T.STROKE)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--progress", type=float, default=1.0)
    ap.add_argument("--compare", default=None, help="second period's folder, e.g. the week before a patch")
    ap.add_argument("--label", default=None, help='legend name for week_dir (default: its week label)')
    ap.add_argument("--compare-label", default="before the patch")
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    s = duration_summary(data_dir)
    cmp = duration_summary(Path(args.compare)) if args.compare else None
    label = args.label or T.week_label_from_dir(data_dir)
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), s["n"])
    name = "match_duration_compare.png" if cmp else "match_duration.png"
    out = data_dir / "graphs" / name
    T.save_card(render(s, meta, args.progress, cmp, label, args.compare_label), out)
    msg = (f"median {s['median']:.1f}, IQR {s['q1']:.1f}–{s['q3']:.1f}, "
           f"75% {s['p125']:.1f}–{s['p875']:.1f} ({s['source']})")
    if cmp:
        msg += f"  vs {args.compare_label}: median {cmp['median']:.1f} ({s['median'] - cmp['median']:+.1f} min)"
    print(f"Saved {out}  | {msg}")


if __name__ == "__main__":
    main()