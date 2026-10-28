"""
pick_rate_winrate_scatter.py — 1080x1080 card: hero pick rate (x) vs win rate (y),
with a "balanced zone" box:  49–51% win rate  x  median pick rate ± k·SD.

Usage (same contract as the other scripts, so gen_visuals.py can call it):
    python3 pick_rate_winrate_scatter.py [week_dir] [--highlight "Victor,Mirage"] [--progress 0.5]

Reusable API for video/animation:
    df, box = prepare(pd.read_csv(...))
    fig = render(df, box, meta, progress=0.4, highlight={"Victor"})
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

import theme as T
from chart_stats import MIN_N, add_pick_rate, add_winrate_ci, total_matches_from_hero_rows

# --- CONFIG ------------------------------------------------------------------
CSV_NAME = "hero_winrates.csv"
OUTPUT_NAME = "pick_rate_winrate_scatter.png"
TITLE = "PICK RATE vs WIN RATE"

BOX_WR = (49.0, 51.0)       # balanced win-rate band (%)
BOX_PICK_METHOD = "iqr"     # "iqr": Q1–Q3 (middle 50% of heroes) | "sd": median ± K·SD
BOX_PICK_K = 1.0            # only used by "sd"
FIT_LIMITS = True           # zoom so the extreme heroes sit on the borders
SYMMETRIC_WR = False        # True: keep 50% centred vertically (honest above/below balance)
EDGE_PAD_PX = 34            # room past the extreme heroes for icon radius + whiskers
ICON_PX = 50                # icon diameter on the card
DOT_PX = 22                 # fallback dot diameter when an icon is missing
DIM_ALPHA = 0.18            # non-highlighted heroes
SHOW_CI = True              # vertical 95% CI whiskers

QUAD_BLUE = "#60a5fa"       # not an archetype color (assassin cyan / mystic purple)
QUAD_YELLOW = "#facc15"
QUADRANT_TINT_ALPHA = T.ZONE_TINT_ALPHA   # shared with the other charts (theme.py)
QUADRANT_LABELS = {          # (axes-fraction x, y, ha, va, color)
    "HIDDEN GEMS\nstrong, rarely picked": (0.015, 0.985, "left", "top", QUAD_BLUE),
    "META PICKS\nstrong & popular": (0.985, 0.985, "right", "top", T.GREEN),
    "STRUGGLING\nweak & rarely picked": (0.015, 0.015, "left", "bottom", T.RED),
    "POPULAR TRAPS\npicked a lot, losing a lot": (0.985, 0.015, "right", "bottom", QUAD_YELLOW),
}


# --- DATA --------------------------------------------------------------------
def prepare(raw: pd.DataFrame, method: str = BOX_PICK_METHOD,
            box_k: float = BOX_PICK_K) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    df = add_pick_rate(df)
    df = add_winrate_ci(df)

    ok = df[df["sufficient"]]["pick_rate"]
    med = float(ok.median())
    if method == "iqr":
        x0, x1 = float(ok.quantile(0.25)), float(ok.quantile(0.75))
        rule = "middle 50% of pick rates"
    else:
        sd = float(ok.std(ddof=1))
        x0, x1 = max(0.0, med - box_k * sd), med + box_k * sd
        rule = f"median ± {box_k:g} SD"
    box = {"x0": x0, "x1": x1, "y0": BOX_WR[0], "y1": BOX_WR[1],
           "median_pick": med, "rule": rule}
    df["in_box"] = (df["pick_rate"].between(box["x0"], box["x1"])
                    & df["winrate"].between(box["y0"], box["y1"]) & df["sufficient"])
    return df.reset_index(drop=True), box


def auto_subtitle(df: pd.DataFrame) -> str:
    n_ok = int(df["sufficient"].sum())
    k = int(df["in_box"].sum())
    return f"Only {k} of {n_ok} heroes land in the balanced zone"


# --- AXIS LIMITS -------------------------------------------------------------
def _padded(lo: float, hi: float, pad_px: float, axis_px: float) -> tuple[float, float]:
    """Expand [lo, hi] so pad_px of screen space remains past each extreme."""
    span = hi - lo
    total = span / max(1 - 2 * pad_px / axis_px, 0.5)
    extra = (total - span) / 2
    return lo - extra, hi + extra


def compute_limits(df: pd.DataFrame, ax, fit: bool = FIT_LIMITS,
                   symmetric_wr: bool = SYMMETRIC_WR):
    """Fitted: the most/least picked and best/worst heroes define the borders."""
    ok = df[df["sufficient"]] if df["sufficient"].any() else df
    bb = ax.get_window_extent()
    if not fit:
        span = np.ceil(max((ok["wr_hi"] - 50).abs().max(), (ok["wr_lo"] - 50).abs().max()) + 0.6)
        return (0, np.ceil((ok["pick_rate"].max() + 6) / 10) * 10), (50 - span, 50 + span)

    xlim = _padded(ok["pick_rate"].min(), ok["pick_rate"].max(), EDGE_PAD_PX, bb.width)
    y_lo, y_hi = ok["wr_lo"].min(), ok["wr_hi"].max()   # include whiskers, not just dots
    if symmetric_wr:
        half = max(50 - y_lo, y_hi - 50)
        y_lo, y_hi = 50 - half, 50 + half
    ylim = _padded(y_lo, y_hi, EDGE_PAD_PX * 0.6, bb.height)
    # keep the 50% line and the balanced box on screen
    ylim = (min(ylim[0], BOX_WR[0] - 0.3), max(ylim[1], BOX_WR[1] + 0.3))
    return xlim, _clear_corner_labels(ok, xlim, ylim, bb)


CORNER_W_FRAC = 0.30        # corner label zone: 30% of plot width...
CORNER_H_PX = 62            # ...by this many pixels tall


def _clear_corner_labels(ok: pd.DataFrame, xlim, ylim, bb):
    """Push the top/bottom border out if an extreme hero would sit under a corner label."""
    x_lo, x_hi = xlim
    left_edge = x_lo + (x_hi - x_lo) * CORNER_W_FRAC
    right_edge = x_hi - (x_hi - x_lo) * CORNER_W_FRAC
    in_corner_x = (ok["pick_rate"] <= left_edge) | (ok["pick_rate"] >= right_edge)
    y_lo, y_hi = ylim
    r_px = ICON_PX / 2
    for _ in range(4):                       # a couple of passes converge
        per_px = (y_hi - y_lo) / bb.height
        band = (CORNER_H_PX + r_px) * per_px
        bottom = ok.loc[in_corner_x, "wr_lo"].min() if in_corner_x.any() else np.inf
        top = ok.loc[in_corner_x, "wr_hi"].max() if in_corner_x.any() else -np.inf
        new_lo = min(y_lo, bottom - band) if bottom < y_lo + band else y_lo
        new_hi = max(y_hi, top + band) if top > y_hi - band else y_hi
        if (new_lo, new_hi) == (y_lo, y_hi):
            break
        y_lo, y_hi = new_lo, new_hi
    return y_lo, y_hi


# --- LABEL PLACEMENT ---------------------------------------------------------
def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if (w > 0 and h > 0) else 0.0


def place_labels(ax, xs, ys, names, font_px: float, marker_r: float):
    """Greedy: try 8 spots around each point, keep the one with the least overlap."""
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    ax_box = ax.get_window_extent()
    markers = [(x - marker_r, y - marker_r, x + marker_r, y + marker_r) for x, y in pts]
    gap = marker_r + 3
    cands = [(gap, 0, "left", "center"), (-gap, 0, "right", "center"),
             (0, gap, "center", "bottom"), (0, -gap, "center", "top"),
             (gap * .75, gap * .75, "left", "bottom"), (-gap * .75, gap * .75, "right", "bottom"),
             (gap * .75, -gap * .75, "left", "top"), (-gap * .75, -gap * .75, "right", "top")]
    placed, out = [], []
    for i, ((px, py), name) in enumerate(zip(pts, names)):
        w, h = len(name) * font_px * 0.62, font_px * 1.15
        best, best_cost = None, None
        for dx, dy, ha, va in cands:
            x0 = px + dx - (w if ha == "right" else w / 2 if ha == "center" else 0)
            y0 = py + dy - (h if va == "top" else h / 2 if va == "center" else 0)
            bb = (x0, y0, x0 + w, y0 + h)
            cost = sum(_overlap(bb, o) for o in placed)
            cost += sum(_overlap(bb, m) for j, m in enumerate(markers) if j != i) * 0.6
            if bb[0] < ax_box.x0 or bb[2] > ax_box.x1 or bb[1] < ax_box.y0 or bb[3] > ax_box.y1:
                cost += 1e6
            if best_cost is None or cost < best_cost:
                best, best_cost = (dx, dy, ha, va, bb), cost
            if cost == 0:
                break
        placed.append(best[4])
        out.append(best[:4])
    return out


# --- RENDER ------------------------------------------------------------------
def render(df: pd.DataFrame, box: dict, meta: T.ChartMeta, progress: float = 1.0,
           highlight: set[str] | None = None, title: str = TITLE,
           subtitle: str | None = None,
           limits: tuple[tuple[float, float], tuple[float, float]] | None = None) -> "T.Figure":
    """progress 0..1: frame/box fade in, then heroes fly from the center to their spots.
    Pass `limits` to lock the axes (e.g. the union of two weeks for a week-over-week animation)."""
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0.00, 0.15))
    p_box = T.ease_out_cubic(T.phase(progress, 0.08, 0.28))

    fig, ax = T.new_card()
    T.draw_header(fig, title, subtitle or auto_subtitle(df), alpha=p_frame)
    legend_end = T.draw_archetype_legend(fig, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    xlim, ylim = limits or compute_limits(df, ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Pick rate  (% of matches the hero is in)", labelpad=12)
    ax.set_ylabel("Win rate", labelpad=10)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_alpha(p_frame)

    # Quadrant tints (split at median pick rate and 50% WR)
    mx = box["median_pick"]
    for (qx0, qx1, qy0, qy1, col) in [
        (xlim[0], mx, 50, ylim[1], QUAD_BLUE), (mx, xlim[1], 50, ylim[1], T.GREEN),
        (xlim[0], mx, ylim[0], 50, T.RED), (mx, xlim[1], ylim[0], 50, QUAD_YELLOW),
    ]:
        ax.add_patch(Rectangle((qx0, qy0), qx1 - qx0, qy1 - qy0, facecolor=col, edgecolor="none",
                               alpha=QUADRANT_TINT_ALPHA * p_frame, zorder=0.5))

    # Quadrant split lines (50% WR, median pick) + corner labels
    ax.axhline(50, color=T.DIVIDER, lw=1.6, zorder=1, alpha=p_frame)
    ax.axvline(box["median_pick"], color=T.DIVIDER, lw=1.6, ls=(0, (6, 4)), zorder=1, alpha=p_frame)
    for text, (qx, qy, ha, va, col) in QUADRANT_LABELS.items():
        head, sub = text.split("\n")
        head_y, sub_y = (qy, qy - 0.03) if va == "top" else (qy + 0.028, qy)
        ax.text(qx, head_y, head, transform=ax.transAxes, ha=ha, va=va, color=col,
                fontsize=T.px_to_pt(17), fontweight="bold", alpha=0.75 * p_frame, zorder=2)
        ax.text(qx, sub_y, sub, transform=ax.transAxes, ha=ha, va=va, color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(13), alpha=0.7 * p_frame, zorder=2)

    # Balanced zone
    ax.add_patch(Rectangle((box["x0"], box["y0"]), box["x1"] - box["x0"], box["y1"] - box["y0"],
                           facecolor=T.GOLD, alpha=0.08 * p_box, edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((box["x0"], box["y0"]), box["x1"] - box["x0"], box["y1"] - box["y0"],
                           facecolor="none", edgecolor=T.GOLD, lw=2.2, ls=(0, (7, 4)),
                           alpha=0.9 * p_box, zorder=3))
    # Zone definition lives in the legend row (keeps the plot area free for hero labels)
    zone = fig.text(1 - T.MARGIN_X, T.HEADER_LEGEND_Y,
                    f"Balanced zone: {box['y0']:.0f}–{box['y1']:.0f}% WR · "
                    f"{box['x0']:.0f}–{box['x1']:.0f}% pick ({box['rule']})",
                    color=T.GOLD, fontsize=T.px_to_pt(15), fontweight="bold",
                    ha="right", va="center", alpha=p_box)
    for size in (14, 13, 12, 11):      # shrink until it clears the archetype legend
        if 1 - T.MARGIN_X - T.text_width_frac(fig, zone) > legend_end:
            break
        zone.set_fontsize(T.px_to_pt(size))

    # Heroes: staggered fly-in from the centre point, strongest first
    order = df.sort_values("winrate", ascending=False).index.tolist()
    n = len(order)
    start_xy = np.array([box["median_pick"], 50.0])
    xs, ys, names, label_alpha, is_hl = [], [], [], [], []
    for rank, i in enumerate(order):
        row = df.loc[i]
        s = 0.25 + 0.45 * rank / max(n - 1, 1)
        t = T.ease_out_cubic(T.phase(progress, s, s + 0.30))
        if t <= 0:
            continue
        end_xy = np.array([row["pick_rate"], row["winrate"]])
        x, y = start_xy + (end_xy - start_xy) * t
        hl = highlight is None or row["hero_name"].lower() in highlight
        a = t * (1.0 if hl else DIM_ALPHA)
        ring = T.archetype_color(row["hero_type"]) if row["sufficient"] else T.NEUTRAL

        if SHOW_CI and t >= 1.0:
            ax.vlines(x, row["wr_lo"], row["wr_hi"], color=T.TEXT, lw=2, alpha=0.35 * a, zorder=4)

        icon = T.hero_icon(row["hero_name"], ICON_PX, ring, round(a, 2),
                           state=T.icon_state(row["winrate"]))
        if icon is not None:
            ab = AnnotationBbox(OffsetImage(icon, zoom=1, dpi_cor=False), (x, y),
                                frameon=False, zorder=8 if hl else 5)
            ax.add_artist(ab)
            r = ICON_PX / 2
        else:
            size = (T.px_to_pt(DOT_PX)) ** 2
            face = ring if row["sufficient"] else "none"
            ax.scatter([x], [y], s=size, color=face, edgecolors=ring if not row["sufficient"] else T.BG,
                       linewidths=1.5, alpha=a, zorder=8 if hl else 5)
            r = DOT_PX / 2

        xs.append(x); ys.append(y); names.append(row["hero_name"])
        label_alpha.append(T.phase(t, 0.85, 1.0) * (1.0 if hl else DIM_ALPHA)); is_hl.append(hl)

    # Name labels (all heroes normally; only highlighted ones when highlighting)
    if xs:
        marker_r = (ICON_PX if T.ICON_DIR.exists() else DOT_PX) / 2
        if highlight:
            idx = [k for k, h in enumerate(is_hl) if h]
            font = T.SIZE_LABEL_BIG
        else:
            idx = list(range(len(xs)))
            font = T.SIZE_LABEL
        spots = place_labels(ax, [xs[k] for k in idx], [ys[k] for k in idx],
                             [names[k] for k in idx], font, marker_r)
        for k, (dx, dy, ha, va) in zip(idx, spots):
            ax.annotate(names[k], (xs[k], ys[k]), xytext=(dx, dy), textcoords="offset pixels",
                        ha=ha, va=va, color=T.TEXT, fontsize=T.px_to_pt(font), fontweight="bold",
                        alpha=label_alpha[k], zorder=9, path_effects=T.STROKE)
    return fig


# --- CLI ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--highlight", default="", help='comma-separated hero names, e.g. "Victor,Mirage"')
    ap.add_argument("--progress", type=float, default=1.0)
    ap.add_argument("--box", choices=["iqr", "sd"], default=BOX_PICK_METHOD)
    ap.add_argument("--k", type=float, default=BOX_PICK_K, help="for --box sd: median ± k*SD")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    raw = pd.read_csv(data_dir / CSV_NAME)
    df, box = prepare(raw, method=args.box, box_k=args.k)
    meta = T.ChartMeta(week_label=T.week_label_from_dir(data_dir),
                       n_matches=total_matches_from_hero_rows(df["total_matches"]), min_n=MIN_N)

    highlight = {h.strip() for h in args.highlight.split(",") if h.strip()} or None
    fig = render(df, box, meta, progress=args.progress, highlight=highlight)
    out = Path(args.out) if args.out else data_dir / "graphs" / OUTPUT_NAME
    T.save_card(fig, out)
    print(f"Saved {out}  | balanced zone: pick {box['x0']:.1f}–{box['x1']:.1f}%, "
          f"{int(df['in_box'].sum())} heroes inside")


if __name__ == "__main__":
    main()