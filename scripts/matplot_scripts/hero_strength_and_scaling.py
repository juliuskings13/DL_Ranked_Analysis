"""
hero_strength_and_scaling.py — how hero win rate depends on match length (1080x1080 cards).

Three questions, three outputs (they deliberately don't overlap):

    hero_early_vs_late.png                 WHO WINS EARLY, WHO WINS LATE (performance + scaling)
        x = win rate in shorter games, y = win rate in longer games. Right of / above 50% =
        actually winning; above the diagonal = better late than early. A 45%→48% hero lands
        bottom-left (scales, still losing); a 51%→53% hero lands top-right.

    hero_duration_winrate_<archetype>.png  WHO IS WINNING AT EACH LENGTH (absolute)
        actual win rate per 5-min bracket, colored around 50%.

    hero_duration_scaling_<archetype>.png  WHO GETS BETTER OR WORSE AS GAMES GO ON (relative)
        win rate per bracket minus the hero's own average, so a weak hero can still be green
        where it is "less bad than usual".

Early / late = shorter / longer than the median match (chart_stats.PHASE_SPLIT; "quartiles"
compares the shortest vs longest 25%). The bracket containing the cut is split using the
per-minute match-length histogram, not rounded.

Input: hero_duration_scaling.parquet (or .csv) + avg_duration.csv + hero_winrates.csv
       (+ match_duration_minutes.csv if present, for exact quantiles)
Icons use each hero's solo win rate (critical / base / gloat variants).

Usage: python3 hero_strength_and_scaling.py [week_dir] [--highlight "Haze,Lady Geist"]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

import theme as T
from chart_stats import (DURATION_BRACKETS, _bracket_edges, duration_summary, load_hero_meta,
                         total_matches_from_hero_rows, wilson_ci)
from chart_stats import Insight, normal_cdf
from pick_rate_winrate_scatter import place_labels

BRACKETS = DURATION_BRACKETS
MIN_CELL_N = 300                             # heatmap cells below this are masked
MIN_PHASE_N = 1000                           # scatter: heroes need this many games on each side
DELTA_SPAN = 5.0                             # scaling heatmap color range: ±5 points
WR_SPAN = 6.0                                # win-rate heatmap color range: 50 ± 6
DIM_ALPHA = 0.22
SCATTER_ICON_PX = 46
EARLY_COLOR, LATE_COLOR = "#60a5fa", "#c084fc"   # same as match_duration.py
IQR_COLOR = T.GOLD
QUAD_BLUE, QUAD_YELLOW = "#60a5fa", "#facc15"     # same as the pick-rate scatter

CMAP = mcolors.LinearSegmentedColormap.from_list("rg", [T.RED, "#3a4444", T.GREEN])


# --- DATA --------------------------------------------------------------------
def load(data_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pq, csv = data_dir / "hero_duration_scaling.parquet", data_dir / "hero_duration_scaling.csv"
    dur = pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv)
    dur.columns = dur.columns.str.strip().str.lower()
    if dur["winrate"].max() <= 1.0:
        dur["winrate"] *= 100
    dur["wins"] = dur["total_games"] * dur["winrate"] / 100
    heroes = load_hero_meta(data_dir)
    dur = dur.merge(heroes[["hero_name", "hero_type"]], on="hero_name", how="left")
    dist = pd.read_csv(data_dir / "avg_duration.csv")
    dist.columns = dist.columns.str.strip().str.lower()
    return dur, dist, heroes


def phase_table(dur: pd.DataFrame, early_w: dict, late_w: dict) -> pd.DataFrame:
    """Per hero: pooled early and late win rates with CIs, and the late − early change.
    early_w / late_w = share of each bracket's games on that side of the cut (0..1)."""
    def pooled(w, tag):
        d = dur.assign(_w=dur["duration_bracket"].map(w).fillna(0.0))
        d = d.assign(_wins=d["wins"] * d["_w"], _n=d["total_games"] * d["_w"])
        g = d.groupby("hero_name")[["_wins", "_n"]].sum()
        lo, hi = wilson_ci(g["_wins"], g["_n"])
        return pd.DataFrame({f"{tag}_wr": g["_wins"] / g["_n"] * 100, f"{tag}_n": g["_n"],
                             f"{tag}_lo": lo * 100, f"{tag}_hi": hi * 100})

    t = pooled(early_w, "early").join(pooled(late_w, "late"), how="inner")
    t["delta"] = t["late_wr"] - t["early_wr"]
    types = dur.drop_duplicates("hero_name").set_index("hero_name")["hero_type"]
    t["hero_type"] = types.reindex(t.index)
    return t.reset_index()


def bracket_matrices(dur: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(absolute win rate, win rate minus the hero's own average, games) per hero × bracket."""
    wr = dur.pivot(index="hero_name", columns="duration_bracket", values="winrate").reindex(columns=BRACKETS)
    n = dur.pivot(index="hero_name", columns="duration_bracket", values="total_games").reindex(columns=BRACKETS)
    overall = dur.groupby("hero_name").apply(lambda g: g["wins"].sum() / g["total_games"].sum() * 100,
                                             include_groups=False)
    return wr, wr.sub(overall, axis=0), n


def fmt_min(m: float) -> str:
    secs = int(round(m * 60))
    return f"{secs // 60}:{secs % 60:02d}"


def phase_labels(summ: dict) -> tuple[str, str, str]:
    """('under 37 min', 'over 37 min', 'split at the median match length (37:09)')"""
    e, l = summ["early_cut"], summ["late_cut"]
    if summ["mode"] == "median":
        return (f"under {e:.0f} min", f"over {l:.0f} min", f"split at the median match length ({fmt_min(e)})")
    return (f"under {e:.0f} min", f"over {l:.0f} min",
            f"shortest 25% (< {fmt_min(e)}) vs longest 25% (> {fmt_min(l)})")


def bracket_x(minute: float, gx0: float, cw: float) -> float:
    """Table x of a match length in minutes on the heatmap's bracket columns."""
    for j, b in enumerate(BRACKETS):
        lo, hi = _bracket_edges(b)
        if minute < hi or j == len(BRACKETS) - 1:
            return gx0 + (j + np.clip((minute - lo) / (hi - lo), 0, 1)) * cw
    return gx0


def late_weighted(values: pd.DataFrame, cnt: pd.DataFrame, summ: dict) -> pd.Series:
    """Each hero's average of `values` over the late side of the cut, weighted by its games."""
    w = cnt.where(values.notna()).mul(pd.Series(summ["late_weights"]).reindex(BRACKETS), axis=1)
    return (values * w).sum(axis=1) / w.sum(axis=1)


# --- CARD 1: EARLY vs LATE SCATTER -------------------------------------------------
def render_early_late(t: pd.DataFrame, summ: dict, solo_wr: dict, meta: T.ChartMeta,
                      progress: float = 1.0, highlight: set[str] | None = None) -> "T.Figure":
    e_lab, l_lab, split_note = phase_labels(summ)
    t = t[(t["early_n"] >= MIN_PHASE_N) & (t["late_n"] >= MIN_PHASE_N)].copy()
    both = t[(t["early_wr"] > 50) & (t["late_wr"] > 50)]
    late_only = t[(t["early_wr"] <= 50) & (t["late_wr"] > 50)]
    early_only = t[(t["early_wr"] > 50) & (t["late_wr"] <= 50)]
    corners = {"LATE-GAME HEROES": "lose early, win late", "STRONG ALL GAME": "win early and late",
               "WEAK ALL GAME": "lose early and late", "EARLY-GAME HEROES": "win early, lose late"}
    return render_split_scatter(
        t, meta, "WHO WINS EARLY, WHO WINS LATE",
        f"{len(both)} heroes win in both halves  ·  {len(late_only)} only late  ·  {len(early_only)} only early",
        f"Win rate in games {e_lab}", f"Win rate in games {l_lab}", corners,
        f"dashed line = same win rate early & late  ·  {split_note}", solo_wr, progress, highlight)


def render_split_scatter(t: pd.DataFrame, meta: T.ChartMeta, title: str, subtitle: str, x_label: str,
                         y_label: str, corners: dict, note: str, solo_wr: dict, progress: float = 1.0,
                         highlight: set[str] | None = None) -> "T.Figure":
    """Each hero at (win rate in one half, win rate in the other): x = early_wr, y = late_wr columns.
    corners = {label: caption} for top-left, top-right, bottom-left, bottom-right, in that order.
    Shared by the early/late card and the low/high rank card."""
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    fig, ax = T.new_card()
    T.draw_header(fig, title, subtitle, alpha=p_frame)
    T.draw_archetype_legend(fig, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    # same range on both axes so the diagonal is a true 45° "no change" line
    lo = np.floor(min(t["early_wr"].min(), t["late_wr"].min(), 50) - 1.6)   # room for the corner labels
    hi = np.ceil(max(t["early_wr"].max(), t["late_wr"].max(), 50) + 0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(2 if hi - lo > 10 else 1))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_alpha(p_frame)

    # quadrants split at 50% / 50%, tinted like the pick-rate scatter
    a = T.ZONE_TINT_ALPHA * p_frame
    for (x0, x1, y0, y1, col) in [(lo, 50, 50, hi, QUAD_BLUE), (50, hi, 50, hi, T.GREEN),
                                  (lo, 50, lo, 50, T.RED), (50, hi, lo, 50, QUAD_YELLOW)]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=col, edgecolor="none", alpha=a, zorder=0.5))
    ax.axhline(50, color=T.DIVIDER, lw=1.6, alpha=p_frame, zorder=1)
    ax.axvline(50, color=T.DIVIDER, lw=1.6, alpha=p_frame, zorder=1)
    ax.plot([lo, hi], [lo, hi], color=T.TEXT_MUTED, lw=1.6, ls=(0, (6, 4)), alpha=0.8 * p_frame, zorder=1)
    spots_ = [(0.015, 0.985, "left", "top", QUAD_BLUE), (0.985, 0.985, "right", "top", T.GREEN),
              (0.015, 0.015, "left", "bottom", T.RED), (0.985, 0.015, "right", "bottom", QUAD_YELLOW)]
    for (head, sub), (qx, qy, ha, va, col) in zip(corners.items(), spots_):
        hy, sy = (qy, qy - 0.032) if va == "top" else (qy + 0.03, qy)
        ax.text(qx, hy, head, transform=ax.transAxes, ha=ha, va=va, color=col,
                fontsize=T.px_to_pt(17), fontweight="bold", alpha=0.85 * p_frame, zorder=2)
        ax.text(qx, sy, sub, transform=ax.transAxes, ha=ha, va=va, color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(13), alpha=0.8 * p_frame, zorder=2)

    # heroes fly in from the center
    order = t.sort_values("late_wr", ascending=False).index.tolist()
    xs, ys, names, l_alpha, is_hl = [], [], [], [], []
    for rank, i in enumerate(order):
        r = t.loc[i]
        s = 0.2 + 0.45 * rank / max(len(order) - 1, 1)
        tt = T.ease_out_cubic(T.phase(progress, s, s + 0.3))
        if tt <= 0:
            continue
        x = 50 + (r["early_wr"] - 50) * tt
        y = 50 + (r["late_wr"] - 50) * tt
        hl = highlight is None or r["hero_name"].lower() in highlight
        al = tt * (1.0 if hl else DIM_ALPHA)
        T.add_icon(ax, T.icon_any(r["hero_name"], SCATTER_ICON_PX, T.archetype_color(r["hero_type"]), al,
                                  wr=solo_wr.get(r["hero_name"])), (x, y), zorder=8 if hl else 5)
        xs.append(x); ys.append(y); names.append(r["hero_name"])
        l_alpha.append(T.phase(tt, 0.85, 1.0) * (1.0 if hl else DIM_ALPHA)); is_hl.append(hl)

    if xs:
        idx = [k for k, h in enumerate(is_hl) if h] if highlight else list(range(len(xs)))
        font = T.SIZE_LABEL_BIG if highlight else T.SIZE_LABEL
        spots = place_labels(ax, [xs[k] for k in idx], [ys[k] for k in idx], [names[k] for k in idx],
                             font, SCATTER_ICON_PX / 2)
        for k, (dx, dy, ha, va) in zip(idx, spots):
            ax.annotate(names[k], (xs[k], ys[k]), xytext=(dx, dy), textcoords="offset pixels", ha=ha, va=va,
                        color=T.TEXT, fontsize=T.px_to_pt(font), fontweight="bold", alpha=l_alpha[k],
                        zorder=9, path_effects=T.STROKE)
    ax.text(0.5, -0.115, note,
            transform=ax.transAxes, ha="center", va="top", color=T.TEXT_MUTED, fontsize=T.px_to_pt(14),
            alpha=p_frame)
    return fig


# --- CARDS 2 & 3: PER-ARCHETYPE HEATMAPS (absolute or relative) -------------------
def render_heatmap(values: pd.DataFrame, n: pd.DataFrame, dist: pd.DataFrame, htype: str, summ: dict,
                   meta: T.ChartMeta, heroes: pd.DataFrame, mode: str, progress: float = 1.0,
                   highlight: set[str] | None = None) -> "T.Figure":
    """mode='winrate': cells = actual win rate (colored around 50%).
       mode='scaling': cells = win rate minus the hero's own average."""
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    hz = heroes[heroes["hero_type"] == htype].sort_values("winrate", ascending=False)
    names = [h for h in hz["hero_name"] if h in values.index]
    solo = dict(zip(hz["hero_name"], hz["winrate"]))
    v, cnt = values.loc[names], n.loc[names]
    shown = v.where(cnt >= MIN_CELL_N)
    late = late_weighted(shown, cnt, summ)
    best = late.idxmax()

    plural = {"marksman": "MARKSMEN"}.get(htype, f"{htype.upper()}S")
    fig, ax = T.new_table_card()
    if mode == "winrate":
        norm, fmt = mcolors.Normalize(50 - WR_SPAN, 50 + WR_SPAN), "{:.1f}"
        T.draw_header(fig, f"{plural}: WIN RATE BY MATCH LENGTH",
                      f"Actual win rate in each 5-min bracket  ·  best in longer games: {best} "
                      f"({late[best]:.1f}%)", alpha=p_frame)
    else:
        norm, fmt = mcolors.Normalize(-DELTA_SPAN, DELTA_SPAN), "{:+.1f}"
        T.draw_header(fig, f"{plural}: WHO SCALES?",
                      f"Win rate vs each hero's own average  ·  biggest late-game boost: {best} "
                      f"({late[best]:+.1f})", alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    rows = len(names)
    row = T.table_rows(fig, ax, rows, header_rows=1.7, footer_rows=0.8)
    W, _ = T.content_px(fig)
    icon_px = min(row * 0.80, 64)
    font = T.fit_font_px(names, 230, row * 0.42, max_px=24, min_px=13)
    name_w = max(T.text_px(h, font) for h in names)
    cols = T.layout_columns({"icon": icon_px, "name": name_w, "grid": None}, W, gap_px=16)
    gx0, gx1 = cols["grid"]
    cw = (gx1 - gx0) / len(BRACKETS)
    cell_px = cw * W
    cell_font = T.fit_font_px(["+10.0", "55.5"], cell_px * 0.82, row * 0.40, max_px=26, min_px=12)
    hdr_font = T.fit_font_px([b.replace("m", "") for b in BRACKETS], cell_px * 0.9, None, max_px=20, min_px=12)

    games = dist.set_index("duration_bracket")["total_games"].reindex(BRACKETS)
    iqr_b = summ["iqr_brackets"]
    for j, b in enumerate(BRACKETS):
        cx = gx0 + (j + 0.5) * cw
        ax.text(cx, -1.45, b.replace("m", ""), ha="center", va="center",
                color=IQR_COLOR if b in iqr_b else T.TEXT_MUTED,
                fontsize=T.px_to_pt(hdr_font), fontweight="bold", alpha=p_frame)
        g = games.get(b, np.nan)
        ax.text(cx, -0.95, f"{g / 1000:.1f}k" if g >= 1000 else f"{g:.0f}", ha="center", va="center",
                color=T.TEXT_MUTED, fontsize=T.px_to_pt(hdr_font * 0.78), alpha=p_frame)
    ax.text(gx0 - 0.008, -0.95, "games", ha="right", va="center", color=T.TEXT_MUTED,
            fontsize=T.px_to_pt(hdr_font * 0.78), alpha=p_frame)

    # early / late underline ending at the exact cut, and the gold middle-50% box
    xe, xl = bracket_x(summ["early_cut"], gx0, cw), bracket_x(summ["late_cut"], gx0, cw)
    ax.plot([gx0 + 0.003, xe - 0.003], [-0.62, -0.62], color=EARLY_COLOR, lw=4, alpha=p_frame,
            solid_capstyle="round")
    ax.plot([xl + 0.003, gx1 - 0.003], [-0.62, -0.62], color=LATE_COLOR, lw=4, alpha=p_frame,
            solid_capstyle="round")
    if summ["mode"] == "median":
        ax.plot([xe, xe], [-0.75, -0.49], color=T.TEXT, lw=2, alpha=p_frame)
    if iqr_b:
        j0, j1 = BRACKETS.index(iqr_b[0]), BRACKETS.index(iqr_b[-1]) + 1
        ax.add_patch(Rectangle((gx0 + j0 * cw - 0.003, -1.8), (j1 - j0) * cw + 0.006, rows - 0.5 + 1.8 + 0.06,
                               facecolor="none", edgecolor=IQR_COLOR, lw=3.2, alpha=p_frame, zorder=5))
    split = (f"shorter / longer than the median ({fmt_min(summ['median'])})" if summ["mode"] == "median"
             else "shortest 25% / longest 25%")
    note = f"gold box = middle 50% of games   ·   blue / purple = {split}"
    ax.text(0.5 * (gx0 + gx1), rows + 0.08, note, ha="center", va="center", color=T.TEXT_MUTED,
            fontsize=T.px_to_pt(T.fit_font_px([note], (gx1 - gx0 + 0.2) * W, None, max_px=16, min_px=11,
                                              bold=False)), alpha=p_frame)

    for i, name in enumerate(names):
        s = 0.15 + 0.5 * i / max(rows - 1, 1)
        tt = T.ease_out_cubic(T.phase(progress, s, s + 0.3))
        hl = highlight is None or name.lower() in highlight
        a = (1.0 if hl else DIM_ALPHA) * tt
        if tt > 0:
            T.add_icon(ax, T.icon_any(name, icon_px, T.archetype_color(htype), a, wr=solo.get(name)),
                       ((cols["icon"][0] + cols["icon"][1]) / 2, i), zorder=6)
        ax.text(cols["name"][0], i, name, ha="left", va="center", color=T.TEXT, fontsize=T.px_to_pt(font),
                fontweight="bold", alpha=a)
        for j, b in enumerate(BRACKETS):
            x = gx0 + j * cw
            val, c = v.loc[name, b], cnt.loc[name, b]
            box = dict(xy=(x + 0.002, i - 0.46), width=cw - 0.004, height=0.92)
            if pd.isna(val) or c < MIN_CELL_N:
                ax.add_patch(Rectangle(**box, facecolor=T.BG, edgecolor=T.DIVIDER, hatch="///", lw=0.8,
                                       alpha=a, zorder=2))
                continue
            ax.add_patch(Rectangle(**box, color=CMAP(norm(val)), alpha=a, lw=0, zorder=2))
            ax.text(x + cw / 2, i, fmt.format(val), ha="center", va="center", color=T.TEXT,
                    fontsize=T.px_to_pt(cell_font), fontweight="bold", alpha=a, zorder=3,
                    path_effects=T.STROKE)
    return fig


def insights(data_dir: Path, **_) -> list[Insight]:
    """The clearest late-game and early-game heroes (win rate after vs before the median length)."""
    dur, _, heroes = load(data_dir)
    summ = duration_summary(data_dir)
    t = phase_table(dur, summ["early_weights"], summ["late_weights"])
    t = t[(t["early_n"] >= MIN_PHASE_N) & (t["late_n"] >= MIN_PHASE_N)].copy()
    se = np.hypot((t["early_hi"] - t["early_lo"]) / 3.92, (t["late_hi"] - t["late_lo"]) / 3.92)
    t["conf"] = normal_cdf(t["delta"].abs() / se)
    cut = fmt_min(summ["median"])
    out = []
    types = dict(zip(heroes["hero_name"], heroes["hero_type"]))
    t["hero_type"] = t["hero_name"].map(types)
    for htype in T.ARCHETYPE_ORDER:
        g = t[(t["hero_type"] == htype) & (t["conf"] >= 0.8)]
        if g.empty:
            continue
        plural = "marksmen" if htype == "marksman" else f"{htype}s"
        late, early, top = g.loc[g["delta"].idxmax()], g.loc[g["delta"].idxmin()], g.loc[g["late_wr"].idxmax()]
        if late["delta"] > 0:
            out.append(Insight(f"hero_duration_scaling_{htype}", "class_late", hero=late["hero_name"],
                               highlight=(late["hero_name"],),
                               headline=f"{late['hero_name']} scales hardest of the {plural}: "
                                        f"{late['delta']:+.1f} pts after {cut}",
                               detail=f"{late['early_wr']:.1f}% before, {late['late_wr']:.1f}% after",
                               effect=abs(late["delta"]), confidence=float(late["conf"])))
        if early["delta"] < 0:
            out.append(Insight(f"hero_duration_scaling_{htype}", "class_early", hero=early["hero_name"],
                               highlight=(early["hero_name"],),
                               headline=f"{early['hero_name']} fades the most of the {plural}: "
                                        f"{early['delta']:+.1f} pts after {cut}",
                               detail=f"{early['early_wr']:.1f}% before, {early['late_wr']:.1f}% after",
                               effect=abs(early["delta"]), confidence=float(early["conf"])))
        se_l = (top["late_hi"] - top["late_lo"]) / 3.92
        out.append(Insight(f"hero_duration_winrate_{htype}", "class_late_best", hero=top["hero_name"],
                           highlight=(top["hero_name"],),
                           headline=f"Best {htype} in long games: {top['hero_name']} at {top['late_wr']:.1f}%",
                           detail=f"games longer than {cut}; {top['early_wr']:.1f}% in shorter ones",
                           effect=abs(top["late_wr"] - 50),
                           confidence=float(normal_cdf(abs(top["late_wr"] - 50) / se_l))))
    for kind, r in (("late_scaler", t.loc[t["delta"].idxmax()]), ("early_hero", t.loc[t["delta"].idxmin()])):
        late = kind == "late_scaler"
        out.append(Insight("hero_early_vs_late", kind, hero=r["hero_name"], highlight=(r["hero_name"],),
                           headline=f"{r['hero_name']} {'gets stronger' if late else 'falls off'} the longer games go: "
                                    f"{r['early_wr']:.1f}% before {cut}, {r['late_wr']:.1f}% after",
                           detail=f"{abs(r['delta']):.1f}-point swing between short and long games",
                           effect=abs(r["delta"]), confidence=float(r["conf"])))
    return out

def cards(data_dir: Path, **_) -> dict:
    """{output name: draw(progress=1.0, highlight=None) -> Figure} for every card this script makes."""
    dur, dist, heroes = load(data_dir)
    summ = duration_summary(data_dir)
    solo = dict(zip(heroes["hero_name"], heroes["winrate"]))
    n_matches = total_matches_from_hero_rows(heroes["total_matches"])
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), n_matches)
    t = phase_table(dur, summ["early_weights"], summ["late_weights"])
    out = {"hero_early_vs_late": lambda progress=1.0, highlight=None: render_early_late(
        t, summ, solo, meta, progress, highlight)}

    wr, delta, n = bracket_matrices(dur)
    meta_h = T.ChartMeta(T.week_label_from_dir(data_dir), n_matches, min_n=MIN_CELL_N,
                         min_n_unit="games per cell (hatched = fewer)")
    for htype in T.ARCHETYPE_ORDER:
        if (heroes["hero_type"] == htype).any():
            out[f"hero_duration_winrate_{htype}"] = (
                lambda progress=1.0, highlight=None, h=htype: render_heatmap(
                    wr, n, dist, h, summ, meta_h, heroes, "winrate", progress, highlight))
            out[f"hero_duration_scaling_{htype}"] = (
                lambda progress=1.0, highlight=None, h=htype: render_heatmap(
                    delta, n, dist, h, summ, meta_h, heroes, "scaling", progress, highlight))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--highlight", default="")
    ap.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    out = data_dir / "graphs"
    T.save_cards(cards(data_dir), out, args.progress, T.parse_names(args.highlight))
    summ = duration_summary(data_dir)
    print(f"Saved duration cards to {out}  (split: {summ['mode']}, early < {summ['early_cut']:.1f} min, "
          f"late > {summ['late_cut']:.1f} min; IQR {summ['q1']:.1f}–{summ['q3']:.1f})")


if __name__ == "__main__":
    main()