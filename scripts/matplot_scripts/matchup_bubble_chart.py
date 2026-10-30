"""
matchup_bubble_chart.py — archetype count matchups (1080x1080, one card per archetype).

Rows = how many of the class YOUR team has, columns = how many the ENEMY has.
The query exports every match from both sides, so (2 vs 1) and (1 vs 2) are the same
games with complementary win rates. Each matchup is shown once, from the side with MORE
of the class, which removes the mirror and the double count. Equal-count matchups are 50%
by construction and drawn as "even". Low-sample cells are hatched.

Complements archetype_stacking.py: that card shows the *difference* (+1, +2 …); this one
shows whether the absolute counts matter (is 3 vs 1 different from 2 vs 0?).

Input: <archetype>_comp_wr.csv (t0_count, t1_count, total_matchups, t0_winrate)
Usage: python3 matchup_bubble_chart.py [week_dir]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import theme as T
from chart_stats import Insight, load_hero_meta, total_matches_from_hero_rows, wilson_ci, wr_confidence

MIN_CELL_N = 200
WR_SPAN = 6.0            # color range ±6 points around 50%
PLURAL = {"brawler": "BRAWLERS", "marksman": "MARKSMEN", "assassin": "ASSASSINS", "mystic": "MYSTICS"}
CMAP = mcolors.LinearSegmentedColormap.from_list("wr", [T.RED, "#3a4444", T.GREEN])


def prepare(data_dir, htype: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{htype}_comp_wr.csv")
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={"t0_count": "mine", "t1_count": "theirs", "total_matchups": "n",
                            "t0_winrate": "wr"})
    df = df[df["mine"] > df["theirs"]].copy()          # one side of each mirror pair
    lo, hi = wilson_ci(df["wr"] / 100 * df["n"], df["n"])
    df["lo"], df["hi"] = lo * 100, hi * 100
    df["shown"] = df["n"] >= MIN_CELL_N
    return df


def render(df: pd.DataFrame, htype: str, meta: T.ChartMeta, progress: float = 1.0) -> "T.Figure":
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    s = df[df["shown"]]
    worst, best = s.loc[s["wr"].idxmin()], s.loc[s["wr"].idxmax()]
    plural = PLURAL.get(htype, htype.upper())
    fig, ax = T.new_card()
    one = "marksman" if htype == "marksman" else htype
    worst_txt = f"worst: {int(worst['mine'])} vs {int(worst['theirs'])} at {worst['wr']:.1f}%"
    best_txt = f"best: {int(best['mine'])} vs {int(best['theirs'])} at {best['wr']:.1f}%"
    # strongest claim the data supports: every CI clears 50% > every estimate on one side > mixed
    if (s["hi"] < 50).all():
        sub = f"Every {one} advantage loses  ·  {worst_txt}"
    elif (s["lo"] > 50).all():
        sub = f"Every {one} advantage wins  ·  {best_txt}"
    elif (s["wr"] < 50).all():
        sub = f"No {one} advantage wins more than half its games  ·  {worst_txt}"
    elif (s["wr"] > 50).all():
        sub = f"No {one} advantage loses more than half its games  ·  {best_txt}"
    else:
        sub = (f"Best: {int(best['mine'])} vs {int(best['theirs'])} wins {best['wr']:.1f}%  ·  "
               f"worst: {int(worst['mine'])} vs {int(worst['theirs'])} wins {worst['wr']:.1f}%")
    T.draw_header(fig, f"MORE {plural} THAN THE ENEMY?", sub, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    max_mine = int(s["mine"].max())
    max_theirs = max_mine - 1
    ax.set_xlim(-0.5, max_theirs + 0.5)
    ax.set_ylim(0.5, max_mine + 0.5)
    ax.grid(False)
    ax.set_xticks(range(0, max_theirs + 1))
    ax.set_yticks(range(1, max_mine + 1))
    ax.set_xlabel(f"{plural.capitalize()} on the ENEMY team", labelpad=12)
    ax.set_ylabel(f"{plural.capitalize()} on YOUR team", labelpad=12)
    ax.tick_params(labelsize=T.px_to_pt(20))
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    norm = mcolors.Normalize(50 - WR_SPAN, 50 + WR_SPAN)

    # text sized to the cells: ~75px of the content box goes to tick labels + axis titles
    W, H = T.content_px(fig)
    cell_w, cell_h = (W - 80) / (max_theirs + 1) * 0.94, (H - 80) / max_mine * 0.94
    subs = [f"±{(r.hi - r.lo) / 2:.1f}  ·  {r.n / 1000:.1f}k games" for r in s.itertuples()]
    main_px = T.fit_font_px(["55.5%"], cell_w * 0.72, cell_h * 0.34, max_px=50, min_px=18)
    sub_px = T.fit_font_px(subs, cell_w * 0.9, cell_h * 0.15, max_px=21, min_px=11, bold=False)

    cells = [(m, e) for m in range(1, max_mine + 1) for e in range(0, max_theirs + 1)]
    for k, (m, e) in enumerate(cells):
        st = 0.15 + 0.55 * k / max(len(cells) - 1, 1)
        t = T.ease_out_cubic(T.phase(progress, st, st + 0.25))
        if t <= 0:
            continue
        box = dict(xy=(e - 0.47, m - 0.47), width=0.94, height=0.94)
        if e >= m:
            if e == m:
                ax.add_patch(Rectangle(**box, facecolor=T.PANEL, alpha=0.6 * t, lw=0))
                ax.text(e, m, "even", ha="center", va="center", color=T.TEXT_MUTED,
                        fontsize=T.px_to_pt(sub_px * 1.2), alpha=t)
            continue
        row = df[(df["mine"] == m) & (df["theirs"] == e)]
        if row.empty or not bool(row["shown"].iloc[0]):
            ax.add_patch(Rectangle(**box, facecolor=T.BG, edgecolor=T.DIVIDER, hatch="///", lw=0.8, alpha=t))
            if not row.empty:
                ax.text(e, m, f"only {int(row['n'].iloc[0])} games", ha="center", va="center",
                        color=T.TEXT_MUTED, fontsize=T.px_to_pt(sub_px * 1.1), alpha=t)
            continue
        r = row.iloc[0]
        ax.add_patch(Rectangle(**box, color=CMAP(norm(r["wr"])), alpha=t, lw=0))
        ax.text(e, m + 0.07, f"{r['wr']:.1f}%", ha="center", va="center", color=T.TEXT,
                fontsize=T.px_to_pt(main_px), fontweight="bold", alpha=t, path_effects=T.STROKE)
        ax.text(e, m - 0.25, f"±{(r['hi'] - r['lo']) / 2:.1f}  ·  {r['n'] / 1000:.1f}k games"
                if r["n"] >= 1000 else f"±{(r['hi'] - r['lo']) / 2:.1f}  ·  {int(r['n'])} games",
                ha="center", va="center", color=T.TEXT, fontsize=T.px_to_pt(sub_px), alpha=0.85 * t,
                path_effects=T.STROKE)
    return fig


def insights(data_dir: Path, **_) -> list[Insight]:
    """Per class: the most extreme reliable 'X of this class vs Y' result."""
    out = []
    for htype in T.ARCHETYPE_ORDER:
        if not (data_dir / f"{htype}_comp_wr.csv").exists():
            continue
        df = prepare(data_dir, htype)
        df = df[df["shown"] & ((df["hi"] < 50) | (df["lo"] > 50))]
        if df.empty:
            continue
        r = df.loc[(df["wr"] - 50).abs().idxmax()]
        plural = "marksmen" if htype == "marksman" else f"{htype}s"
        out.append(Insight(f"archetype_count_matrix_{htype}", "class_count", highlight=(),
                           headline=f"{int(r['mine'])} {plural} vs {int(r['theirs'])}: {r['wr']:.1f}% win rate",
                           detail=f"{int(r['n']):,} matches; 95% range {r['lo']:.1f}–{r['hi']:.1f}%",
                           effect=abs(r["wr"] - 50), confidence=float(wr_confidence(r["wr"], r["n"]))))
    return out

def cards(data_dir: Path, **_) -> dict:
    """{output name: draw(progress=1.0, highlight=None) -> Figure} for every card this script makes."""
    heroes = load_hero_meta(data_dir)
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), total_matches_from_hero_rows(heroes["total_matches"]),
                       min_n=MIN_CELL_N, min_n_unit="games per cell (hatched = fewer)")
    out = {}
    for htype in T.ARCHETYPE_ORDER:
        if (data_dir / f"{htype}_comp_wr.csv").exists():
            df = prepare(data_dir, htype)
            out[f"archetype_count_matrix_{htype}"] = (
                lambda progress=1.0, highlight=None, df=df, h=htype: render(df, h, meta, progress))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    T.save_cards(cards(data_dir), data_dir / "graphs", args.progress)
    print(f"Saved archetype count matrices to {data_dir / 'graphs'}")


if __name__ == "__main__":
    main()