"""
balance_wr_distribution.py — "How balanced is Deadlock?" card (1080x1080).

Every hero is an icon at its exact win rate (stacked only where icons would overlap). The gold band is
where *sampling noise alone* would put heroes if every hero were truly 50%: with ~15–57k
games per hero, a perfectly balanced hero still wobbles by a few tenths of a point.
Heroes outside the band are imbalanced for real, not by chance.

Usage: python3 balance_wr_distribution.py [week_dir] [--highlight "Victor,Mirage"]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import theme as T
from chart_stats import Insight, TIER_BANDS, add_winrate_ci, load_hero_meta, total_matches_from_hero_rows

Z_BAND = 2.58            # 99% of noise-only outcomes fall inside the band
ICON_PX = 50
DIM_ALPHA = 0.2
# Tier zones: (lo, hi, name, tint color, tint strength × ZONE_TINT_ALPHA, label color),
# from the shared definitions so this chart, the tier list and insights always agree.
TIERS = [(lo, hi, name, *T.TIER_STYLE[name]) for lo, hi, name in TIER_BANDS]


def prepare(data_dir) -> tuple[pd.DataFrame, dict]:
    df = add_winrate_ci(load_hero_meta(data_dir))
    se = np.sqrt(0.25 / df["total_matches"]) * 100          # SE of a true-50% hero, in points
    half = float(Z_BAND * se.max())                           # widest (smallest-sample) hero
    band = {"lo": 50 - half, "hi": 50 + half}
    df["in_band"] = df["winrate"].between(band["lo"], band["hi"])
    df = df.sort_values("winrate").reset_index(drop=True)
    df["stack"] = _beeswarm_levels(df["winrate"].to_numpy(), MIN_GAP_PTS)
    return df, band


MIN_GAP_PTS = 0.68       # icons closer than this (in win-rate points) stack vertically


def _beeswarm_levels(xs: np.ndarray, gap: float) -> list[int]:
    """Each hero sits at its exact win rate; stack upward only when it would overlap."""
    placed: list[tuple[float, int]] = []
    levels = []
    for x in xs:
        lvl = 0
        while any(abs(x - px) < gap and pl == lvl for px, pl in placed):
            lvl += 1
        placed.append((x, lvl))
        levels.append(lvl)
    return levels


def render(df: pd.DataFrame, band: dict, meta: T.ChartMeta, progress: float = 1.0,
           highlight: set[str] | None = None) -> "T.Figure":
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    p_band = T.ease_out_cubic(T.phase(progress, 0.75, 0.95))
    n_in, n = int(df["in_band"].sum()), len(df)
    title = f"ONLY {n_in} OF {n} HEROES ARE BALANCED"
    sub = (f"Perfect balance would put every hero between {band['lo']:.1f}% and {band['hi']:.1f}%"
           if n_in < n else "Every hero sits inside the noise band")

    fig, ax = T.new_card()
    T.draw_header(fig, title, sub, alpha=p_frame)
    T.draw_archetype_legend(fig, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    x_lo, x_hi = np.floor(df["winrate"].min() - 0.5), np.ceil(df["winrate"].max() + 0.5)
    max_stack = int(df["stack"].max()) + 1
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.1, max_stack + 1.7)                        # just enough room for tier labels on top
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ticks = np.arange(np.ceil(x_lo), np.floor(x_hi) + 1, 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}%" for t in ticks])
    ax.set_xlabel("Win rate", labelpad=12)
    for tl in ax.get_xticklabels():
        tl.set_alpha(p_frame)

    # Tier zones: tinted full-height columns with solid dividers and colored names on top
    y0, top = ax.get_ylim()
    for lo, hi, name, tint, strength, label_col in TIERS:
        a, b = max(lo, x_lo), min(hi, x_hi)
        if a >= b:
            continue
        ax.axvspan(a, b, color=tint, alpha=T.ZONE_TINT_ALPHA * strength * p_frame, lw=0, zorder=0.3)
        if a > x_lo:
            ax.plot([a, a], [y0, top], color=T.TEXT_MUTED, lw=1.2, alpha=0.45 * p_frame, zorder=0.4)
        ax.text((a + b) / 2, top - 0.55, name, ha="center", va="center", color=label_col,
                fontsize=T.px_to_pt(18), fontweight="bold", alpha=p_frame, path_effects=T.STROKE)

    # Noise-only band: outlined + hatched (not filled) so it doesn't double the BALANCED tint
    from matplotlib.patches import Rectangle
    band_top = top - 1.1
    ax.add_patch(Rectangle((band["lo"], y0), band["hi"] - band["lo"], band_top - y0, facecolor="none",
                           edgecolor=T.GOLD, hatch="/", lw=1.8, ls=(0, (6, 4)), alpha=0.75 * p_band,
                           zorder=1))
    ax.axvline(50, color=T.GOLD, lw=1.8, alpha=0.8 * p_frame, zorder=1)
    ax.text(50, band_top - 0.12, "noise-only\nrange", ha="center", va="top", color=T.GOLD,
            fontsize=T.px_to_pt(16), fontweight="bold", alpha=p_band, linespacing=1.1, zorder=7,
            path_effects=T.STROKE)

    # Hero icons drop in, left to right
    for i, r in df.iterrows():
        s = 0.12 + 0.55 * (r["winrate"] - df["winrate"].min()) / max(df["winrate"].max() - df["winrate"].min(), 1)
        t = T.ease_out_cubic(T.phase(progress, s, s + 0.25))
        if t <= 0:
            continue
        hl = highlight is None or r["hero_name"].lower() in highlight
        a = t * (1.0 if hl else DIM_ALPHA)
        y_final = r["stack"] + 0.5
        y = y_final + (1 - t) * 3
        ring = T.archetype_color(r["hero_type"])
        T.add_icon(ax, T.icon_any(r["hero_name"], ICON_PX, ring, a, wr=r["winrate"]),
                   (r["winrate"], y), zorder=6 if hl else 4)
        if highlight and hl and t >= 1:
            ax.text(r["winrate"], y + 0.62, r["hero_name"], ha="center", va="bottom",
                    color=T.TEXT, fontsize=T.px_to_pt(17), fontweight="bold", zorder=7,
                    path_effects=T.STROKE)
    return fig


def insights(data_dir: Path, **_) -> list[Insight]:
    df, band = prepare(data_dir)
    n_in, n = int(df["in_band"].sum()), len(df)
    return [Insight("deadlock_balance_chart", "balance", context=True,
                    headline=f"{n_in} of {n} heroes are balanced",
                    detail=f"Inside the {band['lo']:.1f}–{band['hi']:.1f}% range that chance alone would produce; "
                           f"{n - n_in} are imbalanced for real")]

def cards(data_dir: Path, **_) -> dict:
    """{output name: draw(progress=1.0, highlight=None) -> Figure} for every card this script makes."""
    df, band = prepare(data_dir)
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), total_matches_from_hero_rows(df["total_matches"]))
    return {"deadlock_balance_chart": lambda progress=1.0, highlight=None: render(df, band, meta, progress=progress,
                                                                                    highlight=highlight)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--highlight", default="")
    ap.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    T.save_cards(cards(data_dir), data_dir / "graphs", args.progress, T.parse_names(args.highlight))
    df, band = prepare(data_dir)
    print(f"Saved balance chart  | band {band['lo']:.2f}–{band['hi']:.2f}%, {int(df['in_band'].sum())} heroes inside")


if __name__ == "__main__":
    main()