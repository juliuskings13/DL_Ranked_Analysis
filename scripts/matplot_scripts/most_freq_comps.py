"""
most_freq_comps.py — "most played team comps" card (1080x1080).

Top 8 exact 6-hero comps by games played (ROWS). Icons are sorted by archetype (brawler →
marksman → assassin → mystic) then name, so rows can be compared column by column.
Win rates carry 95% CIs: with ~20–40 games each, most comps are "too close to call",
which is itself the finding (no comp is a proven meta).

Reads top_20_freq_comps.csv only. The old win-filtered / lose-filtered lists
(top_20_freq_win_comps / lose_comps) select on the outcome, which guarantees extreme
win rates, so they're no longer drawn.

Usage: python3 most_freq_comps.py [week_dir] [--highlight "Haze"]
"""
from __future__ import annotations

import argparse

import pandas as pd
from matplotlib.patches import Rectangle

import theme as T
from chart_stats import load_hero_meta, total_matches_from_hero_rows, wilson_ci

ROWS = 8                    # 10 felt cluttered; icons/fonts scale automatically if you change it
ICON_FILL = 0.66            # icon diameter as a share of row height (the rest is breathing room)
ROW_CARD_H = 0.84           # each row sits on its own panel with a gap between panels
DIM_ALPHA = 0.25


def prepare(data_dir) -> tuple[pd.DataFrame, dict]:
    heroes = load_hero_meta(data_dir)
    htype = dict(zip(heroes["hero_name"], heroes["hero_type"]))
    df = pd.read_csv(data_dir / "top_20_freq_comps.csv")
    df.columns = df.columns.str.strip().str.lower()
    df = df.sort_values("total_matches", ascending=False).head(ROWS).reset_index(drop=True)
    df["heroes"] = df["comp_names"].map(
        lambda s: sorted((h.strip() for h in s.split(",")),
                         key=lambda h: (T.archetype_rank(htype.get(h, "")), h)))
    lo, hi = wilson_ci(df["winrate"] / 100 * df["total_matches"], df["total_matches"])
    df["lo"], df["hi"] = lo * 100, hi * 100
    return df, htype


def render(df: pd.DataFrame, htype: dict, meta: T.ChartMeta, team_sides: int,
           progress: float = 1.0, highlight: set[str] | None = None) -> "T.Figure":
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    top = df.iloc[0]
    n_clear = int(((df["lo"] > 50) | (df["hi"] < 50)).sum())
    fig, ax = T.new_table_card()
    T.draw_header(fig, "THE MOST PLAYED TEAM COMPS",
                  f"Even the #1 comp appeared in just {int(top['total_matches'])} of "
                  f"{team_sides:,} team line-ups", alpha=p_frame)
    T.draw_archetype_legend(fig, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)
    n = len(df)
    row = T.table_rows(fig, ax, n, header_rows=0.8, footer_rows=0.55)
    W, _ = T.content_px(fig)
    icon_px = row * ICON_FILL
    font = min(row * 0.34, 30)
    wr_font = min(font * 1.2, 34)
    hdr_px = 17
    widths = {
        "rank": T.text_px("10", font),
        "comp": None,
        "games": max(T.text_px("GAMES", hdr_px), T.text_px("999", font, bold=False)),
        "wr": max(T.text_px("WIN %", hdr_px), T.text_px("100%", wr_font)),
        "range": max(T.text_px("95% RANGE", hdr_px), T.text_px("100–100%", font, bold=False)),
    }
    cols = T.layout_columns(widths, W, gap_px=26)
    c0, c1 = cols["comp"]
    # spread the six icons evenly across the comp column, but never wider than icon + 40% gap
    step = min((c1 - c0 - icon_px / W) / 5, icon_px * 1.4 / W)
    icon_x = [c0 + icon_px / 2 / W + k * step for k in range(6)]

    hdr = dict(color=T.GOLD, fontsize=T.px_to_pt(hdr_px), fontweight="bold", va="center", alpha=p_frame)
    ax.text((icon_x[0] + icon_x[-1]) / 2, -0.95, "COMP", ha="center", **hdr)
    ax.text(cols["games"][1], -0.95, "GAMES", ha="right", **hdr)
    ax.text(cols["wr"][1], -0.95, "WIN %", ha="right", **hdr)
    ax.text(cols["range"][1], -0.95, "95% RANGE", ha="right", **hdr)

    ax.text(1.0, n - 0.35, f"{n_clear} of {n} win rates clearly differ from 50% (colored)  ·  "
            "gray = too few games to call", ha="right", va="center", color=T.TEXT_MUTED,
            fontsize=T.px_to_pt(16), alpha=p_frame)

    for i, r in df.iterrows():
        s = 0.15 + 0.5 * i / max(n - 1, 1)
        t = T.ease_out_cubic(T.phase(progress, s, s + 0.3))
        hl = highlight is None or bool({h.lower() for h in r["heroes"]} & highlight)
        a = (1.0 if hl else DIM_ALPHA) * t
        ax.add_patch(Rectangle((0, i - ROW_CARD_H / 2), 1, ROW_CARD_H, color=T.PANEL,
                               alpha=0.75 * p_frame, lw=0, zorder=0))
        if t <= 0:
            continue
        ax.text(cols["rank"][0] + 8 / W, i, f"{i + 1}", ha="left", va="center", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(font), fontweight="bold", alpha=a)
        for k, h in enumerate(r["heroes"]):
            T.add_icon(ax, T.icon_any(h, icon_px, T.archetype_color(htype.get(h, "")), a,
                                      wr=r["winrate"]),          # team (comp) win rate
                       (icon_x[k], i), zorder=6)
        ax.text(cols["games"][1], i, f"{int(r['total_matches'])}", ha="right", va="center",
                color=T.TEXT_MUTED, fontsize=T.px_to_pt(font), alpha=a)
        col = T.wr_color(r["winrate"], r["lo"], r["hi"])
        ax.text(cols["wr"][1], i, f"{r['winrate']:.0f}%", ha="right", va="center", color=col,
                fontsize=T.px_to_pt(wr_font), fontweight="bold", alpha=a)
        ax.text(cols["range"][1] - 8 / W, i, f"{r['lo']:.0f}–{r['hi']:.0f}%", ha="right", va="center",
                color=T.TEXT_MUTED, fontsize=T.px_to_pt(font), alpha=a)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--highlight", default="")
    ap.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    df, htype = prepare(data_dir)
    heroes = load_hero_meta(data_dir)
    n_matches = total_matches_from_hero_rows(heroes["total_matches"])
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), n_matches)
    hl = {h.strip() for h in args.highlight.split(",") if h.strip()} or None
    out = data_dir / "graphs" / "most_played_comps.png"
    T.save_card(render(df, htype, meta, team_sides=2 * n_matches, progress=args.progress, highlight=hl), out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()