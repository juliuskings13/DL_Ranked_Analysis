"""
hero_wr_total_matches_split_column.py — hero win-rate cards (1080x1080).

Bars diverge from 50%, so "how far from balanced" is the bar length; 95% CI whiskers;
pick rate printed per row. Icons, fonts and columns are sized to the space available.
Outputs:
    hero_winrates_top_bottom.png          10 best + 10 worst heroes
    hero_winrates_<archetype>.png         one card per archetype (≤12 heroes each)

Usage: python3 hero_wr_total_matches_split_column.py [week_dir] [--highlight "Victor,Mirage"]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import theme as T
from chart_stats import add_pick_rate, add_winrate_ci, load_hero_meta, total_matches_from_hero_rows

TOP_N = 10
BAR_H = 0.62            # fraction of a row
DIM_ALPHA = 0.25


def prepare(data_dir) -> pd.DataFrame:
    df = load_hero_meta(data_dir)
    df = add_winrate_ci(add_pick_rate(df))
    return df.sort_values("winrate", ascending=False).reset_index(drop=True)


def render(rows: pd.DataFrame, meta: T.ChartMeta, title: str, subtitle: str,
           gap_after: int | None = None, span: float | None = None,
           progress: float = 1.0, highlight: set[str] | None = None) -> "T.Figure":
    """rows: already ordered top→bottom. gap_after: insert a separator after this row index.
    span: ±percentage points shown around 50% (shared across cards keeps them comparable)."""
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    fig, ax = T.new_table_card()
    T.draw_header(fig, title, subtitle, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    n = len(rows) + (1 if gap_after is not None else 0)
    row = T.table_rows(fig, ax, n, header_rows=1.0, footer_rows=0.8)
    W, _ = T.content_px(fig)
    span = span or float(np.ceil(max((rows["wr_hi"] - 50).abs().max(), (rows["wr_lo"] - 50).abs().max()) + 0.5))

    icon_px = min(row * 0.84, 70)
    font = T.fit_font_px(rows["hero_name"], 250, row * 0.50, max_px=28, min_px=13)
    name_w = max(T.text_px(h, font) for h in rows["hero_name"])
    val_font = min(font * 0.95, 26)
    pick_w = max(T.text_px("PICK", 17), T.text_px("100%", val_font, bold=False))
    cols = T.layout_columns({"icon": icon_px, "name": name_w, "bars": None, "pick": pick_w}, W)
    # leave room inside the bar column for the % label beyond each whisker
    label_w = T.text_px("55.5%", val_font) + 12
    b0, b1 = cols["bars"][0] + label_w / W, cols["bars"][1] - label_w / W

    def to_x(wr):                           # win rate → table x
        return b0 + (np.asarray(wr) - (50 - span)) / (2 * span) * (b1 - b0)

    hdr = dict(color=T.GOLD, fontsize=T.px_to_pt(17), fontweight="bold", va="center", alpha=p_frame)
    ax.text(cols["name"][0], -1.0, "HERO", ha="left", **hdr)
    ax.text(to_x(50), -1.0, "WIN RATE", ha="center", **hdr)
    ax.text(1.0, -1.0, "PICK", ha="right", **hdr)
    tick_font = min(16, val_font * 0.75)
    for v in np.arange(50 - span, 50 + span + 0.01, 2 if span >= 4 else 1):
        ax.plot([to_x(v)] * 2, [-0.5, n - 0.5], color=T.GRID, lw=1, zorder=0, alpha=p_frame)
        ax.text(to_x(v), n - 0.4, f"{v:.0f}%", ha="center", va="top", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(tick_font), alpha=p_frame)
    ax.plot([to_x(50)] * 2, [-0.5, n - 0.5], color=T.TEXT_MUTED, lw=1.8, zorder=1, alpha=p_frame)

    y = 0
    for i, (_, r) in enumerate(rows.iterrows()):
        if gap_after is not None and i == gap_after + 1:
            ax.plot([0, 1], [y, y], color=T.DIVIDER, lw=1.5, ls=(0, (6, 4)), alpha=p_frame)
            y += 1
        s = 0.15 + 0.5 * i / max(len(rows) - 1, 1)
        t = T.ease_out_cubic(T.phase(progress, s, s + 0.3))
        hl = highlight is None or r["hero_name"].lower() in highlight
        a = (1.0 if hl else DIM_ALPHA) * t
        col = T.archetype_color(r["hero_type"])

        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y - 0.5), 1, 1, color=T.PANEL, alpha=0.55 * p_frame, zorder=0, lw=0))
        if t > 0:
            T.add_icon(ax, T.icon_any(r["hero_name"], icon_px, col, a, wr=r["winrate"]),
                       ((cols["icon"][0] + cols["icon"][1]) / 2, y), zorder=6)
        ax.text(cols["name"][0], y, r["hero_name"], ha="left", va="center", color=T.TEXT,
                fontsize=T.px_to_pt(font), fontweight="bold", alpha=a)

        wr_now = 50 + (r["winrate"] - 50) * t
        x50, xw = to_x(50), to_x(wr_now)
        bar_col = T.wr_color(r["winrate"], r["wr_lo"], r["wr_hi"])
        ax.add_patch(Rectangle((min(x50, xw), y - BAR_H / 2), abs(xw - x50), BAR_H,
                               color=bar_col, alpha=0.9 * a, zorder=3, lw=0))
        if t >= 1:
            ax.plot([to_x(r["wr_lo"]), to_x(r["wr_hi"])], [y, y], color=T.TEXT, lw=2.2, alpha=0.5 * a, zorder=4)
            right = r["winrate"] >= 50
            ax.text(to_x(r["wr_hi"] if right else r["wr_lo"]) + (6 if right else -6) / W, y,
                    f"{r['winrate']:.1f}%", ha="left" if right else "right", va="center",
                    color=bar_col if bar_col != T.TEXT_MUTED else T.TEXT, fontsize=T.px_to_pt(val_font),
                    fontweight="bold", alpha=a, zorder=5, path_effects=T.STROKE)
        ax.text(1.0, y, f"{r['pick_rate']:.0f}%", ha="right", va="center", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(val_font), alpha=a)
        y += 1
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--highlight", default="")
    ap.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    out_dir = data_dir / "graphs"
    df = prepare(data_dir)
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), total_matches_from_hero_rows(df["total_matches"]))
    hl = {h.strip() for h in args.highlight.split(",") if h.strip()} or None
    span = float(np.ceil(max((df["wr_hi"] - 50).abs().max(), (df["wr_lo"] - 50).abs().max()) + 0.5))

    tb = pd.concat([df.head(TOP_N), df.tail(TOP_N)])
    sub = (f"{df.iloc[0]['hero_name']} leads at {df.iloc[0]['winrate']:.1f}%  ·  "
           f"{df.iloc[-1]['hero_name']} trails at {df.iloc[-1]['winrate']:.1f}%")
    T.save_card(render(tb, meta, f"BEST & WORST {TOP_N} HEROES", sub, gap_after=TOP_N - 1,
                       span=span, progress=args.progress, highlight=hl),
                out_dir / "hero_winrates_top_bottom.png")

    for htype in T.ARCHETYPE_ORDER:
        g = df[df["hero_type"] == htype]
        if g.empty:
            continue
        above = int((g["wr_lo"] > 50).sum())
        below = int((g["wr_hi"] < 50).sum())
        sub = f"{above} clearly above 50%  ·  {below} clearly below  ·  {len(g) - above - below} too close to call"
        title = ("MARKSMEN" if htype == "marksman" else f"{htype.upper()}S") + ": WIN RATE"
        T.save_card(render(g, meta, title, sub, span=span, progress=args.progress, highlight=hl),
                    out_dir / f"hero_winrates_{htype}.png")
    print(f"Saved hero win-rate cards to {out_dir}")


if __name__ == "__main__":
    main()