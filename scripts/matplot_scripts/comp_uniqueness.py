"""
comp_uniqueness.py — "team comps almost never repeat" card (1080x1080).

Replaces unique_comps_pie_charts.py and unique_comps_vs_match_volume_decay.py.
Two 100% bars on the same buckets:
    distinct comps  — what share of all distinct 6-hero comps were seen 1, 2–4, 5–9, 10+ times
    team line-ups   — what share of all team line-ups used a comp from each bucket
Note the unit: a match has two team line-ups, so line-ups = 2 × matches.

Input: unique_comp_freq_match.csv (frequency_bucket, unique_team_comps, total_games_played, ...)
Usage: python3 comp_uniqueness.py [week_dir]
"""
from __future__ import annotations

import argparse

import pandas as pd
from matplotlib.patches import Rectangle

import theme as T
from chart_stats import load_hero_meta, total_matches_from_hero_rows

GROUPS = [  # (label, buckets from the SQL, color)
    ("seen once", ["single-occurrence"], "#2f7d6d"),
    ("2–4 times", ["2 - 4 occurrences"], "#33d69f"),
    ("5–9 times", ["5 - 9 occurrences"], T.GOLD),
    ("10+ times", ["10 - 19 occurrences", "20 - 29 occurrences", "30 - 39 occurrences", "40+ occurrences"], T.RED),
]
BAR_Y = {"comps": 1.1, "lineups": 4.3}   # table units; ylim below fits both sections
Y_RANGE = (5.85, -0.35)
BAR_H = 1.35
MIN_INSIDE_LABEL = 0.10      # segments narrower than this get their label below the bar
SECTION_PX, CALLOUT_PX = 24, 20


def prepare(data_dir) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(data_dir / "unique_comp_freq_match.csv")
    raw.columns = raw.columns.str.strip().str.lower()
    rows = []
    for label, buckets, col in GROUPS:
        sub = raw[raw["frequency_bucket"].isin(buckets)]
        rows.append({"label": label, "color": col, "comps": int(sub["unique_team_comps"].sum()),
                     "lineups": int(sub["total_games_played"].sum())})
    g = pd.DataFrame(rows)
    stats = {"comps": int(g["comps"].sum()), "lineups": int(g["lineups"].sum()),
             "repeat_20": int(raw[raw["frequency_bucket"].isin(
                 ["20 - 29 occurrences", "30 - 39 occurrences", "40+ occurrences"])]["unique_team_comps"].sum())}
    g["comps_pct"] = g["comps"] / stats["comps"]
    g["lineups_pct"] = g["lineups"] / stats["lineups"]
    return g, stats


def render(g: pd.DataFrame, stats: dict, meta: T.ChartMeta, progress: float = 1.0) -> "T.Figure":
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    once = g.iloc[0]
    fig, ax = T.new_table_card()
    T.draw_header(fig, f"{once['comps_pct'] * 100:.0f}% OF TEAM COMPS ARE ONE-OFFS",
                  "There is no meta comp: almost every 6-hero line-up is played exactly once", alpha=p_frame)
    T.draw_kpi_row(fig, [(f"{stats['comps']:,}", "distinct comps", T.GOLD),
                         (f"{stats['lineups']:,}", "team line-ups", T.GOLD),
                         (f"{stats['repeat_20']}", "comps played 20+ times", T.RED)], alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)
    ax.set_ylim(*Y_RANGE)
    W, H = T.content_px(fig)
    bar_px = BAR_H * H / (Y_RANGE[0] - Y_RANGE[1])

    for key, title in (("comps", "Share of distinct comps"), ("lineups", "Share of team line-ups")):
        y0 = BAR_Y[key]
        ax.text(0, y0 - BAR_H / 2 - 0.4, title, ha="left", va="center", color=T.TEXT, fontweight="bold",
                fontsize=T.px_to_pt(SECTION_PX), alpha=p_frame)
        x = 0.0
        below = []
        for i, r in g.iterrows():
            st = (0.2 if key == "comps" else 0.5) + 0.06 * i
            t = T.ease_out_cubic(T.phase(progress, st, st + 0.2))
            w = r[f"{key}_pct"] * t
            ax.add_patch(Rectangle((x, y0 - BAR_H / 2), w, BAR_H, color=r["color"], lw=0, alpha=max(t, 0.001)))
            if t >= 1:
                pct = r[f"{key}_pct"] * 100
                text = f"{pct:.1f}%" if pct < 10 else f"{pct:.0f}%"
                if r[f"{key}_pct"] >= MIN_INSIDE_LABEL:
                    seg_px = w * W
                    fs = T.fit_font_px([text, r["label"]], seg_px * 0.86, bar_px * 0.32, max_px=30, min_px=14)
                    ax.text(x + w / 2, y0, f"{text}\n{r['label']}", ha="center", va="center",
                            color=T.BG if r["color"] in (T.GOLD, "#33d69f") else T.TEXT,
                            fontsize=T.px_to_pt(fs), fontweight="bold", linespacing=1.15)
                else:
                    below.append((x + w / 2, f"{r['label']}: {text}", r["color"]))
            x += w
        # small segments: labels stacked under the right end of the bar
        for k, (xm, text, col) in enumerate(below):
            yk = y0 + BAR_H / 2 + 0.3 + 0.42 * k
            ax.plot([xm, xm], [y0 + BAR_H / 2, yk], color=col, lw=1.8)
            ax.text(xm - 0.008, yk, text, ha="right", va="center",
                    color=col, fontsize=T.px_to_pt(CALLOUT_PX), fontweight="bold")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    g, stats = prepare(data_dir)
    heroes = load_hero_meta(data_dir)
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), total_matches_from_hero_rows(heroes["total_matches"]))
    out = data_dir / "graphs" / "comp_uniqueness.png"
    T.save_card(render(g, stats, meta, args.progress), out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()