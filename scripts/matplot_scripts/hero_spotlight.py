"""
hero_spotlight.py — one 1080x1080 card per hero, for hero spotlight videos and shorts.

    header      hero name; tier, rank and power; big icon
    tiles       win rate, pick rate, ban rate, power, each with the change since last week
    trend       win rate over the last 8 weeks (from the week folders next to this one), 95% range
    game length win rate in each 5-minute bracket, against 50%
    partners    best and worst duo partners (win rate vs what their solo win rates predict)

The roster is split into four fixed groups (hero_groups.json next to the week folders, created
on first run, mixed classes in each). insights.py makes one spotlight video per group, rotating
weekly, so every hero is covered about once a month.

Outputs: graphs/heroes/hero_<name>.png
Usage: python3 hero_spotlight.py [week_dir] [--heroes "Haze,Victor"]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import theme as T
from chart_stats import (DURATION_BRACKETS, Insight, add_power, duration_summary, hero_deltas, hero_history,
                         load_hero_meta, previous_week_dir, tier_of)

GROUPS = 4
TREND_WEEKS = 8
MIN_BRACKET_N = 150          # brackets with fewer games are drawn hatched
PARTNERS_BEST, PARTNERS_WORST = 3, 2
TIER_COLOR = {name: style[2] for name, style in T.TIER_STYLE.items()}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# --- DATA --------------------------------------------------------------------------
def load_all(data_dir: Path) -> dict:
    """Everything the cards need, loaded once per week."""
    prev = previous_week_dir(data_dir)
    if prev is not None:
        d = hero_deltas(data_dir, prev)
    else:
        d = add_power(load_hero_meta(data_dir).rename(columns={}))
        d["rank"] = d["power"].rank(ascending=False, method="min").astype(int)
        d = d.rename(columns={"total_matches": "games"})
    out = {"d": d.set_index("hero_name"), "n_heroes": len(d), "hist": hero_history(data_dir, TREND_WEEKS),
           "dur": None, "duos": None, "prev_label": T.week_label_from_dir(prev) if prev is not None else None}
    try:
        from hero_strength_and_scaling import load
        out["dur"], _, _ = load(data_dir)
        out["median"] = duration_summary(data_dir)["median"]
    except FileNotFoundError:
        pass
    try:
        from hero_duo_synergy_table import prepare as duo_prepare
        out["duos"] = duo_prepare(data_dir)
    except FileNotFoundError:
        pass
    return out


def partners(duos: pd.DataFrame | None, hero: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if duos is None:
        return pd.DataFrame(), pd.DataFrame()
    mine = duos[(duos["hero_name_a"] == hero) | (duos["hero_name_b"] == hero)].copy()
    mine["partner"] = np.where(mine["hero_name_a"] == hero, mine["hero_name_b"], mine["hero_name_a"])
    best = mine[mine["lift_lo"] > 0].nlargest(PARTNERS_BEST, "lift_lo")
    worst = mine[mine["lift_hi"] < 0].nsmallest(PARTNERS_WORST, "lift_hi")
    return best, worst


def hero_groups(data_dir: Path, heroes: pd.DataFrame) -> list[list[str]]:
    """Four fixed groups with a mix of classes. Stored next to the week folders so membership
    stays the same week to week; new heroes join the smallest group."""
    path = Path(data_dir).parent / "hero_groups.json"
    try:
        groups = json.loads(path.read_text())
    except (OSError, ValueError):
        groups = [[] for _ in range(GROUPS)]
    known = {h for g in groups for h in g}
    new = heroes[~heroes["hero_name"].isin(known)].sort_values(["hero_type", "hero_name"])["hero_name"]
    for h in new:
        min(groups, key=len).append(h)
    if len(new):
        try:
            path.write_text(json.dumps(groups, indent=2))
        except OSError:
            pass
    present = set(heroes["hero_name"])
    return [[h for h in g if h in present] for g in groups]


# --- CARD --------------------------------------------------------------------------
def _fmt_delta(v, unit="", digits=1) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    arrow = "▲" if v > 0 else "▼" if v < 0 else "■"
    return f"{arrow} {abs(v):.{digits}f}{unit} vs last week"


def _panel(fig, rect, title: str, alpha: float):
    ax = fig.add_axes(rect)
    ax._no_fit = True
    ax.set_facecolor(T.BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=6, labelsize=T.px_to_pt(15), colors=T.TEXT_MUTED)
    for sp in ax.spines.values():
        sp.set_color(T.DIVIDER)
        sp.set_alpha(alpha)
    fig.text(rect[0], rect[1] + rect[3] + 0.012, title, color=T.GOLD, fontsize=T.px_to_pt(18),
             fontweight="bold", va="bottom", ha="left", alpha=alpha)
    return ax


def render(hero: str, data: dict, meta: T.ChartMeta, progress: float = 1.0, highlight=None) -> "T.Figure":
    r = data["d"].loc[hero]
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    p_line = T.ease_out_cubic(T.phase(progress, 0.15, 0.55))
    p_bars = T.ease_out_cubic(T.phase(progress, 0.40, 0.75))
    p_part = T.ease_out_cubic(T.phase(progress, 0.65, 0.95))

    fig = T.plt.figure(figsize=(T.CARD_IN, T.CARD_IN), dpi=T.DPI, facecolor=T.BG)
    tier = r["tier"]
    T.draw_header(fig, hero.upper(), f"{tier} · #{int(r['rank'])} of {data['n_heroes']} on the tier list · "
                                     f"power {r['power']:+.1f}", alpha=p_frame)
    # big icon, top right
    icon_px = 96
    icon = T.icon_any(hero, icon_px, T.archetype_color(r["hero_type"]), p_frame, wr=r["winrate"])
    iax = fig.add_axes([1 - T.MARGIN_X - icon_px / T.CARD_PX, 1 - 0.022 - icon_px / T.CARD_PX,
                        icon_px / T.CARD_PX, icon_px / T.CARD_PX])
    iax._no_fit = True
    iax.imshow(icon)
    iax.axis("off")

    ban = r.get("ban_rate")
    has_prev = "winrate_prev" in r.index and pd.notna(r.get("winrate_prev"))
    tiles = [
        (f"{r['winrate']:.1f}%", "win rate", T.wr_color(r["winrate"]),
         _fmt_delta(r["wr_delta"], " pts") if has_prev else None),
        (f"{r['pick_rate']:.0f}%", "of matches picked", T.TEXT,
         _fmt_delta(r["pick_delta"], " pts", 0) if has_prev else None),
        (f"{ban:.0f}%" if pd.notna(ban) else "—", "of matches banned", T.TEXT,
         (_fmt_delta(ban - r["ban_rate_prev"], " pts", 0) if has_prev and pd.notna(r.get("ban_rate_prev"))
          and pd.notna(ban) else ("not bannable yet" if pd.notna(ban) and ban == 0 else None))),
        (f"{r['power']:+.1f}", "power", TIER_COLOR.get(tier, T.TEXT),
         _fmt_delta(r["power_delta"]) if has_prev else None),
    ]
    T.draw_kpi_row(fig, tiles, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    top = fig._content_top - 0.06
    bottom = T.FOOTER_LINE_Y + 0.05
    left, right = T.MARGIN_X + 0.055, 1 - T.MARGIN_X
    split = bottom + (top - bottom) * 0.52            # trend above, game length + partners below

    # --- trend
    ax = _panel(fig, [left, split + 0.045, right - left, top - split - 0.045], "WIN RATE, LAST 8 WEEKS", p_frame)
    h = data["hist"]
    h = h[h["hero_name"] == hero] if len(h) else h
    if len(h) >= 2:
        x = h["week"].to_numpy(float)                   # real week numbers: gaps stay gaps
        ax.axhline(50, color=T.TEXT_MUTED, lw=1.2, alpha=0.6 * p_frame)
        ax.fill_between(x, h["wr_lo"], h["wr_hi"], color=T.GOLD, alpha=0.12 * p_frame, lw=0)
        upto = x[0] + p_line * (x[-1] - x[0])
        k = int(np.searchsorted(x, upto, side="right"))
        xs = list(x[:k]) + ([upto] if upto < x[-1] else [])
        ys = list(h["winrate"].iloc[:k]) + ([np.interp(upto, x, h["winrate"])] if upto < x[-1] else [])
        ax.plot(xs, ys, color=T.GOLD, lw=3, solid_capstyle="round", zorder=3)
        for i in range(k):
            v = h["winrate"].iloc[i]
            ax.scatter([x[i]], [v], s=46, color=T.wr_color(v), zorder=4, edgecolor=T.BG, lw=1.5)
            if i in (0, len(h) - 1):
                ax.text(x[i], v + 0.35, f"{v:.1f}%", ha="center", va="bottom", color=T.TEXT,
                        fontsize=T.px_to_pt(16), fontweight="bold", path_effects=T.STROKE, zorder=5)
        weeks = np.arange(int(x[0]), int(x[-1]) + 1)
        ax.set_xticks(weeks, [f"W{w}" for w in weeks])
        lo, hi = min(h["wr_lo"].min(), 49), max(h["wr_hi"].max(), 51)
        ax.set_ylim(lo - 0.6, hi + 0.9)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Trend appears once earlier weeks are in the project folder", ha="center",
                va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(16), transform=ax.transAxes,
                alpha=p_frame)
    ax.grid(axis="y", color=T.GRID, alpha=p_frame)

    # --- game length
    gw = 0.54
    ax = _panel(fig, [left, bottom, (right - left) * gw, split - bottom - 0.035], "WIN RATE BY GAME LENGTH", p_frame)
    dur = data["dur"]
    rows = dur[dur["hero_name"] == hero].set_index("duration_bracket") if dur is not None else pd.DataFrame()
    if len(rows):
        rows = rows.reindex([b for b in DURATION_BRACKETS if b in rows.index])
        x = np.arange(len(rows))
        ok_rows = rows[rows["total_games"] >= MIN_BRACKET_N]
        span = max(4.0, float(np.nanmax(np.abs(ok_rows["winrate"] - 50))) + 1) if len(ok_rows) else 6.0
        for i, (b, rr) in enumerate(rows.iterrows()):
            v = 50 + float(np.clip(rr["winrate"] - 50, -span, span)) * p_bars
            ok = rr["total_games"] >= MIN_BRACKET_N
            ax.bar(i, v - 50, bottom=50, width=0.72, color=T.wr_color(rr["winrate"]) if ok else T.NEUTRAL,
                   alpha=0.9 if ok else 0.35, hatch=None if ok else "//", edgecolor=T.BG, lw=0, zorder=3)
        ax.axhline(50, color=T.TEXT_MUTED, lw=1.2, alpha=p_frame)
        ax.set_ylim(50 - span, 50 + span)
        ax.set_xticks(x, [b.replace("m", "").replace("-", "–") for b in rows.index], rotation=0,
                      fontsize=T.px_to_pt(12))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_xlabel("minutes", color=T.TEXT_MUTED, fontsize=T.px_to_pt(14), labelpad=4)
        ax.grid(axis="y", color=T.GRID, alpha=p_frame)
        ax.text(1.0, 1.02, "hatched = fewer than 150 games", transform=ax.transAxes, ha="right", va="bottom",
                color=T.TEXT_MUTED, fontsize=T.px_to_pt(12), alpha=p_frame)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No game-length export this week", ha="center", va="center", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(15), transform=ax.transAxes, alpha=p_frame)

    # --- partners
    px0 = left + (right - left) * gw + 0.04
    ax = _panel(fig, [px0, bottom, right - px0, split - bottom - 0.035], "BEST & WORST PARTNERS", p_frame)
    ax.axis("off")
    ax.set_xlim(0, 1)
    best, worst = partners(data["duos"], hero)
    rows = [(rr, True) for _, rr in best.iterrows()] + [(rr, False) for _, rr in worst.iterrows()]
    n = max(len(rows), 1)
    ax.set_ylim(PARTNERS_BEST + PARTNERS_WORST - 0.5, -0.5)          # fixed row height, rows from the top
    if not rows:
        ax.text(0.5, 1.5, "No duo with enough games\nis clearly above or below expectation",
                ha="center", va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(15), alpha=p_frame)
    for i, (rr, good) in enumerate(rows):
        t = T.ease_out_cubic(T.phase(p_part, i / (n + 1), i / (n + 1) + 0.4))
        if t <= 0:
            continue
        col = T.GREEN if good else T.RED
        T.add_icon(ax, T.icon_any(rr["partner"], 50, T.archetype_color(
            data["d"].loc[rr["partner"], "hero_type"] if rr["partner"] in data["d"].index else ""), t,
            wr=rr["wr"]), (0.09, i))
        ax.text(0.2, i - 0.12, rr["partner"], ha="left", va="center", color=T.TEXT, alpha=t,
                fontsize=T.px_to_pt(19), fontweight="bold")
        ax.text(0.2, i + 0.22, f"{rr['wr']:.1f}% together · {int(rr['n']):,} games", ha="left", va="center",
                color=T.TEXT_MUTED, alpha=t, fontsize=T.px_to_pt(13))
        ax.text(1.0, i, f"{rr['lift']:+.1f}", ha="right", va="center", color=col, alpha=t,
                fontsize=T.px_to_pt(24), fontweight="bold")
    return fig


# --- INSIGHTS / ENTRY POINTS --------------------------------------------------------
def insights(data_dir: Path, **_) -> list[Insight]:
    """A one-line profile per hero (context on its spotlight card)."""
    data = load_all(data_dir)
    out = []
    for hero, r in data["d"].iterrows():
        best, _ = partners(data["duos"], hero)
        bits = [f"{r['winrate']:.1f}% win rate in {r['pick_rate']:.0f}% of matches"]
        if pd.notna(r.get("ban_rate")):
            bits.append(f"banned in {r['ban_rate']:.0f}%")
        if len(best):
            bits.append(f"best partner {best.iloc[0]['partner']} ({best.iloc[0]['lift']:+.1f} pts)")
        out.append(Insight(f"hero_{slug(hero)}", "hero_profile", context=True,
                           headline=f"{hero}: {r['tier']}, #{int(r['rank'])} of {data['n_heroes']}",
                           detail="; ".join(bits)))
    return out


def cards(data_dir: Path, heroes: list[str] | None = None, **_) -> dict:
    """{output name: draw(progress=1.0, highlight=None) -> Figure} for every card this script makes."""
    data = load_all(data_dir)
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), None)
    names = heroes or list(data["d"].index)
    return {f"hero_{slug(h)}": (lambda progress=1.0, highlight=None, h=h: render(h, data, meta, progress))
            for h in names if h in data["d"].index}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--heroes", default="", help='only these heroes, e.g. "Haze,Victor"')
    ap.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    names = [h.strip() for h in args.heroes.split(",") if h.strip()] or None
    saved = T.save_cards(cards(data_dir, heroes=names), data_dir / "graphs" / "heroes", args.progress)
    hero_groups(data_dir, load_hero_meta(data_dir))
    print(f"Saved {len(saved)} hero cards to {data_dir / 'graphs' / 'heroes'}")


if __name__ == "__main__":
    main()
