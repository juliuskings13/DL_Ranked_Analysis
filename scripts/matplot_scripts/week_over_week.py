"""
week_over_week.py — what changed since last week (1080x1080 cards + hero_deltas.csv).

Compares this week's hero_winrates.csv with the previous week's folder (week<N-1>_analysis),
or with any folder passed as --base (e.g. the last week before a patch).

Outputs
    hero_movers.png       biggest win-rate risers and fallers, with 95% ranges
    tier_list.png         every hero in its tier by POWER: win-rate edge scaled by how present
                          the hero is in the meta, picks plus a share of bans
                          (chart_stats.add_power), with arrows for heroes that changed tier
                          and NEW badges
    hero_deltas.csv       one row per hero: both weeks' games / win rate / pick rate / rank /
                          tier, the change, its 95% range and how sure it is (read by insights.py)

Without a previous week, the tier list is still drawn (no arrows) and the movers card is skipped.

Usage: python3 week_over_week.py [week_dir] [--base DIR] [--highlight "Haze,Paige"]
"""
from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

import theme as T
from chart_stats import (BAN_WEIGHT, POWER_ALPHA, TIER_BANDS, TIER_NAMES, Insight, add_power, hero_deltas,
                         load_hero_meta, normal_cdf, previous_week_dir, total_matches_from_hero_rows,
                         wr_confidence)

MOVER_ROWS = 6          # risers and fallers each
BAR_H = 0.62
DIM_ALPHA = 0.25
NOISE_ALPHA = 0.45      # tier arrows whose change is within noise


# --- DATA --------------------------------------------------------------------------
def prepare(data_dir: Path, base: Path | None = None) -> tuple[pd.DataFrame, str | None]:
    """(deltas, previous label). Without a previous week: this week's table, label None."""
    prev = Path(base) if base else previous_week_dir(data_dir)
    if prev is None or not (prev / "hero_winrates.csv").exists():
        df = add_power(load_hero_meta(data_dir))
        df.attrs["ban_weight"] = BAN_WEIGHT if "ban_rate" in df and df["ban_rate"].notna().any() else 0.0
        df["status"], df["tier_change"], df["power_clear"] = "returning", np.nan, False
        return df.sort_values("power", ascending=False).reset_index(drop=True), None
    return hero_deltas(data_dir, prev), T.week_label_from_dir(prev)


def pick_movers(d: pd.DataFrame, n: int = MOVER_ROWS) -> pd.DataFrame:
    """Most confident risers and fallers (ranked by the cautious end of their 95% range,
    like the duo cards), shown best to worst."""
    d = d[d["status"] == "returning"]
    up = d[d["wr_delta"] > 0].nlargest(n, "delta_lo")
    down = d[d["wr_delta"] < 0].nsmallest(n, "delta_hi")
    return (pd.concat([up.sort_values("wr_delta", ascending=False),
                       down.sort_values("wr_delta", ascending=False)]), len(up))


# --- MOVERS CARD -------------------------------------------------------------------
def render_movers(d: pd.DataFrame, meta: T.ChartMeta, prev_label: str,
                  progress: float = 1.0, highlight: set[str] | None = None) -> "T.Figure":
    highlight = {h.lower() for h in highlight} if highlight else None
    rows, n_up = pick_movers(d)
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    fig, ax = T.new_table_card()
    sub = []
    if n_up:
        r = rows.iloc[0]
        sub.append(f"{r['hero_name']} {r['wr_delta']:+.1f} pts")
    if len(rows) > n_up:
        r = rows.iloc[-1]
        sub.append(f"{r['hero_name']} {r['wr_delta']:+.1f} pts")
    T.draw_header(fig, f"BIGGEST MOVERS SINCE {prev_label.upper()}",
                  "  ·  ".join(sub) + "  (win rate, percentage points)", alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    gap = 1 if 0 < n_up < len(rows) else 0
    n = len(rows) + gap
    row = T.table_rows(fig, ax, n, header_rows=1.0, footer_rows=0.9)
    W, _ = T.content_px(fig)
    span = float(np.ceil(max(rows["delta_hi"].abs().max(), rows["delta_lo"].abs().max()) + 0.3))

    icon_px = min(row * 0.84, 70)
    font = T.fit_font_px(rows["hero_name"], 250, row * 0.50, max_px=28, min_px=13)
    name_w = max(T.text_px(h, font) for h in rows["hero_name"])
    val_font = min(font * 0.95, 25)
    wr_w = T.text_px("55.5% → 55.5%", val_font * 0.9, bold=False)
    cols = T.layout_columns({"icon": icon_px, "name": name_w, "wr": wr_w, "bars": None}, W)
    label_w = T.text_px("+5.5", val_font) + 12
    b0, b1 = cols["bars"][0] + label_w / W, cols["bars"][1] - label_w / W

    def to_x(v):
        return b0 + (np.asarray(v) + span) / (2 * span) * (b1 - b0)

    hdr = dict(color=T.GOLD, fontsize=T.px_to_pt(17), fontweight="bold", va="center", alpha=p_frame)
    ax.text(cols["name"][0], -1.0, "HERO", ha="left", **hdr)
    ax.text(cols["wr"][1], -1.0, f"{prev_label.upper()} → NOW", ha="right", **hdr)
    ax.text(to_x(0), -1.0, "CHANGE", ha="center", **hdr)
    step = 1 if span <= 4 else 2
    tick_font = min(16, val_font * 0.75)
    for v in np.arange(-(span // step) * step, span + 0.01, step):     # ticks centered on 0
        ax.plot([to_x(v)] * 2, [-0.5, n - 0.5], color=T.GRID, lw=1, zorder=0, alpha=p_frame)
        ax.text(to_x(v), n - 0.4, f"{v:+.0f}" if v else "0", ha="center", va="top", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(tick_font), alpha=p_frame)
    ax.plot([to_x(0)] * 2, [-0.5, n - 0.5], color=T.TEXT_MUTED, lw=1.8, zorder=1, alpha=p_frame)
    ax.text(1.0, n + 0.35, "gray = could be noise (95% range crosses 0)", ha="right", va="top",
            color=T.TEXT_MUTED, fontsize=T.px_to_pt(14), alpha=p_frame)

    y = 0
    for i, (_, r) in enumerate(rows.iterrows()):
        if gap and i == n_up:
            ax.plot([0, 1], [y, y], color=T.DIVIDER, lw=1.5, ls=(0, (6, 4)), alpha=p_frame)
            y += 1
        s = 0.15 + 0.5 * i / max(len(rows) - 1, 1)
        t = T.ease_out_cubic(T.phase(progress, s, s + 0.3))
        hl = highlight is None or r["hero_name"].lower() in highlight
        a = (1.0 if hl else DIM_ALPHA) * t
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y - 0.5), 1, 1, color=T.PANEL, alpha=0.55 * p_frame, zorder=0, lw=0))
        if t > 0:
            T.add_icon(ax, T.icon_any(r["hero_name"], icon_px, T.archetype_color(r["hero_type"]), a,
                                      wr=r["winrate"]),
                       ((cols["icon"][0] + cols["icon"][1]) / 2, y), zorder=6)
        ax.text(cols["name"][0], y, r["hero_name"], ha="left", va="center", color=T.TEXT,
                fontsize=T.px_to_pt(font), fontweight="bold", alpha=a)
        now_s = f"{r['winrate']:.1f}%"
        ax.text(cols["wr"][1], y, now_s, ha="right", va="center", color=T.TEXT,
                fontsize=T.px_to_pt(val_font * 0.9), fontweight="bold", alpha=a)
        ax.text(cols["wr"][1] - (T.text_px(now_s, val_font * 0.9) + 4) / W, y,
                f"{r['winrate_prev']:.1f}% →", ha="right", va="center", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(val_font * 0.9), alpha=a)

        clear = bool(r["clear"])
        col = (T.GREEN if r["wr_delta"] > 0 else T.RED) if clear else T.TEXT_MUTED
        dv = r["wr_delta"] * t
        x0, xv = to_x(0), to_x(dv)
        ax.add_patch(Rectangle((min(x0, xv), y - BAR_H / 2), abs(xv - x0), BAR_H,
                               color=col, alpha=0.9 * a, zorder=3, lw=0))
        if t >= 1:
            ax.plot([to_x(r["delta_lo"]), to_x(r["delta_hi"])], [y, y], color=T.TEXT, lw=2.2,
                    alpha=0.5 * a, zorder=4)
            right = r["wr_delta"] >= 0
            ax.text(to_x(r["delta_hi"] if right else r["delta_lo"]) + (6 if right else -6) / W, y,
                    f"{r['wr_delta']:+.1f}", ha="left" if right else "right", va="center",
                    color=col if col != T.TEXT_MUTED else T.TEXT, fontsize=T.px_to_pt(val_font),
                    fontweight="bold", alpha=a, zorder=5, path_effects=T.STROKE)
        y += 1
    return fig


# --- TIER LIST CARD ----------------------------------------------------------------
def _tier_layout(counts: list[int], area_w: float, area_h: float) -> tuple[float, int, float]:
    """Largest icon size (px) whose wrapped rows fit: returns (icon px, icons per line, line px)."""
    for s in range(88, 30, -2):
        g = s * 0.22
        per_line = max(1, int((area_w + g) // (s + g)))
        line_h = s * 1.42                                   # icon + win-rate label + breathing room
        lines = sum(max(1, ceil(c / per_line)) if c else 0.6 for c in counts)
        if lines * line_h + len(counts) * 10 <= area_h:
            return float(s), per_line, area_h / lines            # stretch rows to fill the card
    return 30.0, max(1, int(area_w // 37)), 30 * 1.42


def tier_subtitle(d: pd.DataFrame) -> str:
    if "tier_prev" not in d or d["tier_change"].isna().all():
        r = d.iloc[0]
        return f"{r['hero_name']} tops the list: {r['winrate']:.1f}% win rate in {r['pick_rate']:.0f}% of matches"
    moved = d[d["tier_change"].fillna(0) != 0]
    parts = []
    if (moved["tier_change"] > 0).any():
        r = moved.sort_values(["tier_change", "wr_delta"], ascending=False).iloc[0]
        parts.append(f"{r['hero_name']} climbs {r['tier_prev']} → {r['tier']}")
    if (moved["tier_change"] < 0).any():
        r = moved.sort_values(["tier_change", "wr_delta"]).iloc[0]
        parts.append(f"{r['hero_name']} drops {r['tier_prev']} → {r['tier']}")
    return "  ·  ".join(parts) or "No hero changed tier this week"


def render_tier_list(d: pd.DataFrame, meta: T.ChartMeta, prev_label: str | None,
                     progress: float = 1.0, highlight: set[str] | None = None) -> "T.Figure":
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    p_badges = T.ease_out_cubic(T.phase(progress, 0.85, 1.0))
    fig, ax = T.new_table_card()
    T.draw_header(fig, f"{meta.week_label.upper()} TIER LIST", tier_subtitle(d), alpha=p_frame)
    T.draw_archetype_legend(fig, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    W, H = T.content_px(fig)
    note_h = 52
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                                       # pixel coordinates, y down
    tiers = TIER_NAMES[::-1]                                # GOD TIER at the top
    groups = {t: d[d["tier"] == t].sort_values("power", ascending=False) for t in tiers}
    label_w = max(T.text_px(t, 24) for t in tiers) + 26
    area_x0 = label_w + 14
    area_w = W - area_x0 - 8
    s, per_line, line_h = _tier_layout([len(g) for g in groups.values()], area_w, H - note_h - 10)
    g = s * 0.22
    wr_font = max(12, s * 0.24)

    order = [(t, j) for t in tiers for j in range(len(groups[t]))]
    total = max(len(order), 1)
    k = 0
    y = 0.0
    for tier in tiers:
        grp = groups[tier]
        n_lines = max(1, ceil(len(grp) / per_line)) if len(grp) else 0.6
        h = n_lines * line_h
        tint, strength, label_col = T.TIER_STYLE[tier]
        ax.add_patch(Rectangle((0, y), W, h, facecolor=tint, lw=0, zorder=0,
                               alpha=T.ZONE_TINT_ALPHA * strength * p_frame))
        ax.add_patch(Rectangle((0, y), label_w, h, facecolor=tint, lw=0, zorder=0,
                               alpha=min(1, T.ZONE_TINT_ALPHA * strength * 2.2) * p_frame))
        ax.text(label_w / 2, y + h / 2 - 9, tier, ha="center", va="center", color=label_col,
                fontsize=T.px_to_pt(24 if len(tier) < 8 else 21), fontweight="bold", alpha=p_frame)
        lo, hi = next((lo - 50, hi - 50) for lo, hi, name in TIER_BANDS if name == tier)
        rng = (f"below {hi:+g}" if not np.isfinite(lo) else
               f"{lo:+g} and up" if not np.isfinite(hi) else f"{lo:+g} to {hi:+g}")
        ax.text(label_w / 2, y + h / 2 + 15, rng, ha="center", va="center", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(15), alpha=p_frame)
        if grp.empty:
            ax.text(area_x0, y + h / 2, "nobody this week", ha="left", va="center", color=T.TEXT_MUTED,
                    fontsize=T.px_to_pt(16), style="italic", alpha=p_frame)
        for j, (_, r) in enumerate(grp.iterrows()):
            line, col = divmod(j, per_line)
            cx = area_x0 + s / 2 + col * (s + g)
            cy = y + line * line_h + s / 2 + (line_h - s * 1.3) / 2
            st = 0.15 + 0.68 * k / total
            t = T.ease_out_cubic(T.phase(progress, st, st + 0.12))
            k += 1
            if t <= 0:
                continue
            hl = highlight is None or r["hero_name"].lower() in highlight
            a = (1.0 if hl else DIM_ALPHA) * t
            size = s * (0.7 + 0.3 * t)                      # icons pop in
            T.add_icon(ax, T.icon_any(r["hero_name"], size, T.archetype_color(r["hero_type"]), a,
                                      wr=r["winrate"]), (cx, cy), zorder=6)
            ax.text(cx, cy + s / 2 + 3, f"{r['winrate']:.1f}%", ha="center", va="top", color=T.TEXT,
                    fontsize=T.px_to_pt(wr_font), fontweight="bold", alpha=a, zorder=7)
            # movement badges
            badge_a = a * p_badges
            if badge_a <= 0:
                continue
            bx, by = cx + s * 0.38, cy - s * 0.36
            if r.get("status") == "new" and prev_label:
                ax.add_patch(FancyBboxPatch((bx - 17, by - 9), 34, 18, boxstyle="round,pad=0,rounding_size=6",
                                            facecolor=T.GOLD, lw=0, alpha=badge_a, zorder=8))
                ax.text(bx, by, "NEW", ha="center", va="center", color=T.BG, fontsize=T.px_to_pt(11),
                        fontweight="bold", alpha=badge_a, zorder=9)
            elif pd.notna(r.get("tier_change")) and r["tier_change"] != 0:
                up = r["tier_change"] > 0
                faded = 1.0 if r.get("power_clear", False) else NOISE_ALPHA
                ax.text(bx, by, "▲" if up else "▼", ha="center", va="center",
                        color=T.GREEN if up else T.RED, fontsize=T.px_to_pt(max(16, s * 0.32)),
                        fontweight="bold", alpha=badge_a * faded, zorder=9, path_effects=T.STROKE)
        y += h
    bw = d.attrs.get("ban_weight", BAN_WEIGHT if "ban_rate" in d and d["ban_rate"].notna().any() else 0)
    presence = f"pick rate + {bw:g} × ban rate" if bw else "pick rate"
    note = f"Power = (win rate − 50) × (presence ÷ median)^{POWER_ALPHA:g}  ·  presence = {presence}"
    note += "\nNumber under icon = win rate"
    if prev_label:
        note += f"  ·  ▲▼ changed tier since {prev_label}  ·  faded = change within noise"
    ax.text(W, H - 4, note, ha="right", va="bottom", color=T.TEXT_MUTED, fontsize=T.px_to_pt(14),
            alpha=p_frame, linespacing=1.4)
    return fig


# --- INSIGHTS ----------------------------------------------------------------------
WATCH_N = 3             # nerf-watch / buff-watch heroes per week


def _tags(r) -> tuple:
    tags = []
    if r.get("status") == "new":
        tags.append("new hero")
    if pd.notna(r.get("games_prev")) and r["games_prev"] < 0.5 * r["games"]:
        tags.append("partial week last time (likely released mid-week)")
    if pd.notna(r.get("ban_rate_prev")) and r["ban_rate_prev"] == 0 and r.get("ban_rate", 0) > 0:
        tags.append("entered the ban pool: bans now add to its presence")
    elif pd.notna(r.get("ban_rate")) and r["ban_rate"] == 0:
        tags.append("not bannable yet (new hero)")
    return tuple(tags)


def insights(data_dir: Path, base: str | None = None, **_) -> list[Insight]:
    d, prev_label = prepare(data_dir, Path(base) if base else None)
    out: list[Insight] = []
    has_bans = "ban_rate" in d and d["ban_rate"].notna().any()

    def where(r):
        ban = f", banned {r['ban_rate']:.0f}%" if has_bans and pd.notna(r.get("ban_rate")) else ""
        return f"{r['winrate']:.1f}% win rate in {r['pick_rate']:.0f}% of matches{ban}"

    top = d.head(3)
    out.append(Insight("tier_list", "tier_top", context=True,
                       headline=f"{', '.join(top['hero_name'][:-1])} and {top['hero_name'].iloc[-1]} top the tier list",
                       detail="; ".join(f"{r['hero_name']}: {where(r)}" for _, r in top.iterrows()),
                       highlight=tuple(top["hero_name"])))

    # nerf / buff watch: the extremes of the power ranking
    for label, rows, kind in (("Nerf watch", d[d["tier"] == "GOD TIER"].head(WATCH_N), "nerf_watch"),
                              ("Buff watch", d[d["tier"] == "TRASH"].tail(WATCH_N).iloc[::-1], "buff_watch")):
        for _, r in rows.iterrows():
            out.append(Insight("tier_list", kind, hero=r["hero_name"], highlight=(r["hero_name"],),
                               headline=f"{label}: {r['hero_name']} ({r['tier']}, power {r['power']:+.1f})",
                               detail=where(r).capitalize(), effect=abs(r["power"]),
                               confidence=float(wr_confidence(r["winrate"], r["games" if "games" in r else "total_matches"])),
                               tags=_tags(r)))
    if not prev_label:
        return out

    movers_card, _ = pick_movers(d)
    on_movers = set(movers_card["hero_name"])
    for _, r in d.iterrows():
        if r["status"] == "new":
            out.append(Insight("tier_list", "new_hero", hero=r["hero_name"], highlight=(r["hero_name"],),
                               headline=f"New: {r['hero_name']} debuts in {r['tier']}",
                               detail=where(r).capitalize(), effect=abs(r["power"]) + 1,
                               confidence=float(wr_confidence(r["winrate"], r["games"])), tags=_tags(r)))
        elif pd.notna(r["tier_change"]) and r["tier_change"] != 0:
            up = r["tier_change"] > 0
            se = np.hypot(r["power_se"], r["power_se_prev"])
            move = f"{r['hero_name']} {'climbs' if up else 'drops'} {r['tier_prev']} → {r['tier']}"
            wr_s = f"{r['winrate_prev']:.1f}% → {r['winrate']:.1f}%"
            pres_s = f"{r['presence_prev']:.0f}% → {r['presence']:.0f}% of matches"
            if abs(r["delta_from_presence"]) > abs(r["delta_from_wr"]):
                # popularity, not win rate, moved this hero
                more = r["presence"] > r["presence_prev"]
                why = ("more players picking a winning hero" if more and r["winrate"] > 50 else
                       "more players picking a losing hero" if more else
                       "fewer players picking a losing hero" if r["winrate"] < 50 else
                       "fewer players picking a winning hero")
                headline, detail = f"{move}, mostly from {why}", f"Presence {pres_s}; win rate {wr_s}"
            else:
                headline = move
                detail = f"Win rate {wr_s}" + (f"; presence {pres_s}" if abs(r["presence"] - r["presence_prev"]) >= 3 else "")
            out.append(Insight("tier_list", "tier_change", hero=r["hero_name"], highlight=(r["hero_name"],),
                               headline=headline, detail=detail, effect=abs(r["power_delta"]),
                               confidence=float(normal_cdf(abs(r["power_delta"]) / se)), tags=_tags(r)))
        elif r["clear"]:
            up = r["wr_delta"] > 0
            out.append(Insight("hero_movers" if r["hero_name"] in on_movers else "tier_list", "mover",
                               hero=r["hero_name"], highlight=(r["hero_name"],),
                               headline=f"{r['hero_name']} {'up' if up else 'down'} {abs(r['wr_delta']):.1f} pts "
                                        f"to {r['winrate']:.1f}% (still {r['tier']})",
                               detail=f"{r['winrate_prev']:.1f}% {prev_label} → {r['winrate']:.1f}% now, "
                                      f"in {r['pick_rate']:.0f}% of matches",
                               effect=abs(r["wr_delta"]), confidence=float(r["p_change"]), tags=_tags(r)))
    return out


# --- ENTRY POINTS ------------------------------------------------------------------
def cards(data_dir: Path, base: str | None = None, **_) -> dict:
    """{output name: draw(progress=1.0, highlight=None) -> Figure} for every card this script makes."""
    d, prev_label = prepare(data_dir, Path(base) if base else None)
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), total_matches_from_hero_rows(d["games"]
                       if "games" in d else d["total_matches"]))
    out = {"tier_list": lambda progress=1.0, highlight=None: render_tier_list(
        d, meta, prev_label, progress, highlight)}
    if prev_label and (d["status"] == "returning").any():
        out["hero_movers"] = lambda progress=1.0, highlight=None: render_movers(
            d, meta, prev_label, progress, highlight)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--base", default=None, help="folder to compare against (default: previous week)")
    ap.add_argument("--highlight", default="")
    ap.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    d, prev_label = prepare(data_dir, Path(args.base) if args.base else None)
    if prev_label:
        d.to_csv(data_dir / "hero_deltas.csv", index=False)
        print(f"Saved {data_dir / 'hero_deltas.csv'}  (vs {prev_label}; "
              f"{int(d['clear'].sum())} of {len(d)} heroes changed beyond noise)")
    else:
        print("No previous week found: tier list only (pass --base DIR to compare with a folder)")
    for p in T.save_cards(cards(data_dir, args.base), data_dir / "graphs", args.progress,
                          T.parse_names(args.highlight)):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()