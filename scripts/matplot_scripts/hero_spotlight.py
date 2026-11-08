"""
hero_spotlight.py — one 1080x1080 card per hero, for hero spotlight videos and shorts.

    header      hero name; tier, rank and power; big icon
    tiles       win rate, pick rate, ban rate, power, each with the change since last week
    trend       win rate by week (last 8 weeks; every week of a period such as a season), 95% range,
                patches as dashed lines colored by what each did to the hero, NET first-to-last change
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
PARTNERS_BEST, PARTNERS_WORST = 3, 3
PARTNER_CONF = 0.80          # how sure the lift is real; below 95% the row is marked "could be noise"
WORST_NEEDS_SUB50 = True     # a "worst" partner must also lose together (< 50%); False = any clear drop
ICON_PX, TITLE_PX, SUBTITLE_PX = 136, 58, 24      # header (title and icon ~40% larger than other cards)
NAME_ART_PX = 72             # height of the hero's name logo (assets/hero_names/), when there is one
TIER_COLOR = {name: style[2] for name, style in T.TIER_STYLE.items()}

# --- animation: everything appears in reading order --------------------------------
BUILD_S = 15.0               # seconds the card takes to build itself (animate.py / assemble_video use it)
P_HEADER = (0.0, 0.06)       # name, subtitle, icon (fractions of the build)
P_TILES = (0.05, 0.22)       # the four tiles, left to right, one after another
P_PANELS = [("trend", 0.22, 0.42), ("rank", 0.42, 0.61), ("length", 0.61, 0.80), ("partners", 0.80, 0.98)]
PATCH_COLORS = {"buff": T.GREEN, "nerf": T.RED, "mixed": T.GOLD, "rework": T.GOLD}


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
           "is_period": "week" not in Path(data_dir).name.lower(),
           "dur": None, "duos": None, "prev_label": T.week_label_from_dir(prev) if prev is not None else None}
    try:
        from hero_strength_and_scaling import load
        out["dur"], _, _ = load(data_dir)
        out["median"] = duration_summary(data_dir)["median"]
    except FileNotFoundError:
        pass
    try:
        from rank_performance import curve_table, load_tier_winrates
        out["ranks"] = load_tier_winrates(data_dir)
        out["rank_net"] = curve_table(out["ranks"]).set_index("hero_name")
    except FileNotFoundError:
        out["ranks"] = out["rank_net"] = None
    try:
        from hero_duo_synergy_table import prepare as duo_prepare
        out["duos"] = duo_prepare(data_dir)
    except FileNotFoundError:
        pass
    out["patches"] = load_patches(data_dir)
    return out


def _week_x(t: pd.Timestamp) -> float:
    """A moment in time on the weekly trend's x axis: week N's point sits mid-week at x = N."""
    launch = pd.Timestamp(T.RANKED_START_DATE, tz="UTC")
    return (pd.Timestamp(t).tz_convert("UTC") - launch).total_seconds() / 86400 / 7 + 0.5


def _patch_calls(folder: Path) -> dict:
    """hero -> buff / nerf / mixed / rework from a patch-notes folder's overall.csv (your reviewed calls)."""
    try:
        o = pd.read_csv(folder / "overall.csv", dtype=str, keep_default_na=False)
    except (OSError, ValueError):
        return {}
    if "subject" not in o.columns:
        return {}
    o = o[o.get("kind", "hero") == "hero"] if "kind" in o.columns else o
    call = o["overall"].where(o.get("overall", "") != "", o.get("auto", "")) if "overall" in o.columns else o["auto"]
    return {h: c.strip().lower() for h, c in zip(o["subject"], call) if c and c.strip()}


def load_patches(data_dir: Path) -> list[dict]:
    """Balance patches for the weekly trend: [{x, when, name, calls: {hero: buff|nerf|...}}].
    Dates from the period's patches.csv export (sync_patches.py events) and from the patches/ folders
    (patch_notes.py); the per-hero buff / nerf calls from those folders' overall.csv."""
    events = []
    try:
        p = pd.read_csv(Path(data_dir) / "patches.csv", parse_dates=["start_time"])
        for _, r in p.iterrows():
            if r.get("kind") != "hero_release":
                events.append({"when": pd.Timestamp(r["start_time"]), "name": r.get("name", ""), "calls": {}})
    except (OSError, ValueError, KeyError):
        pass
    folders = []
    for f in sorted((T.PROJECT_ROOT / "patches").glob("20??-??-??*")):
        try:
            folders.append((pd.Timestamp(f.name[:10], tz="UTC") + pd.Timedelta(hours=12), f))
        except ValueError:
            continue
    for when, f in folders:
        near = [e for e in events if abs((e["when"] - when).total_seconds()) < 1.5 * 86400]
        if near:
            near[0]["calls"] = _patch_calls(f)
        else:
            events.append({"when": when, "name": f.name, "calls": _patch_calls(f)})
    for e in events:
        e["x"] = _week_x(e["when"])
    return sorted(events, key=lambda e: e["x"])


def partners(duos: pd.DataFrame | None, hero: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if duos is None:
        return pd.DataFrame(), pd.DataFrame()
    mine = duos[(duos["hero_name_a"] == hero) | (duos["hero_name_b"] == hero)].copy()
    mine["partner"] = np.where(mine["hero_name_a"] == hero, mine["hero_name_b"], mine["hero_name_a"])
    sure = mine["conf"] >= PARTNER_CONF
    # "best" = they win together (over 50%) AND more than their solo win rates predict;
    # "worst" = they lose together (under 50%) AND less than predicted
    # ranked by win rate together, the number the card shows biggest
    best = mine[sure & (mine["wr"] > 50) & (mine["lift"] > 0)].nlargest(PARTNERS_BEST, "wr")
    losing = (mine["wr"] < 50) if WORST_NEEDS_SUB50 else True
    worst = mine[sure & losing & (mine["lift"] < 0)].nsmallest(PARTNERS_WORST, "wr")
    return best, worst.iloc[::-1]          # worst last


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


def _fade_ticks(ax, alpha: float) -> None:
    """Tick labels and axis labels follow the panel's fade-in (matplotlib doesn't tie them to anything)."""
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_alpha(alpha)
    ax.xaxis.label.set_alpha(alpha)


def _rank_panel(ax, hero: str, data: dict, p_line: float, p_frame: float) -> None:
    """Actual win rate at each of the 11 ranks (hollow = thin sample) and the hero's NET change,
    the same numbers as the rank_curve cards."""
    from rank_performance import RANK_POINT_MIN, RANK_TIER_NAMES
    r = data.get("ranks")
    rows = r[r["hero_name"] == hero].set_index("rank_tier") if r is not None else pd.DataFrame()
    net = data["rank_net"].loc[hero] if data.get("rank_net") is not None and hero in data["rank_net"].index else None
    if rows.empty or net is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No rank data this week", ha="center", va="center", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(15), transform=ax.transAxes, alpha=p_frame)
        return
    tiers = sorted(RANK_TIER_NAMES)
    rows = rows.reindex(tiers)
    wr, n = rows["winrate"].to_numpy(float), rows["games"].fillna(0).to_numpy(float)
    x = np.arange(len(tiers))
    ok = ~np.isnan(wr)
    xi, yi = x[ok], wr[ok]
    ax.axhline(50, color=T.TEXT_MUTED, lw=1.2, alpha=0.6 * p_frame)
    upto = p_line * (len(tiers) - 1)
    keep = (xi <= upto + 1e-9) & (p_line > 0)
    xs, ys = list(xi[keep]), list(yi[keep])
    if len(xs) and upto > xs[-1] and (~keep).any():
        xs.append(upto); ys.append(float(np.interp(upto, xi, yi)))
    ax.plot(xs, ys, color=T.GOLD, lw=2.6, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    for j in xi[keep]:
        few = n[j] < RANK_POINT_MIN
        c = T.wr_color(wr[j])
        ax.scatter([j], [wr[j]], s=40, zorder=4, color=T.BG if few else c, edgecolor=c, lw=1.8)
    lo, hi = min(np.nanmin(wr), 49), max(np.nanmax(wr), 51)
    ax.set_ylim(lo - 0.8, hi + 2.0)
    ax.set_xlim(-0.5, len(tiers) - 0.5)
    ax.set_xticks(x, [RANK_TIER_NAMES[k] for k in tiers], rotation=40, ha="right", rotation_mode="anchor",
                  fontsize=T.px_to_pt(12))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", color=T.GRID, alpha=p_frame)
    clear = net["conf"] >= 0.95
    col = (T.GREEN if net["net"] > 0 else T.RED) if clear else T.TEXT_MUTED
    ax.text(1.0, 1.02, f"NET {net['net']:+.1f} pts", transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=T.px_to_pt(17), fontweight="bold", alpha=p_frame)
    ax.text(0.025, 0.97, f"hollow = under {RANK_POINT_MIN:,} games", transform=ax.transAxes, ha="left", va="top",
            color=T.TEXT_MUTED, fontsize=T.px_to_pt(11), alpha=p_frame)


def _place(fig, arr: np.ndarray, x_px: float, y_top_px: float, zorder: float = 4) -> None:
    """An image array (already at real pixels) with its top-left corner at card px (x_px, y_top_px)."""
    fig.figimage(arr, xo=int(round(T.dev_px(x_px))), yo=int(round(T.OUT_PX - T.dev_px(y_top_px) - arr.shape[0])),
                 origin="upper", zorder=zorder)


def _phase(progress: float, a: float, b: float) -> float:
    return T.ease_out_cubic(T.phase(progress, a, b))


def render(hero: str, data: dict, meta: T.ChartMeta, progress: float = 1.0, highlight=None) -> "T.Figure":
    r = data["d"].loc[hero]
    p_frame = _phase(progress, *P_HEADER)
    panel = {k: (_phase(progress, a, a + 0.03), _phase(progress, a + 0.02, b)) for k, a, b in P_PANELS}

    fig = T.plt.figure(figsize=(T.CARD_IN, T.CARD_IN), dpi=T.DPI, facecolor=T.BG)
    tier = r["tier"]
    icon_px = ICON_PX
    x0 = T.MARGIN_X * T.CARD_PX
    room = T.CARD_PX * (1 - 2 * T.MARGIN_X) - icon_px - 30
    art = T.name_art(hero, NAME_ART_PX, int(room))
    if art is not None:                                # the hero's own name logo
        a = art.copy()
        a[..., 3] *= p_frame
        h_px = a.shape[0] / T.RENDER_SCALE
        top_px = 54 - h_px / 2                         # centered where the text title would be
        _place(fig, a, x0 - 10, top_px)
        fig._content_top = min(getattr(fig, "_content_top", 1.0), 1 - (top_px + h_px - 10) / T.CARD_PX)
    else:
        title_px = T.fit_font_px([hero.upper()], room, max_px=TITLE_PX, min_px=34)
        t = fig.text(T.MARGIN_X, 0.946, hero.upper(), color=T.GOLD, fontsize=T.px_to_pt(title_px),
                     fontweight="bold", va="center", ha="left", alpha=p_frame)
        T._mark_header_bottom(fig, t)
    # subtitle: "OVER · [#13 of 44] on the tier list · power +1.5", the tier and the placing highlighted
    y, fs = 0.884, T.px_to_pt(SUBTITLE_PX)
    x = T.MARGIN_X
    tcol = TIER_COLOR.get(tier, T.TEXT)
    parts = [(f"{tier}", dict(color=tcol, fontweight="bold")),
             ("  ·  ", dict(color=T.TEXT_MUTED)),
             (f"#{int(r['rank'])} of {data['n_heroes']}", dict(color=T.BG, fontweight="bold",
                                                              bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.3",
                                                                        fc=T.GOLD, ec="none", alpha=p_frame))),
             (f"  on the tier list  ·  power {r['power']:+.1f}", dict(color=T.TEXT_MUTED))]
    for txt, kw in parts:
        pad = 0.28 * SUBTITLE_PX / T.CARD_PX if "bbox" in kw else 0      # the pill's padding, each side
        art_ = fig.text(x + pad, y, txt, fontsize=fs, va="center", ha="left", alpha=p_frame, **kw)
        x += T.text_width_frac(fig, art_) + 2 * pad              # the measured width excludes the pill
        T._mark_header_bottom(fig, art_)
    # big icon, top right
    icon = T.icon_any(hero, icon_px, T.archetype_color(r["hero_type"]), p_frame, wr=r["winrate"])
    iax = fig.add_axes([1 - T.MARGIN_X - icon_px / T.CARD_PX, 1 - 14 / T.CARD_PX - icon_px / T.CARD_PX,
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
    a0, a1 = P_TILES
    step = (a1 - a0) / len(tiles)
    T.draw_kpi_row(fig, tiles, alpha=[_phase(progress, a0 + k * step, a0 + k * step + step * 1.6)
                                      for k in range(len(tiles))])
    T.draw_footer(fig, meta, alpha=p_frame)

    top = fig._content_top - 0.06
    bottom = T.FOOTER_LINE_Y + 0.05
    left, right = T.MARGIN_X + 0.038, 1 - T.MARGIN_X
    split = bottom + (top - bottom) * 0.52            # trend above, game length + partners below

    # --- trend (top left) and rank (top right)
    tw = 0.47
    row_gap = 0.085                                   # room for the rotated rank names above the lower panels
    h = data["hist"]
    pf, pl = panel["trend"]
    trend_title = ("WIN RATE BY WEEK" if data.get("is_period") else f"WIN RATE, LAST {TREND_WEEKS} WEEKS")
    ax = _panel(fig, [left, split + row_gap, (right - left) * tw, top - split - row_gap], trend_title, pf)
    h = h[h["hero_name"] == hero] if len(h) else h
    if len(h) >= 2:
        h = h.sort_values("week")
        x = h["week"].to_numpy(float)                   # real week numbers: gaps stay gaps
        ax.axhline(50, color=T.TEXT_MUTED, lw=1.2, alpha=0.6 * pf)
        ax.fill_between(x, h["wr_lo"], h["wr_hi"], color=T.GOLD, alpha=0.12 * pl, lw=0)
        upto = x[0] + pl * (x[-1] - x[0])
        # balance patches: dashed, colored by what the patch did to THIS hero; drawn as the line reaches them
        shown = False
        for e in data.get("patches") or []:
            if not (x[0] - 0.5 <= e["x"] <= x[-1] + 0.5) or e["x"] > upto + 0.5 or pl <= 0:
                continue
            call = e["calls"].get(hero)
            col = PATCH_COLORS.get(call or "", T.TEXT_MUTED)
            ax.axvline(e["x"], color=col, ls=(0, (4, 3)), lw=1.8 if call else 1.0,
                       alpha=(0.9 if call else 0.45) * pf, zorder=2)
            shown = True
        k = int(np.searchsorted(x, upto, side="right")) if pl > 0 else 0
        xs = list(x[:k]) + ([upto] if upto < x[-1] and k else [])
        ys = list(h["winrate"].iloc[:k]) + ([np.interp(upto, x, h["winrate"])] if upto < x[-1] and k else [])
        ax.plot(xs, ys, color=T.GOLD, lw=3, solid_capstyle="round", zorder=3)
        for i in range(k):
            v = h["winrate"].iloc[i]
            ax.scatter([x[i]], [v], s=46, color=T.wr_color(v), zorder=4, edgecolor=T.BG, lw=1.5)
            if i in (0, len(h) - 1):
                ax.text(x[i], v + 0.35, f"{v:.1f}%", ha="center", va="bottom", color=T.TEXT,
                        fontsize=T.px_to_pt(16), fontweight="bold", path_effects=T.STROKE, zorder=5)
        weeks = np.arange(int(x[0]), int(x[-1]) + 1)
        step = 1 if len(weeks) <= 16 else 2          # plain week numbers: the panel title says "by week"
        ax.set_xticks(weeks[::step], [str(w) for w in weeks[::step]])
        lo, hi = min(h["wr_lo"].min(), 49), max(h["wr_hi"].max(), 51)
        ax.set_ylim(lo - 0.6, hi + 0.9)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
        # NET: first week to last week, colored only when the change is clearly more than noise
        first, last = h.iloc[0], h.iloc[-1]
        net = last["winrate"] - first["winrate"]
        se = np.hypot((first["wr_hi"] - first["wr_lo"]) / 3.92, (last["wr_hi"] - last["wr_lo"]) / 3.92)
        clear = abs(net) > 1.96 * se
        col = (T.GREEN if net > 0 else T.RED) if clear else T.TEXT_MUTED
        ax.text(1.0, 1.02, f"NET {net:+.1f} pts", transform=ax.transAxes, ha="right", va="bottom",
                color=col, fontsize=T.px_to_pt(17), fontweight="bold", alpha=pl)
        if shown:
            ax.text(0.02, 0.97, "dashed = patch (green buffed, red nerfed, gold mixed)", transform=ax.transAxes,
                    ha="left", va="top", color=T.TEXT_MUTED, fontsize=T.px_to_pt(11), alpha=pf)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Trend appears once earlier weeks are in the project folder", ha="center",
                va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(16), transform=ax.transAxes,
                alpha=pf)
    ax.grid(axis="y", color=T.GRID, alpha=pf)
    _fade_ticks(ax, pf)

    rx0 = left + (right - left) * tw + 0.055
    pf, pl = panel["rank"]
    ax = _panel(fig, [rx0, split + row_gap, right - rx0, top - split - row_gap], "WIN RATE BY RANK", pf)
    _rank_panel(ax, hero, data, pl, pf)
    _fade_ticks(ax, pf)

    # --- game length
    gw = 0.54
    p_frame, p_bars = panel["length"]
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
        _fade_ticks(ax, p_frame)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No game-length export this week", ha="center", va="center", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(15), transform=ax.transAxes, alpha=p_frame)

    # --- partners
    px0 = left + (right - left) * gw + 0.04
    p_frame, p_part = panel["partners"]
    ax = _panel(fig, [px0, bottom, right - px0, split - bottom - 0.035], "BEST & WORST PARTNERS", p_frame)
    ax.axis("off")
    ax.set_xlim(0, 1)
    best, worst = partners(data["duos"], hero)
    rows = [(rr, True) for _, rr in best.iterrows()] + [(rr, False) for _, rr in worst.iterrows()]
    n = max(len(rows), 1)
    slots = PARTNERS_BEST + PARTNERS_WORST
    ax.set_ylim(slots - 0.5 + 0.3, -0.5)               # fixed row height, rows from the top
    if not rows:
        ax.text(0.5, 2.5, "No duo with enough games is\nclearly better or worse together",
                ha="center", va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(15), alpha=p_frame)
    if any(r_["conf"] < 0.95 for r_, _ in rows):
        ax.text(0, -0.03, "faded = could be noise (less than 95% sure)", transform=ax.transAxes, ha="left",
                va="top", color=T.TEXT_MUTED, fontsize=T.px_to_pt(12), alpha=p_frame)
    if len(best) and len(worst):                        # divider between the good and the bad
        ax.plot([0, 1], [len(best) - 0.5 + 0.15] * 2, color=T.DIVIDER, lw=1.2, ls=(0, (5, 4)), alpha=p_frame)
    for i, (rr, good) in enumerate(rows):
        t = T.ease_out_cubic(T.phase(p_part, i / (n + 1), i / (n + 1) + 0.4))
        if t <= 0:
            continue
        y = i + (0.3 if i >= len(best) and len(best) else 0)
        noisy = rr["conf"] < 0.95
        fade = 0.55 if noisy else 1.0
        T.add_icon(ax, T.icon_any(rr["partner"], 44, T.archetype_color(
            data["d"].loc[rr["partner"], "hero_type"] if rr["partner"] in data["d"].index else ""), t * fade,
            wr=rr["wr"]), (0.07, y))
        ax.text(0.17, y - 0.13, rr["partner"], ha="left", va="center", color=T.TEXT, alpha=t * fade,
                fontsize=T.px_to_pt(18), fontweight="bold")
        fade = 0.55 if noisy else 1.0
        ax.text(0.17, y + 0.23, f"{int(rr['n']):,} games together", ha="left", va="center",
                color=T.TEXT_MUTED, alpha=t * fade, fontsize=T.px_to_pt(13))
        ax.text(1.0, y - 0.1, f"{rr['wr']:.1f}%", ha="right", va="center", color=T.wr_color(rr["wr"]),
                alpha=t * fade, fontsize=T.px_to_pt(23), fontweight="bold")
        ax.text(1.0, y + 0.27, f"{rr['lift']:+.1f} vs expected", ha="right", va="center",
                color=T.GREEN if rr["lift"] > 0 else T.RED, alpha=t * fade, fontsize=T.px_to_pt(13),
                fontweight="bold")
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
        rn = data.get("rank_net")
        if rn is not None and hero in rn.index and rn.loc[hero, "conf"] >= 0.8:
            v = rn.loc[hero, "net"]
            bits.append(f"{'gets better' if v > 0 else 'gets worse'} at higher ranks ({v:+.1f} pts Initiate to Eternus)")
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
    out = {f"hero_{slug(h)}": (lambda progress=1.0, highlight=None, h=h: render(h, data, meta, progress))
           for h in names if h in data["d"].index}
    for draw in out.values():
        draw.build_s = BUILD_S                 # header, tiles one by one, then each panel in turn
    return out


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