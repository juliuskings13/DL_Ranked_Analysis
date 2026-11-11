"""
hero_spotlight.py — one card per hero, for hero spotlight videos and shorts, in three shapes:

    hero_<name>        square 1440x1440 (talking-head videos)
    hero_<name>_wide   2560x1440 (16:9, no-voiceover long form): the square card on the left, unchanged,
                       and the lane / MVP / per-minute panels on the right
    hero_<name>_tall   1440x2560 (9:16, shorts): the square card's header and tiles, then full-width trend /
                       rank / game length, MVP, per-minute and lanes. No duo or lane-partner lists, and the
                       bottom of the card is left clear for the Shorts title and buttons.

Square card:

    header      hero name; tier, rank and power; big icon
    tiles       win rate, pick rate, ban rate, power, each with the change since last week
    trend       win rate by week (last 8 weeks; every week of a period such as a season), 95% range,
                patches as dashed lines colored by what each did to the hero, NET first-to-last change
    game length win rate in each 5-minute bracket, against 50%
    partners    best and worst duo partners (win rate vs what their solo win rates predict)

The roster is split into four fixed groups (hero_groups.json next to the week folders, created
on first run, mixed classes in each). insights.py makes one spotlight video per group, rotating
weekly, so every hero is covered about once a month.

Wide / tall extra panels (from query_transform's hero_lanes, hero_lane_partners, hero_lane_matchups,
hero_mvp and hero_per_min exports; any that are missing show a note instead):
    per minute  souls, damage, objective damage, healing + barriers per minute, against every hero
    MVP         how often the hero is the match MVP / in the top 3
    lanes       where the hero starts, its win rate there and how often that lane is won
                (lane won = took the enemy's Tier 1 tower in that lane first)
    lane partners / lane matchups   best and worst heroes to share / face a lane with

Outputs: graphs/heroes/hero_<name>[_wide|_tall].png
Usage: python3 hero_spotlight.py [week_dir] [--heroes "Haze,Victor"] [--shapes square,wide,tall]
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
from chart_stats import inv_logit_pct, logit
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
BUILD_S = 16.0               # seconds the card takes to build itself (animate.py / assemble_video use it)
P_HEADER = (0.0, 0.06)       # name, subtitle, icon (fractions of the build)
P_TILES = (0.05, 0.20)       # the four tiles, left to right, one after another
# panels in reading order (fractions of the build); partners get the most time: 6 rows to read
P_PANELS = [("trend", 0.20, 0.38), ("rank", 0.38, 0.55), ("length", 0.55, 0.72), ("partners", 0.72, 0.98)]
PATCH_COLORS = {"buff": T.GREEN, "nerf": T.RED, "mixed": T.GOLD, "rework": T.GOLD}

# --- wide (16:9) and tall (9:16) cards ----------------------------------------------
SHAPES = {"square": (T.CARD_PX, T.CARD_PX), "wide": (1920, 1080), "tall": (1080, 1920)}   # card px
# seconds each extra panel takes to reveal its rows (lane partners / matchups match the square's duos list,
# ~0.7 s a row). The next panel's title starts EXT_OVERLAP_S before the previous one's last row lands, so the
# eased tail of one panel never reads as a pause.
EXT_PANELS = {"per_min": 5.0, "mvp": 2.4, "lanes": 3.0, "partners": 2.8, "matchups": 2.8}
EXT_OVERLAP_S = 0.5
EXT_BUILD_S = BUILD_S + sum(EXT_PANELS.values()) - EXT_OVERLAP_S * (len(EXT_PANELS) - 1) + 0.5
SQUARE_SHARE = BUILD_S / EXT_BUILD_S        # fraction of the build the square part takes
EXT_START = SQUARE_SHARE * P_PANELS[-1][2]  # the extra panels start the moment the square's last row is in
# assigned_lane -> (name, colour). Lane numbers 1/4/6 are the API's; the names are the in-game lane colours.
LANES = {1: ("Yellow", "#e8c547"), 4: ("Blue", "#4f8fe0"), 6: ("Green", "#5cb85c")}
LANE_MIN_GAMES = 200         # a lane needs this many games to be called the hero's best
PAIR_MIN_GAMES = 150         # lane partners / opponents below this aren't listed
PAIRS_BEST, PAIRS_WORST = 2, 2
TIER_ORDER = ["GOD TIER", "STRONG", "OVER", "BALANCED", "UNDER", "WEAK", "TRASH"]   # best -> worst
PERIOD_MAX_PX, PERIOD_MIN_PX = 64, 28       # the period label between the name and the icon
# Avg stats per min. Every stat the hero_per_min export has, by key: (label, columns summed, format).
# Each hero is ranked against the whole roster on every row, whatever its class.
PER_MIN_STATS = {
    "souls": ("Souls / min", ["souls_pm"], "{:,.0f}"),
    "damage": ("Damage / min", ["damage_pm"], "{:,.0f}"),
    "objective": ("Objective damage / min", ["obj_damage_pm"], "{:,.0f}"),
    # damage taken + damage resisted: everything the hero soaked
    "tanked": ("Damage tanked + mitigated / min", ["tanked_pm", "mitigated_pm"], "{:,.0f}"),
    # the hero's own healing + its shields on itself (allies not counted)
    "sustain": ("Self healing + shields / min", ["self_heal_pm", "self_shield_pm"], "{:,.0f}"),
    # supports: all healing + all shields/barriers, on allies and on themselves
    "heal_shield": ("Healing + shields / min", ["heal_pm", "barrier_pm"], "{:,.0f}"),
    "heal_denied": ("Healing denied / min", ["heal_denied_pm"], "{:,.0f}"),
    "kp": ("Kill participation (per match)", ["kill_participation"], "{:.1f}%"),
}
# Every hero shows souls + damage, then its class's stats (up to 5 rows). Hybrids ("carry/bruiser") show their
# main class's first two and their second class's first one that isn't already there.
CLASS_STATS = {
    "tank": ["tanked", "kp"],
    "bruiser": ["sustain", "tanked", "kp"],
    "carry": ["objective", "heal_denied"],
    "assassin": ["kp", "objective"],
    "mage": ["objective", "kp"],
    "support": ["heal_shield", "kp"],
}
DEFAULT_STATS = ["tanked", "objective", "kp"]             # a hero missing from HERO_CLASS
# hero -> class, main class first (from the sub-class sheet, Nov 2026)
HERO_CLASS = {
    "Grey Talon": "carry", "Mo & Krill": "tank", "Shiv": "bruiser", "Ivy": "support/carry",
    "Warden": "bruiser", "Yamato": "bruiser/assassin", "Lash": "assassin", "Viscous": "support/mage",
    "Pocket": "assassin", "Mirage": "carry", "Vyper": "carry", "Sinclair": "mage", "Mina": "assassin",
    "Drifter": "assassin/carry", "Venator": "carry/bruiser", "Victor": "bruiser/carry", "Paige": "support/mage",
    "The Doorman": "mage/assassin", "Billy": "bruiser", "Graves": "carry/mage", "Apollo": "bruiser/mage",
    "Deadman Danny": "assassin/mage", "Infernus": "carry/mage", "Rem": "support/mage",
    "Silver": "bruiser/assassin", "Seven": "carry/mage", "Vindicta": "carry", "Lady Geist": "bruiser/mage",
    "Abrams": "tank/bruiser", "Wraith": "carry", "McGinnis": "mage/bruiser", "Paradox": "assassin",
    "Dynamo": "support/tank", "Kelvin": "support/tank", "Haze": "carry/assassin", "Holliday": "assassin/mage",
    "Bebop": "tank/bruiser", "Calico": "assassin", "Celeste": "mage/bruiser", "Rat King": "tank/carry",
    "Solomon": "mage", "Violet": "mage", "Nurse Harrow": "support", "Baba": "tank/mage",
}


def hero_stat_keys(hero: str) -> list[str]:
    """The five per-minute rows for a hero: souls, damage, then its class's stats."""
    classes = [c.strip() for c in HERO_CLASS.get(hero, "").split("/") if c.strip() in CLASS_STATS]
    keys = ["souls", "damage"]
    if not classes:
        return keys + DEFAULT_STATS
    main = CLASS_STATS[classes[0]]
    keys += main[:2]
    for second in classes[1:2]:
        extra = next((k for k in CLASS_STATS[second] if k not in keys), None)
        if extra:
            keys.append(extra)
    for k in main[2:]:                                      # pure class, or nothing new from the second class
        if len(keys) < 5 and k not in keys:
            keys.append(k)
    return keys[:5]


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
    for key in ("hero_lanes", "hero_lane_partners", "hero_lane_matchups", "hero_mvp", "hero_per_min"):
        try:
            out[key] = pd.read_csv(Path(data_dir) / f"{key}.csv")
        except (OSError, ValueError):
            out[key] = None                      # older period folder: the panel says so
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
    fig.text(rect[0], rect[1] + rect[3] + 13 / getattr(fig, "_card_h", T.CARD_PX), title, color=T.GOLD, fontsize=T.px_to_pt(18),
             fontweight="bold", va="bottom", ha="left", alpha=alpha)
    return ax


def _fade_ticks(ax, alpha: float) -> None:
    """Tick labels and axis labels follow the panel's fade-in (matplotlib doesn't tie them to anything)."""
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_alpha(alpha)
    ax.xaxis.label.set_alpha(alpha)


def _rank_panel(ax, hero: str, data: dict, p_line: float, p_frame: float, rotate: bool = True,
                compact: bool = False) -> None:
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
    if rotate:
        ax.set_xticks(x, [RANK_TIER_NAMES[k] for k in tiers], rotation=40, ha="right", rotation_mode="anchor",
                      fontsize=T.px_to_pt(12))
    else:                                              # wide enough to read them flat
        ax.set_xticks(x, [RANK_TIER_NAMES[k] for k in tiers], fontsize=T.px_to_pt(13))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", color=T.GRID, alpha=p_frame)
    clear = net["conf"] >= 0.95
    col = (T.GREEN if net["net"] > 0 else T.RED) if clear else T.TEXT_MUTED
    ax.text(1.0, 1.02, f"NET {net['net']:+.1f} pts", transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=T.px_to_pt(17), fontweight="bold", alpha=p_frame)
    if not compact:
        ax.text(0.025, 0.97, f"hollow = under {RANK_POINT_MIN:,} games", transform=ax.transAxes, ha="left",
                va="top", color=T.TEXT_MUTED, fontsize=T.px_to_pt(11), alpha=p_frame)


def _place(fig, arr: np.ndarray, x_px: float, y_top_px: float, zorder: float = 4) -> None:
    """An image array (already at real pixels) with its top-left corner at card px (x_px, y_top_px)."""
    fig.figimage(arr, xo=int(round(T.dev_px(x_px))), yo=int(round(T.OUT_PX - T.dev_px(y_top_px) - arr.shape[0])),
                 origin="upper", zorder=zorder)


def _phase(progress: float, a: float, b: float) -> float:
    return T.ease_out_cubic(T.phase(progress, a, b))


def _stagger(p: float, i: int, n: int) -> float:
    """Item i of n inside a section's (linear) progress p: each item eases in over its own slice, the slices
    overlap, and the last one finishes exactly at p = 1, so a section never sits idle before the next starts."""
    d = min(1.0, 1.8 / max(n, 1))
    start = i * (1 - d) / max(n - 1, 1)
    return T.ease_out_cubic(T.phase(p, start, start + d))


def _trend_panel(fig, rect, hero: str, data: dict, pf: float, pl: float, compact: bool = False) -> None:
    """Weekly win rate with its 95% band, patch lines and the NET change."""
    h = data["hist"]
    trend_title = ("WIN RATE BY WEEK" if data.get("is_period") else f"WIN RATE, LAST {TREND_WEEKS} WEEKS")
    ax = _panel(fig, rect, trend_title, pf)
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
        ax.set_ylim(lo - 0.6, hi + (1.6 if compact else 0.9))       # compact: bigger text needs headroom
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        pad = 0.75 if compact else 0.4                  # room for the first / last value labels
        ax.set_xlim(x[0] - pad, x[-1] + pad)
        # NET: first week to last week, colored only when the change is clearly more than noise
        first, last = h.iloc[0], h.iloc[-1]
        net = last["winrate"] - first["winrate"]
        se = np.hypot((first["wr_hi"] - first["wr_lo"]) / 3.92, (last["wr_hi"] - last["wr_lo"]) / 3.92)
        clear = abs(net) > 1.96 * se
        col = (T.GREEN if net > 0 else T.RED) if clear else T.TEXT_MUTED
        ax.text(1.0, 1.02, f"NET {net:+.1f} pts", transform=ax.transAxes, ha="right", va="bottom",
                color=col, fontsize=T.px_to_pt(17), fontweight="bold", alpha=pl)
        if shown and not compact:
            ax.text(0.02, 0.97, "dashed = patch (green buffed, red nerfed, gold mixed)", transform=ax.transAxes,
                    ha="left", va="top", color=T.TEXT_MUTED, fontsize=T.px_to_pt(11), alpha=pf)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Trend appears once earlier weeks are in the project folder", ha="center",
                va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(16), transform=ax.transAxes,
                alpha=pf)
    ax.grid(axis="y", color=T.GRID, alpha=pf)
    _fade_ticks(ax, pf)


def _length_panel(ax, hero: str, data: dict, p_frame: float, p_bars: float, compact: bool = False) -> None:
    """Win rate by game-length bracket, bars growing one after another."""
    dur = data["dur"]
    rows = dur[dur["hero_name"] == hero].set_index("duration_bracket") if dur is not None else pd.DataFrame()
    if len(rows):
        rows = rows.reindex([b for b in DURATION_BRACKETS if b in rows.index])
        x = np.arange(len(rows))
        ok_rows = rows[rows["total_games"] >= MIN_BRACKET_N]
        span = max(4.0, float(np.nanmax(np.abs(ok_rows["winrate"] - 50))) + 1) if len(ok_rows) else 6.0
        for i, (b, rr) in enumerate(rows.iterrows()):
            v = 50 + float(np.clip(rr["winrate"] - 50, -span, span)) * _stagger(p_bars, i, len(rows))
            ok = rr["total_games"] >= MIN_BRACKET_N
            ax.bar(i, v - 50, bottom=50, width=0.72, color=T.wr_color(rr["winrate"]) if ok else T.NEUTRAL,
                   alpha=0.9 if ok else 0.35, hatch=None if ok else "//", edgecolor=T.BG, lw=0, zorder=3)
        ax.axhline(50, color=T.TEXT_MUTED, lw=1.2, alpha=p_frame)
        ax.set_ylim(50 - span, 50 + span)
        names = [b.replace("m", "").replace("-", "–") for b in rows.index]
        if compact:                                    # bigger text: just each bracket's first minute
            names = [n.split("–")[0] for n in names]
        ax.set_xticks(x, names, rotation=0, fontsize=T.px_to_pt(12))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_xlabel("minutes", color=T.TEXT if compact else T.TEXT_MUTED, fontsize=T.px_to_pt(14), labelpad=4)
        ax.grid(axis="y", color=T.GRID, alpha=p_frame)
        if not compact:
            ax.text(1.0, 1.02, "hatched = fewer than 150 games", transform=ax.transAxes, ha="right", va="bottom",
                    color=T.TEXT_MUTED, fontsize=T.px_to_pt(12), alpha=p_frame)
        _fade_ticks(ax, p_frame)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No game-length export this week", ha="center", va="center", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(15), transform=ax.transAxes, alpha=p_frame)


def render(hero: str, data: dict, meta: T.ChartMeta, progress: float = 1.0, highlight=None,
           footer: bool = True, header_only: bool = False) -> "T.Figure":
    r = data["d"].loc[hero]
    p_frame = _phase(progress, *P_HEADER)
    # (frame fade, content progress): content is linear; each bar / row eases in on its own (_stagger)
    panel = {k: (_phase(progress, a, a + 0.03), T.phase(progress, a + 0.01, b)) for k, a, b in P_PANELS}

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
        name_right = x0 - 20 + a.shape[1] / T.RENDER_SCALE
    else:
        title_px = T.fit_font_px([hero.upper()], room, max_px=TITLE_PX, min_px=34)
        t = fig.text(T.MARGIN_X, 0.946, hero.upper(), color=T.GOLD, fontsize=T.px_to_pt(title_px),
                     fontweight="bold", va="center", ha="left", alpha=p_frame)
        T._mark_header_bottom(fig, t)
        name_right = x0 + T.text_px(hero.upper(), title_px)
    # the period ("SEASON 1", "WEEK 12") big, centred between the name and the icon; left out if it won't fit
    period = meta.week_label.upper()
    icon_left = T.CARD_PX * (1 - T.MARGIN_X) - icon_px
    gap = icon_left - name_right - 2 * 30
    size = min(PERIOD_MAX_PX, gap / max(T.display_w_per_px(period), 1e-6))
    if size >= PERIOD_MIN_PX:
        T.outlined_text(fig, (name_right + icon_left) / 2 / T.CARD_PX, 1 - 59 / T.CARD_PX, period, size,
                        T.TEXT, alpha=p_frame, display=True, ha="center", va="center")
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
    alphas = [_phase(progress, a0 + k * step, a0 + k * step + step * 1.6) for k in range(len(tiles))]
    if header_only:
        # shorts: short white labels as big as the tiles allow, no change-vs-before line (too small on a phone)
        short = ["win rate", "pick rate", "ban rate", "power"]
        tiles = [(v, short[k], c) for k, (v, _, c, *_) in enumerate(tiles)]
        T.draw_kpi_row(fig, tiles, height_px=TALL_TILE_PX, alpha=alphas, label_max_px=48, label_color=T.TEXT,
                       label_h_frac=0.30)
    else:
        T.draw_kpi_row(fig, tiles, alpha=alphas)
    fig._header_bottom_px = (1 - fig._content_top) * T.CARD_PX     # bottom of the tiles
    if header_only:                                   # the tall card draws its own (bigger) panels
        return fig
    if footer:
        T.draw_footer(fig, meta, alpha=p_frame)

    top = fig._content_top - 0.06
    bottom = T.FOOTER_LINE_Y + 0.05
    left, right = T.MARGIN_X + 0.038, 1 - T.MARGIN_X
    split = bottom + (top - bottom) * 0.52            # trend above, game length + partners below

    # --- trend (top left) and rank (top right)
    tw = 0.47
    row_gap = 0.085                                   # room for the rotated rank names above the lower panels
    pf, pl = panel["trend"]
    _trend_panel(fig, [left, split + row_gap, (right - left) * tw, top - split - row_gap], hero, data, pf, pl)

    rx0 = left + (right - left) * tw + 0.055
    pf, pl = panel["rank"]
    ax = _panel(fig, [rx0, split + row_gap, right - rx0, top - split - row_gap], "WIN RATE BY RANK", pf)
    _rank_panel(ax, hero, data, pl, pf)
    _fade_ticks(ax, pf)

    # --- game length
    gw = 0.54
    p_frame, p_bars = panel["length"]
    ax = _panel(fig, [left, bottom, (right - left) * gw, split - bottom - 0.035], "WIN RATE BY GAME LENGTH", p_frame)
    _length_panel(ax, hero, data, p_frame, p_bars)

    # --- partners
    px0 = left + (right - left) * gw + 0.04
    p_frame, p_part = panel["partners"]
    ax = _panel(fig, [px0, bottom, right - px0, split - bottom - 0.035], "BEST & WORST DUOS", p_frame)
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
        t = _stagger(p_part, i, n)
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


# --- WIDE / TALL CARDS ---------------------------------------------------------------
# The square card is drawn exactly as above (same sizes) and pasted in as an image; the extra panels
# are drawn around it on the bigger figure. Positions below are card px of the whole wide/tall card.
EXT_LAYOUT = {   # panel -> (x, y_top, w, h)
    "wide": {"per_min": (1100, 22, 800, 336), "mvp": (1100, 384, 800, 150), "lanes": (1100, 560, 800, 196),
             "partners": (1100, 786, 380, 234), "matchups": (1520, 786, 380, 234)},
}
EXT_ORDER = {"wide": ["per_min", "mvp", "lanes", "partners", "matchups"]}    # reveal order (reading order)
EXT_TITLE_PX = 18            # same as the square card's panel titles
_SQUARE_DONE: dict = {}      # (hero, shape) -> finished square image (frames after the square part is built)


class _Canvas:
    """Card-px coordinates (top-left origin) on a W x H card."""

    def __init__(self, fig, w: int, h: int):
        self.fig, self.W, self.H = fig, w, h

    def rect(self, x, y_top, w, h) -> list[float]:
        return [x / self.W, 1 - (y_top + h) / self.H, w / self.W, h / self.H]

    def text(self, x, y, s, **kw):
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)


def _ext_panel(cv: _Canvas, box, title: str, alpha: float, note: str | None = None):
    """A titled panel: gold title on top, an axes (0..1 wide, rows downward) below it."""
    x, y, w, h = box
    cv.text(x, y, title, color=T.GOLD, fontsize=T.px_to_pt(EXT_TITLE_PX), fontweight="bold", va="top",
            ha="left", alpha=alpha)
    if note:
        cv.text(x + w, y + 2, note, color=T.TEXT_MUTED, fontsize=T.px_to_pt(12), va="top", ha="right",
                alpha=alpha)
    gap = 34 * getattr(cv.fig, "_font_scale", 1.0)     # room for the title
    ax = cv.fig.add_axes(cv.rect(x, y + gap, w, h - gap))
    ax._no_fit = True
    ax.axis("off")
    ax.set_xlim(0, 1)
    return ax


def _missing(ax, text: str, alpha: float) -> None:
    ax.set_ylim(1, 0)
    ax.text(0.5, 0.5, text, ha="center", va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(14),
            alpha=alpha, linespacing=1.4)


def _rank_of(values: pd.Series, hero: str) -> tuple[int, int]:
    v = values.dropna()
    return int(v.rank(ascending=False, method="min")[hero]), len(v)


def _band_color(rank: int, total: int) -> str:
    """Tier-list colour for a rank among `total` heroes: seven bands, the outer six equal and the middle one
    taking the rest (44 heroes -> 6,6,6,8,6,6,6)."""
    base = total // 7
    if base == 0:
        return T.TEXT
    sizes = [base] * 3 + [total - 6 * base] + [base] * 3
    edge = np.cumsum(sizes)
    return TIER_COLOR[TIER_ORDER[int(np.searchsorted(edge, rank, side="left"))]]


def _stat_row(key: str, pm: pd.DataFrame) -> tuple:
    """PER_MIN_STATS[key], falling back for exports made before the newer stats columns were ingested."""
    label, cols, fmt = PER_MIN_STATS[key]
    if key == "sustain" and not ("self_shield_pm" in pm.columns and pm["self_shield_pm"].notna().any()):
        return "Self healing / min", ["self_heal_pm"], fmt
    return label, cols, fmt


def _per_min_panel(ax, hero: str, data: dict, p_frame: float, p_rows: float,
                   strip=(0.36, 0.80), compact: bool = False) -> None:
    """compact (shorts): one line per stat (label, value, rank), no roster strip."""
    pm = data.get("hero_per_min")
    if pm is None or hero not in set(pm["hero_name"]):
        _missing(ax, "No per-minute export for this period\n(re-run query_transform)", p_frame)
        return
    pm = pm.set_index("hero_name")
    rows = [_stat_row(k, pm) for k in hero_stat_keys(hero)]
    n = len(rows)
    ax.set_ylim(n - 0.5, -0.5)
    for i, (label, cols, fmt) in enumerate(rows):
        t = _stagger(p_rows, i, n)
        ax.text(0, i - 0.3 if compact else i - 0.2, label, ha="left", va="center", color=T.TEXT,
                fontsize=T.px_to_pt(18 if compact else 14), alpha=p_frame)    # compact: same size as lane names
        if not all(c in pm.columns for c in cols) or pm[cols].isna().all(axis=0).any():
            why = ("re-run query_transform" if not all(c in pm.columns for c in cols)
                   else "needs ingest --with-objectives" if "obj_damage_pm" in cols else "needs ingest --with-stats")
            if compact:
                ax.text(1.0, i, why, ha="right", va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(12),
                        alpha=p_frame)
                continue
            ax.text(0, i + 0.18, "—", ha="left", va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(22),
                    fontweight="bold", alpha=p_frame)
            ax.text(0.40, i, why, ha="left", va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(13),
                    alpha=p_frame)
            continue
        vals = pm[cols].sum(axis=1, min_count=len(cols))
        if pd.isna(vals.get(hero)):
            continue
        v = float(vals[hero])
        rank, total = _rank_of(vals, hero)
        lo, hi = float(vals.min()), float(vals.max())
        frac = (v - lo) / (hi - lo) if hi > lo else 0.5
        col = _band_color(rank, total)
        if compact:
            # a strip under the label: every hero as a tier-coloured tick, the median, and this hero's dot
            x0, x1, ys = 0.0, COMPACT_STRIP_W, i + 0.26
            ranks = vals.dropna().rank(ascending=False, method="min").astype(int)
            xs = x0 + (vals[ranks.index].to_numpy(float) - lo) / max(hi - lo, 1e-9) * (x1 - x0)
            med = x0 + (float(vals.median()) - lo) / max(hi - lo, 1e-9) * (x1 - x0)
            ax.plot([x0, x1], [ys] * 2, color=T.DIVIDER, lw=2, alpha=p_frame, solid_capstyle="round")
            ax.scatter(xs, [ys] * len(xs), marker="|", s=230, lw=1.5, alpha=0.6 * p_frame,
                       color=[_band_color(k, total) for k in ranks])
            ax.plot([med, med], [ys - 0.2, ys + 0.2], color=T.TEXT_MUTED, lw=1.8, alpha=0.85 * p_frame)
            if t > 0:
                ax.scatter([x0 + frac * t * (x1 - x0)], [ys], s=150, color=col, edgecolor=T.BG, lw=1.8,
                           zorder=5, alpha=t)
            ax.text(COMPACT_VALUE_X, i, fmt.format(v * t), ha="right", va="center", color=col,
                    fontsize=T.px_to_pt(22), fontweight="bold", alpha=t)
            ax.text(1.0, i, f"#{rank} of {total}", ha="right", va="center", color=col,
                    fontsize=T.px_to_pt(17), fontweight="bold", alpha=t)
            continue
        ranks = vals.dropna().rank(ascending=False, method="min").astype(int)
        ax.text(0, i + 0.18, fmt.format(v * t), ha="left", va="center", color=col, fontsize=T.px_to_pt(24),
                fontweight="bold", alpha=t)
        # every hero's value as a tick, this hero as a dot: where it sits in the whole roster
        x0, x1 = strip
        xs = x0 + (vals[ranks.index].to_numpy(float) - lo) / max(hi - lo, 1e-9) * (x1 - x0)
        ax.plot([x0, x1], [i + 0.08] * 2, color=T.DIVIDER, lw=2, alpha=p_frame, solid_capstyle="round")
        ax.scatter(xs, [i + 0.08] * len(xs), marker="|", s=110, lw=1.4, alpha=0.6 * p_frame,
                   color=[_band_color(k, total) for k in ranks])      # ticks in tier-list colours
        med = x0 + (float(vals.median()) - lo) / max(hi - lo, 1e-9) * (x1 - x0)
        ax.plot([med, med], [i - 0.12, i + 0.28], color=T.TEXT_MUTED, lw=1.6, alpha=0.8 * p_frame)
        if t > 0:
            ax.scatter([x0 + frac * t * (x1 - x0)], [i + 0.08], s=150, color=col, edgecolor=T.BG, lw=2, zorder=5,
                       alpha=t)
        ax.text(1.0, i + 0.08, f"#{rank} of {total}", ha="right", va="center", color=col,
                fontsize=T.px_to_pt(17), fontweight="bold", alpha=t)
    if compact:
        return
        ax.text(1.0, -0.75, "each tick = one hero, coloured by tier-list band · line = median", ha="right",
                va="bottom",
                color=T.TEXT_MUTED, fontsize=T.px_to_pt(10), alpha=p_frame)
        return
    ax.text(sum(strip) / 2, -0.5, "each tick = one hero, coloured by tier-list band · line = median", ha="center", va="bottom", color=T.TEXT_MUTED,
            fontsize=T.px_to_pt(11), alpha=p_frame)


def _mvp_panel(ax, hero: str, data: dict, p_frame: float, p_in: float, vertical: bool = False) -> None:
    from matplotlib.patches import FancyBboxPatch
    m = data.get("hero_mvp")
    if m is None or hero not in set(m["hero_name"]):
        _missing(ax, "No MVP export for this period", p_frame)
        return
    m = m.set_index("hero_name")
    ax.set_ylim(1, 0)
    # compared with the average hero (plain mean of every hero's rate), not with an average player
    tiles = [("mvp_rate", "MVP of the match"), ("top3_rate", "top 3 of the match"),
             ("top3_loss_rate", "top 3 in losses")]
    tiles = [(c, lab) for c, lab in tiles if c in m.columns and m[c].notna().any()]   # older exports: no losses col
    if vertical:                                       # tiles stacked: value left, label + rank right
        gap = 0.06
        hgt = (1 - gap * (len(tiles) - 1)) / len(tiles)
        for k, (col, label) in enumerate(tiles):
            t = _stagger(p_in, k, len(tiles))
            if t <= 0:
                continue
            y0 = k * (hgt + gap)
            ax.add_patch(FancyBboxPatch((0, y0), 1, hgt, boxstyle="round,pad=0,rounding_size=0.02",
                                        transform=ax.transAxes, facecolor=T.PANEL, edgecolor="none", alpha=t))
            avg = float(m[col].mean())
            v = float(m.loc[hero, col])
            rank, total = _rank_of(m[col], hero)
            c = T.GREEN if v > avg * 1.1 else T.RED if v < avg * 0.9 else T.TEXT
            ym = y0 + hgt / 2
            # value hugs the left edge, the words the right edge, so long values never run into them
            ax.text(0.025, ym, f"{v:.1f}%", ha="left", va="center", color=c, fontsize=T.px_to_pt(28),
                    fontweight="bold", alpha=t)
            ax.text(0.975, ym - hgt * 0.2, label, ha="right", va="center", color=T.TEXT,
                    fontsize=T.px_to_pt(15), alpha=t)
            ax.text(0.975, ym + hgt * 0.22, f"#{rank} of {total} · hero avg {avg:.1f}%", ha="right",
                    va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(12), alpha=t)
        return
    gap = 0.025
    w = (1 - gap * (len(tiles) - 1)) / len(tiles)
    for k, (col, label) in enumerate(tiles):
        t = _stagger(p_in, k, len(tiles))
        if t <= 0:
            continue
        x = k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.02), w, 0.96, boxstyle="round,pad=0,rounding_size=0.02",
                                    transform=ax.transAxes, facecolor=T.PANEL, edgecolor="none", alpha=t))
        avg = float(m[col].mean())
        v = float(m.loc[hero, col])
        rank, total = _rank_of(m[col], hero)
        c = T.GREEN if v > avg * 1.1 else T.RED if v < avg * 0.9 else T.TEXT
        ax.text(x + 0.03, 0.27, f"{v:.1f}%", ha="left", va="center", color=c, fontsize=T.px_to_pt(30),
                fontweight="bold", alpha=t)
        ax.text(x + 0.03, 0.60, label, ha="left", va="center", color=T.TEXT, fontsize=T.px_to_pt(14),
                alpha=t)
        ax.text(x + 0.03, 0.83, f"#{rank} of {total} · hero avg {avg:.1f}%", ha="left", va="center",
                color=T.TEXT_MUTED, fontsize=T.px_to_pt(12), alpha=t)


def _lanes_panel(ax, hero: str, data: dict, p_frame: float, p_rows: float, cols=(0.58, 0.80, 1.0)):
    """Returns the overall 'lanes won' text for the panel title, or None."""
    ln = data.get("hero_lanes")
    rows = ln[(ln["hero_name"] == hero) & ln["lane"].isin(list(LANES))].set_index("lane") if ln is not None else None
    if rows is None or rows.empty:
        _missing(ax, "No lane export for this period", p_frame)
        return None
    lanes = [k for k in LANES if k in rows.index]
    n = len(lanes)
    ax.set_ylim(n - 0.5, -0.8)                      # column headers right under the title, rows right after
    has_lane_wr = rows["lane_games"].fillna(0).sum() > 0
    most = rows["lane_share"].idxmax()
    ok = rows[rows["games"] >= LANE_MIN_GAMES]
    best = ok["winrate"].idxmax() if len(ok) > 1 else None
    c_play, c_wr, c_lane = cols
    for x, lab in ((c_play, "played"), (c_wr, "win rate"), (c_lane, "lane won")):
        ax.text(x, -0.62, lab, ha="right", va="center", color=T.TEXT_MUTED, fontsize=T.px_to_pt(12), alpha=p_frame)
    for i, lane in enumerate(lanes):
        rr = rows.loc[lane]
        name, color = LANES[lane]
        t = _stagger(p_rows, i, n)
        if t <= 0:
            continue
        ax.scatter([0.025], [i], s=170, color=color, edgecolor=T.BG, lw=1.5, alpha=t, zorder=4)
        tags = (["most played"] if lane == most else []) + (["best"] if lane == best else [])
        ax.text(0.065, i - (0.14 if tags else 0), name, ha="left", va="center", color=T.TEXT,
                fontsize=T.px_to_pt(18), fontweight="bold", alpha=t)
        if tags:                                       # under the lane name, so narrow panels have room
            ax.text(0.068, i + 0.3, " · ".join(tags).upper(), ha="left", va="center",
                    color=T.BG, fontsize=T.px_to_pt(10), fontweight="bold", alpha=t,
                    bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.4", fc=T.GOLD, ec="none", alpha=t))
        share = float(rr["lane_share"])
        ax.text(c_play, i, f"{share * t:.0f}%", ha="right", va="center", color=T.TEXT, fontsize=T.px_to_pt(18),
                fontweight="bold", alpha=t)
        ax.text(c_wr, i, f"{rr['winrate']:.1f}%", ha="right", va="center", color=T.wr_color(rr["winrate"]),
                fontsize=T.px_to_pt(18), fontweight="bold", alpha=t)
        lw = rr["lane_winrate"] if has_lane_wr and pd.notna(rr["lane_winrate"]) else None
        ax.text(c_lane, i, f"{lw:.1f}%" if lw is not None else "—", ha="right", va="center",
                color=T.wr_color(lw) if lw is not None else T.TEXT_MUTED, fontsize=T.px_to_pt(18),
                fontweight="bold", alpha=t)
    if not has_lane_wr:
        return "lane won: needs ingest --objectives-only"
    won = rows["lane_wins"].sum() / max(rows["lane_games"].sum(), 1) * 100
    return f"lanes won {won:.1f}%"


def _lane_wr(lanes: pd.DataFrame | None) -> pd.Series:
    """Each hero's own lane win rate (% of decided lanes it won), all lanes together."""
    if lanes is None or "lane_games" not in lanes.columns:
        return pd.Series(dtype=float)
    g = lanes.groupby("hero_name")[["lane_wins", "lane_games"]].sum()
    return (100 * g["lane_wins"] / g["lane_games"].where(g["lane_games"] > 0)).dropna()


def _pairs(df: pd.DataFrame | None, hero: str, other: str, lane_wr: pd.Series, opponents: bool):
    """Best / worst lane partners (or opponents), by LANE win rate (took the enemy Tier 1 first), against the
    lane win rate expected from the two heroes' own lane win rates (log-odds added, like the duos panel;
    an opponent's edge counts against). Same rules as the duos: best = wins the lane (> 50%) and beats
    expected, worst = loses it and falls short; must be PARTNER_CONF sure, faded below 95%."""
    from math import erf
    if df is None or hero not in lane_wr.index or "lane_winrate" not in df.columns:
        return None, None
    d = df[(df["hero_name"] == hero) & (df["lane_games"] >= PAIR_MIN_GAMES) & df[other].isin(lane_wr.index)].copy()
    if d.empty:
        return d, d
    sign = -1 if opponents else 1
    d["expected"] = inv_logit_pct(logit(lane_wr[hero]) + sign * logit(lane_wr[d[other]].to_numpy()))
    d["lift"] = d["lane_winrate"] - d["expected"]
    p = d["expected"] / 100
    d["z"] = d["lift"] / (100 * np.sqrt(p * (1 - p) / d["lane_games"]))
    conf_up = d["z"].map(lambda z: 0.5 * (1 + erf(z / np.sqrt(2))))
    best = d[(d["lane_winrate"] > 50) & (d["lift"] > 0) & (conf_up >= PARTNER_CONF)].nlargest(PAIRS_BEST,
                                                                                                "lane_winrate")
    losing = (d["lane_winrate"] < 50) if WORST_NEEDS_SUB50 else True
    worst = d[losing & (d["lift"] < 0) & (1 - conf_up >= PARTNER_CONF)].nsmallest(PAIRS_WORST, "lane_winrate")
    return best.rename(columns={other: "other"}), worst.iloc[::-1].rename(columns={other: "other"})


def _pairs_panel(ax, data: dict, best, worst, p_frame: float, p_rows: float, what: str) -> None:
    if best is None:
        _missing(ax, f"No {what} lane-win data for this period\n(ingest --objectives-only, then re-run the transform)",
                 p_frame)
        return
    rows = [(rr, True) for _, rr in best.iterrows()] + [(rr, False) for _, rr in worst.iterrows()]
    slots = PAIRS_BEST + PAIRS_WORST
    ax.set_ylim(slots - 0.5 + 0.3, -0.5)
    if not rows:
        ax.text(0.5, 1.5, f"No {what} with {PAIR_MIN_GAMES}+ games", ha="center", va="center",
                color=T.TEXT_MUTED, fontsize=T.px_to_pt(14), alpha=p_frame)
        return
    if len(best) and len(worst):
        ax.plot([0, 1], [len(best) - 0.5 + 0.15] * 2, color=T.DIVIDER, lw=1.2, ls=(0, (5, 4)), alpha=p_frame)
    n = len(rows)
    for i, (rr, good) in enumerate(rows):
        t = _stagger(p_rows, i, n)
        if t <= 0:
            continue
        y = i + (0.3 if i >= len(best) and len(best) else 0)
        fade = 0.55 if abs(rr["z"]) < 1.96 else 1.0
        name = rr["other"]
        htype = data["d"].loc[name, "hero_type"] if name in data["d"].index else ""
        lw = rr["lane_winrate"]
        T.add_icon(ax, T.icon_any(name, 40, T.archetype_color(htype), t * fade, wr=lw), (0.075, y))
        ax.text(0.17, y - 0.15, name, ha="left", va="center", color=T.TEXT, alpha=t * fade,
                fontsize=T.px_to_pt(17), fontweight="bold")
        ax.text(0.17, y + 0.22, f"{int(rr['lane_games']):,} lanes · match won {rr['winrate']:.0f}%", ha="left",
                va="center", color=T.TEXT_MUTED, alpha=t * fade, fontsize=T.px_to_pt(12))
        ax.text(1.0, y - 0.12, f"{lw:.1f}%", ha="right", va="center", color=T.wr_color(lw),
                alpha=t * fade, fontsize=T.px_to_pt(21), fontweight="bold")
        ax.text(1.0, y + 0.24, f"{rr['lift']:+.1f} vs expected", ha="right", va="center",
                color=T.GREEN if rr["lift"] > 0 else T.RED, alpha=t * fade, fontsize=T.px_to_pt(12),
                fontweight="bold")


def _ext_footer(cv: _Canvas, meta: T.ChartMeta, alpha: float) -> None:
    """The square card's footer, stretched across the whole wide / tall card."""
    y_line = cv.H - T.FOOTER_LINE_Y * T.CARD_PX
    y_text = cv.H - T.FOOTER_TEXT_Y * T.CARD_PX
    mx = T.MARGIN_X * T.CARD_PX
    cv.fig.add_artist(T.plt.Line2D([mx / cv.W, 1 - mx / cv.W], [1 - y_line / cv.H] * 2, color=T.DIVIDER,
                                   linewidth=1, transform=cv.fig.transFigure, alpha=alpha))
    fs = T.px_to_pt(T.SIZE_FOOTER)
    cv.text(mx, y_text, meta.left_text(), color=T.FOOTER, fontsize=fs, va="center", ha="left", alpha=alpha)
    cv.text(cv.W - mx, y_text, f"Analysis by {meta.handle}", color=T.FOOTER, fontsize=fs, va="center",
            ha="right", alpha=alpha)


def render_shaped(hero: str, data: dict, meta: T.ChartMeta, shape: str, progress: float = 1.0,
                  highlight=None) -> "T.Figure":
    """The wide (16:9) or tall (9:16) card: the square card, untouched in size, plus the extra panels."""
    W, H = SHAPES[shape]
    p_sq = min(1.0, progress / SQUARE_SHARE) if progress < EXT_START else 1.0
    key = (hero, shape)
    if p_sq >= 1.0 and key in _SQUARE_DONE:
        sq = _SQUARE_DONE[key]
    else:
        sq = T.fig_to_array(render(hero, data, meta, p_sq, footer=False))
        if p_sq >= 1.0:
            _SQUARE_DONE.clear()                 # keep only the current hero's (memory)
            _SQUARE_DONE[key] = sq
    fig = T.plt.figure(figsize=(W / T.DESIGN_DPI, H / T.DESIGN_DPI), dpi=T.DPI, facecolor=T.BG)
    fig.figimage(sq, xo=0, yo=int(round(T.dev_px(H))) - sq.shape[0], origin="upper", zorder=0)
    cv = _Canvas(fig, W, H)
    _ext_footer(cv, meta, _phase(progress, 0, P_HEADER[1] * SQUARE_SHARE))

    # extra panels: one after another once the square part is done
    # extra panels back to back, each as long as its rows need; title and first row come in together
    now = progress * EXT_BUILD_S - EXT_START * EXT_BUILD_S          # seconds into the extra panels
    ph, a = {}, 0.0
    for k in EXT_ORDER[shape]:
        dur = EXT_PANELS[k]
        ph[k] = (_phase(now, a, a + 0.4), T.phase(now, a, a + dur))   # 0.4 s title fade, rows over `dur`
        a += dur - EXT_OVERLAP_S
    box = EXT_LAYOUT[shape]
    lane_wr = _lane_wr(data.get("hero_lanes"))

    pf, pr = ph["per_min"]
    if pf > 0:
        _per_min_panel(_ext_panel(cv, box["per_min"], "HERO AVG STATS", pf), hero, data, pf, pr,
                       strip=(0.36, 0.80) if shape == "wide" else (0.42, 0.74))
    pf, pr = ph["mvp"]
    if pf > 0:
        _mvp_panel(_ext_panel(cv, box["mvp"], "MVP RATE", pf), hero, data, pf, pr)
    pf, pr = ph["lanes"]
    if pf > 0:
        ax = _ext_panel(cv, box["lanes"], "LANES", pf)
        note = _lanes_panel(ax, hero, data, pf, pr)
        if note:
            x, y, w, _ = box["lanes"]
            cv.text(x + w, y, note, color=T.TEXT if "won" in note[:9] else T.TEXT_MUTED,
                    fontsize=T.px_to_pt(16 if note.startswith("lanes won") else 12), fontweight="bold",
                    va="top", ha="right", alpha=pr)
    for k, src, other, title, what in (("partners", "hero_lane_partners", "partner_name", "BEST & WORST LANE PARTNERS",
                                        "lane partner"),
                                       ("matchups", "hero_lane_matchups", "opponent_name", "BEST & WORST LANE MATCHUPS",
                                        "lane opponent")):
        pf, pr = ph[k]
        if pf <= 0:
            continue
        best, worst = _pairs(data.get(src), hero, other, lane_wr, opponents=(k == "matchups"))
        ax = _ext_panel(cv, box[k], title, pf)
        _pairs_panel(ax, data, best, worst, pf, pr, what)
        if best is not None:
            noisy = any(abs(r_["z"]) < 1.96 for df in (best, worst) for _, r_ in df.iterrows())
            x, y, w, h = box[k]
            cv.text(x, y + h + 4, "% = lanes won (took the enemy Tier 1 first)" + (" · faded = could be noise"
                    if noisy else ""), color=T.TEXT_MUTED, fontsize=T.px_to_pt(11), va="top", ha="left", alpha=pf)
    return fig


# --- TALL (9:16 SHORTS) CARD ----------------------------------------------------------
# The square card's header and tiles, then bigger full-width charts. No duo / lane partner lists: on a phone
# the bottom ~15% sits under the short's title and the right edge under the like / comment buttons, so
# nothing important goes there (TALL_SAFE_*). Boxes are (x, y_top, w, h) in card px of the 1080x1920 card;
# for the matplotlib panels (trend, rank, length) the box is the plot area and the title sits above it.
TALL_FONT_SCALE = 1.5            # every font, line and marker on the shorts card, vs the square card
TALL_TILE_PX = 150                      # taller header tiles on the shorts card (square: 118)
TALL_LAYOUT = {
    "trend": (75, 375, 985, 165),
    "rank": (75, 630, 985, 100),
    "length": (75, 870, 500, 120),
    "mvp": (625, 825, 435, 230),        # title + stacked tiles
    "per_min": (40, 1075, 900, 450),
    "lanes": (40, 1545, 640, 210),
}
TALL_LANE_COLS = (0.56, 0.78, 1.0)
COMPACT_STRIP_W = 0.55                   # shorts stat strips: share of the panel width
COMPACT_VALUE_X = 0.80                   # right edge of the value; the "#N of M" sits right after it       # played / win rate / lane won, close to the lane names
TALL_SAFE_BOTTOM, TALL_SAFE_RIGHT = 170, 140     # px kept clear for the Shorts UI (documentation; layout above)
# (start s, end s) of each section; the header and tiles keep the square card's timing (first 3.2 s)
TALL_TIMES = {"trend": (3.2, 6.1), "rank": (6.1, 8.8), "length": (8.8, 11.0), "mvp": (10.6, 13.0),
              "per_min": (12.5, 17.5), "lanes": (17.0, 20.0)}
TALL_BUILD_S = 20.5


def _white_xticks(ax, px: float) -> None:
    for lab in ax.get_xticklabels():
        lab.set_color(T.TEXT)
        lab.set_fontsize(T.px_to_pt(px))


def render_tall(hero: str, data: dict, meta: T.ChartMeta, progress: float = 1.0, highlight=None) -> "T.Figure":
    W, H = SHAPES["tall"]
    now = progress * TALL_BUILD_S
    key = (hero, "tall_header")
    header_done = now >= P_TILES[1] * BUILD_S + 1.0
    if header_done and key in _SQUARE_DONE:
        head = _SQUARE_DONE[key]
    else:
        f = render(hero, data, meta, min(1.0, now / BUILD_S), header_only=True)
        arr = T.fig_to_array(f)
        head = arr[: int(round(T.dev_px(f._header_bottom_px + 12)))]
        if header_done:
            _SQUARE_DONE.clear()                 # keep only the current hero's (memory)
            _SQUARE_DONE[key] = head
    # same pixel size, higher dpi: every font / line / marker comes out TALL_FONT_SCALE bigger
    k = TALL_FONT_SCALE
    fig = T.plt.figure(figsize=(W / T.DESIGN_DPI / k, H / T.DESIGN_DPI / k), dpi=T.DPI * k, facecolor=T.BG)
    fig._card_h, fig._font_scale = H, k
    fig.figimage(head, xo=0, yo=int(round(T.dev_px(H))) - head.shape[0], origin="upper", zorder=0)
    cv = _Canvas(fig, W, H)
    _ext_footer(cv, meta, _phase(progress, 0, P_HEADER[1] * BUILD_S / TALL_BUILD_S))

    ph = {k: (_phase(now, a, a + 0.48), T.phase(now, a + 0.16, b)) for k, (a, b) in TALL_TIMES.items()}
    box = TALL_LAYOUT

    pf, pl = ph["trend"]
    if pf > 0:
        _trend_panel(fig, cv.rect(*box["trend"]), hero, data, pf, pl, compact=True)
        _white_xticks(fig.axes[-1], 17)
    pf, pl = ph["rank"]
    if pf > 0:
        ax = _panel(fig, cv.rect(*box["rank"]), "WIN RATE BY RANK", pf)
        _rank_panel(ax, hero, data, pl, pf, rotate=True, compact=True)
        _white_xticks(ax, 14)
        _fade_ticks(ax, pf)
    pf, pl = ph["length"]
    if pf > 0:
        ax = _panel(fig, cv.rect(*box["length"]), "WIN RATE BY GAME LENGTH", pf)
        _length_panel(ax, hero, data, pf, pl, compact=True)
        _white_xticks(ax, 15)
    pf, pl = ph["mvp"]
    if pf > 0:
        _mvp_panel(_ext_panel(cv, box["mvp"], "MVP RATE", pf), hero, data, pf, pl, vertical=True)
    pf, pl = ph["per_min"]
    if pf > 0:
        _per_min_panel(_ext_panel(cv, box["per_min"], "HERO AVG STATS", pf), hero, data, pf, pl,
                       compact=True)
    pf, pl = ph["lanes"]
    if pf > 0:
        ax = _ext_panel(cv, box["lanes"], "LANES", pf)
        note = _lanes_panel(ax, hero, data, pf, pl, cols=TALL_LANE_COLS)
        if note:
            x, y, w, _ = box["lanes"]
            cv.text(x + w, y, note, color=T.TEXT if note.startswith("lanes won") else T.TEXT_MUTED,
                    fontsize=T.px_to_pt(16 if note.startswith("lanes won") else 12), fontweight="bold",
                    va="top", ha="right", alpha=pl)
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


def cards(data_dir: Path, heroes: list[str] | None = None, shapes=("square", "wide", "tall"), **_) -> dict:
    """{output name: draw(progress=1.0, highlight=None) -> Figure} for every card this script makes:
    hero_<name> (square), hero_<name>_wide (16:9) and hero_<name>_tall (9:16)."""
    data = load_all(data_dir)
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), None)
    names = [h for h in (heroes or list(data["d"].index)) if h in data["d"].index]
    out = {}
    for h in names:
        if "square" in shapes:
            draw = lambda progress=1.0, highlight=None, h=h: render(h, data, meta, progress)
            draw.build_s = BUILD_S             # header, tiles one by one, then each panel in turn
            out[f"hero_{slug(h)}"] = draw
        if "wide" in shapes:
            draw = lambda progress=1.0, highlight=None, h=h: render_shaped(h, data, meta, "wide", progress)
            draw.build_s = EXT_BUILD_S         # the square part at its usual pace, then the extra panels
            out[f"hero_{slug(h)}_wide"] = draw
        if "tall" in shapes:
            draw = lambda progress=1.0, highlight=None, h=h: render_tall(h, data, meta, progress)
            draw.build_s = TALL_BUILD_S
            out[f"hero_{slug(h)}_tall"] = draw
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--heroes", default="", help='only these heroes, e.g. "Haze,Victor"')
    ap.add_argument("--progress", type=float, default=1.0)
    ap.add_argument("--shapes", default="square", help="any of square,wide,tall (comma separated)")
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    names = [h.strip() for h in args.heroes.split(",") if h.strip()] or None
    shapes = tuple(s.strip() for s in args.shapes.split(",") if s.strip())
    bad = [s for s in shapes if s not in SHAPES]
    if bad:
        raise SystemExit(f"unknown shape(s): {', '.join(bad)} (square, wide, tall)")
    saved = T.save_cards(cards(data_dir, heroes=names, shapes=shapes), data_dir / "graphs" / "heroes", args.progress)
    hero_groups(data_dir, load_hero_meta(data_dir))
    print(f"Saved {len(saved)} hero cards to {data_dir / 'graphs' / 'heroes'}")


if __name__ == "__main__":
    main()