"""
rank_performance.py — how heroes do at different ranks (1080x1080 cards).

Rank = the match's average badge (tier 1 Initiate ... 11 Eternus), grouped into four brackets of
roughly equal size (chart_stats.RANK_BRACKETS).

Outputs
    rank_best_heroes.png          the 6 best heroes in each bracket
    rank_worst_heroes.png         the 6 worst heroes in each bracket
    rank_curve_<class>.png        every hero of a class across the 11 ranks (actual win rate per rank),
                                  with each hero's net change from low to high rank

Reads hero_rank_winrates.csv and rank_tier_matches.csv (query_transform.py).
Usage: python3 rank_performance.py [week_dir] [--highlight "Celeste,Victor"]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import theme as T
from chart_stats import (HIGH_BRACKETS, LOW_BRACKETS, RANK_BRACKETS, RANK_TIER_NAMES, Insight, load_hero_meta,
                         load_rank_winrates, low_high_table, wr_confidence, normal_cdf)

MIN_GAMES = 1000          # a hero needs this many games in a bracket to be ranked there
RANK_POINT_MIN = 1000     # rank-curve points with fewer games are drawn hollow
TOP_N = 6
DIM_ALPHA = 0.25


def _bracket_label(idx) -> str:
    names = [RANK_BRACKETS[i][0] for i in idx]
    return f"{names[0].split('–')[0]}–{names[-1].split('–')[-1]}"


LOW_LABEL, HIGH_LABEL = _bracket_label(LOW_BRACKETS), _bracket_label(HIGH_BRACKETS)


# --- CARD 1: BEST HEROES PER BRACKET ------------------------------------------------
def render_best(g: pd.DataFrame, meta: T.ChartMeta, progress: float = 1.0,
                highlight: set[str] | None = None, worst: bool = False) -> "T.Figure":
    """The TOP_N best (or worst) heroes in each rank bracket."""
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    pick = (lambda d: d.nsmallest(TOP_N, "winrate")) if worst else (lambda d: d.nlargest(TOP_N, "winrate"))
    tops = {b: pick(g[(g["bracket"] == b) & (g["games"] >= MIN_GAMES)]) for b in range(len(RANK_BRACKETS))}
    everywhere = set.intersection(*(set(t["hero_name"]) for t in tops.values()))
    first_low, first_high = tops[0].iloc[0]["hero_name"], tops[len(RANK_BRACKETS) - 1].iloc[0]["hero_name"]
    names = sorted(everywhere)
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] if names else ""
    where = f"{'bottom' if worst else 'top'} {TOP_N} at every rank"
    first = "last place" if worst else "#1"
    sub = (f"{listed} {'is' if len(names) == 1 else 'are'} {where}" if everywhere else
           f"{first.capitalize()} goes from {first_low} at the bottom to {first_high} at the top")
    fig, ax = T.new_table_card()
    T.draw_header(fig, f"{'WORST' if worst else 'BEST'} HEROES AT EVERY RANK", sub, alpha=p_frame)
    T.draw_archetype_legend(fig, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)

    nb = len(RANK_BRACKETS)
    row = T.table_rows(fig, ax, TOP_N, header_rows=1.25, footer_rows=0.4)
    W, _ = T.content_px(fig)
    ax.set_xlim(0, nb)
    col_px = W / nb
    icon_px = min(row * 0.78, col_px * 0.3, 72)
    text_w = col_px - icon_px - 22
    name_font = T.fit_font_px(g["hero_name"].unique(), text_w, row * 0.3, max_px=22, min_px=11)
    matches = g.groupby("bracket")["games"].sum() / 12
    for b, (name, _) in enumerate(RANK_BRACKETS):
        if b % 2 == 0:
            ax.add_patch(Rectangle((b, -1.75), 1, TOP_N + 1.55, color=T.PANEL, alpha=0.5 * p_frame, lw=0, zorder=0))
        ax.text(b + 0.5, -1.2, name.upper(), ha="center", va="center", color=T.GOLD,
                fontsize=T.px_to_pt(T.fit_font_px([name.upper()], col_px * 0.92, max_px=19, min_px=11)),
                fontweight="bold", alpha=p_frame)
        ax.text(b + 0.5, -0.78, f"{matches.get(b, 0) / 1000:.0f}k matches", ha="center", va="center",
                color=T.TEXT_MUTED, fontsize=T.px_to_pt(14), alpha=p_frame)
        x_icon = b + (8 + icon_px / 2) / col_px
        x_text = b + (14 + icon_px) / col_px
        for k, (_, r) in enumerate(tops[b].iterrows()):
            s = 0.15 + 0.6 * (k * nb + b) / (TOP_N * nb)
            t = T.ease_out_cubic(T.phase(progress, s, s + 0.2))
            if t <= 0:
                continue
            hl = highlight is None or r["hero_name"].lower() in highlight
            a = t * (1.0 if hl else DIM_ALPHA)
            T.add_icon(ax, T.icon_any(r["hero_name"], icon_px, T.archetype_color(r["hero_type"]), a,
                                      wr=r["winrate"]), (x_icon, k), zorder=6)
            ax.text(x_text, k - 0.15, r["hero_name"], ha="left", va="center", color=T.TEXT, alpha=a,
                    fontsize=T.px_to_pt(name_font), fontweight="bold")
            ax.text(x_text, k + 0.2, f"{r['winrate']:.1f}%", ha="left", va="center",
                    color=T.wr_color(r["winrate"], r["wr_lo"], r["wr_hi"]), alpha=a,
                    fontsize=T.px_to_pt(name_font * 0.95), fontweight="bold")
    ax.text(nb / 2, TOP_N - 0.15, f"rank = the match's average badge  ·  heroes need {MIN_GAMES:,}+ games in a bracket",
            ha="center", va="top", color=T.TEXT_MUTED, fontsize=T.px_to_pt(14), alpha=p_frame)
    return fig


# --- CARD 2: RANK CURVES PER CLASS -------------------------------------------------
def _spread(ys: list[float], min_gap: float, lo: float, hi: float) -> list[float]:
    """Nudge sorted label positions apart so none are closer than min_gap, staying within [lo, hi]."""
    order = np.argsort(ys)
    pos = np.array(ys, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    if len(pos) and pos[-1] > hi:                      # pushed off the top: shift everything down
        pos -= pos[-1] - hi
    for i in range(len(pos) - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - min_gap)
    if len(pos) and pos[0] < lo:
        pos += lo - pos[0]
    out = np.empty_like(pos)
    out[order] = pos
    return list(out)


def load_tier_winrates(data_dir: Path) -> pd.DataFrame:
    """hero_rank_winrates.csv per individual rank (Initiate 1 ... Eternus 11), unranked rows dropped."""
    h = pd.read_csv(Path(data_dir) / "hero_rank_winrates.csv")
    return h[h["rank_tier"].isin(RANK_TIER_NAMES)].copy()


# rank groups on the curve cards
CURVE_GROUPS = [("LOW RANK", (1, 2, 3)), ("MID RANK", (4, 5, 6, 7)), ("HIGH RANK", (8, 9, 10, 11))]
SMOOTH_SIGMA = 0.9          # width of the neighbour estimate used for NET, in ranks (games-weighted)
TRUST_GAMES = 1000          # NET endpoints: a rank with this many games is trusted 50/50 with its neighbours
CURVE_MIN_GAMES = 3000      # a hero needs this many ranked games in total to get a line
CURVE_MARGIN_X = 0.018      # ~20px side margins (the default card uses ~40px)
HERO_COLORS = ["#ff6b6b", "#4dabf7", "#ffd43b", "#69db7c", "#da77f2", "#ffa94d", "#66d9e8",
               "#f783ac", "#a9e34b", "#748ffc", "#e9ecef", "#38d9a9", "#ffc9c9", "#c0a36e"]


def endpoint_estimate(wr: np.ndarray, n: np.ndarray, at: int) -> tuple[float, float]:
    """Win rate at one end of the rank ladder. That rank's actual win rate is trusted in proportion to
    its games, n / (n + TRUST_GAMES); the rest comes from a games-weighted Gaussian estimate over the
    neighbouring ranks. Well-sampled ends stay close to their real number, thin ones (often Eternus)
    lean on their neighbours. Returns (value, SE), in %."""
    ok = ~np.isnan(wr) & (n > 0)
    p = np.where(ok, wr / 100, 0.0)
    nn = np.where(ok, n, 0.0)
    kern = np.exp(-0.5 * ((np.arange(len(wr)) - at) / SMOOTH_SIGMA) ** 2) * nn
    trust = nn[at] / (nn[at] + TRUST_GAMES)
    weights = trust * (np.arange(len(wr)) == at) + (1 - trust) * kern / kern.sum()
    val = float((weights * p).sum())
    var = float((weights[ok] ** 2 * p[ok] * (1 - p[ok]) / nn[ok]).sum())
    return val * 100, np.sqrt(var) * 100


def curve_table(r: pd.DataFrame) -> pd.DataFrame:
    """Per hero: first and last rank with data, their actual win rates, the blended endpoint estimates,
    the net change between those (what the NET column shows) and how sure we are it's real."""
    tiers = sorted(RANK_TIER_NAMES)
    out = []
    for (h, ht), d in r.groupby(["hero_name", "hero_type"]):
        d = d.set_index("rank_tier").reindex(tiers)
        if d["games"].sum() < CURVE_MIN_GAMES:
            continue
        wr, n = d["winrate"].to_numpy(float), d["games"].fillna(0).to_numpy(float)
        have = np.flatnonzero(~np.isnan(wr) & (n > 0))
        first, last = int(have[0]), int(have[-1])
        v0, se0 = endpoint_estimate(wr, n, first)
        v1, se1 = endpoint_estimate(wr, n, last)
        net = v1 - v0
        out.append({"hero_name": h, "hero_type": ht, "first": first, "last": last,
                    "first_wr": wr[first], "last_wr": wr[last], "last_n": int(n[last]),
                    "start_wr": v0, "end_wr": v1, "net": net, "games": int(n.sum()),
                    "conf": float(normal_cdf(abs(net) / np.hypot(se0, se1)))})
    return pd.DataFrame(out)


def render_curve(r: pd.DataFrame, c_t: pd.DataFrame, htype: str, meta: T.ChartMeta, progress: float = 1.0,
                 highlight: set[str] | None = None) -> "T.Figure":
    """Every hero of a class across the 11 ranks: actual win rate per rank joined by straight lines,
    with each hero's net change (blended Eternus estimate minus blended Initiate estimate)."""
    highlight = {h.lower() for h in highlight} if highlight else None
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    p_lines = T.ease_out_cubic(T.phase(progress, 0.15, 0.75))
    p_labels = T.ease_out_cubic(T.phase(progress, 0.7, 0.95))
    plural = "MARKSMEN" if htype == "marksman" else f"{htype.upper()}S"
    rows = c_t[c_t["hero_type"] == htype].sort_values("last_wr", ascending=False).reset_index(drop=True)
    sure = rows[rows["conf"] >= 0.8]                  # only name changes that are probably real
    parts = []
    if (sure["net"] > 0).any():
        x = sure.loc[sure["net"].idxmax()]
        parts.append(f"{x['hero_name']} gains the most ({x['net']:+.1f} pts)")
    if (sure["net"] < 0).any():
        x = sure.loc[sure["net"].idxmin()]
        parts.append(f"{x['hero_name']} loses the most ({x['net']:+.1f} pts)")
    sub = "  ·  ".join(parts) or f"No {plural.lower()} clearly change with rank this week"

    fig, ax = T.new_card()
    fig._margin_x = CURVE_MARGIN_X
    T.draw_header(fig, f"{plural} ACROSS THE RANKS", sub, alpha=p_frame)
    T.draw_footer(fig, meta, alpha=p_frame)
    # the note sits right on the footer rule; the plot fills everything above it
    note_y = T.FOOTER_LINE_Y + 9 / T.CARD_PX
    fig.text(CURVE_MARGIN_X, note_y, f"hollow = under {RANK_POINT_MIN:,} games  ·  net = Eternus minus Initiate; ranks with fewer "
             "games lean on their neighbouring ranks  ·  gray net = could be noise",
             ha="left", va="bottom", color=T.TEXT_MUTED, fontsize=T.px_to_pt(14), alpha=p_frame)
    fig._content_bottom = note_y + 26 / T.CARD_PX

    tiers = sorted(RANK_TIER_NAMES)
    nx = len(tiers)
    cls = r[r["hero_type"] == htype]
    wr = cls.pivot(index="hero_name", columns="rank_tier", values="winrate").reindex(columns=tiers)
    games = cls.pivot(index="hero_name", columns="rank_tier", values="games").reindex(columns=tiers).fillna(0)
    vals = wr.loc[rows["hero_name"]].to_numpy(float)
    lo_y = float(np.floor(min(np.nanmin(vals), 50) - 0.5))
    hi_y = float(np.ceil(max(np.nanmax(vals), 50) + 1.2))

    # horizontal budget (px): plot | icon | name | net. The name column hugs the NET column.
    ax_w_px = (1 - 2 * CURVE_MARGIN_X) * T.CARD_PX - 62          # minus the y tick labels (approx.)
    name_px = 21
    name_w = max(T.text_px(h, name_px) for h in rows["hero_name"])
    net_w = T.text_px("+0.0", name_px * 1.1) + 4
    gaps = 14 + 10 + 18
    plot_h_px = (T.HEADER_SUB_Y - 0.06 - fig._content_bottom) * T.CARD_PX
    icon_px = int(np.clip(plot_h_px / max(len(rows), 1) * 0.8, 28, 46))
    block_frac = (icon_px + name_w + net_w + gaps + 30) / ax_w_px
    x0, x1 = -0.35, nx - 1 + 0.35                                 # data range of the plot itself
    xmax = x1 + (x1 - x0) * block_frac / (1 - block_frac)
    to_x = lambda frac: x0 + frac * (xmax - x0)                   # axes fraction -> data x
    ax.set_xlim(x0, xmax)
    ax.set_ylim(lo_y, hi_y)
    x_net = to_x(1.0)
    x_name = to_x(1 - (net_w + 14 + name_w) / ax_w_px)
    x_icon = to_x(1 - (net_w + 14 + name_w + 10 + icon_px / 2) / ax_w_px)

    ax.set_xticks(range(nx), [RANK_TIER_NAMES[k] for k in tiers], rotation=35, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(np.arange(np.ceil(lo_y), hi_y + 0.01, 2 if hi_y - lo_y > 9 else 1))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_bounds(x0, x1)
    ax.tick_params(axis="x", pad=4)
    for tl in ax.get_xticklabels():
        tl.set_alpha(p_frame); tl.set_fontsize(T.px_to_pt(18)); tl.set_color(T.TEXT)
    for tl in ax.get_yticklabels():
        tl.set_alpha(p_frame); tl.set_fontsize(T.px_to_pt(20)); tl.set_color(T.TEXT); tl.set_fontweight("bold")
    ax.vlines(range(nx), lo_y, hi_y, colors=T.DIVIDER, linestyles=(0, (3, 4)), lw=0.9, alpha=p_frame, zorder=0)
    ax.plot([x0, x1], [50, 50], color=T.TEXT_MUTED, lw=1.6, alpha=p_frame, zorder=1)
    for gi, (gname, gt) in enumerate(CURVE_GROUPS):
        a_, b_ = tiers.index(gt[0]) - 0.5, tiers.index(gt[-1]) + 0.5
        if gi == 1:
            ax.axvspan(max(a_, x0), b_, color=T.TEXT, alpha=0.035 * p_frame, lw=0, zorder=0)
        ax.text((max(a_, x0) + min(b_, x1)) / 2, hi_y - 0.15, gname, ha="center", va="top", color=T.GOLD,
                fontsize=T.px_to_pt(15), fontweight="bold", alpha=0.85 * p_frame)
    ax.text(x_net, hi_y - 0.15, "NET", ha="right", va="top", color=T.GOLD, fontsize=T.px_to_pt(15),
            fontweight="bold", alpha=p_frame)

    # labels: in end-of-line order, nudged apart so icons never overlap
    px_per_unit = plot_h_px / (hi_y - lo_y)
    ends = list(rows["last_wr"])
    pad = icon_px / 2 / px_per_unit
    label_y = _spread(ends, (icon_px + 5) / px_per_unit, lo_y + pad, hi_y - pad - 26 / px_per_unit)

    for i, row in rows.iterrows():
        h = row["hero_name"]
        col = HERO_COLORS[i % len(HERO_COLORS)]
        hl = highlight is None or h.lower() in highlight
        a = 1.0 if hl else DIM_ALPHA
        ys = wr.loc[h].to_numpy(float)
        n = games.loc[h].to_numpy(float)
        ok = ~np.isnan(ys)
        xi, yi = np.arange(nx)[ok], ys[ok]
        upto = p_lines * (nx - 1)
        keep = xi <= upto + 1e-9
        xs, yv = list(xi[keep]), list(yi[keep])
        if len(xs) and upto > xs[-1] and (~keep).any():
            xs.append(upto); yv.append(float(np.interp(upto, xi, yi)))
        ax.plot(xs, yv, color=col, lw=3.4 if (highlight and hl) else 2.2, alpha=0.95 * a,
                zorder=4 if hl else 3, solid_capstyle="round", solid_joinstyle="round")
        for j in xi[keep]:
            few = n[j] < RANK_POINT_MIN
            ax.scatter([j], [ys[j]], s=30, zorder=5, color=T.BG if few else col, edgecolor=col, lw=1.6, alpha=a)
        if p_labels <= 0:
            continue
        la = a * p_labels
        clear = row["conf"] >= 0.95
        ax.plot([row["last"], x_icon - 0.25], [row["last_wr"], label_y[i]],
                color=col, lw=1.3, alpha=0.7 * la, zorder=2)
        T.add_icon(ax, T.icon_any(h, icon_px, col, la, wr=row["last_wr"]), (x_icon, label_y[i]), zorder=6)
        ax.text(x_name, label_y[i], h, ha="left", va="center", color=col, alpha=la,
                fontsize=T.px_to_pt(name_px), fontweight="bold", zorder=6)
        ax.text(x_net, label_y[i], f"{row['net']:+.1f}", ha="right", va="center",
                color=(T.GREEN if row["net"] > 0 else T.RED) if clear else T.TEXT_MUTED, alpha=la,
                fontsize=T.px_to_pt(name_px * 1.1), fontweight="bold", zorder=6)
    return fig


# --- INSIGHTS / ENTRY POINTS --------------------------------------------------------
def insights(data_dir: Path, **_) -> list[Insight]:
    g = load_rank_winrates(data_dir)
    t = low_high_table(g)
    t = t[(t["early_n"] >= MIN_GAMES) & (t["late_n"] >= MIN_GAMES)]
    ok = g[g["games"] >= MIN_GAMES]
    out = []
    for worst in (False, True):
        card, kind = ("rank_worst_heroes", "rank_worst") if worst else ("rank_best_heroes", "rank_best")
        tops = []
        for b, (name, _) in enumerate(RANK_BRACKETS):
            sub = ok[ok["bracket"] == b]
            if sub.empty:
                continue
            tops.append(set((sub.nsmallest if worst else sub.nlargest)(TOP_N, "winrate")["hero_name"]))
            r = sub.loc[sub["winrate"].idxmin() if worst else sub["winrate"].idxmax()]
            out.append(Insight(card, kind, hero=r["hero_name"], highlight=(r["hero_name"],),
                               headline=f"{'Worst' if worst else 'Best'} hero in {name}: {r['hero_name']} "
                                        f"at {r['winrate']:.1f}%",
                               detail=f"{int(r['games']):,} games, in {r['pick_rate']:.0f}% of that bracket's matches",
                               effect=abs(r["winrate"] - 50),
                               confidence=float(wr_confidence(r["winrate"], r["games"]))))
        every = sorted(set.intersection(*tops)) if tops else []
        if every:
            out.append(Insight(card, f"{kind}_everywhere", context=True, highlight=tuple(every),
                               headline=f"{', '.join(every)} {'is' if len(every) == 1 else 'are'} "
                                        f"{'bottom' if worst else 'top'} {TOP_N} at every rank"))
    ct = curve_table(load_tier_winrates(data_dir))
    for htype in T.ARCHETYPE_ORDER:
        c = ct[(ct["hero_type"] == htype) & (ct["conf"] >= 0.8)]
        plural = "marksmen" if htype == "marksman" else f"{htype}s"
        for pick, verb in ((c[c["net"] > 0].nlargest(1, "net"), "gains the most"),
                           (c[c["net"] < 0].nsmallest(1, "net"), "loses the most")):
            for _, r in pick.iterrows():
                out.append(Insight(f"rank_curve_{htype}", "class_rank", hero=r["hero_name"],
                                   highlight=(r["hero_name"],),
                                   headline=f"Of the {plural}, {r['hero_name']} {verb} going up the ranks: "
                                            f"{r['net']:+.1f} pts",
                                   detail=f"{r['first_wr']:.1f}% at Initiate, {r['last_wr']:.1f}% at Eternus "
                                          f"({r['last_n']:,} games); net leans on neighbouring ranks "
                                          f"where a rank has few games",
                                   effect=abs(r["net"]), confidence=float(r["conf"])))
    return out


def cards(data_dir: Path, **_) -> dict:
    """{output name: draw(progress=1.0, highlight=None) -> Figure} for every card this script makes."""
    g = load_rank_winrates(data_dir)
    t = low_high_table(g)
    heroes = load_hero_meta(data_dir)
    solo = dict(zip(heroes["hero_name"], heroes["winrate"]))
    meta = T.ChartMeta(T.week_label_from_dir(data_dir), int(round(g["games"].sum() / 12)),
                       min_n=MIN_GAMES, min_n_unit="games per hero per bracket")
    r = load_tier_winrates(data_dir)
    ct = curve_table(r)
    curve_meta = T.ChartMeta(meta.week_label, meta.n_matches, min_n=CURVE_MIN_GAMES,
                             min_n_unit="ranked games per hero")
    out = {"rank_best_heroes": lambda progress=1.0, highlight=None: render_best(g, meta, progress, highlight),
           "rank_worst_heroes": lambda progress=1.0, highlight=None: render_best(g, meta, progress, highlight,
                                                                                  worst=True)}
    for htype in T.ARCHETYPE_ORDER:
        if (ct["hero_type"] == htype).any():
            out[f"rank_curve_{htype}"] = (lambda progress=1.0, highlight=None, h=htype:
                                          render_curve(r, ct, h, curve_meta, progress, highlight))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", nargs="?", default=None)
    ap.add_argument("--highlight", default="")
    ap.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()
    data_dir = T.resolve_target_dir(args.week_dir)
    for p in T.save_cards(cards(data_dir), data_dir / "graphs", args.progress, T.parse_names(args.highlight)):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
