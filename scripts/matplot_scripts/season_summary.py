"""
season_summary.py — Season 1 in data: placements vs final ranks, climbing vs win rate and games played,
points won / lost / kept, what demotion shields actually did, the ladder week by week, missing matches,
and the Monte Carlo sims vs reality.

Two steps:
    python season_summary.py export [--start 2026-07-30] [--end 2026-11-05] [--name season_1]
        one pass over match_player (needs the rank progress columns) -> <project>/season_1/:
        players.parquet (one row per player), weekly_ranks.csv, losses_by_rank.csv, rank_changes.csv,
        season_meta.json
    python season_summary.py cards [season_dir] [--sim DIR]
        the cards -> <season_dir>/graphs/ (animate.py and assemble_video.py work on the folder too)

Definitions (rank progress: 1,000 points per tier, 7,000 per rank, Eternus from 70,000)
    placement match   calibration_games > 0 going in
    placed at         progress after the last placement match (calibration_games = 1 going in; the earlier ones
                      sit near 0 until it), else the first ranked match after placements
    ended at          progress after the player's last recorded match;  peak = highest progress after a match
    climb             ended - placed, in ranks. Exact even with missing matches: every match moves the same
                      progress number, recorded or not
    win rate          wins / (wins + losses) over the player's RECORDED non-placement matches (the API misses
                      some matches, so this is a sample); win-rate cards use players with MIN_GAMES of them,
                      in windows around the sims' win rates (45% = 44 to under 46, ...)
    points won / lost from recorded matches only, per 100 recorded games (gaps don't inflate them)
    points saved      on a shielded loss: actual change - what the loss should have taken (300 - x)
    missing matches   between two recorded matches, rank going in != rank coming out of the previous one means
                      at least one match is missing (a gap). A missed win + a missed loss can cancel out, so
                      the counts are a floor

Cards (video order: orders/season_recap.txt)
    season_numbers               the season in six numbers
    season_rank_distribution     where players placed vs where they ended, per rank
    season_placement_outcomes    for each placement rank: stopped early / fell / stayed / climbed 1, 2, 3+
    season_underplaced           for each placement rank: share winning 55%+ and how far they climbed
    season_rank_over_time        active players per week by low / mid / high rank (counts, shares labelled)
    season_climb_by_winrate      what happened to the players at each win rate (fell ... +3 ranks)
    season_climb_by_games        median climb by games played, one line per win rate
    season_points_kept           points won, lost and kept per 100 games (+ the sim's "kept")
    season_shields               shields per 100 games and points per shield, real vs sim
    season_games_per_rank        games needed per tier and per rank at each win rate, real vs the sims
    season_below_peak            share who finished below their best rank, real vs sim
    season_sim_vs_real           points per 100 games: real vs both sims, with games needed per rank
    season_missing_matches       how many ranked matches the public data is missing
    season_missing_skew          recorded win rate vs estimated real win rate once missing games are added back
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import theme as T
from chart_stats import RANK_TIER_NAMES

RANK_PTS = 7000                 # progress per rank
ETERNUS_PTS = 70000
POINTS_PER_LOSS = 300
MIN_GAMES = 30                  # recorded non-placement matches a player needs for the win-rate cards
MIN_POINT_N = 100               # chart points with fewer players are drawn hollow
SIM_GAMES = 210
# win-rate windows centred on the sims' win rates: (label, sim win rate, lo, hi) with lo <= wr < hi
WR_WINDOWS = [(f"{w}%", w, w - 1, w + 1) for w in (45, 48, 50, 52, 55, 60)]
WR_COLORS = ["#ef6363", "#f0a35e", "#ffd073", "#9bd17f", "#41d67e", "#4dabf7"]
GAME_EDGES = [30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 100_000]
GAME_LABELS = ["30–49", "50–74", "75–99", "100–124", "125–149", "150–174", "175–199", "200–224", "225–249", "250+"]
CLIMB_EDGES = [-np.inf, 0, 7000, 14000, 21000, np.inf]          # rank points gained since placement
CLIMB_LABELS = ["lost points", "+0–7k", "+7k–14k", "+14k–21k", "+21k or more"]
CLIMB_COLORS = [T.RED, "#8a8f98", "#9bd17f", "#41d67e", "#4dabf7"]
RANK_GROUPS = [("Low (Initiate–Acolyte)", (1, 2, 3), "#4dabf7"), ("Mid (Sentinel–Emissary)", (4, 5, 6, 7), "#ffd43b"),
               ("High (Oracle+)", (8, 9, 10, 11), "#da77f2")]
SIM_FILES = {"with shields": "ranked_monte_carlo_results_3_dailies_realistic.parquet",
             "no shields": "ranked_monte_carlo_results_shieldless_3_dailies.parquet"}
SIM_COLORS = {"with shields": "#4dabf7", "no shields": T.RED}
SEASON_START = datetime(2026, 7, 30, tzinfo=timezone.utc)


def rank_of(progress) -> np.ndarray:
    p = np.asarray(progress, dtype=float)
    return np.where(p >= ETERNUS_PTS, 11, np.floor(p / RANK_PTS) + 1).clip(1, 11)


def tier_index(progress) -> np.ndarray:
    """Badge position on one scale (6 tiers per rank, VI = two steps), the sims' (rank, tier) equivalent."""
    p = np.asarray(progress, dtype=float)
    step = np.floor(p / 1000)
    idx = np.floor(step / 7) * 6 + np.minimum(step % 7, 5)
    return np.where(p >= ETERNUS_PTS, 60, idx)


# =============================================================================
# EXPORT (PostgreSQL -> season folder)
# =============================================================================
# Tier start in flat progress: 1,000 points per step, 7 steps per rank, tier VI = the last two steps.
TIER_START = "((pi / 1000 - CASE WHEN (pi / 1000) % 7 = 6 THEN 1 ELSE 0 END) * 1000)"
# Valve's match record leaves out one kind of demotion: a loss at a tier's 0 points with no shields is stored as
# no change (final = initial, and "shield consumed"), but the player's next match starts 300 lower with fresh
# shields. Those rows are corrected to initial + desired change, and their shield flag dropped.
SQL_STAGE = f"""
CREATE TEMP TABLE season_m AS
WITH r AS (
    SELECT account_id, match_id, start_time,
           rank_progress_initial AS pi, rank_progress_final AS pf_raw, rank_change_desired AS d,
           COALESCE(shield_consumed, false) AS sc_raw, COALESCE(calibration_games, 0) AS cal,
           shields_initial AS si, rank_outcome AS o
    FROM match_player
    WHERE start_time >= %(start)s AND start_time < %(end)s AND rank_progress_final IS NOT NULL
), c AS (
    SELECT *, (d < 0 AND COALESCE(si, 0) = 0 AND pf_raw = pi AND pi BETWEEN 1000 AND 69999
               AND pi = {TIER_START.replace('%', '%%')}) AS hidden_demotion
    FROM r
)
SELECT account_id, start_time, pi,
       CASE WHEN hidden_demotion THEN pi + d ELSE pf_raw END AS pf,
       d, (sc_raw AND NOT hidden_demotion) AS sc, cal > 0 AS placement, cal, si, o, hidden_demotion,
       pi - CASE WHEN pi < 70000 THEN {TIER_START.replace('%', '%%')} ELSE 70000 END AS in_tier,
       lag(CASE WHEN hidden_demotion THEN pi + d ELSE pf_raw END) OVER w AS prev_pf,
       lag(cal > 0) OVER w AS prev_placement
FROM c
WINDOW w AS (PARTITION BY account_id ORDER BY start_time, match_id);
ANALYZE season_m;
"""
SQL_PLAYERS = """
SELECT account_id,
       count(*) FILTER (WHERE placement)                         AS placement_games,
       count(*) FILTER (WHERE NOT placement)                     AS games,
       count(*) FILTER (WHERE NOT placement AND o = 'Win')       AS wins,
       count(*) FILTER (WHERE NOT placement AND o = 'Loss')      AS losses,
       (array_agg(pf ORDER BY start_time DESC) FILTER (WHERE cal = 1))[1]   AS placed_progress,
       (array_agg(pi ORDER BY start_time) FILTER (WHERE NOT placement))[1]   AS first_progress,
       (array_agg(pf ORDER BY start_time DESC))[1]                            AS end_progress,
       max(pf) FILTER (WHERE NOT placement)                      AS peak_progress,
       COALESCE(sum(pf - pi) FILTER (WHERE NOT placement), 0)    AS recorded_change,
       COALESCE(sum(pf - pi) FILTER (WHERE NOT placement AND pf > pi), 0) AS points_won,
       COALESCE(sum(pi - pf) FILTER (WHERE NOT placement AND pf < pi), 0) AS points_lost,
       count(*) FILTER (WHERE sc)                                AS shields_used,
       COALESCE(sum((pf - pi) - d) FILTER (WHERE sc AND d < 0), 0) AS points_saved,
       count(*) FILTER (WHERE NOT placement AND prev_pf IS NOT NULL AND NOT prev_placement) AS transitions,
       count(*) FILTER (WHERE NOT placement AND prev_pf IS NOT NULL AND NOT prev_placement
                        AND prev_pf <> pi)                       AS gaps,
       COALESCE(sum(abs(pi - prev_pf)) FILTER (WHERE NOT placement AND prev_pf IS NOT NULL
                                               AND NOT prev_placement AND prev_pf <> pi), 0) AS gap_points,
       COALESCE(sum(pi - prev_pf) FILTER (WHERE NOT placement AND prev_pf IS NOT NULL
                                          AND NOT prev_placement AND prev_pf <> pi), 0) AS gap_net,
       count(*) FILTER (WHERE NOT placement AND prev_pf IS NOT NULL AND NOT prev_placement
                        AND pi - prev_pf BETWEEN 1 AND 430)      AS gap_win1,
       count(*) FILTER (WHERE NOT placement AND prev_pf IS NOT NULL AND NOT prev_placement
                        AND pi - prev_pf BETWEEN -300 AND -1)    AS gap_loss1,
       min(start_time) AS first_match, max(start_time) AS last_match
FROM season_m GROUP BY account_id
"""
SQL_WEEKLY = """
SELECT week, LEAST(pf / 7000 + 1, 11) AS rank, count(*) AS players
FROM (SELECT DISTINCT ON (account_id, week) account_id, week, pf
      FROM (SELECT account_id, start_time, pf,
                   floor(extract(epoch FROM start_time - %(start)s) / 604800)::int + 1 AS week
            FROM season_m WHERE NOT placement) w
      ORDER BY account_id, week, start_time DESC) last_of_week
GROUP BY 1, 2 ORDER BY 1, 2
"""
SQL_LOSSES = """
SELECT LEAST(pi / 7000 + 1, 11) AS rank,
       count(*)                                    AS games,
       count(*) FILTER (WHERE d < 0)               AS losses,
       count(*) FILTER (WHERE sc)                  AS shielded,
       COALESCE(sum((pf - pi) - d) FILTER (WHERE sc AND d < 0), 0) AS points_saved,
       COALESCE(sum(-d) FILTER (WHERE d < 0), 0)   AS points_at_stake
FROM season_m WHERE NOT placement GROUP BY 1 ORDER BY 1
"""
# Losses that would take a player below their tier (fewer than 300 points into it): how often they were
# protected or demoted, by shields held going in and by being at exactly 0 points
SQL_PROTECTION = """
SELECT LEAST(si, 5) AS shields_held, (in_tier = 0) AS at_zero,
       count(*)                                         AS losses,
       count(*) FILTER (WHERE sc)                       AS shielded,
       count(*) FILTER (WHERE pf < pi - in_tier)        AS demoted,
       count(*) FILTER (WHERE hidden_demotion)          AS corrected,
       avg(pf - pi)                                     AS avg_change
FROM season_m
WHERE NOT placement AND d = -300 AND pi BETWEEN 1000 AND 69999 AND in_tier < 300 AND si IS NOT NULL
GROUP BY 1, 2 ORDER BY 2, 1
"""
SQL_CHANGES = """
SELECT d AS desired, o AS outcome, sc AS shield, count(*) AS n
FROM season_m WHERE NOT placement GROUP BY 1, 2, 3 ORDER BY n DESC
"""


def export(start: datetime, end: datetime, out: Path) -> None:
    from pipeline_common import connect
    out.mkdir(parents=True, exist_ok=True)
    conn = connect()
    try:
        with conn.cursor() as cur:
            t0 = datetime.now()
            print(f"Staging {start:%Y-%m-%d} -> {end:%Y-%m-%d} ...", flush=True)
            cur.execute(SQL_STAGE, {"start": start, "end": end})
            cur.execute("SELECT count(*), count(DISTINCT account_id) FROM season_m")
            rows, players = cur.fetchone()
            if not rows:
                raise SystemExit("No rows with rank progress in that range: run the rank backfill first "
                                 "(ingest_dl_comm_api_parquet.py --ranks-only --weeks ...)")
            print(f"  {rows:,} player-matches, {players:,} players ({(datetime.now() - t0).seconds}s)")
            for name, sql in (("players", SQL_PLAYERS), ("weekly_ranks", SQL_WEEKLY),
                              ("losses_by_rank", SQL_LOSSES), ("rank_changes", SQL_CHANGES),
                              ("loss_protection", SQL_PROTECTION)):
                cur.execute(sql, {"start": start})
                df = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
                if name == "players":
                    df.to_parquet(out / "players.parquet", index=False)
                else:
                    df.to_csv(out / f"{name}.csv", index=False)
                print(f"  {name}: {len(df):,} rows ({(datetime.now() - t0).seconds}s)")
            cur.execute("SELECT count(DISTINCT match_id) FROM match_player WHERE start_time >= %s AND start_time < %s",
                        (start, end))
            matches = cur.fetchone()[0]
        conn.rollback()                                  # only a temp table was written
    finally:
        conn.close()
    meta = {"name": out.name, "start": start.isoformat(), "end": end.isoformat(), "matches": int(matches),
            "player_matches": int(rows), "players": int(players),
            "date_range": f"{start:%b %-d} – {(end - timedelta(seconds=1)):%b %-d}"}
    (out / "season_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"Done -> {out}")


# =============================================================================
# LOAD + DERIVE
# =============================================================================
def load(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "season_meta.json").read_text())
    p = pd.read_parquet(data_dir / "players.parquet")
    p["placed_at"] = p["placed_progress"].fillna(p["first_progress"])
    # started placements in the data but no placement result and no ranked match afterwards
    p["unfinished"] = (p["placement_games"] > 0) & p["placed_progress"].isna() & (p["games"] == 0)
    p["climb"] = (p["end_progress"] - p["placed_at"]) / RANK_PTS
    p["placed_rank"] = rank_of(p["placed_at"].fillna(0))
    p["end_rank"] = rank_of(p["end_progress"].fillna(0))
    p["decided"] = p["wins"] + p["losses"]
    p["wr"] = 100 * p["wins"] / p["decided"].replace(0, np.nan)
    g = p["games"].replace(0, np.nan)
    p["per100"] = 100 * p["recorded_change"] / g
    for c in ("points_won", "points_lost", "shields_used"):
        if c in p:
            p[f"{c}_100"] = 100 * p[c] / g
    if "gap_net" in p:
        estimate_true_wr(p)
    if "peak_progress" in p:
        p["below_peak"] = tier_index(p["peak_progress"].fillna(0)) > tier_index(p["end_progress"].fillna(0))
    return {"meta": meta, "players": p,
            "weekly": pd.read_csv(data_dir / "weekly_ranks.csv"),
            "losses": pd.read_csv(data_dir / "losses_by_rank.csv"),
            "changes": pd.read_csv(data_dir / "rank_changes.csv"),
            "protection": (pd.read_csv(data_dir / "loss_protection.csv")
                           if (data_dir / "loss_protection.csv").exists() else None)}


def estimate_true_wr(p: pd.DataFrame) -> None:
    """Missing matches per player: how many (a floor: points unaccounted for / 300, at least one per gap), the
    exact net points they were worth (rank change between recorded matches), how many of them were wins (net points
    solved with the player's own average win and loss value), and the win rate with them added back in.
    Losses at a tier's 0-point floor leave no trace, so missing losses are undercounted: wr_adj leans high."""
    n = np.ceil(p["gap_points"] / POINTS_PER_LOSS).clip(lower=p["gaps"])
    win_val = (p["points_won"] / p["wins"].replace(0, np.nan)).fillna(320)
    loss_val = (p["points_lost"] / p["losses"].replace(0, np.nan)).fillna(200)
    w = ((p["gap_net"] + n * loss_val) / (win_val + loss_val)).clip(lower=0).clip(upper=n)
    p["est_missing"] = n
    p["missing_wins"] = w
    p["wr_adj"] = 100 * (p["wins"] + w) / (p["decided"] + n).replace(0, np.nan)


def has(d: dict, *cols) -> bool:
    return all(c in d["players"] for c in cols)


def chart_meta(d: dict, min_n: int | None = None) -> T.ChartMeta:
    m = d["meta"]
    return T.ChartMeta(m["name"].replace("_", " ").title(), m["matches"], min_n=min_n,
                       min_n_unit="recorded games per player", date_range=m.get("date_range"))


def qualified(p: pd.DataFrame) -> pd.DataFrame:
    return p[p["games"] >= MIN_GAMES]


def in_window(p: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    return p[(p["wr"] >= lo) & (p["wr"] < hi)]


def numbers(d: dict) -> dict:
    p = d["players"]
    placed = p[p["placed_progress"].notna() & (p["games"] > 0)]
    lo = d["losses"]
    losses, shielded = lo["losses"].sum(), lo["shielded"].sum()
    return {"players": len(p), "matches": d["meta"]["matches"],
            "climbed": 100 * (placed["climb"] > 0).mean() if len(placed) else np.nan,
            "fell": 100 * (placed["climb"] < 0).mean() if len(placed) else np.nan,
            "shielded": 100 * shielded / losses if losses else np.nan,
            "saved_per_shield": lo["points_saved"].sum() / shielded if shielded else np.nan,
            "placed_n": len(placed)}


def outcomes_by_wr(p: pd.DataFrame) -> pd.DataFrame:
    q = qualified(p)
    rows = []
    for label, w, lo, hi in WR_WINDOWS:
        g = in_window(q, lo, hi)
        pts = (g["climb"] * RANK_PTS).to_numpy()
        n = max(len(g), 1)
        counts = np.histogram(pts, CLIMB_EDGES)[0]
        rows.append({"window": label, "n": len(g), **{k: 100 * v / n for k, v in zip(CLIMB_LABELS, counts)}})
    return pd.DataFrame(rows)


def climb_by_games(p: pd.DataFrame) -> pd.DataFrame:
    q = qualified(p).copy()
    q["gbin"] = pd.cut(q["games"], GAME_EDGES, labels=GAME_LABELS, right=False)
    rows = []
    for (label, w, lo, hi), color in zip(WR_WINDOWS, WR_COLORS):
        s = in_window(q, lo, hi).groupby("gbin", observed=False)["climb"]
        for b, n, med in zip(GAME_LABELS, s.size().values, s.median().values):
            rows.append({"window": label, "bin": b, "n": n, "median": med, "color": color})
    return pd.DataFrame(rows)


def placement_outcomes(p: pd.DataFrame) -> pd.DataFrame:
    """Players with recorded placements, per placement rank: stopped early (< MIN_GAMES after placing), and for
    the rest the change in rank (by rank name, not points)."""
    q = p[p["placed_progress"].notna()]
    rows = []
    for r in range(1, 12):
        g = q[q["placed_rank"] == r]
        if len(g) < MIN_POINT_N:
            continue
        stayed = g[g["games"] >= MIN_GAMES]
        delta = stayed["end_rank"] - stayed["placed_rank"]
        n = len(g)
        rows.append({"rank": r, "n": n, "stopped early": 100 * (n - len(stayed)) / n,
                     "fell": 100 * (delta < 0).sum() / n, "same rank": 100 * (delta == 0).sum() / n,
                     "+1": 100 * (delta == 1).sum() / n, "+2": 100 * (delta == 2).sum() / n,
                     "+3 or more": 100 * (delta >= 3).sum() / n})
    return pd.DataFrame(rows)


def underplaced(p: pd.DataFrame) -> pd.DataFrame:
    q = qualified(p[p["placed_progress"].notna()])
    rows = []
    for r in range(1, 12):
        g = q[q["placed_rank"] == r]
        if len(g) < MIN_POINT_N:
            continue
        hi = g[g["wr"] >= 55]
        rows.append({"rank": r, "n": len(g), "share55": 100 * len(hi) / len(g),
                     "climb55": hi["climb"].median() if len(hi) else np.nan, "median_wr": g["wr"].median(),
                     "climb_all": g["climb"].median()})
    return pd.DataFrame(rows)


def by_window(p: pd.DataFrame, col: str, how: str = "mean") -> list[float]:
    q = qualified(p)
    out = []
    for label, w, lo, hi in WR_WINDOWS:
        s = in_window(q, lo, hi)[col].dropna()
        out.append(float(getattr(s, how)()) if len(s) else np.nan)
    return out


def load_sims(sim_dir: Path | None) -> dict:
    """{label: DataFrame per sim win rate} with per-100-game rates, shields, points per shield, below-peak."""
    out = {}
    if not sim_dir:
        return out
    for label, fname in SIM_FILES.items():
        f = Path(sim_dir) / fname
        if not f.exists():
            continue
        s = pd.read_parquet(f)
        s["below_peak"] = ((s["peak_rank"] - 1) * 6 + s["peak_tier"]) > ((s["final_rank"] - 1) * 6 + s["final_tier"])
        agg = {"net": ("net_points", "mean"), "gross": ("gross_points", "mean"), "below_peak": ("below_peak", "mean")}
        if "shields_used" in s:
            agg |= {"shields": ("shields_used", "mean"), "protected": ("points_protected", "sum"),
                    "shields_sum": ("shields_used", "sum")}
        g = s.groupby("target_winrate").agg(**agg)
        g["net100"] = g["net"] / SIM_GAMES * 100
        g["won100"] = g["gross"] / SIM_GAMES * 100
        g["lost100"] = g["won100"] - g["net100"]
        g["below_peak"] *= 100
        if "shields" in g:
            g["shields100"] = g["shields"] / SIM_GAMES * 100
            g["per_shield"] = g["protected"] / g["shields_sum"].replace(0, np.nan)
        out[label] = g
    return out


def sim_table(sims: dict, p: pd.DataFrame) -> pd.DataFrame | None:
    if not sims:
        return None
    rows = []
    real = by_window(p, "per100")
    ns = [len(in_window(qualified(p), lo, hi)) for _, _, lo, hi in WR_WINDOWS]
    for (label, w, lo, hi), r, n in zip(WR_WINDOWS, real, ns):
        row = {"wr": w, "real": r, "real_n": n}
        for k, g in sims.items():
            row[k] = float(g["net100"].get(w, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# CARD HELPERS
# =============================================================================
def _phases(progress: float) -> tuple[float, float, float]:
    return (T.ease_out_cubic(T.phase(progress, 0, 0.15)), T.ease_out_cubic(T.phase(progress, 0.15, 0.85)),
            T.ease_out_cubic(T.phase(progress, 0.8, 0.98)))


def _count(v: float) -> str:
    return f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}"


def _k(v, _=None) -> str:
    return f"{v / 1000:g}k" if v else "0"


def _ax_style(ax, p_frame: float) -> None:
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_alpha(p_frame)
        tl.set_color(T.TEXT)
        tl.set_fontsize(T.px_to_pt(T.SIZE_TICK))


def _note(fig, text: str, alpha: float) -> None:
    y = T.FOOTER_LINE_Y + 9 / T.CARD_PX
    fig.text(T.margin_x(fig), y, text, ha="left", va="bottom", color=T.TEXT_MUTED,
             fontsize=T.px_to_pt(14), alpha=alpha)
    fig._content_bottom = y + 28 / T.CARD_PX


def _label(ax, text: str, axis: str, p_frame: float) -> None:
    kw = dict(color=T.TEXT_MUTED, fontsize=T.px_to_pt(17), alpha=p_frame)
    (ax.set_xlabel if axis == "x" else ax.set_ylabel)(text, **kw)


def _legend(ax, p_frame: float, **kw) -> None:
    leg = ax.legend(frameon=False, fontsize=T.px_to_pt(kw.pop("size", 16)), labelcolor=T.TEXT, **kw)
    for t in leg.get_texts():
        t.set_alpha(p_frame)
    for h in leg.legend_handles:
        h.set_alpha(p_frame)


def _card(d: dict, title: str, sub: str, note: str, p_frame: float, min_n: int | None = None):
    fig, ax = T.new_card()
    T.draw_header(fig, title, sub, alpha=p_frame)
    T.draw_footer(fig, chart_meta(d, min_n), alpha=p_frame)
    _note(fig, note, p_frame)
    return fig, ax


def _window_ticks(ax, x) -> None:
    ax.set_xticks(x, [f"{lab}\n({lo}–{hi})" for lab, _, lo, hi in WR_WINDOWS])


def _stacked_h(ax, labels: list[str], table: pd.DataFrame, cols: list[str], colors: list[str], p_bars: float,
               p_lab: float, min_label: float = 6) -> None:
    """Horizontal 100% stacked bars, top row first, revealed left to right."""
    y = np.arange(len(table))[::-1]
    left = np.zeros(len(table))
    for col, color in zip(cols, colors):
        v = table[col].to_numpy(float)
        ax.barh(y, v * p_bars, 0.72, left=left * p_bars, color=color, zorder=3, label=col)
        for yi, l, vi in zip(y, left, v):
            if vi >= min_label and p_lab > 0:
                ax.text((l + vi / 2) * p_bars, yi, f"{vi:.0f}%", ha="center", va="center", color=T.BG,
                        fontsize=T.px_to_pt(15), fontweight="bold", alpha=p_lab)
        left += v
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", visible=False)


# =============================================================================
# CARDS
# =============================================================================
def render_numbers(d: dict, progress: float = 1.0, highlight=None):
    n = numbers(d)
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    fig = T.plt.figure(figsize=(T.CARD_IN, T.CARD_IN), dpi=T.DPI, facecolor=T.BG)
    T.draw_header(fig, "SEASON 1 IN NUMBERS", f"{d['meta']['date_range']}  ·  every ranked match in the data",
                  alpha=p_frame)
    T.draw_footer(fig, chart_meta(d), alpha=p_frame)
    tiles = [(f"{n['players']:,}", "players tracked", T.TEXT),
             (f"{n['matches']:,}", "ranked matches", T.TEXT),
             (f"{n['climbed']:.0f}%", "ended above their placement", T.GREEN),
             (f"{n['fell']:.0f}%", "ended below their placement", T.RED),
             (f"{n['shielded']:.0f}%", "of losses softened by a shield", T.GOLD),
             (f"{n['saved_per_shield']:.0f}", "points a shield saved, on average", T.GOLD)]
    top = getattr(fig, "_content_top", 0.86) - T.GAP_BELOW_HEADER
    bottom = T.FOOTER_LINE_Y + T.GAP_ABOVE_FOOTER
    cols, rows_n, gap = 2, 3, 18 / T.CARD_PX
    w = (1 - 2 * T.MARGIN_X - gap) / cols
    h = (top - bottom - gap * (rows_n - 1)) / rows_n
    v_px = T.fit_font_px([t[0] for t in tiles], w * T.CARD_PX * 0.86, h * T.CARD_PX * 0.42, max_px=96, min_px=30)
    l_px = T.fit_font_px([t[1] for t in tiles], w * T.CARD_PX * 0.88, h * T.CARD_PX * 0.16, max_px=24,
                         min_px=13, bold=False)
    for k, (val, lab, color) in enumerate(tiles):
        a = T.ease_out_cubic(T.phase(progress, 0.12 + 0.11 * k, 0.32 + 0.11 * k))
        x = T.MARGIN_X + (k % cols) * (w + gap)
        y = top - (k // cols + 1) * h - (k // cols) * gap
        fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, facecolor=T.PANEL, edgecolor="none", alpha=a))
        fig.text(x + w / 2, y + h * 0.58, val, ha="center", va="center", color=color,
                 fontsize=T.px_to_pt(v_px), fontweight="bold", alpha=a)
        fig.text(x + w / 2, y + h * 0.22, lab, ha="center", va="center", color=T.TEXT_MUTED,
                 fontsize=T.px_to_pt(l_px), alpha=a)
    return fig


def render_rank_distribution(d: dict, progress: float = 1.0, highlight=None):
    p = d["players"]
    q = p[p["placed_progress"].notna() & (p["games"] > 0)]
    placed = pd.Series(q["placed_rank"]).value_counts(normalize=True).reindex(range(1, 12), fill_value=0) * 100
    now = pd.Series(q["end_rank"]).value_counts(normalize=True).reindex(range(1, 12), fill_value=0) * 100
    p_frame, p_bars, p_lab = _phases(progress)
    fig, ax = _card(d, "WHERE PLAYERS PLACED VS WHERE THEY ENDED",
                    f"{len(q):,} players whose placement matches are in the data",
                    "share of those players in each rank  ·  placed = rank after the last placement match  ·  "
                    "ended = rank after their last recorded match", p_frame)
    x = np.arange(1, 12)
    bw = 0.4
    ax.bar(x - bw / 2, placed.values * p_bars, bw, color=T.NEUTRAL, label="placed", zorder=3)
    ax.bar(x + bw / 2, now.values * p_bars, bw, color=T.GOLD, label="ended", zorder=3)
    for xi, a, b in zip(x, placed.values, now.values):
        for xo, v, c in ((-bw / 2, a, T.TEXT_MUTED), (bw / 2, b, T.GOLD)):
            if v >= 0.5 and p_lab > 0:
                ax.text(xi + xo, v * p_bars + 0.3, f"{v:.0f}", ha="center", va="bottom", color=c,
                        fontsize=T.px_to_pt(13), fontweight="bold", alpha=p_lab)
    ax.set_xticks(x, [RANK_TIER_NAMES[k] for k in x], rotation=35, ha="right", rotation_mode="anchor")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylim(0, max(placed.max(), now.max()) * 1.15 + 1)
    ax.grid(axis="x", visible=False)
    _legend(ax, p_frame, loc="upper right", size=18)
    _ax_style(ax, p_frame)
    return fig


def render_placement_outcomes(d: dict, progress: float = 1.0, highlight=None):
    t = placement_outcomes(d["players"])
    cols = ["stopped early", "fell", "same rank", "+1", "+2", "+3 or more"]
    colors = ["#4a5157", T.RED, "#8a8f98", "#9bd17f", T.GREEN, "#4dabf7"]
    stop = (t["stopped early"] * t["n"]).sum() / t["n"].sum() if len(t) else np.nan
    unfinished = int(d["players"]["unfinished"].sum())
    p_frame, p_bars, p_lab = _phases(progress)
    fig, ax = _card(d, "WHAT HAPPENED AFTER PLACEMENTS",
                    f"{stop:.0f}% stopped within {MIN_GAMES} games  ·  {unfinished:,} more never finished "
                    "placements",
                    f"stopped early = under {MIN_GAMES} recorded games after placing  ·  everyone else: rank now vs "
                    "rank placed  ·  players whose placements are in the data", p_frame)
    labels = [f"{RANK_TIER_NAMES[int(r)]}  ({int(n):,})" for r, n in zip(t["rank"], t["n"])]
    _stacked_h(ax, labels, t, cols, colors, p_bars, p_lab)
    _legend(ax, p_frame, loc="upper center", bbox_to_anchor=(0.45, 1.075), ncol=6, size=14, handlelength=1,
            columnspacing=0.9)
    _ax_style(ax, p_frame)
    return fig


def render_underplaced(d: dict, progress: float = 1.0, highlight=None):
    t = underplaced(d["players"])
    p_frame, p_bars, p_lab = _phases(progress)
    fig, ax = _card(d, "WERE PLAYERS PLACED TOO LOW?",
                    "share of each placement rank winning 55%+ afterwards, and the rank points they gained",
                    f"players with {MIN_GAMES}+ recorded games after placing  ·  label = median rank points the 55%+ "
                    "players gained (7,000 = one rank)", p_frame, MIN_GAMES)
    x = np.arange(len(t))
    v = t["share55"].to_numpy(float)
    ax.bar(x, v * p_bars, 0.66, color=T.GREEN, zorder=3)
    for xi, vi, c, n in zip(x, v, t["climb55"], t["n"]):
        if p_lab > 0:
            ax.text(xi, vi * p_bars + max(v) * 0.02, f"{vi:.0f}%", ha="center", va="bottom", color=T.TEXT,
                    fontsize=T.px_to_pt(17), fontweight="bold", alpha=p_lab)
            if not np.isnan(c) and vi > max(v) * 0.18:
                ax.text(xi, vi * p_bars / 2, f"+{c * RANK_PTS / 1000:.1f}k\npts", ha="center", va="center", color=T.BG,
                        fontsize=T.px_to_pt(14), fontweight="bold", alpha=p_lab, linespacing=0.95)
    ax.set_xticks(x, [f"{RANK_TIER_NAMES[int(r)]}\n{int(n):,}" for r, n in zip(t["rank"], t["n"])], rotation=0)
    ax.yaxis.set_major_formatter(lambda v_, _: f"{v_:.0f}%")
    ax.set_ylim(0, max(max(v), 1) * 1.15)
    _label(ax, "placement rank  (players with 30+ games)", "x", p_frame)
    ax.grid(axis="x", visible=False)
    _ax_style(ax, p_frame)
    for tl in ax.get_xticklabels():
        tl.set_fontsize(T.px_to_pt(14))
    return fig


def render_rank_over_time(d: dict, progress: float = 1.0, highlight=None):
    w = d["weekly"]
    piv = w.pivot_table(index="week", columns="rank", values="players", aggfunc="sum", fill_value=0)
    piv = piv.reindex(columns=range(1, 12), fill_value=0)
    counts = pd.DataFrame({name: piv[list(ranks)].sum(axis=1) for name, ranks, _ in RANK_GROUPS})
    total = counts.sum(axis=1)
    shares = counts.div(total, axis=0) * 100
    p_frame, p_bars, p_lab = _phases(progress)
    fig, ax = _card(d, "HOW THE LADDER MOVED, WEEK BY WEEK",
                    f"active players: {int(total.iloc[0]):,} in week 1 → {int(total.iloc[-1]):,} in week "
                    f"{int(total.index[-1])}  ·  labels = players, share of that week",
                    "each player counted once per week they played, at their rank after their last match that week",
                    p_frame)
    weeks = counts.index.to_numpy()
    n_show = int(np.ceil(p_bars * len(weeks)))
    bottom = np.zeros(len(weeks))
    for (name, _, color), col in zip(RANK_GROUPS, counts.columns):
        v = counts[col].to_numpy(float)
        ax.bar(weeks[:n_show], v[:n_show], 0.78, bottom=bottom[:n_show], color=color, zorder=3, label=name)
        for xi, b, vi, s in zip(weeks[:n_show], bottom[:n_show], v[:n_show], shares[col].to_numpy()[:n_show]):
            if vi >= total.max() * 0.055:          # room for two lines (~30px)
                ax.text(xi, b + vi / 2, f"{_count(vi)}\n{s:.0f}%", ha="center", va="center", color=T.BG,
                        fontsize=T.px_to_pt(12), fontweight="bold", linespacing=1.0)
            elif vi >= total.max() * 0.028:
                ax.text(xi, b + vi / 2, f"{s:.0f}%", ha="center", va="center", color=T.BG,
                        fontsize=T.px_to_pt(12), fontweight="bold")
        bottom += v
    ax.set_xticks(weeks, [str(int(k)) for k in weeks])
    ax.set_xlim(weeks.min() - 0.6, weeks.max() + 0.6)
    ax.set_ylim(0, total.max() * 1.12)
    ax.yaxis.set_major_formatter(_k)
    _label(ax, "week of the season", "x", p_frame)
    _label(ax, "active players", "y", p_frame)
    ax.grid(axis="x", visible=False)
    _legend(ax, p_frame, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3, size=15, handlelength=1.2,
            columnspacing=1.2)
    _ax_style(ax, p_frame)
    return fig


def render_climb_by_wr(d: dict, progress: float = 1.0, highlight=None):
    t = outcomes_by_wr(d["players"])
    p_frame, p_bars, p_lab = _phases(progress)
    neg = qualified(d["players"])
    neg = neg[neg["wr"] < 50]
    neg_up = 100 * (neg["climb"] > 0).mean() if len(neg) else np.nan
    fig, ax = _card(d, "DO LOSING PLAYERS CLIMB?",
                    f"{neg_up:.0f}% of players under 50% still gained rank points after placements",
                    "rank points gained since placement (7,000 = one rank)  ·  win rate ±1%, from recorded matches",
                    p_frame, MIN_GAMES)
    x = np.arange(len(t))
    bottom = np.zeros(len(t))
    for col, color in zip(CLIMB_LABELS, CLIMB_COLORS):
        v = t[col].to_numpy(float)
        ax.bar(x, v * p_bars, 0.7, bottom=bottom * p_bars, color=color, zorder=3, label=col)
        for xi, b, vi in zip(x, bottom, v):
            if vi >= 5 and p_lab > 0:
                ax.text(xi, (b + vi / 2) * p_bars, f"{vi:.0f}%", ha="center", va="center", color=T.BG,
                        fontsize=T.px_to_pt(15), fontweight="bold", alpha=p_lab)
        bottom += v
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xticks(x, [f"{lab}\n{int(n):,}" for (lab, *_), n in zip(WR_WINDOWS, t["n"])])
    _label(ax, "season win rate  (players)", "x", p_frame)
    ax.grid(visible=False)
    _legend(ax, p_frame, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=5, size=15, handlelength=1,
            columnspacing=1)
    _ax_style(ax, p_frame)
    return fig


def render_climb_by_games(d: dict, progress: float = 1.0, highlight=None):
    t = climb_by_games(d["players"])
    p_frame, p_lines, _ = _phases(progress)
    fig, ax = _card(d, "DOES PLAYING MORE MAKE YOU CLIMB?",
                    "median rank points gained since placement, by games played, for each win rate",
                    f"win rate ±1%, games and win rate from recorded matches  ·  hollow = under {MIN_POINT_N} players",
                    p_frame, MIN_GAMES)
    x = np.arange(len(GAME_LABELS))
    for (label, *_), color in zip(WR_WINDOWS, WR_COLORS):
        g = t[t["window"] == label].set_index("bin").reindex(GAME_LABELS)
        y = g["median"].to_numpy(float) * RANK_PTS
        upto = int(np.floor(p_lines * (len(x) - 1))) + 1
        ax.plot(x[:upto], y[:upto], color=color, lw=4.5, zorder=3, solid_capstyle="round",
                path_effects=[T.pe.Stroke(linewidth=7, foreground=T.BG), T.pe.Normal()])
        for xi, yi, n in zip(x[:upto], y[:upto], g["n"].to_numpy()[:upto]):
            if not np.isnan(yi):
                ax.scatter([xi], [yi], s=120, zorder=4, color=color if n >= MIN_POINT_N else T.BG,
                           edgecolors=color, linewidths=2.5)
        ok = np.where(~np.isnan(y))[0]
        if len(ok) and p_lines > 0.95:
            ax.text(ok[-1] + 0.2, y[ok[-1]], label, color=color, fontsize=T.px_to_pt(18), fontweight="bold",
                    va="center", ha="left", path_effects=T.STROKE)
    med = t["median"].to_numpy(float) * RANK_PTS
    lo_y, hi_y = min(0.0, np.nanmin(med)), np.nanmax(med)
    lo_t, hi_t = int(np.floor(lo_y / RANK_PTS)) * RANK_PTS, int(np.ceil(hi_y / RANK_PTS)) * RANK_PTS
    ax.set_yticks(np.arange(lo_t, hi_t + 1, RANK_PTS))                      # one tick per rank (7,000)
    ax.set_ylim(lo_y * 1.15 if lo_y < 0 else 0, hi_y * 1.06)     # after the ticks, which would widen it
    ax.yaxis.set_major_formatter(lambda v, _: "0" if v == 0 else
                                 f"{v / 1000:+.0f}k\n{v / RANK_PTS:+.0f} rank" + ("s" if abs(v) > RANK_PTS else ""))
    ax.set_xlim(-0.4, len(x) - 1 + 1.1)
    ax.set_xticks(x, GAME_LABELS, rotation=35, ha="right", rotation_mode="anchor")
    _label(ax, "recorded ranked games this season", "x", p_frame)
    _label(ax, "rank points gained (median)", "y", p_frame)
    ax.grid(axis="both", visible=True, color=T.DIVIDER, lw=1.0, alpha=0.8 * p_frame, zorder=1)
    _ax_style(ax, p_frame)
    return fig


def render_points_kept(d: dict, sims: dict, progress: float = 1.0, highlight=None):
    p = d["players"]
    won, lost, kept = by_window(p, "points_won_100"), by_window(p, "points_lost_100"), by_window(p, "per100")
    sim = sims.get("with shields")
    sv = [float(sim["net100"].get(w, np.nan)) for _, w, _, _ in WR_WINDOWS] if sim is not None else None
    p_frame, p_bars, p_lab = _phases(progress)
    fig, ax = _card(d, "POINTS WON, LOST AND KEPT",
                    f"per 100 games: a 45% player wins {won[0]:,.0f}, loses {lost[0]:,.0f} and keeps {kept[0]:,.0f}",
                    "real players, recorded matches only, win rate ±1%" +
                    ("  ·  kept bar: blue = what the simulation keeps, gold on top = what real players keep beyond it"
                     if sv else ""), p_frame, MIN_GAMES)
    x = np.arange(len(WR_WINDOWS))
    bw = 0.26
    fs = T.px_to_pt(12)
    for k, (vals, label, color) in enumerate(((won, "won", T.GREEN), (lost, "lost", T.RED))):
        ax.bar(x + (k - 1) * bw, np.nan_to_num(vals) * p_bars, bw * 0.92, color=color, zorder=3, label=label)
        if p_lab > 0:
            for xi, v in zip(x, vals):
                if not np.isnan(v):
                    ax.text(xi + (k - 1) * bw, v * p_bars + 250, f"{v / 1000:.1f}k", ha="center", va="bottom",
                            color=color, fontsize=fs, fontweight="bold", alpha=p_lab)
    xk = x + bw
    if sv is None:
        ax.bar(xk, np.nan_to_num(kept) * p_bars, bw * 0.92, color=T.GOLD, zorder=3, label="kept")
    else:
        lower = np.fmin(kept, sv)
        upper = np.abs(np.subtract(kept, sv))
        top_color = [T.GOLD if r >= s_ else "#4dabf7" for r, s_ in zip(kept, sv)]
        base_color = ["#4dabf7" if r >= s_ else T.GOLD for r, s_ in zip(kept, sv)]
        ax.bar(xk, np.nan_to_num(lower) * p_bars, bw * 0.92, color=base_color, zorder=3)
        ax.bar(xk, np.nan_to_num(upper) * p_bars, bw * 0.92, bottom=np.nan_to_num(lower) * p_bars, color=top_color,
               zorder=3)
        ax.bar([np.nan], [0], color=T.GOLD, label="kept (real)")
        ax.bar([np.nan], [0], color="#4dabf7", label="kept (simulation)")
        if p_lab > 0:
            for xi, r, s_ in zip(xk, kept, sv):
                if np.isnan(r) or np.isnan(s_):
                    continue
                hi_v, lo_v = max(r, s_), min(r, s_)
                hi_c, lo_c = (T.GOLD, "#4dabf7") if r >= s_ else ("#4dabf7", T.GOLD)
                ax.text(xi, hi_v * p_bars + 250, f"{hi_v / 1000:.1f}k", ha="center", va="bottom", color=hi_c,
                        fontsize=fs, fontweight="bold", alpha=p_lab)
                if lo_v > 900:
                    ax.text(xi, lo_v * p_bars / 2, f"{lo_v / 1000:.1f}k", ha="center", va="center", color=T.BG,
                            fontsize=fs, fontweight="bold", alpha=p_lab, rotation=90)
    ax.axhline(0, color=T.DIVIDER, lw=1.2, zorder=2)
    _window_ticks(ax, x)
    ax.yaxis.set_major_formatter(_k)
    _label(ax, "rank points per 100 games  (7,000 = one rank)", "y", p_frame)
    ax.grid(axis="x", visible=False)
    _legend(ax, p_frame, loc="upper left", size=16)
    _ax_style(ax, p_frame)
    return fig


def render_shields(d: dict, sims: dict, progress: float = 1.0, highlight=None):
    p = d["players"]
    q = qualified(p)
    real100, saved100 = [], []
    for _, _, lo, hi in WR_WINDOWS:
        g = in_window(q, lo, hi)
        games = g["games"].sum()
        real100.append(100 * g["shields_used"].sum() / games if games else np.nan)
        saved100.append(100 * g["points_saved"].sum() / games if games else np.nan)
    sim = sims.get("with shields")
    has_sim = sim is not None and "shields100" in sim
    p_frame, p_bars, p_lab = _phases(progress)
    lo_ = d["losses"]
    avg = lo_["points_saved"].sum() / max(lo_["shielded"].sum(), 1)
    sim_avg = (sim["protected"].sum() / sim["shields_sum"].sum()) if has_sim else None
    fig, ax = _card(d, "WHAT SHIELDS ACTUALLY DID",
                    f"a shield saved {avg:.0f} points on average" +
                    (f"  ·  the simulation: {sim_avg:.0f}" if sim_avg else ""),
                    "bars = shields used per 100 games  ·  label = points they saved per 100 games (a loss takes 300; "
                    "a shield only takes what's left in the tier)", p_frame, MIN_GAMES)
    x = np.arange(len(WR_WINDOWS))
    series = [(real100, saved100, "real players", T.GOLD)]
    if has_sim:
        series.append(([float(sim["shields100"].get(w, np.nan)) for _, w, _, _ in WR_WINDOWS],
                       [float(sim["shields100"].get(w, np.nan) * sim["per_shield"].get(w, np.nan))
                        for _, w, _, _ in WR_WINDOWS], "simulation", "#4dabf7"))
    bw = 0.8 / len(series)
    for k, (vals, pts, label, color) in enumerate(series):
        xs = x - 0.4 + bw * (k + 0.5)
        ax.bar(xs, np.nan_to_num(vals) * p_bars, bw * 0.92, color=color, zorder=3, label=label)
        if p_lab > 0:
            for xi, v, s_ in zip(xs, vals, pts):
                if not np.isnan(v):
                    ax.text(xi, v * p_bars + 0.15, f"{v:.1f}", ha="center", va="bottom", color=color,
                            fontsize=T.px_to_pt(14), fontweight="bold", alpha=p_lab)
                    if not np.isnan(s_) and v > 2:
                        ax.text(xi, v * p_bars / 2, f"{s_ / 1000:.1f}k\npts", ha="center", va="center",
                                color=T.BG, fontsize=T.px_to_pt(13), fontweight="bold", alpha=p_lab,
                                linespacing=0.9)
    top = np.nanmax([np.nanmax(s_[0]) for s_ in series])
    ax.set_ylim(0, top * 1.2)
    _window_ticks(ax, x)
    _label(ax, "shields used per 100 games", "y", p_frame)
    ax.grid(axis="x", visible=False)
    _legend(ax, p_frame, loc="upper right", size=16)
    _ax_style(ax, p_frame)
    return fig


def render_below_peak(d: dict, sims: dict, progress: float = 1.0, highlight=None):
    real = [100 * v for v in by_window(d["players"], "below_peak")]
    series = [("real players", real, T.GOLD)]
    for k in ("with shields", "no shields"):
        if k in sims:
            series.append((f"sim, {k}", [float(sims[k]["below_peak"].get(w, np.nan)) for _, w, _, _ in WR_WINDOWS],
                           SIM_COLORS[k]))
    q = qualified(d["players"])
    overall = 100 * q["below_peak"].mean() if len(q) else np.nan
    p_frame, p_bars, p_lab = _phases(progress)
    fig, ax = _card(d, "FINISHED BELOW THEIR BEST",
                    f"{overall:.0f}% of players ended the season below the highest rank they reached",
                    "below = a lower tier at their last match than at their peak  ·  sims: 210-game seasons; "
                    "real players have fewer recorded games", p_frame, MIN_GAMES)
    x = np.arange(len(WR_WINDOWS))
    bw = 0.8 / len(series)
    for k, (label, vals, color) in enumerate(series):
        xs = x - 0.4 + bw * (k + 0.5)
        ax.bar(xs, np.nan_to_num(vals) * p_bars, bw * 0.92, color=color, zorder=3, label=label)
        if p_lab > 0:
            for xi, v in zip(xs, vals):
                if not np.isnan(v):
                    ax.text(xi, v * p_bars + 1, f"{v:.0f}", ha="center", va="bottom", color=color,
                            fontsize=T.px_to_pt(13), fontweight="bold", alpha=p_lab)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    _window_ticks(ax, x)
    ax.grid(axis="x", visible=False)
    _legend(ax, p_frame, loc="upper right", size=16)
    _ax_style(ax, p_frame)
    return fig


def render_sim_vs_real(d: dict, sim: pd.DataFrame, progress: float = 1.0, highlight=None):
    p_frame, p_bars, p_lab = _phases(progress)
    series = [("real", "real players", T.GOLD)] + [(k, f"sim, {k}", SIM_COLORS[k]) for k in SIM_FILES if k in sim]
    r45 = sim["real"].iloc[0]
    s45 = sim["with shields"].iloc[0] if "with shields" in sim else np.nan
    sub = "rank points gained per 100 games, by win rate"
    if not np.isnan(r45) and not np.isnan(s45):
        sub = f"at 45%, real players keep {r45:,.0f} points per 100 games; the simulation keeps {s45:,.0f}"
    fig, ax = _card(d, "THE SIMULATION VS REAL PLAYERS", sub,
                    "real: ±1% win rate, recorded matches only  ·  sims: 210-game seasons, placed like the real ladder",
                    p_frame, MIN_GAMES)
    x = np.arange(len(sim))
    bw = 0.8 / len(series)
    for k, (col, label, color) in enumerate(series):
        v = sim[col].to_numpy(float)
        xs = x - 0.4 + bw * (k + 0.5)
        ax.bar(xs, np.nan_to_num(v) * p_bars, bw * 0.92, color=color, zorder=3, label=label)
    ax.axhline(0, color=T.DIVIDER, lw=1.2, zorder=2)
    _window_ticks(ax, x)
    vals = sim[[c for c, _, _ in series]].to_numpy(float)
    lo_y, hi_y = min(0, np.nanmin(vals)), np.nanmax(vals)
    lo_t, hi_t = int(np.floor(lo_y / RANK_PTS)) * RANK_PTS, int(np.ceil(hi_y / RANK_PTS)) * RANK_PTS
    ax.set_yticks(np.arange(lo_t, hi_t + 1, RANK_PTS))
    ax.set_ylim(lo_y * 1.15 if lo_y < 0 else 0, hi_y * 1.08)     # after the ticks, which would widen it
    ax.yaxis.set_major_formatter(lambda v, _: "0" if v == 0 else
                                 f"{v / 1000:+.0f}k\n{v / RANK_PTS:+.0f} rank" + ("s" if abs(v) > RANK_PTS else ""))
    for k_ in range(int(np.floor(lo_y / 1000)), int(np.ceil(hi_y / 1000)) + 1):
        if k_ % 2 and (k_ * 1000) % RANK_PTS:
            ax.axhline(k_ * 1000, color=T.DIVIDER, lw=0.8, ls=(0, (4, 4)), alpha=0.55 * p_frame, zorder=1)
    _label(ax, "rank points per 100 games", "y", p_frame)
    ax.grid(axis="x", visible=False)
    _legend(ax, p_frame, loc="upper left", size=16)
    _ax_style(ax, p_frame)
    return fig


def render_games_per_rank(d: dict, sim: pd.DataFrame, progress: float = 1.0, highlight=None):
    """Table: games needed per tier (1,000 points) and per rank (7,000) at each win rate, real vs the sims."""
    cols = [("real", "REAL PLAYERS", T.GOLD)] + [(k, f"SIM, {k.upper()}", SIM_COLORS[k]) for k in SIM_FILES if k in sim]
    p_frame = T.ease_out_cubic(T.phase(progress, 0, 0.15))
    fig, ax = T.new_table_card()
    r45 = sim["real"].iloc[0]
    s45 = sim["with shields"].iloc[0] if "with shields" in sim else np.nan
    sub = "games needed to climb, from the points kept per 100 games"
    if r45 > 0 and s45 > 0:
        sub = (f"at 45%: one rank every ~{100 * RANK_PTS / r45:.0f} games for real players, "
               f"~{100 * RANK_PTS / s45:.0f} in the simulation")
    T.draw_header(fig, "HOW MANY GAMES TO CLIMB?", sub, alpha=p_frame)
    T.draw_footer(fig, chart_meta(d, MIN_GAMES), alpha=p_frame)
    _note(fig, "tier = 1,000 points (tier VI is 2,000)  ·  rank = 7,000  ·  real: ±1% win rate, recorded matches only",
          p_frame)
    n_rows = len(sim) + 2
    ax.set_ylim(n_rows, 0)
    ax.set_xlim(0, 1)
    x0, col_w = 0.17, (1 - 0.17) / len(cols)
    head_px, cell_px = 17, 26
    for j, (_, name, color) in enumerate(cols):
        cx = x0 + col_w * (j + 0.5)
        ax.text(cx, 0.45, name, ha="center", va="center", color=color, fontsize=T.px_to_pt(head_px),
                fontweight="bold", alpha=p_frame)
        ax.text(cx, 1.15, "per tier  ·  per rank", ha="center", va="center", color=T.TEXT_MUTED,
                fontsize=T.px_to_pt(14), alpha=p_frame)
    for i, (_, r) in enumerate(sim.iterrows()):
        a = T.ease_out_cubic(T.phase(progress, 0.15 + 0.1 * i, 0.35 + 0.1 * i))
        y = i + 2
        if i % 2 == 0:
            ax.add_patch(plt_rect(-0.01, y - 0.5, 1.02, 1, T.PANEL, a))
        ax.text(0.01, y, f"{int(r['wr'])}%", ha="left", va="center", color=T.TEXT, fontsize=T.px_to_pt(cell_px),
                fontweight="bold", alpha=a)
        for j, (col, _, color) in enumerate(cols):
            v = r[col]
            txt = "never" if not v > 0 else f"{100 * 1000 / v:.0f}  ·  {100 * RANK_PTS / v:,.0f}"
            ax.text(x0 + col_w * (j + 0.5), y, txt, ha="center", va="center", color=color if v > 0 else T.RED,
                    fontsize=T.px_to_pt(cell_px), fontweight="bold", alpha=a)
    return fig


def plt_rect(x, y, w, h, color, alpha):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=alpha, zorder=0)


def missing_stats(p: pd.DataFrame) -> dict:
    q = qualified(p)
    trans, gaps = q["transitions"].sum(), q["gaps"].sum()
    est_missing = np.ceil(q["gap_points"] / POINTS_PER_LOSS).clip(lower=q["gaps"])   # at least 1 per gap
    share = 100 * est_missing / (est_missing + q["games"])
    out = {"gap_rate": 100 * gaps / trans if trans else np.nan, "share": share,
           "median": float(share.median()), "overall": 100 * est_missing.sum() / (est_missing.sum() + q["games"].sum()),
           "n": len(q)}
    if "gap_net" in q:
        single = q["gap_win1"].sum() + q["gap_loss1"].sum()
        out |= {"net_median": float(q["gap_net"].median()), "net_mean": float(q["gap_net"].mean()),
                "per_missing": float(q["gap_net"].sum() / max(est_missing.sum(), 1)),
                "single_win_share": 100 * q["gap_win1"].sum() / single if single else np.nan,
                "missing_win_share": 100 * q["missing_wins"].sum() / max(q["est_missing"].sum(), 1),
                "recorded_wr": 100 * q["wins"].sum() / max(q["decided"].sum(), 1)}
    return out


def skew_table(p: pd.DataFrame) -> pd.DataFrame:
    q = qualified(p)
    rows = []
    for label, w, lo, hi in WR_WINDOWS:
        g = in_window(q, lo, hi)
        rows.append({"window": label, "wr": w, "n": len(g), "recorded": g["wr"].mean(),
                     "estimated": g["wr_adj"].median(), "missing_share": (100 * g["est_missing"] /
                     (g["est_missing"] + g["games"])).median(), "missing_net": g["gap_net"].median()})
    return pd.DataFrame(rows)


def render_missing_skew(d: dict, progress: float = 1.0, highlight=None):
    m = missing_stats(d["players"])
    t = skew_table(d["players"])
    t["gap"] = t["estimated"] - t["wr"]
    p_frame, p_bars, p_lab = _phases(progress)
    first = t.iloc[0]
    fig, ax = _card(d, "ARE PLAYERS BETTER THAN THEIR STATS?",
                    f"{m['missing_win_share']:.0f}% of the missing games were wins  ·  a {first['window']} player in "
                    f"the data is about {first['estimated']:.1f}% for real",
                    "bar = estimated real win rate minus the recorded one (median player)  ·  missing games' wins "
                    "estimated from the rank points they moved", p_frame, MIN_GAMES)
    x = np.arange(len(t))
    v = t["gap"].to_numpy(float)
    ax.bar(x, np.nan_to_num(v) * p_bars, 0.62, color=[T.GOLD if g >= 0 else T.RED for g in v], zorder=3)
    span = max(np.nanmax(np.abs(v)), 0.5)
    if p_lab > 0:
        for xi, g, est in zip(x, v, t["estimated"]):
            if not np.isnan(g):
                ax.text(xi, g * p_bars + (span * 0.03 if g >= 0 else -span * 0.03), f"{g:+.1f}",
                        ha="center", va="bottom" if g >= 0 else "top", color=T.GOLD if g >= 0 else T.RED,
                        fontsize=T.px_to_pt(20), fontweight="bold", alpha=p_lab)
    ax.axhline(0, color=T.DIVIDER, lw=1.2, zorder=2)
    ax.set_ylim(0 if np.nanmin(v) >= 0 else np.nanmin(v) - span * 0.15, max(0, np.nanmax(v)) + span * 0.2)
    ax.set_xticks(x, [f"{lab}\n→ {e:.1f}%" for lab, e in zip(t["window"], t["estimated"])])
    _label(ax, "recorded win rate  →  estimated real win rate", "x", p_frame)
    ax.yaxis.set_major_formatter(lambda v_, _: f"{v_:+.1f}")
    _label(ax, "win-rate points missing from the public data", "y", p_frame)
    ax.grid(axis="x", visible=False)
    _ax_style(ax, p_frame)
    return fig


MISSING_X_MAX = 80          # missing-matches histogram stops here; the last bar collects everyone above


def render_missing(d: dict, progress: float = 1.0, highlight=None):
    m = missing_stats(d["players"])
    p_frame, p_bars, p_lab = _phases(progress)
    fig, ax = _card(d, "HOW MANY MATCHES ARE MISSING?",
                    f"at least {m['overall']:.0f}% of ranked games aren't in the public data",
                    f"gap = rank going in ≠ rank after the last recorded match ({m['gap_rate']:.0f}% of them)  ·  "
                    "missing points ÷ 300 = games  ·  a floor: a missed win + loss cancel  ·  "
                    f"last bar includes everyone above {MISSING_X_MAX}%", p_frame, MIN_GAMES)
    share = m["share"].clip(upper=MISSING_X_MAX - 0.1)       # the last bar holds everyone at 75% or more
    edges = np.arange(0, MISSING_X_MAX + 1, 5)
    counts, _ = np.histogram(share, edges)
    pct = 100 * counts / max(counts.sum(), 1)
    ax.bar(edges[:-1] + 2.5, pct * p_bars, 4.4, color=T.GOLD, zorder=3)
    if p_lab > 0:
        ax.axvline(m["median"], color=T.TEXT, lw=2, ls="--", zorder=4, alpha=p_lab)
        ax.text(m["median"] + 1, max(pct) * 1.02, f"median player:\n{m['median']:.0f}% missing", color=T.TEXT,
                fontsize=T.px_to_pt(17), fontweight="bold", va="top", ha="left", alpha=p_lab,
                path_effects=T.STROKE)
    ax.set_xlim(0, MISSING_X_MAX)
    ax.set_ylim(0, max(pct) * 1.12 + 0.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    _label(ax, "share of a player's ranked games missing from the data (at least)", "x", p_frame)
    _label(ax, "share of players", "y", p_frame)
    ax.grid(axis="x", visible=False)
    _ax_style(ax, p_frame)
    return fig


def render_protection(d: dict, progress: float = 1.0, highlight=None):
    t = d["protection"].copy()
    t["demoted%"] = 100 * t["demoted"] / t["losses"]
    t["at_zero"] = t["at_zero"].astype(str).str.lower().isin(["true", "t", "1"])
    p_frame, p_bars, p_lab = _phases(progress)
    z = t[t["at_zero"] & (t["shields_held"] == 0)]
    z_rate = 100 * z["demoted"].sum() / max(z["losses"].sum(), 1)
    fig, ax = _card(d, "WHEN DOES A LOSS ACTUALLY DEMOTE YOU?",
                    f"with no shields left, {z_rate:.0f}% of losses at a tier's 0 points dropped a tier",
                    "losses of 300 that would drop below the tier  ·  demotions at 0 points are missing from the "
                    "match record: rebuilt from the player's next match", p_frame)
    x = np.arange(6)
    groups = [(False, "1–299 points into the tier", T.RED), (True, "exactly 0 points", T.GOLD)]
    bw = 0.38
    for k, (zero, label, color) in enumerate(groups):
        g = t[t["at_zero"] == zero].set_index("shields_held").reindex(range(6))
        v = g["demoted%"].to_numpy(float)
        xs = x + (k - 0.5) * bw
        ax.bar(xs, np.nan_to_num(v) * p_bars, bw * 0.92, color=color, zorder=3, label=label)
        if p_lab > 0:
            for xi, vi, n in zip(xs, v, g["losses"].to_numpy()):
                if not np.isnan(vi):
                    ax.text(xi, vi * p_bars + 1.5, f"{vi:.0f}%", ha="center", va="bottom", color=color,
                            fontsize=T.px_to_pt(15), fontweight="bold", alpha=p_lab)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xticks(x, [str(k) if k < 5 else "5+" for k in x])
    _label(ax, "shields held going into the loss", "x", p_frame)
    _label(ax, "losses that demoted", "y", p_frame)
    ax.grid(axis="x", visible=False)
    _legend(ax, p_frame, loc="upper right", size=16)
    _ax_style(ax, p_frame)
    return fig


def find_sim_dir(data_dir: Path, sim_dir: Path | None = None) -> Path | None:
    for c in ([Path(sim_dir)] if sim_dir else []) + [Path(data_dir) / "sim_results", T.PROJECT_ROOT / "sim_results"]:
        if any((c / f).exists() for f in SIM_FILES.values()):
            return c
    return None


def cards(data_dir: Path, sim_dir: Path | None = None, **_) -> dict:
    """{card name: draw(progress=1.0, highlight=None)} for a season folder (made by `export`)."""
    data_dir = Path(data_dir)
    if not (data_dir / "season_meta.json").exists():
        raise FileNotFoundError(data_dir / "season_meta.json")
    d = load(data_dir)
    sims = load_sims(find_sim_dir(data_dir, sim_dir))
    out = {"season_numbers": lambda progress=1.0, highlight=None: render_numbers(d, progress),
           "season_rank_distribution": lambda progress=1.0, highlight=None: render_rank_distribution(d, progress),
           "season_placement_outcomes": lambda progress=1.0, highlight=None: render_placement_outcomes(d, progress),
           "season_underplaced": lambda progress=1.0, highlight=None: render_underplaced(d, progress),
           "season_rank_over_time": lambda progress=1.0, highlight=None: render_rank_over_time(d, progress),
           "season_climb_by_winrate": lambda progress=1.0, highlight=None: render_climb_by_wr(d, progress),
           "season_climb_by_games": lambda progress=1.0, highlight=None: render_climb_by_games(d, progress)}
    if has(d, "points_won", "points_lost"):
        out["season_points_kept"] = lambda progress=1.0, highlight=None: render_points_kept(d, sims, progress)
    out["season_shields"] = lambda progress=1.0, highlight=None: render_shields(d, sims, progress)

    if has(d, "peak_progress"):
        out["season_below_peak"] = lambda progress=1.0, highlight=None: render_below_peak(d, sims, progress)
    sim = sim_table(sims, d["players"])
    if sim is not None:
        out["season_sim_vs_real"] = lambda progress=1.0, highlight=None: render_sim_vs_real(d, sim, progress)
        out["season_games_per_rank"] = lambda progress=1.0, highlight=None: render_games_per_rank(d, sim, progress)
    if has(d, "gaps", "transitions"):
        out["season_missing_matches"] = lambda progress=1.0, highlight=None: render_missing(d, progress)
    if has(d, "gap_net", "wr_adj"):
        out["season_missing_skew"] = lambda progress=1.0, highlight=None: render_missing_skew(d, progress)
    return out


def summary(data_dir: Path, sim_dir: Path | None = None) -> str:
    """The numbers behind the cards, as text (for writing the script)."""
    d = load(data_dir)
    p = d["players"]
    n = numbers(d)
    sims = load_sims(find_sim_dir(data_dir, sim_dir))
    lines = [f"{n['players']:,} players, {n['matches']:,} matches ({d['meta']['date_range']})",
             f"{n['placed_n']:,} players with recorded placements: {n['climbed']:.1f}% climbed, {n['fell']:.1f}% fell",
             f"{n['shielded']:.1f}% of losses shielded; {n['saved_per_shield']:.0f} points saved per shield",
             f"games per player: median {p['games'].median():.0f}, max {p['games'].max():,}; "
             f"{len(qualified(p)):,} players with {MIN_GAMES}+ recorded games",
             "", "Outcomes by win rate (% of players):", outcomes_by_wr(p).round(1).to_string(index=False),
             "", "Median climb by games played:",
             climb_by_games(p).pivot(index="bin", columns="window", values="median").reindex(GAME_LABELS)
             .round(2).to_string(),
             "", "After placements (% of players placed in each rank):",
             placement_outcomes(p).round(1).to_string(index=False),
             "", "Placed too low? (players with 30+ games):", underplaced(p).round(2).to_string(index=False),
             "", f"Started placements, no result and no ranked match after: {int(p['unfinished'].sum()):,}"]
    if d["protection"] is not None:
        lines += ["", "Losses that would drop below the tier (shields held, at 0 points):",
                  d["protection"].round(1).to_string(index=False)]
    if has(d, "points_won"):
        t = pd.DataFrame({"window": [w[0] for w in WR_WINDOWS], "won100": by_window(p, "points_won_100"),
                          "lost100": by_window(p, "points_lost_100"), "kept100": by_window(p, "per100"),
                          "shields100": by_window(p, "shields_used_100"),
                          "below_peak%": [100 * v for v in by_window(p, "below_peak")]})
        lines += ["", "Per 100 games by win rate (real):", t.round(1).to_string(index=False)]
    if has(d, "gaps"):
        m = missing_stats(p)
        lines += ["", f"Missing matches: gaps in {m['gap_rate']:.1f}% of transitions; at least {m['overall']:.1f}% "
                      f"of games missing overall; median player {m['median']:.1f}%"]
        if "net_median" in m:
            lines += [f"Points from missing matches per player: median {m['net_median']:+,.0f}, mean "
                      f"{m['net_mean']:+,.0f}; {m['per_missing']:+.0f} per missing game",
                      f"Single missing games we can classify: {m['single_win_share']:.1f}% wins;  all missing games "
                      f"(estimated): {m['missing_win_share']:.1f}% wins  vs  {m['recorded_wr']:.1f}% in recorded games",
                      "Recorded vs estimated real win rate (players with 30+ games):",
                      skew_table(p).round(2).to_string(index=False)]
    lines += ["", "Most common rank changes (desired change, outcome, shield):", d["changes"].head(12).to_string(index=False)]
    sim = sim_table(sims, p)
    if sim is not None:
        lines += ["", "Points per 100 games, real vs sims:", sim.round(0).to_string(index=False)]
    for k, g in sims.items():
        cols = [c for c in ("net100", "won100", "lost100", "shields100", "per_shield", "below_peak") if c in g]
        lines += ["", f"Sim ({k}):", g[cols].round(1).to_string()]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Season summary: export from PostgreSQL, then cards")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--start", default=SEASON_START.date().isoformat())
    e.add_argument("--end", default=None, help="YYYY-MM-DD, exclusive (default: now)")
    e.add_argument("--name", default="season_1")
    c = sub.add_parser("cards")
    c.add_argument("season_dir", nargs="?", default=None)
    c.add_argument("--sim", type=Path, default=None, help="folder with the Monte Carlo parquet results")
    c.add_argument("--progress", type=float, default=1.0)
    args = ap.parse_args()
    if args.cmd == "export":
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end
               else datetime.now(timezone.utc))
        export(start, end, T.PROJECT_ROOT / args.name)
        print("\n" + summary(T.PROJECT_ROOT / args.name))
        return 0
    data_dir = Path(args.season_dir) if args.season_dir else T.PROJECT_ROOT / "season_1"
    for old in (data_dir / "graphs").glob("season_*.png"):        # cards that no longer exist don't linger
        old.unlink()
    for p in T.save_cards(cards(data_dir, args.sim), data_dir / "graphs", args.progress):
        print(f"Saved {p}")
    print("\n" + summary(data_dir, args.sim))
    return 0


if __name__ == "__main__":
    sys.exit(main())