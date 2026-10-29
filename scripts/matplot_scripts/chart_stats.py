"""
chart_stats.py — shared statistics for charts: sample thresholds, confidence
intervals, pick rate. Keep chart scripts free of ad-hoc math.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLAYERS_PER_MATCH = 12      # 6v6
MIN_N = 1000                # below this a win rate is drawn as "insufficient sample"


def wilson_ci(wins, n, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for a win proportion. Returns (lo, hi) in 0..1."""
    wins = np.asarray(wins, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = wins / n
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def add_winrate_ci(df: pd.DataFrame, wr_col: str = "winrate", n_col: str = "total_matches") -> pd.DataFrame:
    """Adds wr_lo / wr_hi (percent) and a `sufficient` flag. wr_col is in percent."""
    out = df.copy()
    wins = out[wr_col] / 100.0 * out[n_col]
    lo, hi = wilson_ci(wins, out[n_col])
    out["wr_lo"] = lo * 100
    out["wr_hi"] = hi * 100
    out["sufficient"] = out[n_col] >= MIN_N
    return out


def total_matches_from_hero_rows(hero_matches: pd.Series) -> int:
    """Every match has 12 hero slots, so sum(hero appearances) / 12 = matches."""
    return int(round(hero_matches.sum() / PLAYERS_PER_MATCH))


def logit(p_pct):
    p = np.clip(np.asarray(p_pct, dtype=float) / 100.0, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inv_logit_pct(x):
    return 100.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def expected_pair_wr(wr_a_pct, wr_b_pct):
    """Win rate expected for two heroes on the same team if they had no special synergy:
    each hero's edge over 50% is added on the log-odds scale."""
    return inv_logit_pct(logit(wr_a_pct) + logit(wr_b_pct))


def load_hero_meta(data_dir) -> pd.DataFrame:
    """hero_winrates.csv with normalized columns: hero_name, hero_type, total_matches, winrate.
    Adds ban_rate (percent of matches with recorded bans) from hero_bans.csv when it exists."""
    from pathlib import Path
    df = pd.read_csv(Path(data_dir) / "hero_winrates.csv")
    df.columns = df.columns.str.strip().str.lower()
    df["hero_type"] = df["hero_type"].str.strip().str.lower()
    bans = Path(data_dir) / "hero_bans.csv"
    if bans.exists():
        b = pd.read_csv(bans)
        df = df.merge(b[["hero_name", "ban_rate"]], on="hero_name", how="left")
    return df


# =============================================================================
# MATCH DURATION  (single source of truth for "early" / "late" across all charts)
# =============================================================================
DURATION_BRACKETS = ["20-25m", "25-30m", "30-35m", "35-40m", "40-45m", "45-50m", "50-55m", "55-60m", "60m+"]
BRACKET_W = 5                     # minutes per bracket in the bucketed exports
OPEN_END_MIN = 75                 # assumed upper edge of the open "60m+" bracket for interpolation


def _bracket_edges(label: str) -> tuple[float, float]:
    s = label.replace("m", "").replace("+", "")
    if "-" in s:
        a, b = s.split("-")
        return float(a), float(b)
    return float(s), float(OPEN_END_MIN)


def duration_histogram(data_dir) -> tuple[pd.DataFrame, str]:
    """Games per minute bin: columns start_min, end_min, games.
    Uses match_duration_minutes.csv (1-minute bins, exact) when present,
    otherwise falls back to the 5-minute avg_duration.csv (estimate)."""
    from pathlib import Path
    d = Path(data_dir)
    exact = d / "match_duration_minutes.csv"
    if exact.exists():
        df = pd.read_csv(exact)
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns={"duration_min": "start_min"})
        df["end_min"] = df["start_min"] + 1
        return df[["start_min", "end_min", "games"]].sort_values("start_min").reset_index(drop=True), "exact"
    df = pd.read_csv(d / "avg_duration.csv")
    df.columns = df.columns.str.strip().str.lower()
    edges = df["duration_bracket"].map(_bracket_edges)
    df["start_min"] = edges.map(lambda e: e[0])
    df["end_min"] = edges.map(lambda e: e[1])
    df = df.rename(columns={"total_games": "games"})
    return df[["start_min", "end_min", "games"]].sort_values("start_min").reset_index(drop=True), "buckets"


def histogram_quantile(hist: pd.DataFrame, q: float) -> float:
    """Quantile of match length from binned counts (linear within the bin it falls in)."""
    cum = hist["games"].cumsum().to_numpy(float)
    target = q * cum[-1]
    i = int(np.searchsorted(cum, target))
    prev = cum[i - 1] if i > 0 else 0.0
    r = hist.iloc[i]
    frac = (target - prev) / r["games"] if r["games"] else 0.0
    return float(r["start_min"] + frac * (r["end_min"] - r["start_min"]))


# How "early game" and "late game" are split for the hero-scaling charts:
#   "median"    -> early = games shorter than the median match, late = longer (50% / 50%)
#   "quartiles" -> early = shorter than Q1, late = longer than Q3 (extremes only, ~25% each)
# Cut-offs are exact minutes, not rounded to the 5-min brackets (see bracket_weights).
PHASE_SPLIT = "median"
SHORT_GAME_MIN, LONG_GAME_MIN = 30, 45     # fixed thresholds for the "% of short / long games" stats


def games_between(hist: pd.DataFrame, a: float, b: float) -> float:
    """Games with length in [a, b), assuming games spread evenly inside each histogram bin."""
    lo, hi = hist["start_min"].to_numpy(float), hist["end_min"].to_numpy(float)
    overlap = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None) / (hi - lo)
    return float((overlap * hist["games"].to_numpy(float)).sum())


def bracket_weights(hist: pd.DataFrame, cut: float, side: str) -> dict[str, float]:
    """For each 5-min hero bracket, the share of its games that fall below (side='below') or
    above (side='above') `cut`. The bracket containing the cut gets a fraction (e.g. 0.42),
    taken from the per-minute match-length histogram, so no rounding to bracket edges."""
    out = {}
    for b in DURATION_BRACKETS:
        lo, hi = _bracket_edges(b)
        total = games_between(hist, lo, hi)
        below = games_between(hist, lo, min(max(cut, lo), hi))
        frac = below / total if total else float(cut >= hi)
        out[b] = frac if side == "below" else 1.0 - frac
    return out


def phase_cuts(s: dict, mode: str = PHASE_SPLIT) -> tuple[float, float]:
    return (s["median"], s["median"]) if mode == "median" else (s["q1"], s["q3"])


def duration_summary(data_dir, mode: str = PHASE_SPLIT) -> dict:
    """Median, quartiles (IQR), 12.5/87.5 (middle 75%) and 10/90 percentiles of match length,
    plus the early/late cut-offs every duration chart uses (see PHASE_SPLIT)."""
    hist, source = duration_histogram(data_dir)
    q = {k: histogram_quantile(hist, v) for k, v in
         {"p10": 0.10, "p125": 0.125, "q1": 0.25, "median": 0.50, "q3": 0.75,
          "p875": 0.875, "p90": 0.90}.items()}
    q["mid75"] = q["p875"] - q["p125"]             # width of the middle 75% of matches
    s = {**q, "iqr": q["q3"] - q["q1"], "hist": hist, "source": source, "mode": mode}
    total = float(hist["games"].sum())
    early_cut, late_cut = phase_cuts(s, mode)
    end = float(hist["end_min"].max())
    start = float(hist["start_min"].min())
    s.update({
        "n": int(total), "early_cut": early_cut, "late_cut": late_cut,
        "early_weights": bracket_weights(hist, early_cut, "below"),
        "late_weights": bracket_weights(hist, late_cut, "above"),
        "early_share": games_between(hist, start, early_cut) / total * 100,
        "late_share": games_between(hist, late_cut, end) / total * 100,
        "short_share": games_between(hist, start, SHORT_GAME_MIN) / total * 100,
        "long_share": games_between(hist, LONG_GAME_MIN, end) / total * 100,
        # brackets that overlap the middle 50% of matches (for the heatmaps' highlight box)
        "iqr_brackets": [b for b in DURATION_BRACKETS
                         if _bracket_edges(b)[1] > q["q1"] and _bracket_edges(b)[0] < q["q3"]],
    })
    return s


def add_pick_rate(df: pd.DataFrame, n_col: str = "total_matches") -> pd.DataFrame:
    """Pick (presence) rate: % of matches the hero appears in, on either team."""
    out = df.copy()
    out["pick_rate"] = out[n_col] / total_matches_from_hero_rows(out[n_col]) * 100
    return out


# =============================================================================
# TIERS  (single source of truth: balance chart zones, tier list, insights)
# =============================================================================
# (lower bound inclusive, upper bound exclusive, name), worst to best, win rate in percent
TIER_BANDS = [(-np.inf, 46, "TRASH"), (46, 48, "WEAK"), (48, 49, "UNDER"), (49, 51, "BALANCED"),
              (51, 52, "OVER"), (52, 54, "STRONG"), (54, np.inf, "GOD TIER")]
TIER_NAMES = [name for _, _, name in TIER_BANDS]


def tier_index(wr: float) -> int:
    """0 = TRASH ... 6 = GOD TIER."""
    for i, (lo, hi, _) in enumerate(TIER_BANDS):
        if lo <= wr < hi:
            return i
    return len(TIER_BANDS) - 1


def tier_of(wr: float) -> str:
    return TIER_NAMES[tier_index(wr)]


# =============================================================================
# POWER SCORE  (what the tier list ranks by)
# =============================================================================
# Tier lists are about meta impact: a strong hero that is everywhere is oppressive, and those
# are the heroes that get nerfed. Power scales a hero's win-rate edge by how present it is in
# the meta, relative to the week's median hero:
#     presence = pick rate + BAN_WEIGHT * ban rate
#     power    = (win rate - 50) * (presence / median presence) ** POWER_ALPHA
# A hero exactly as present as the median hero scores its win rate - 50, so power uses the same
# tier cutoffs as win rate, shifted by 50 (STRONG = +2 to +4 points).
# POWER_ALPHA: 0 = pure win rate, 1 = pure "extra wins it adds across all games".
# BAN_WEIGHT: bans are recorded only for matches whose replay the API processed (15-50% of
# ranked), so ban rates are an estimate from that sample and count at a reduced weight. Raise it
# once ban coverage (validate_week.py, period_meta.json) is reliably high.
POWER_ALPHA = 0.9
BAN_WEIGHT = 0.25


def add_power(df: pd.DataFrame, alpha: float = POWER_ALPHA, ban_weight: float = BAN_WEIGHT,
              n_col: str = "total_matches") -> pd.DataFrame:
    """Adds pick_rate (if missing), presence, power, power_se (points) and power tier columns.
    Without a ban_rate column (older weeks), presence is the pick rate alone."""
    out = df if "pick_rate" in df else add_pick_rate(df, n_col)
    out = out.copy()
    bans = out["ban_rate"].fillna(0) if "ban_rate" in out else 0.0
    out["presence"] = out["pick_rate"] + ban_weight * bans
    weight = (out["presence"] / out["presence"].median()) ** alpha
    p = out["winrate"] / 100
    out["power"] = (out["winrate"] - 50) * weight
    out["power_se"] = np.sqrt(p * (1 - p) / out[n_col]) * 100 * weight   # win-rate noise only
    out["tier"] = (out["power"] + 50).map(tier_of)
    out["wr_tier"] = out["winrate"].map(tier_of)
    return out


# =============================================================================
# WEEK-OVER-WEEK DELTAS
# =============================================================================
def normal_cdf(x):
    """Standard normal CDF, elementwise (no scipy dependency)."""
    from math import erf, sqrt
    return np.vectorize(lambda v: 0.5 * (1 + erf(v / sqrt(2))) if np.isfinite(v) else np.nan)(np.asarray(x, float))


def previous_week_dir(data_dir) -> "Path | None":
    """week<N-1>_analysis next to week<N>_analysis, if it exists."""
    import re
    from pathlib import Path
    data_dir = Path(data_dir)
    m = re.search(r"week(\d+)", data_dir.name)
    if not m:
        return None
    prev = data_dir.with_name(data_dir.name.replace(m.group(0), f"week{int(m.group(1)) - 1}", 1))
    return prev if (prev / "hero_winrates.csv").exists() else None


def _with_wins(df: pd.DataFrame) -> pd.DataFrame:
    """Older exports only have the rounded win rate; rebuild wins from it when needed."""
    if "wins" not in df.columns:
        df = df.assign(wins=(df["winrate"] / 100 * df["total_matches"]).round())
    return df


def hero_deltas(now_dir, prev_dir, z: float = 1.96) -> pd.DataFrame:
    """One row per hero in the current week, compared with prev_dir.
    rank / tier come from the power score (see add_power); wr_tier is the plain win-rate tier.

    wr_delta is in percentage points. Its 95% range uses the difference of two independent
    proportions: SE = sqrt(p1(1-p1)/n1 + p2(1-p2)/n2). `clear` = the range excludes 0.
    With ~40 heroes, expect about 2 heroes per week to look "clear" by chance alone, so
    rank by p_change rather than trusting every flag. Heroes missing last week get
    status "new" and NaN deltas."""
    now_raw, prev_raw = _with_wins(load_hero_meta(now_dir)), _with_wins(load_hero_meta(prev_dir))
    # bans only count when both weeks have them; otherwise the formula change itself would move tiers
    both_bans = all("ban_rate" in d and d["ban_rate"].notna().any() for d in (now_raw, prev_raw))
    bw = BAN_WEIGHT if both_bans else 0.0
    now, prev = add_power(now_raw, ban_weight=bw), add_power(prev_raw, ban_weight=bw)
    for d in (now, prev):
        d["rank"] = d["power"].rank(ascending=False, method="min").astype(int)
        if "ban_rate" not in d:
            d["ban_rate"] = np.nan
    cols = ["hero_name", "total_matches", "wins", "winrate", "pick_rate", "ban_rate", "presence",
            "power", "power_se", "rank", "tier", "wr_tier"]
    out = now[["hero_type"] + cols].merge(prev[cols], on="hero_name", how="left", suffixes=("", "_prev"))
    out = out.rename(columns={"total_matches": "games", "total_matches_prev": "games_prev"})

    p1, n1 = out["wins"] / out["games"], out["games"]
    p0, n0 = out["wins_prev"] / out["games_prev"], out["games_prev"]
    out["wr_delta"] = (p1 - p0) * 100
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) * 100
    out["delta_lo"] = out["wr_delta"] - z * se
    out["delta_hi"] = out["wr_delta"] + z * se
    out["z"] = out["wr_delta"] / se
    out["p_change"] = normal_cdf(out["z"].abs())        # how sure the change is real, either way
    out["clear"] = (out["delta_lo"] > 0) | (out["delta_hi"] < 0)
    out["pick_delta"] = out["pick_rate"] - out["pick_rate_prev"]
    out["power_delta"] = out["power"] - out["power_prev"]
    out["power_clear"] = out["power_delta"].abs() > z * np.sqrt(out["power_se"] ** 2 + out["power_se_prev"] ** 2)
    out["rank_change"] = out["rank_prev"] - out["rank"]        # + = climbed
    tier_now = out["tier"].map(TIER_NAMES.index)
    tier_prev = out["tier_prev"].map(lambda t: TIER_NAMES.index(t) if isinstance(t, str) else np.nan)
    out["tier_change"] = tier_now - tier_prev                  # + = moved up
    out["status"] = np.where(out["games_prev"].isna(), "new", "returning")
    out.attrs["ban_weight"] = bw
    # what moved the power score: the win-rate change at last week's presence, and the rest
    w_prev = out["power_prev"] / (out["winrate_prev"] - 50).where(out["winrate_prev"] != 50)
    out["delta_from_wr"] = (out["winrate"] - out["winrate_prev"]) * w_prev.fillna(1.0)
    out["delta_from_presence"] = out["power_delta"] - out["delta_from_wr"]
    return out.sort_values("power", ascending=False).reset_index(drop=True)


# =============================================================================
# INSIGHTS  (each chart script's insights() returns these; insights.py ranks them)
# =============================================================================
from dataclasses import dataclass, field


@dataclass
class Insight:
    """One finding, as data. `effect` is the size of the finding in win-rate points where that
    makes sense (minutes for match length); `confidence` is how sure we are it's real (0.5-1).
    `highlight` = names animate.py should highlight on `card` (heroes or archetypes)."""
    card: str
    kind: str
    headline: str
    detail: str = ""
    highlight: tuple = ()
    effect: float = 0.0
    confidence: float = 1.0
    tags: tuple = ()
    hero: str | None = None            # main hero, used to merge several findings about one hero
    context: bool = False              # background fact for the video, not a ranked story
    also: list = field(default_factory=list)


def wr_confidence(wr, n):
    """How sure we are a win rate (percent) is really above / below 50%, from its sample size."""
    se = np.sqrt(0.25 / np.asarray(n, float)) * 100
    return normal_cdf(np.abs(np.asarray(wr, float) - 50) / se)


def confidence_label(c: float) -> str:
    return "very likely real" if c >= 0.99 else "likely real" if c >= 0.95 else "could be noise"