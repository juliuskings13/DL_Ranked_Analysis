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
    """hero_winrates.csv with normalized columns: hero_name, hero_type, total_matches, winrate."""
    from pathlib import Path
    df = pd.read_csv(Path(data_dir) / "hero_winrates.csv")
    df.columns = df.columns.str.strip().str.lower()
    df["hero_type"] = df["hero_type"].str.strip().str.lower()
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
PHASE_SPLIT = "quartiles"
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