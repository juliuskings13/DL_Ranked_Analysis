"""Realign drifting match_paths arrays and validate the result against kills.

Usage:  python realign_paths.py <match_id> [<match_id> ...]
        python realign_paths.py --ids-file ids.txt

Steps
  0. Conversion fit: path x_pos/y_pos are compressed integers. On clean players
     (array length ~ duration) the frozen position while dead should equal
     death_pos, so compare candidate conversions against it.
  1. Death-anchored realignment: each zero-health stretch in the array is matched
     to a death in death_details. (stretch start index -> death second + 1) pairs
     are anchors; np.interp maps every array index back to a game second.
  2. Held-out validation: kills were never used in step 1. At every kill second,
     compare the killer's (realigned) path position to killer_pos.
     Reported for clean players (baseline), drifting players before realignment
     and drifting players after realignment.
"""
import argparse
import bz2
import gzip
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

API = "https://api.deadlock-api.com/v1/matches/{}/metadata"
CACHE = Path("meta_cache")
DRIFT_TOL = 2          # |len(array) - duration| <= this counts as clean
REQUEST_GAP_S = 1.1    # be polite to the API
_last_request = 0.0


# ---------------------------------------------------------------- fetching
def fetch(match_id):
    global _last_request
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{match_id}.json.gz"
    if f.exists():
        return json.loads(gzip.decompress(f.read_bytes()))
    wait = REQUEST_GAP_S - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(API.format(match_id), headers={"User-Agent": "dl-realign"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    _last_request = time.time()
    if raw[:3] == b"BZh":
        raw = bz2.decompress(raw)
    f.write_bytes(gzip.compress(raw))
    return json.loads(raw)


def xy(p):
    """death_pos / killer_pos may come as {'x','y','z'} or [x, y, z]."""
    if p is None:
        return None
    if isinstance(p, dict):
        return float(p.get("x", p.get("1"))), float(p.get("y", p.get("2")))
    return float(p[0]), float(p[1])


# ---------------------------------------------------------------- parsing
def load_match(match_id):
    meta = fetch(match_id)
    info = meta.get("match_info", meta)
    duration = int(info["duration_s"])
    paths = {p["player_slot"]: p for p in (info.get("match_paths") or {}).get("paths", [])}
    players = {}
    for pl in info.get("players", []):
        slot = pl["player_slot"]
        path = paths.get(slot)
        if not path:
            continue
        deaths = []
        for d in pl.get("death_details", []):
            deaths.append({
                "t": int(d["game_time_s"]),
                "dur": int(d.get("death_duration_s") or 0),
                "pos": xy(d.get("death_pos")),
                "killer": int(d.get("killer_player_slot") or 0),
                "killer_pos": xy(d.get("killer_pos")),
            })
        hp = np.asarray(path["health"], dtype=float)
        players[slot] = {
            "slot": slot,
            "team": pl.get("team"),
            "hp": hp,
            "x": np.asarray(path["x_pos"], dtype=float),
            "y": np.asarray(path["y_pos"], dtype=float),
            "combat": np.asarray(path.get("combat_type") or np.zeros(len(hp)), dtype=float),
            "move": np.asarray(path.get("move_type") or np.zeros(len(hp)), dtype=float),
            "bounds": (path.get("x_min"), path.get("x_max"), path.get("y_min"), path.get("y_max")),
            "deaths": deaths,
            "drift": len(hp) - (duration + 1),
        }
    return {"match_id": match_id, "duration": duration, "players": players}


def zero_runs(hp, min_len=2):
    """[(start, length)] of consecutive health == 0 stretches."""
    runs, start = [], None
    for i, h in enumerate(np.append(hp, 1)):
        if h == 0 and start is None:
            start = i
        elif h != 0 and start is not None:
            if i - start >= min_len:
                runs.append((start, i - start))
            start = None
    return runs


# ---------------------------------------------------------------- step 0
RES_CANDIDATES = (255, 1023, 4095, 16383, 65535)


def fit_conversion(matches):
    """Compare candidate raw->world conversions on clean players' deaths."""
    raw_max = max(max(p["x"].max(), p["y"].max()) for m in matches for p in m["players"].values())
    res = next(r for r in RES_CANDIDATES if r >= raw_max)

    rows = []  # (raw_x, raw_y, bounds, world_x, world_y)
    for m in matches:
        for p in m["players"].values():
            if abs(p["drift"]) > DRIFT_TOL:
                continue
            for d in p["deaths"]:
                if d["pos"] is None or d["dur"] < 3:
                    continue
                t = d["t"]                                    # last alive sample = death spot
                if t + d["dur"] > len(p["hp"]) or p["hp"][t] == 0 or not np.all(p["hp"][t + 1:t + d["dur"]] == 0):
                    continue
                dead = (p["x"][t + 1:t + d["dur"]], p["y"][t + 1:t + d["dur"]])
                rows.append((p["x"][t], p["y"][t], p["bounds"], *d["pos"], dead))
    if not rows:
        raise SystemExit("No clean deaths to fit the conversion on.")

    # diagnostics: what does the path look like while dead?
    print("\n[step 0] sample deaths: raw pos at death second | raw pos while dead (unique) | death_pos")
    for r in rows[:5]:
        dead_xy = sorted(set(zip(r[5][0].tolist(), r[5][1].tolist())))[:3]
        print(f"  ({r[0]:.0f}, {r[1]:.0f}) | {dead_xy} | ({r[3]:.0f}, {r[4]:.0f})")

    rx = np.array([r[0] for r in rows]); ry = np.array([r[1] for r in rows])
    wx = np.array([r[3] for r in rows]); wy = np.array([r[4] for r in rows])
    ok = np.isfinite(rx) & np.isfinite(ry) & np.isfinite(wx) & np.isfinite(wy)
    if not ok.all():
        print(f"  dropping {np.sum(~ok)} rows with NaN values")
    rows = [r for r, k in zip(rows, ok) if k]
    rx, ry, wx, wy = rx[ok], ry[ok], wx[ok], wy[ok]
    if len(rows) < 5 or np.ptp(rx) == 0 or np.ptp(ry) == 0:
        raise SystemExit(f"Can't fit: {len(rows)} usable rows, raw x range {np.ptp(rx) if len(rx) else 0}, "
                         f"raw y range {np.ptp(ry) if len(ry) else 0}. Paste the sample lines above.")

    # candidate A: per-path bounds formula
    def formula(raw, lo, hi):
        return lo + raw / res * (hi - lo)
    ax = np.array([formula(r[0], r[2][0], r[2][1]) for r in rows])
    ay = np.array([formula(r[1], r[2][2], r[2][3]) for r in rows])
    err_a = np.hypot(ax - wx, ay - wy)

    # candidate B: one global linear fit per axis
    kx = np.polyfit(rx, wx, 1); ky = np.polyfit(ry, wy, 1)
    err_b = np.hypot(np.polyval(kx, rx) - wx, np.polyval(ky, ry) - wy)

    print(f"\n[step 0] conversion fit on {len(rows)} clean deaths, raw max={raw_max:.0f} -> RES={res}")
    print(f"  A bounds formula : median err {np.median(err_a):8.1f}  p90 {np.percentile(err_a, 90):8.1f}")
    print(f"  B global linear  : median err {np.median(err_b):8.1f}  p90 {np.percentile(err_b, 90):8.1f}"
          f"   x={kx[0]:.3f}*raw{kx[1]:+.1f}  y={ky[0]:.3f}*raw{ky[1]:+.1f}")
    if min(np.median(err_a), np.median(err_b)) > 1000:
        print("  !! both conversions are far off: the frozen dead position may not be the death spot")

    if np.median(err_a) <= np.median(err_b):
        print("  using A")
        return lambda p, rx_, ry_: (formula(rx_, p["bounds"][0], p["bounds"][1]),
                                    formula(ry_, p["bounds"][2], p["bounds"][3]))
    print("  using B")
    return lambda p, rx_, ry_: (np.polyval(kx, rx_), np.polyval(ky, ry_))


# ---------------------------------------------------------------- step 1
def find_anchors(p, duration):
    """Match zero-health stretches to deaths. Returns (anchors_i, anchors_s, unmatched)."""
    deaths = [d for d in p["deaths"] if d["dur"] >= 3]
    runs = zero_runs(p["hp"])
    anchors_i, anchors_s, unmatched, r = [0], [0], 0, 0
    for d in deaths:                                   # in-order matching with sanity checks
        expected_start = d["t"] + 1
        matched = False
        while r < len(runs):
            start, length = runs[r]
            r += 1
            if start < expected_start - 2:             # run before this death: stray zeros, skip
                continue
            # lower bound 0.4x: the patron phase-2 mechanic shortens respawns late in the match
            if 0.4 * d["dur"] - 1 <= length <= 1.6 * d["dur"] + 3 and start >= anchors_i[-1]:
                anchors_i.append(start); anchors_s.append(expected_start)
                matched = True
            else:
                r -= 1                                 # leave run for the next death
            break
        unmatched += not matched
    anchors_i.append(len(p["hp"]) - 1); anchors_s.append(duration)   # assume array end = match end
    return anchors_i, anchors_s, unmatched


def realign_linear(p, duration, anchors_i, anchors_s):
    """Method 1: spread the extra entries evenly between anchors."""
    idx = np.interp(np.arange(duration + 1), anchors_s, anchors_i)
    return np.clip(np.rint(idx).astype(int), 0, len(p["hp"]) - 1)


def repeat_flags(p):
    """dup[j]: entry j is identical to entry j-1 in every field, player alive.
    iso[j]: an isolated repeat, i.e. neighbours on both sides differ. A moving,
    alive player should almost never produce one, so these are the suspects."""
    f = np.vstack([p["x"], p["y"], p["hp"], p["combat"], p["move"]])
    same_prev = np.zeros(f.shape[1], bool)
    same_prev[1:] = np.all(f[:, 1:] == f[:, :-1], axis=0)
    alive = (p["hp"] > 0) & ~((p["x"] == 0) & (p["y"] == 0))
    dup = same_prev & alive
    nxt = np.append(same_prev[1:], False)
    prv = np.insert(same_prev[:-1], 0, False)
    iso = dup & ~nxt & ~prv
    return dup, iso


def realign_dedupe(p, duration, anchors_i, anchors_s):
    """Method 2: between each pair of anchors, delete the surplus entries, taking
    isolated repeats first. Falls back to even spreading for any surplus left."""
    _, iso = repeat_flags(p)
    idx = np.zeros(duration + 1, int)
    for (i0, s0), (i1, s1) in zip(zip(anchors_i, anchors_s), zip(anchors_i[1:], anchors_s[1:])):
        seg = np.arange(i0, i1)
        n_sec = s1 - s0
        if n_sec <= 0:
            continue
        surplus = len(seg) - n_sec
        cand = seg[iso[seg]]
        if surplus > 0 and len(cand):
            if len(cand) > surplus:                    # more suspects than needed: take them spread out
                cand = cand[np.linspace(0, len(cand) - 1, surplus).round().astype(int)]
            seg = np.setdiff1d(seg, cand)
        pos = np.linspace(0, len(seg) - 1, n_sec) if len(seg) != n_sec else np.arange(n_sec)
        idx[s0:s1] = seg[np.rint(pos).astype(int)]
    idx[anchors_s[-1]:] = anchors_i[-1]
    return np.clip(idx, 0, len(p["hp"]) - 1)


# ---------------------------------------------------------------- step 2
def validate(matches, to_world):
    groups = {"clean": [], "drift_raw": [], "drift_linear": [], "drift_dedupe": []}
    rep_rows = []
    for m in matches:
        dur = m["duration"]
        lin, ded = {}, {}
        print(f"\n[step 1] match {m['match_id']}")
        print("  slot  type   drift  deaths anchors unmatched | repeats  isolated  isolated/1000s alive")
        for slot, p in sorted(m["players"].items()):
            ai, as_, n_miss = find_anchors(p, dur)
            lin[slot] = realign_linear(p, dur, ai, as_)
            ded[slot] = realign_dedupe(p, dur, ai, as_)
            dup, iso = repeat_flags(p)
            alive_s = max(int(np.sum(p["hp"] > 0)), 1)
            tag = "clean" if abs(p["drift"]) <= DRIFT_TOL else "DRIFT"
            rep_rows.append((tag, p["drift"], int(iso.sum()), alive_s))
            print(f"  {slot:>4}  {tag:5} {p['drift']:+5d}  {len(p['deaths']):>6} {len(ai) - 2:>7} {n_miss:>9} |"
                  f" {int(dup.sum()):>7} {int(iso.sum()):>9} {1000 * iso.sum() / alive_s:>15.1f}")
        for victim in m["players"].values():
            for d in victim["deaths"]:
                k = m["players"].get(d["killer"])
                if k is None or d["killer_pos"] is None or d["t"] > dur:
                    continue
                t = d["t"]
                kx, ky = d["killer_pos"]
                def err(i):
                    i = min(max(int(i), 0), len(k["x"]) - 1)
                    wx, wy = to_world(k, k["x"][i], k["y"][i])
                    return float(np.hypot(wx - kx, wy - ky))
                if abs(k["drift"]) <= DRIFT_TOL:
                    groups["clean"].append(err(t))
                else:
                    groups["drift_raw"].append(err(t))
                    groups["drift_linear"].append(err(lin[k["slot"]][t]))
                    groups["drift_dedupe"].append(err(ded[k["slot"]][t]))

    print("\n[repeats] are the extra entries repeats? compare isolated repeats to drift")
    for tag in ("clean", "DRIFT"):
        rows = [r for r in rep_rows if r[0] == tag]
        if rows:
            iso = sum(r[2] for r in rows); drift = sum(r[1] for r in rows); alive = sum(r[3] for r in rows)
            print(f"  {tag:5}: players {len(rows):3d}  total drift {drift:+6d}  isolated repeats {iso:6d}"
                  f"  ({1000 * iso / alive:.1f} per 1000 alive seconds)")
    print("  if repeats explain the drift: clean rate ~0, and DRIFT isolated repeats ~ total drift")

    print("\n[step 2] killer position error at kill time (map units; held-out: kills unused in step 1)")
    for name, e in groups.items():
        if e:
            e = np.array(e)
            print(f"  {name:13} n={len(e):4d}  median {np.median(e):8.1f}  p90 {np.percentile(e, 90):8.1f}"
                  f"  share<500 {np.mean(e < 500):5.0%}")
    print("  goal: drift_linear / drift_dedupe ~ clean, far below drift_raw")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_ids", nargs="*", type=int)
    ap.add_argument("--ids-file")
    a = ap.parse_args()
    ids = list(a.match_ids)
    if a.ids_file:
        ids += [int(x) for x in Path(a.ids_file).read_text().split() if x.strip()]
    ids = ids or [139742867]
    matches = [load_match(i) for i in ids]
    to_world = fit_conversion(matches)
    validate(matches, to_world)


if __name__ == "__main__":
    main()