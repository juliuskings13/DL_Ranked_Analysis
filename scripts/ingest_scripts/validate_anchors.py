"""Add kill anchors to the death-anchored realignment and validate on purchases.

Usage:  python validate_anchors.py --ids-file ids.txt
Needs realign_paths.py and shop_items.py next to it; reads meta_cache/.

Fit with:      deaths (as before) + kills (new anchors)
Validate on:   1. kills, 5-fold cross-validation (a kill is never an anchor
                  in the run that grades it)
               2. shop purchases/sells: never used as anchors at all
               3. bridge-buff pickups, if the metadata has them
Purchase/pickup check: at the event second the player must be dead or inside
a zone where clean players from *other* matches did the same thing.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from realign_paths import fetch, load_match, find_anchors, realign_linear, DRIFT_TOL
from shop_items import SHOP_ITEM_IDS

KILL_WINDOW = 60        # entries searched around the death-only estimate
KILL_MAX_DIST = 300     # map units: closest path point must be this near killer_pos
ZONE_RADIUS = 600       # map units
ZONE_MIN_POINTS = 5     # clean events from other matches needed nearby to call it a zone
FOLDS = 5
rng = np.random.default_rng(13)


# ---------------------------------------------------------------- data
def world(p, i):
    x0, x1, y0, y1 = p["bounds"]
    return x0 + p["x"][i] / 16383 * (x1 - x0), y0 + p["y"][i] / 16383 * (y1 - y0)


def is_dead(p, i):
    return p["hp"][i] == 0 or (p["x"][i] == 0 and p["y"][i] == 0)


def load_events(match_id, m):
    """Attach shop events and buff pickups to each player of match m."""
    info = fetch(match_id).get("match_info", {})
    seen_keys = set()
    for pl in info.get("players", []):
        p = m["players"].get(pl["player_slot"])
        if p is None:
            continue
        shop = []
        for it in pl.get("items", []):
            seen_keys |= set(it)
            if it.get("item_id") not in SHOP_ITEM_IDS:
                continue
            t = int(it.get("game_time_s") or 0)
            if t >= 10:
                shop.append(t)
            s = int(it.get("sold_time_s") or 0)
            if s >= 10:
                shop.append(s)
        pickups = []
        for b in pl.get("power_up_buffs", []) or []:
            seen_keys |= {"buff." + k for k in b}
            for t in b.get("pickup_times_s", []) or []:
                if int(t) >= 10:
                    pickups.append(int(t))
        p["shop_t"] = sorted(shop)
        p["pickup_t"] = sorted(pickups)
        kills = []
        for victim in m["players"].values():
            for d in victim["deaths"]:
                if d["killer"] == p["slot"] and d["killer_pos"] is not None and d["t"] <= m["duration"]:
                    kills.append({"t": d["t"], "pos": d["killer_pos"]})
        p["kills"] = kills
    return seen_keys


# ---------------------------------------------------------------- anchors
def kill_candidates(p, idx_deaths):
    """For each kill, the array index nearest to killer_pos around the death-only estimate."""
    n = len(p["hp"])
    out = []
    for k, kill in enumerate(p["kills"]):
        c = int(idx_deaths[kill["t"]])
        lo, hi = max(c - KILL_WINDOW, 0), min(c + KILL_WINDOW, n - 1)
        best, best_d = None, None
        for i in range(lo, hi + 1):
            if is_dead(p, i):
                continue
            wx, wy = world(p, i)
            dist = np.hypot(wx - kill["pos"][0], wy - kill["pos"][1])
            # ties: prefer the index closest to the estimate (players can stand still)
            if best_d is None or dist < best_d - 25 or (abs(dist - best_d) <= 25 and abs(i - c) < abs(best - c)):
                best, best_d = i, dist
        if best is not None and best_d <= KILL_MAX_DIST:
            out.append((k, best, kill["t"]))
    return out


def merge_anchors(death_i, death_s, extra):
    """Insert (index, second) kill anchors between death anchors when consistent:
    index strictly increasing, and an index gap never smaller than the time gap - 2
    (drift only adds samples)."""
    anchors = list(zip(death_s, death_i))
    for i, s in sorted(extra, key=lambda a: a[1]):
        pos = np.searchsorted([a[0] for a in anchors], s)
        if pos == 0 or pos >= len(anchors) or anchors[pos][0] == s:
            continue
        (s0, i0), (s1, i1) = anchors[pos - 1], anchors[pos]
        if i0 < i < i1 and (i - i0) >= (s - s0) - 2 and (i1 - i) >= (s1 - s) - 2:
            anchors.insert(pos, (s, i))
    return [a[1] for a in anchors], [a[0] for a in anchors]


# ---------------------------------------------------------------- zones
def zone_points(matches, key):
    """Clean players' alive positions at event seconds, tagged with match id."""
    pts = []
    for m in matches:
        for p in m["players"].values():
            if abs(p["drift"]) > DRIFT_TOL:
                continue
            for t in p[key]:
                if t < len(p["hp"]) and not is_dead(p, t):
                    pts.append((m["match_id"], *world(p, t)))
    return pts


def summarize_zones(pts, cell=500):
    cells = defaultdict(int)
    for _, x, y in pts:
        cells[(int(x // cell), int(y // cell))] += 1
    dense = {c for c, n in cells.items() if n >= ZONE_MIN_POINTS}
    seen, zones = set(), []
    for c in dense:
        if c in seen:
            continue
        stack, members = [c], []
        seen.add(c)
        while stack:
            cx, cy = stack.pop()
            members.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in dense and nb not in seen:
                        seen.add(nb); stack.append(nb)
        n = sum(cells[c_] for c_ in members)
        zx = np.mean([c_[0] for c_ in members]) * cell; zy = np.mean([c_[1] for c_ in members]) * cell
        zones.append((n, round(zx), round(zy), len(members)))
    return sorted(zones, reverse=True)


class ZoneIndex:
    """Grid lookup: how many clean events from *other* matches lie within ZONE_RADIUS."""
    def __init__(self, pts):
        self.cells = defaultdict(list)
        for mid, x, y in pts:
            self.cells[(int(x // ZONE_RADIUS), int(y // ZONE_RADIUS))].append((mid, x, y))
        self.cells = {c: (np.array([v[0] for v in vals]), np.array([[v[1], v[2]] for v in vals]))
                      for c, vals in self.cells.items()}

    def contains(self, match_id, x, y):
        cx, cy, n = int(x // ZONE_RADIUS), int(y // ZONE_RADIUS), 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                c = self.cells.get((cx + dx, cy + dy))
                if c is None:
                    continue
                mids, xy = c
                near = np.hypot(xy[:, 0] - x, xy[:, 1] - y) <= ZONE_RADIUS
                n += int(np.sum(near & (mids != match_id)))   # leave-match-out
                if n >= ZONE_MIN_POINTS:
                    return True
        return False


def event_ok(p, idx, t, zones, match_id):
    """Dead, or in a zone, at the event second (+/-1 s for sampling)."""
    for dt in (-1, 0, 1):
        s = t + dt
        if s < 0 or s >= len(idx):
            continue
        i = int(idx[s])
        if is_dead(p, i) or zones.contains(match_id, *world(p, i)):
            return True
    return False


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_ids", nargs="*", type=int)
    ap.add_argument("--ids-file")
    a = ap.parse_args()
    ids = list(a.match_ids)
    if a.ids_file:
        ids += [int(x) for x in Path(a.ids_file).read_text().split() if x.strip()]

    matches, item_keys = [], set()
    for mid in ids:
        m = load_match(mid)
        item_keys |= load_events(mid, m)
        matches.append(m)
    n_pick = sum(len(p["pickup_t"]) for m in matches for p in m["players"].values())
    print(f"item/buff fields seen: {sorted(item_keys)}   buff pickups found: {n_pick}")

    # ---- mappings per player
    kill_err = defaultdict(list)
    maps = {}                      # (match, slot) -> {method: idx}
    n_kill_anchor = n_kill = 0
    for m in matches:
        dur = m["duration"]
        for slot, p in m["players"].items():
            if p["drift"] < -DRIFT_TOL:
                continue           # truncated (abandon) paths: handled in the pipeline, skipped here
            ident = np.clip(np.arange(dur + 1), 0, len(p["hp"]) - 1)
            di, ds, _ = find_anchors(p, dur)
            idx_d = realign_linear(p, dur, di, ds)
            cands = kill_candidates(p, idx_d)
            n_kill += len(p["kills"]); n_kill_anchor += len(cands)
            ai, as_ = merge_anchors(di, ds, [(i, s) for _, i, s in cands])
            idx_dk = realign_linear(p, dur, ai, as_)
            maps[(m["match_id"], slot)] = {"raw": ident, "deaths": idx_d, "deaths+kills": idx_dk}

            # kill error: deaths-only (kills held out entirely) and 5-fold CV for deaths+kills
            folds = rng.integers(0, FOLDS, len(p["kills"]))
            for f in range(FOLDS):
                train = [(i, s) for k, i, s in cands if folds[k] != f]
                ai_f, as_f = merge_anchors(di, ds, train)
                idx_f = realign_linear(p, dur, ai_f, as_f)
                for k, kill in enumerate(p["kills"]):
                    if folds[k] != f:
                        continue
                    group = "clean" if abs(p["drift"]) <= DRIFT_TOL else "drift"
                    for name, idx in (("raw", ident), ("deaths", idx_d), ("deaths+kills (CV)", idx_f)):
                        if group == "clean" and name != "raw":
                            continue
                        wx, wy = world(p, int(idx[kill["t"]]))
                        kill_err[(group, name)].append(np.hypot(wx - kill["pos"][0], wy - kill["pos"][1]))

    print(f"kills usable as anchors: {n_kill_anchor}/{n_kill} (closest path point within {KILL_MAX_DIST} units)")
    print("\n[kills] killer position error at kill time (map units)")
    for (group, name), e in kill_err.items():
        e = np.array(e)
        print(f"  {group:5} {name:18} n={len(e):5d}  median {np.median(e):7.1f}  p90 {np.percentile(e, 90):7.1f}"
              f"  share<500 {np.mean(e < 500):5.1%}")

    # ---- purchases and pickups
    for key, label in (("shop_t", "shop purchases/sells"), ("pickup_t", "bridge-buff pickups")):
        pts = zone_points(matches, key)
        if len(pts) < 20:
            print(f"\n[{label}] not enough clean events ({len(pts)}), skipped")
            continue
        zindex = ZoneIndex(pts)
        zones = summarize_zones(pts)
        print(f"\n[{label}] {len(pts)} clean alive events -> {len(zones)} zones (events, x, y, cells):")
        for z in zones[:14]:
            print(f"    {z}")

        res = defaultdict(lambda: [0, 0])
        for m in matches:
            for slot, p in m["players"].items():
                mp = maps.get((m["match_id"], slot))
                if mp is None:
                    continue
                group = "clean" if abs(p["drift"]) <= DRIFT_TOL else "drift"
                for t in p[key]:
                    if t > m["duration"]:
                        continue
                    for name, idx in mp.items():
                        if group == "clean" and name != "raw":
                            continue
                        r = res[(group, name)]
                        r[0] += event_ok(p, idx, t, zindex, m["match_id"]); r[1] += 1
        print(f"  share of events where the player is dead or in a zone (+/-1 s):")
        for (group, name), (ok, n) in sorted(res.items()):
            print(f"    {group:5} {name:14} {ok:5d}/{n:5d}  {ok / n:6.1%}")


if __name__ == "__main__":
    main()
