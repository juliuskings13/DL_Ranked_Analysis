"""Diagnostics for match_paths drift: pauses and disconnects.

Usage:  python drift_diagnostics.py --ids-file ids.txt
Reads the meta_cache/ written by realign_paths.py (no new API calls for cached matches).

1. Pause check: during a pause everything freezes, so if the path sampler kept
   running, the array would hold a long run (>= 15 entries) of identical samples
   while the player is alive, at the same moment for every player recorded that way.
2. Disconnect check: a path shorter than the match (negative drift) is missing
   seconds. Where they went missing should show up as one big position jump.
3. Drift sizes: are they random, or do they come in fixed-size steps?
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from realign_paths import load_match, DRIFT_TOL

MIN_FREEZE = 15        # entries; pauses last at least 30 s
JUMP_UNITS = 3000      # a hero can't move this far in one second without a teleport


def frozen_runs(p):
    """[(start, length)] of identical consecutive samples while alive."""
    f = np.vstack([p["x"], p["y"], p["hp"], p["combat"], p["move"]])
    same = np.zeros(f.shape[1], bool)
    same[1:] = np.all(f[:, 1:] == f[:, :-1], axis=0)
    alive = (p["hp"] > 0) & ~((p["x"] == 0) & (p["y"] == 0))
    same &= alive
    runs, start = [], None
    for i, s in enumerate(np.append(same, False)):
        if s and start is None:
            start = i - 1
        elif not s and start is not None:
            if i - start >= MIN_FREEZE:
                runs.append((start, i - start))
            start = None
    return runs


def world(p):
    x0, x1, y0, y1 = p["bounds"]
    return x0 + p["x"] / 16383 * (x1 - x0), y0 + p["y"] / 16383 * (y1 - y0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_ids", nargs="*", type=int)
    ap.add_argument("--ids-file")
    a = ap.parse_args()
    ids = list(a.match_ids)
    if a.ids_file:
        ids += [int(x) for x in Path(a.ids_file).read_text().split() if x.strip()]

    freeze_rows, drifts = [], []
    print("=== 1. long frozen runs (possible pauses) ===")
    for mid in ids:
        m = load_match(mid)
        for slot, p in sorted(m["players"].items()):
            drifts.append(p["drift"])
            runs = frozen_runs(p)
            tag = "clean" if abs(p["drift"]) <= DRIFT_TOL else "DRIFT"
            freeze_rows.append((tag, p["drift"], sum(l for _, l in runs)))
            if runs:
                print(f"  match {mid} slot {slot:>2} {tag} drift {p['drift']:+5d}  runs (start,len): {runs}")

        # 2. disconnects
        for slot, p in sorted(m["players"].items()):
            if p["drift"] >= -DRIFT_TOL:
                continue
            wx, wy = world(p)
            alive = (p["hp"] > 0) & ~((p["x"] == 0) & (p["y"] == 0))
            step = np.hypot(np.diff(wx), np.diff(wy))
            step[~(alive[1:] & alive[:-1])] = 0          # ignore death/respawn teleports
            big = np.argsort(step)[::-1][:3]
            jumps = [(int(i), round(float(step[i]))) for i in big if step[i] > JUMP_UNITS]
            print(f"\n=== 2. match {mid} slot {slot} drift {p['drift']:+d}: "
                  f"{len(p['hp'])} entries for {m['duration']} s; big jumps (index, units): {jumps or 'none'}")

    print("\n=== 1b. do frozen entries account for the drift? ===")
    for tag in ("clean", "DRIFT"):
        rows = [r for r in freeze_rows if r[0] == tag]
        if rows:
            print(f"  {tag:5}: players {len(rows):3d}  total drift {sum(r[1] for r in rows):+6d}"
                  f"  entries in long frozen runs {sum(r[2] for r in rows):6d}")

    print("\n=== 3. drift sizes (positive drifts only) ===")
    pos = [d for d in drifts if d > DRIFT_TOL]
    print(f"  even: {sum(d % 2 == 0 for d in pos)}/{len(pos)}   multiple of 4: {sum(d % 4 == 0 for d in pos)}/{len(pos)}")
    print(f"  most common: {Counter(pos).most_common(12)}")


if __name__ == "__main__":
    main()
