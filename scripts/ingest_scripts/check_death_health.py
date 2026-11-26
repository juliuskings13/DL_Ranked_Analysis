"""Check whether match_paths health stays 0 for a player's whole respawn timer.

Usage:  python check_death_health.py <match_id> [<match_id> ...]
Pulls /v1/matches/{id}/metadata, lines up every death (death_details) with the
per-second path arrays, and reports what health / position do while dead.
"""
import json
import sys
import urllib.request
from collections import Counter

API = "https://api.deadlock-api.com/v1/matches/{}/metadata"


def fetch(match_id):
    req = urllib.request.Request(API.format(match_id), headers={"User-Agent": "dl-deathcheck"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def check(match_id):
    meta = fetch(match_id)
    info = meta.get("match_info", meta)
    mp = info.get("match_paths") or {}
    paths = mp.get("paths") or []
    if not paths:
        print(f"[{match_id}] no match_paths; match_info keys: {sorted(info)}")
        return
    interval = mp.get("interval_s", 1) or 1
    duration = info.get("duration_s")
    print(f"\n=== match {match_id}  duration_s={duration}  interval_s={interval}")
    print(f"path keys: {sorted(paths[0])}")

    by_slot = {p.get("player_slot"): p for p in paths}
    totals = Counter()
    for player in info.get("players", []):
        slot = player.get("player_slot")
        path = by_slot.get(slot)
        if path is None:
            continue
        hp = path.get("health") or []
        xs, ys = path.get("x_pos") or [], path.get("y_pos") or []
        drift = len(hp) - int(duration or len(hp))
        flag = "  <-- length drift (issue #698)" if abs(drift) > 2 else ""
        print(f"slot {slot:>2}: {len(hp)} samples (drift {drift:+d}){flag}")

        for d in player.get("death_details", []):
            t0 = d.get("game_time_s")
            dur = d.get("death_duration_s") or 0
            i0, i1 = int(t0 // interval), int((t0 + dur) // interval)
            window = hp[i0:i1]
            if not window:
                continue
            zero_share = sum(h == 0 for h in window) / len(window)
            first_nonzero = next((k for k, h in enumerate(window) if h != 0), None)
            moved = len({(x, y) for x, y in zip(xs[i0:i1], ys[i0:i1])}) > 1
            after = hp[i1:i1 + 3]
            totals["deaths"] += 1
            totals["all_zero"] += zero_share == 1
            totals["moved_while_dead"] += moved
            print(f"   death t={t0:>4}s respawn={dur:>3}s  hp==0 {zero_share:5.0%}"
                  f"  first_nonzero_at={first_nonzero}  pos_changes={moved}"
                  f"  hp_before={hp[max(i0-2,0):i0]}  hp_after={after}")

    print(f"summary: {dict(totals)}")


if __name__ == "__main__":
    for mid in sys.argv[1:] or ["139742867"]:
        check(int(mid))