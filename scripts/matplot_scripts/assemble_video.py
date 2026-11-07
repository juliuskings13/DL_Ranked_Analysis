"""
assemble_video.py — turn one talking-head recording + the cards it talks about into a finished video
(and shorts), using render_pipeline.py's ffmpeg helpers, encoder settings and hook-text style.

Workflow
  1. Record yourself talking through the cards (one long take is fine: redos and pauses get cut).
  2. In LosslessCut, mark one segment per card and name it after the card, e.g. patch_hero_rat_king
     (".png" on the end is fine). Want a short from a segment too? Add "| HOOK TEXT" to its name,
     exactly like render_pipeline: "patch_hero_rat_king | RAT KING GOT GUTTED". "| -" makes a short
     without hook text. Cutting a stumble out of the middle of a card? Give both pieces the same card
     name: back-to-back segments on one card share one animation (it builds once and keeps going).
  3. Export the segments: File -> Export project -> CSV (Start, End, Name).
  4. Run:
       python assemble_video.py RECORDING.mp4 SEGMENTS.csv patches/2026-10-30_...      (or a week folder)
       python assemble_video.py REC.mp4 SEGS.csv season_2_start/ patches/<patch>/ patch_2026-11-05/ \
           --order season2_patch                       (cards from several folders in one video)
         [--add closer.mp4]                 (a whole recording, no cuts: named after the card it goes with)

What it does
  - renders every card's animation to last exactly as long as you talk about it (text cards reveal their
    lines at their normal speed, charts build in 6s, then the card holds until the take ends; if a take is
    shorter than the build, the camera's first frame is frozen (silent) at the start to make room)
  - puts face + card side by side at 2560x1440 (your face 1120x1440 on the left, taken 1:1 from x 480-1600 of
    the recording; the card upscaled to 1440x1440 on the right) and cuts your recording frame-accurately
    (it's re-encoded once here anyway, so keyframes don't matter)
  - joins every segment into one video (stream copy, no second encode) + YouTube chapters
  - renders a 9:16 short (card on top, face below, hook text + intro sound like your other shorts) for every
    segment whose name has "| ..."

Order and chapter titles come from an order file (format in orders/patch_notes.txt):
  --order auto (default)  order.txt in the cards folder if there is one; else orders/patch_notes.txt for a patch
                          folder; else the order of your CSV
  --order csv             the order of your CSV, chapter titles from the card names
  --order NAME | FILE     orders/NAME.txt next to this script, or any order file

Face position: the recording is treated as 2560x1440 (scaled to it if not). Main video takes x 480-1600 at
full height; shorts take x 300-1700 below the top 300 px. Change them with --face-crop / --short-crop (w:h:x:y).

Takes from another recording (e.g. closing thoughts recorded later), repeatable; sorted into the same order:
  --add closer.mp4                      the whole file, on the card its name matches (closer -> patch_closer)
  --add thoughts.mp4 patch_closer       the whole file, on that card ("patch_closer | HOOK" for a short too)
  --add more.mp4 more.csv               cut by its own LosslessCut CSV
"""
from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- layout -----------------------------------------------------------------------
MAIN_W, MAIN_H = 2560, 1440          # face 1120x1440 on the left + card 1440x1440 on the right
CARD_PX = 1440                       # card panel size; cards already render at 1440 (theme RENDER_SCALE), else lanczos-scaled to this
SOURCE_W, SOURCE_H = 2560, 1440      # recordings are normalized to this first, so the crops below always fit
# camera areas (w:h:x:y on the 2560x1440 recording). Main: x 480-1600, full height (1:1 pixels, no scaling).
# Shorts: x 300-1700, top 300 px cut; trimmed from the bottom to the 1440x1120 panel's shape, then scaled.
FACE_CROP_MAIN = "1120:1440:480:0"
FACE_CROP_SHORT = "1400:1140:300:300"
SHORT_CARD = 1440                    # shorts: 1440x1440 card on top, face 1440x1120 below (render_pipeline OUT_W/H)
SHORT_HOOK_Y = 1450                  # hook text sits at the top of the face area, not over the card
CARD_FPS = 30                        # cards are rendered at 30 fps (frames are converted to the recording's rate)
CHAPTER_MIN_S = 10                   # YouTube needs chapters of 10 s or more; shorter ones merge with the next
PIPELINE_CANDIDATES = [Path(__file__).resolve().parent / "render_pipeline.py",
                       Path.home() / "Streaming" / "render_pipeline.py",
                       Path.home() / "Streaming" / "Scripts" / "render_pipeline.py"]


def load_pipeline(path: str | None):
    """render_pipeline.py as a module (its settings and helpers), from --pipeline or the usual places."""
    cands = [Path(path).expanduser()] if path else PIPELINE_CANDIDATES
    for p in cands:
        if p.exists():
            spec = importlib.util.spec_from_file_location("render_pipeline", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("render_pipeline.py not found. Pass its location with --pipeline /path/to/render_pipeline.py")


# --- segments ---------------------------------------------------------------------
def card_of(name: str) -> str:
    return re.sub(r"\.(png|jpg|mp4|mov)$", "", name.strip(), flags=re.I)


ORDERS_DIR = Path(__file__).resolve().parent / "orders"


def load_order(path: Path) -> list[dict]:
    """Order file -> [{pattern, title, az}], one per line ("pattern | chapter title | options")."""
    rules = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split("|")]
        rules.append({"pattern": parts[0], "title": (parts[1] if len(parts) > 1 else "") or "{name}",
                      "az": len(parts) > 2 and "a-z" in parts[2].lower()})
    if not rules:
        raise SystemExit(f"{path} has no order lines")
    return rules


def resolve_order(order: str, cards_dir: Path, patch: bool) -> tuple[list[dict] | None, str]:
    """(rules or None for CSV order, description)."""
    if order == "csv":
        return None, "CSV order"
    if order == "auto":
        if (cards_dir / "order.txt").exists():
            return load_order(cards_dir / "order.txt"), str(cards_dir / "order.txt")
        if patch:
            return load_order(ORDERS_DIR / "patch_notes.txt"), "patch notes preset"
        return None, "CSV order (no order.txt in the cards folder)"
    path = Path(order) if Path(order).suffix else ORDERS_DIR / f"{order}.txt"
    if not path.exists():
        raise SystemExit(f"Order file not found: {path}")
    return load_order(path), str(path)


def name_part(card: str, pattern: str, display: dict) -> str:
    """The part of the card name the pattern's * matched, as a title: patch_hero_mo_krill_2 -> Mo & Krill."""
    prefix = pattern.split("*", 1)[0]
    rest = card[len(prefix):] if "*" in pattern and card.startswith(prefix) else card
    rest = re.sub(r"_\d+$", "", rest.strip("_")) or card
    return display.get(rest, rest.replace("_", " ").title())


def apply_order(segs: list[dict], rules: list[dict] | None, display: dict) -> list[dict]:
    """Sort segments by the order file and give each its chapter title."""
    def default(card):
        return re.sub(r"_\d+$", "", card).replace("_", " ").title()
    if rules is None:
        for s in segs:
            s["title"] = default(s["card"])
        return segs
    for s in segs:
        k = next((i for i, r in enumerate(rules) if fnmatch.fnmatchcase(s["card"], r["pattern"])), None)
        s["rule"] = len(rules) if k is None else k
        s["title"] = (default(s["card"]) if k is None else
                      rules[k]["title"].replace("{name}", name_part(s["card"], rules[k]["pattern"], display)))
    return sorted(segs, key=lambda s: (s["rule"], s["card"] if s["rule"] < len(rules) and rules[s["rule"]]["az"]
                                       else "", s["row"]))
FREEZE_SPARE_S = 0.5                 # a take shorter than its card's build gets this much extra after the build


def read_segments(rp, takes: list[tuple[str, Path]], card_names: set[str]) -> list[dict]:
    segs, problems = [], []
    for f, (rec, src) in enumerate(takes):
        if str(src).lower().endswith(".csv"):
            segs += _read_one(rp, rec, Path(src), card_names, problems, f)
        else:                                          # whole recording on one card
            lab = rp.parse_label(src or Path(rec).stem)
            card = card_of(lab["name"])
            card = card if card in card_names else f"patch_{card}" if f"patch_{card}" in card_names else card
            if card not in card_names:
                problems.append(f"{Path(rec).name}: no card named {card!r} (add the card name after the file)")
                continue
            dur = rp.get_video_duration(rec)
            hook = lab["hook"]
            segs.append({"card": card, "start": 0.0, "end": dur, "dur": dur, "row": (f, 0), "rec": rec,
                         "short": "|" in str(src), "hook": None if hook.strip() in ("", "-") else hook})
    if problems:
        print("  [!] Skipped (check the names in LosslessCut):\n      " + "\n      ".join(problems))
    return segs


def _read_one(rp, rec: str, csv_path: Path, card_names: set[str], problems: list, f: int) -> list[dict]:
    df = rp.read_csv_clean(str(csv_path))
    segs = []
    for i, row in df.iterrows():
        lab = rp.parse_label(row.get("Name"))
        if not lab["name"]:
            continue
        card = card_of(lab["name"])
        start, end = float(row["Start"]), float(row["End"])
        if card not in card_names:
            problems.append(f"{csv_path.name} row {i + 2}: no card named {card!r}")
            continue
        hook = lab["hook"]
        short = "|" in str(row.get("Name"))
        segs.append({"card": card, "start": start, "end": end, "dur": end - start, "row": (f, i), "rec": rec,
                     "short": short, "hook": None if hook.strip() in ("", "-") else hook})
    return segs


def chapters(segs: list[dict]) -> list[str]:
    """YouTube chapter lines. Consecutive segments with the same title merge; chapters under 10 s merge with
    the next one (names joined), so YouTube accepts the list."""
    blocks, t = [], 0.0
    for s in segs:
        title = s["title"]
        if blocks and blocks[-1]["titles"][-1] == title:
            blocks[-1]["dur"] += s.get("len", s["dur"])
        else:
            blocks.append({"start": t, "dur": s.get("len", s["dur"]), "titles": [title]})
        t += s.get("len", s["dur"])
    merged = []
    for b in blocks:
        if merged and merged[-1]["dur"] < CHAPTER_MIN_S:
            merged[-1]["dur"] += b["dur"]
            merged[-1]["titles"] += b["titles"]
        else:
            merged.append(dict(b))
    if len(merged) > 1 and merged[-1]["dur"] < CHAPTER_MIN_S:
        last = merged.pop()
        merged[-1]["dur"] += last["dur"]
        merged[-1]["titles"] += last["titles"]

    def stamp(x):
        x = int(x)
        return f"{x // 3600}:{x % 3600 // 60:02d}:{x % 60:02d}" if x >= 3600 else f"{x // 60}:{x % 60:02d}"

    out = []
    for b in merged:
        names = b["titles"]
        label = ", ".join(names[:3]) + (f" + {len(names) - 3} more" if len(names) > 3 else "")
        out.append(f"{stamp(b['start'])} {label}")
    return out


# --- card clips -------------------------------------------------------------------
def collect_all(dirs: list[Path]) -> dict:
    """Cards from every folder, merged; on a name clash the first folder listed wins."""
    import animate as A
    out = {}
    for d in dirs:
        for name, draw in A.collect_cards(d).items():
            if name in out:
                print(f"  [!] card {name!r} is in more than one folder; using the one from the first")
                continue
            out[name] = draw
    return out


def render_cards(data_dirs: list[Path], segs: list[dict], tmp: Path, workers: int) -> None:
    import animate as A
    import theme as T
    A._CARDS.clear()
    A._CARDS.update(collect_all(data_dirs))
    # consecutive segments on the same card (e.g. one take with a stumble cut out of the middle) share ONE
    # animation spanning all of them: the card builds once and each later piece picks up where the last ended
    groups: list[list[dict]] = []
    for s in segs:
        if groups and groups[-1][-1]["card"] == s["card"]:
            groups[-1].append(s)
        else:
            groups.append([s])
    with mp.get_context("fork").Pool(workers) as pool:
        for k, g in enumerate(groups):
            draw = A._CARDS[g[0]["card"]]
            build = getattr(draw, "build_s", None) or A.DEFAULT_BUILD_S     # always the card's natural speed
            total = sum(s["dur"] for s in g)
            if k == 0 and build + FREEZE_SPARE_S > total:
                # the video's first card: never open on a frozen, silent camera; build faster to fit the take
                build = max(0.6, total - FREEZE_SPARE_S)
            # takes shorter than the build: freeze the first camera frame (silent) at the start of the group
            pre = max(0.0, build + FREEZE_SPARE_S - total)
            hold = total + pre - build + 0.5                   # a little spare; the composite trims to the length
            out = tmp / f"card_{k:03d}"
            A.render_scene(A.Scene(g[0]["card"], build, hold), pool, out, CARD_FPS, "h264")
            offset = 0.0
            for i, s in enumerate(g):
                s["pre"] = pre if i == 0 else 0.0
                s["len"] = s["dur"] + s["pre"]
                s["clip"], s["clip_offset"], s["build"] = str(out.with_suffix(".mp4")), offset, build
                offset += s["len"]
            print(f"  [+] card {k + 1}/{len(groups)} {g[0]['card']}: {total:.1f}s"
                  + (f" over {len(g)} pieces" if len(g) > 1 else " take") + f", builds in {build:.1f}s"
                  + (f"  (frozen camera for {pre:.1f}s at the start)" if pre else ""))
    T.plt.close("all")


# --- composites -------------------------------------------------------------------
def face_filter(face_crop: str | None, w: int, h: int) -> str:
    """Normalize the recording to 2560x1440, crop the camera area, trim it to the panel's shape (sides
    evenly, or from the bottom, so the top of your head stays put) and scale to the panel."""
    ar = w / h
    pre = f"scale={SOURCE_W}:{SOURCE_H},setsar=1," + (f"crop={face_crop}," if face_crop else "")
    return (f"{pre}crop='if(gt(iw/ih,{ar:.6f}),ih*{ar:.6f},iw)':'if(gt(iw/ih,{ar:.6f}),ih,iw/{ar:.6f})'"
            f":'(iw-ow)/2':0,scale={w}:{h}:flags=lanczos,setsar=1")


def video_tail(rp, use_vaapi: bool) -> str:
    return "format=nv12,hwupload" if use_vaapi else "format=yuv420p"


def enc(rp, use_vaapi: bool, quality: int) -> list[str]:
    return rp.encoder_args(quality) if use_vaapi else rp.software_args()


def freeze(s: dict) -> tuple[str, str]:
    """Filters that hold the take's first frame (and silence) for s['pre'] seconds before it starts."""
    pre = s.get("pre", 0.0)
    if pre <= 0:
        return "", ""
    return f"tpad=start_mode=clone:start_duration={pre:.3f},", f"adelay={int(round(pre * 1000))}:all=1,"


def main_piece_cmd(rp, raw: str, s: dict, out: str, face_crop: str | None, use_vaapi: bool) -> list[str]:
    fps = rp.get_video_fps(raw)
    fps_s = f"{fps.numerator}/{fps.denominator}"
    take, d = s["dur"], s["len"]
    vpad, apad = freeze(s)
    cmd = ["ffmpeg", *rp.FFMPEG_BASE]
    if use_vaapi:
        cmd += ["-vaapi_device", rp.VAAPI_DEVICE]
    cmd += ["-ss", f"{s['start']:.3f}", "-t", f"{take:.3f}", "-i", raw,
            "-ss", f"{s.get('clip_offset', 0.0):.3f}", "-i", s["clip"]]
    a = rp.AUDIO_TRACK if rp.AUDIO_TRACK is not None else 0
    # camera on the LEFT, card on the right
    g = (f"[1:v]fps={fps_s},scale={CARD_PX}:{CARD_PX}:flags=lanczos,setsar=1,tpad=stop_mode=clone:stop_duration=5[c];"
         f"[0:v]fps={fps_s},{vpad}{face_filter(face_crop, MAIN_W - CARD_PX, MAIN_H)}[f];"
         f"[f][c]hstack=inputs=2,trim=duration={d:.3f},setpts=PTS-STARTPTS,{video_tail(rp, use_vaapi)}[v];"
         f"[0:a:{a}]aresample=48000,{apad}apad,atrim=duration={d:.3f},asetpts=PTS-STARTPTS[a]")
    return cmd + ["-filter_complex", g, "-map", "[v]", "-map", "[a]", *enc(rp, use_vaapi, rp.SEGMENT_QUALITY),
                  "-c:a", "aac", "-b:a", "192k", "-t", f"{d:.3f}", out]


def short_cmd(rp, raw: str, s: dict, out: str, face_crop: str | None, use_vaapi: bool, png: str | None, fx) -> list[str]:
    fps = rp.get_video_fps(raw)
    fps_s = f"{fps.numerator}/{fps.denominator}"
    take, d = s["dur"], s["len"]
    vpad, apad = freeze(s)
    W, H = rp.OUT_W, rp.OUT_H
    cmd = ["ffmpeg", *rp.FFMPEG_BASE]
    if use_vaapi:
        cmd += ["-vaapi_device", rp.VAAPI_DEVICE]
    cmd += ["-ss", f"{s['start']:.3f}", "-t", f"{take:.3f}", "-i", raw,
            "-ss", f"{s.get('clip_offset', 0.0):.3f}", "-i", s["clip"]]
    n = 2
    if png:
        cmd += ["-loop", "1", "-framerate", fps_s, "-t", str(rp.HOOK_SHOW_SECONDS + rp.HOOK_FADE_SECONDS), "-i", png]
        png_i, n = n, n + 1
    a = rp.AUDIO_TRACK if rp.AUDIO_TRACK is not None else 0
    g = [f"[1:v]fps={fps_s},scale={W}:{W}:flags=lanczos,setsar=1,tpad=stop_mode=clone:stop_duration=5[c]",
         f"[0:v]fps={fps_s},{vpad}{face_filter(face_crop, W, H - W)}[f]",
         f"[c][f]vstack=inputs=2,trim=duration={d:.3f},setpts=PTS-STARTPTS[base]"]
    last = "base"
    if png:
        g += [f"[{png_i}:v]format=rgba,fade=t=out:st={rp.HOOK_SHOW_SECONDS}:d={rp.HOOK_FADE_SECONDS}:alpha=1[ov]",
              f"[base][ov]overlay=0:{SHORT_HOOK_Y}:eof_action=pass[comp]"]
        last = "comp"
    g.append(f"[{last}]{video_tail(rp, use_vaapi)}[v]")
    g.append(f"[0:a:{a}]aresample=48000,aformat=channel_layouts=stereo,{apad}apad,atrim=duration={d:.3f}[a_main]")
    amap = "[a_main]"
    if fx:
        cmd += ["-i", fx[0]]
        gain = fx[1] + rp.INTRO_FX_MASTER_GAIN_DB
        g.append(f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo,volume={gain}dB[a_fx]")
        g.append("[a_main][a_fx]amix=inputs=2:duration=first:normalize=0:dropout_transition=0,alimiter=limit=0.97[a]")
        amap = "[a]"
    return cmd + ["-filter_complex", ";".join(g), "-map", "[v]", "-map", amap,
                  *enc(rp, use_vaapi, rp.SHORT_QUALITY), "-c:a", "aac", "-b:a", "192k",
                  "-movflags", "+faststart", "-t", f"{d:.3f}", out]


def run_with_fallback(rp, build_cmd, label: str) -> bool:
    """GPU encode first (if enabled), CPU x264 if that fails: same idea as render_pipeline."""
    for use_vaapi in ([True, False] if rp.USE_VAAPI else [False]):
        ok, err = rp.run_ffmpeg(build_cmd(use_vaapi))
        if ok:
            return True
        print(f"  [!] {label} ({'VAAPI' if use_vaapi else 'x264'}) failed: {rp.short_error(err)}")
    return False


def make_main_video(rp, args, segs: list[dict], tmp: Path, out: Path) -> None:
    """Card + face for every segment, joined into the long video, plus its chapters file."""
    pieces = []
    for k, s in enumerate(segs):
        piece = str(tmp / f"piece_{k:03d}.mkv")
        if not run_with_fallback(rp, lambda v, s=s, piece=piece: main_piece_cmd(
                rp, s["rec"], s, piece, args.face_crop, v), s["card"]):
            raise SystemExit(f"Could not render the segment for {s['card']}.")
        pieces.append(piece)
        print(f"  [+] {k + 1}/{len(segs)} {s['card']}")
    # every piece has the same encoder settings, so they join without a second encode (as render_pipeline does)
    lst = tmp / "pieces.txt"
    lst.write_text("".join(f"file {rp.concat_quote(p)}\n" for p in pieces))
    ok, err = rp.run_ffmpeg(["ffmpeg", *rp.FFMPEG_BASE, "-f", "concat", "-safe", "0", "-i", str(lst),
                             "-c", "copy", "-movflags", "+faststart", str(out)])
    if not ok:
        raise SystemExit(f"Joining failed: {rp.short_error(err)}")
    chap = chapters(segs)
    out.with_suffix(".chapters.txt").write_text("\n".join(chap) + "\n")
    print(f"--> Video: {out} ({rp.get_video_duration(str(out)) / 60:.1f} min)\n--> Chapters: "
          f"{out.with_suffix('.chapters.txt')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Recording + cards -> finished video, chapters and shorts")
    ap.add_argument("recording")
    ap.add_argument("segments_csv", type=Path, help="LosslessCut export (Start, End, Name)")
    ap.add_argument("cards_dirs", type=Path, nargs="+", metavar="cards_dir",
                    help="folder(s) the cards come from (patch, week, period, season...). Several can be "
                         "given for one video; the first one's order.txt / patch preset sets the order")
    ap.add_argument("--out", type=Path, default=None, help="finished video (default: <longs_dir>/<csv name>.mp4)")
    ap.add_argument("--order", default="auto",
                    help="auto (default), csv, a preset name from orders/ (e.g. patch_notes), or an order file")
    ap.add_argument("--add", nargs="+", action="append", default=[], metavar="RECORDING [CSV|CARD]",
                    help="another recording: alone (whole file, card from its name), + a card name, "
                         "or + its LosslessCut CSV; repeatable")
    ap.add_argument("--face-crop", default=FACE_CROP_MAIN,
                    help=f"camera area for the main video on the 2560x1440 recording, w:h:x:y (default {FACE_CROP_MAIN})")
    ap.add_argument("--short-crop", default=FACE_CROP_SHORT,
                    help=f"camera area for shorts, w:h:x:y (default {FACE_CROP_SHORT})")
    ap.add_argument("--pipeline", default=None, help="path to render_pipeline.py")
    ap.add_argument("--cpu", action="store_true", help="x264 instead of the GPU encoder")
    ap.add_argument("--no-shorts", action="store_true")
    ap.add_argument("--shorts-only", action="store_true",
                    help="only make the shorts (skips the long video; renders just the cards shorts use)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()
    for k in ("face_crop", "short_crop"):              # accept "crop=1120:1440:630:0" too (ffmpeg filter syntax)
        v = re.sub(r"^\s*crop\s*=\s*", "", getattr(args, k) or "")
        if v and not re.fullmatch(r"\d+:\d+:\d+:\d+", v):
            raise SystemExit(f"--{k.replace('_', '-')} must be w:h:x:y, e.g. 1120:1440:630:0 (got {getattr(args, k)!r})")
        setattr(args, k, v or None)

    rp = load_pipeline(args.pipeline)
    if args.cpu:
        rp.USE_VAAPI = False
    import animate as A
    args.cards_dir = args.cards_dirs[0]
    patch_dirs = [d for d in args.cards_dirs if (d / "changes.csv").exists()]
    patch = bool(patch_dirs)
    cards = collect_all(args.cards_dirs)
    for a in args.add:
        if len(a) > 2:
            raise SystemExit(f"--add takes a recording and at most one CSV or card name, got: {' '.join(a)}")
    takes = [(args.recording, str(args.segments_csv))] + [(a[0], a[1] if len(a) > 1 else "") for a in args.add]
    segs = read_segments(rp, takes, set(cards))
    if not segs:
        raise SystemExit("No segment matched a card.")
    display = {}                                       # card-name part -> proper name for chapter titles
    from hero_spotlight import slug
    try:
        import theme as T
        display = {slug(h): h for h in T.HERO_COLORS} if hasattr(T, "HERO_COLORS") else {}
    except Exception:
        pass
    if patch:
        import patch_notes as P
        for d in patch_dirs:
            display.update({slug(h): h for h in P.load_changes(d)[0]["subject"].unique()})
    rules, how = resolve_order(args.order, args.cards_dir, patch)
    segs = apply_order(segs, rules, display)
    print(f"Order: {how}")
    total = sum(s["dur"] for s in segs)
    print(f"{len(segs)} segments, {total / 60:.1f} min of video, {sum(s['short'] for s in segs)} shorts")

    out = args.out or Path(rp.longs_dir) / (args.segments_csv.stem + ".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(out.parent)) as td:
        tmp = Path(td)
        if args.shorts_only:
            # every piece of a card that has a short, so its animation is timed exactly as in the long video
            short_cards = {s["card"] for s in segs if s["short"]}
            segs = [s for s in segs if s["card"] in short_cards]
            if not segs:
                raise SystemExit("No shorts flagged: name a segment 'card | HOOK TEXT' in LosslessCut.")
        print("--> Rendering card animations to the length of each take...")
        render_cards(args.cards_dirs, segs, tmp, args.workers)

        if not args.shorts_only:
            make_main_video(rp, args, segs, tmp, out)

        if not args.no_shorts:
            # back-to-back pieces of one card with the same hook = ONE short (e.g. a take with a stumble cut
            # out); the hook text and intro sound go on its first piece only
            groups: list[list[dict]] = []
            for s in segs:
                if not s["short"]:
                    continue
                prev = groups[-1][-1] if groups else None
                if prev is not None and prev["card"] == s["card"] and prev["hook"] == s["hook"] \
                        and segs.index(prev) + 1 == segs.index(s):
                    groups[-1].append(s)
                else:
                    groups.append([s])
            made, used = 0, {}
            for g in groups:
                base = rp.clean_filename(g[0]["card"])
                used[base] = used.get(base, 0) + 1                  # two shorts from one card: _short, _short_2
                name = base if used[base] == 1 else f"{base}_{used[base]}"
                dest = Path(rp.shorts_dir) / f"{name}_short.mp4"
                dest.parent.mkdir(parents=True, exist_ok=True)
                hook = g[0]["hook"]
                png = None
                if hook:
                    png = str(tmp / f"hook_{name}.png")
                    rp.generate_text_overlay_image(hook, png)
                fx = rp.pick_intro_fx() if hook else None
                parts, ok = [], True
                for i, s in enumerate(g):
                    out_i = str(dest) if len(g) == 1 else str(tmp / f"short_{name}_{i}.mp4")
                    ok = run_with_fallback(rp, lambda v, s=s, o=out_i, png=(png if i == 0 else None),
                                           fx=(fx if i == 0 else None): short_cmd(
                        rp, s["rec"], s, o, args.short_crop, v, png, fx), f"short {name}")
                    if not ok:
                        break
                    parts.append(out_i)
                if ok and len(parts) > 1:
                    lst = tmp / f"short_{name}.txt"
                    lst.write_text("".join(f"file {rp.concat_quote(x)}\n" for x in parts))
                    ok, err = rp.run_ffmpeg(["ffmpeg", *rp.FFMPEG_BASE, "-f", "concat", "-safe", "0", "-i", str(lst),
                                             "-c", "copy", "-movflags", "+faststart", str(dest)])
                    if not ok:
                        print(f"  [!] joining short {name} failed: {rp.short_error(err)}")
                if ok:
                    made += 1
                    secs = sum(s["len"] for s in g)
                    print(f"  [+] short: {dest}  ({secs:.0f}s" + (f", {len(g)} pieces" if len(g) > 1 else "") + ")")
            if made:
                print(f"--> {made} shorts in {rp.shorts_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())