"""
assemble_video.py — turn one talking-head recording + the cards it talks about into a finished video
(and shorts), using render_pipeline.py's ffmpeg helpers, encoder settings and hook-text style.

Workflow
  1. Record yourself talking through the cards (one long take is fine: redos and pauses get cut).
  2. In LosslessCut, mark one segment per card and name it after the card, e.g. patch_hero_rat_king
     (".png" on the end is fine). Want a short from a segment too? Add "| HOOK TEXT" to its name,
     exactly like render_pipeline: "patch_hero_rat_king | RAT KING GOT GUTTED". "| -" makes a short
     without hook text.
  3. Export the segments: File -> Export project -> CSV (Start, End, Name).
  4. Run:
       python assemble_video.py RECORDING.mp4 SEGMENTS.csv patches/2026-10-30_...      (or a week folder)

What it does
  - renders every card's animation to last exactly as long as you talk about it (text cards reveal their
    lines at their normal speed, charts build in 6s, then the card holds until the take ends; if a take is
    shorter than the build, the camera's first frame is frozen (silent) at the start to make room)
  - puts face + card side by side (16:9: your face 840x1080 on the left, card 1080x1080 on the right) and cuts
    your recording frame-accurately (it's re-encoded once here anyway, so keyframes don't matter)
  - joins every segment into one video (stream copy, no second encode) + YouTube chapters
  - renders a 9:16 short (card on top, face below, hook text + intro sound like your other shorts) for every
    segment whose name has "| ..."

Order: patch folders default to opener -> item overviews -> item cards -> hero overviews -> heroes A-Z ->
general -> closer (patch_closer, the winners & losers card); anything else
uses the CSV order. --order csv forces the CSV order.

Face position: by default the whole recording frame is center-cropped to fit the face panel. If your camera
is only part of the recording, give its rectangle like render_pipeline's layouts: --face-crop 517:291:2045:563
(width:height:x:y). Record the camera at its own full resolution if you can; a small cam region gets blurry
when it's blown up to 840x1080.
"""
from __future__ import annotations

import argparse
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
MAIN_W, MAIN_H = 1920, 1080          # card 1080x1080 + face 840x1080
CARD_PX = 1080
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


# opener -> items overview -> item cards -> heroes overview -> heroes A-Z -> general -> closer
PATCH_ORDER = [r"patch_opener$", r"patch_items$", r"patch_items_size$", r"patch_items_weapon",
               r"patch_items_vitality", r"patch_items_spirit", r"patch_items_", r"patch_heroes$",
               r"patch_heroes_size$", r"patch_hero_", r"patch_general", r"patch_closer$"]
HERO_RANK = PATCH_ORDER.index(r"patch_hero_")
FREEZE_SPARE_S = 0.5                 # a take shorter than its card's build gets this much extra after the build


def read_segments(rp, csv_path: Path, card_names: set[str], order: str, patch: bool) -> list[dict]:
    df = rp.read_csv_clean(str(csv_path))
    segs, problems = [], []
    for i, row in df.iterrows():
        lab = rp.parse_label(row.get("Name"))
        if not lab["name"]:
            continue
        card = card_of(lab["name"])
        start, end = float(row["Start"]), float(row["End"])
        if card not in card_names:
            problems.append(f"row {i + 2}: no card named {card!r}")
            continue
        hook = lab["hook"]
        short = "|" in str(row.get("Name"))
        segs.append({"card": card, "start": start, "end": end, "dur": end - start, "row": i,
                     "short": short, "hook": None if hook.strip() in ("", "-") else hook})
    if problems:
        print("  [!] Skipped (check the names in LosslessCut):\n      " + "\n      ".join(problems))
    if order == "auto" and patch:
        def rank(s):
            k = next((n for n, pat in enumerate(PATCH_ORDER) if re.match(pat, s["card"])), len(PATCH_ORDER))
            return (k, s["card"] if k == HERO_RANK else "", s["row"])        # heroes A-Z, everything else as recorded
        segs.sort(key=rank)
    return segs


def chapter_title(card: str, hero_names: dict) -> str:
    fixed = {"patch_opener": "Intro", "patch_heroes": "Hero changes overview",
             "patch_heroes_size": "How big were the changes", "patch_items": "Item changes overview",
             "patch_items_size": "How big were the item changes", "patch_closer": "Winners & losers"}
    if card in fixed:
        return fixed[card]
    m = re.match(r"patch_hero_(.+?)(?:_\d+)?$", card)
    if m:
        return hero_names.get(m.group(1), m.group(1).replace("_", " ").title())
    m = re.match(r"patch_items_(weapon|vitality|spirit|other)", card)
    if m:
        return f"{m.group(1).title()} items"
    if card.startswith("patch_general"):
        return "General changes"
    return card.replace("_", " ").title()


def chapters(segs: list[dict], hero_names: dict) -> list[str]:
    """YouTube chapter lines. Consecutive segments with the same title merge; chapters under 10 s merge with
    the next one (names joined), so YouTube accepts the list."""
    blocks, t = [], 0.0
    for s in segs:
        title = chapter_title(s["card"], hero_names)
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
def render_cards(data_dir: Path, segs: list[dict], tmp: Path, workers: int) -> None:
    import animate as A
    import theme as T
    A._CARDS.clear()
    A._CARDS.update(A.collect_cards(data_dir))
    with mp.get_context("fork").Pool(workers) as pool:
        for k, s in enumerate(segs):
            draw = A._CARDS[s["card"]]
            build = getattr(draw, "build_s", None) or A.DEFAULT_BUILD_S     # always the card's natural speed
            d = s["dur"]
            # take shorter than the build: freeze the first camera frame (silent) at the start of the segment
            s["pre"] = max(0.0, build + FREEZE_SPARE_S - d)
            s["len"] = d + s["pre"]
            hold = s["len"] - build + 0.5                      # a little spare; the composite trims to the length
            out = tmp / f"card_{k:03d}"
            A.render_scene(A.Scene(s["card"], build, hold), pool, out, CARD_FPS, "h264")
            s["clip"] = str(out.with_suffix(".mp4"))
            s["build"] = build
            print(f"  [+] card {k + 1}/{len(segs)} {s['card']}: {d:.1f}s take, builds in {build:.1f}s"
                  + (f"  (frozen camera for {s['pre']:.1f}s at the start)" if s["pre"] else ""))
    T.plt.close("all")


# --- composites -------------------------------------------------------------------
def face_filter(face_crop: str | None, w: int, h: int) -> str:
    """Crop the face area (optional rectangle), then center-crop to the panel's shape and scale."""
    ar = w / h
    pre = f"crop={face_crop}," if face_crop else ""
    return (f"{pre}crop='if(gt(iw/ih,{ar:.6f}),ih*{ar:.6f},iw)':'if(gt(iw/ih,{ar:.6f}),ih,iw/{ar:.6f})',"
            f"scale={w}:{h},setsar=1")


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
    cmd += ["-ss", f"{s['start']:.3f}", "-t", f"{take:.3f}", "-i", raw, "-i", s["clip"]]
    a = rp.AUDIO_TRACK if rp.AUDIO_TRACK is not None else 0
    # camera on the LEFT, card on the right
    g = (f"[1:v]fps={fps_s},scale={CARD_PX}:{CARD_PX},setsar=1,tpad=stop_mode=clone:stop_duration=5[c];"
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
    cmd += ["-ss", f"{s['start']:.3f}", "-t", f"{take:.3f}", "-i", raw, "-i", s["clip"]]
    n = 2
    if png:
        cmd += ["-loop", "1", "-framerate", fps_s, "-t", str(rp.HOOK_SHOW_SECONDS + rp.HOOK_FADE_SECONDS), "-i", png]
        png_i, n = n, n + 1
    a = rp.AUDIO_TRACK if rp.AUDIO_TRACK is not None else 0
    g = [f"[1:v]fps={fps_s},scale={W}:{W},setsar=1,tpad=stop_mode=clone:stop_duration=5[c]",
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Recording + cards -> finished video, chapters and shorts")
    ap.add_argument("recording")
    ap.add_argument("segments_csv", type=Path, help="LosslessCut export (Start, End, Name)")
    ap.add_argument("cards_dir", type=Path, help="patch folder or week folder the cards come from")
    ap.add_argument("--out", type=Path, default=None, help="finished video (default: <longs_dir>/<csv name>.mp4)")
    ap.add_argument("--order", choices=["auto", "csv"], default="auto")
    ap.add_argument("--face-crop", default=None, help="camera rectangle in the recording, w:h:x:y")
    ap.add_argument("--pipeline", default=None, help="path to render_pipeline.py")
    ap.add_argument("--cpu", action="store_true", help="x264 instead of the GPU encoder")
    ap.add_argument("--no-shorts", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    rp = load_pipeline(args.pipeline)
    if args.cpu:
        rp.USE_VAAPI = False
    import animate as A
    patch = (args.cards_dir / "changes.csv").exists()
    cards = A.collect_cards(args.cards_dir)
    segs = read_segments(rp, args.segments_csv, set(cards), args.order, patch)
    if not segs:
        raise SystemExit("No segment matched a card.")
    hero_names = {}
    if patch:
        import patch_notes as P
        from hero_spotlight import slug
        hero_names = {slug(h): h for h in P.load_changes(args.cards_dir)[0].query("kind == 'hero'")["subject"].unique()}
    total = sum(s["dur"] for s in segs)
    print(f"{len(segs)} segments, {total / 60:.1f} min of video, {sum(s['short'] for s in segs)} shorts")

    out = args.out or Path(rp.longs_dir) / (args.segments_csv.stem + ".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(out.parent)) as td:
        tmp = Path(td)
        print("--> Rendering card animations to the length of each take...")
        render_cards(args.cards_dir, segs, tmp, args.workers)

        print("--> Card + face, one piece per segment...")
        pieces = []
        for k, s in enumerate(segs):
            piece = str(tmp / f"piece_{k:03d}.mkv")
            if not run_with_fallback(rp, lambda v, s=s, piece=piece: main_piece_cmd(
                    rp, args.recording, s, piece, args.face_crop, v), s["card"]):
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
        chap = chapters(segs, hero_names)
        out.with_suffix(".chapters.txt").write_text("\n".join(chap) + "\n")
        print(f"--> Video: {out} ({rp.get_video_duration(str(out)) / 60:.1f} min)\n--> Chapters: "
              f"{out.with_suffix('.chapters.txt')}")

        if not args.no_shorts:
            made = 0
            for s in [s for s in segs if s["short"]]:
                name = rp.clean_filename(s["card"])
                dest = Path(rp.shorts_dir) / f"{name}_short.mp4"
                dest.parent.mkdir(parents=True, exist_ok=True)
                png = None
                if s["hook"]:
                    png = str(tmp / f"hook_{name}.png")
                    rp.generate_text_overlay_image(s["hook"], png)
                fx = rp.pick_intro_fx() if s["hook"] else None
                if run_with_fallback(rp, lambda v, s=s, png=png, fx=fx: short_cmd(
                        rp, args.recording, s, str(dest), args.face_crop, v, png, fx), f"short {name}"):
                    made += 1
                    print(f"  [+] short: {dest}")
            if made:
                print(f"--> {made} shorts in {rp.shorts_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())