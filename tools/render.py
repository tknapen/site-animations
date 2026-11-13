#!/usr/bin/env python3
"""Render the website's clips in the light and the dark site style and prepare them for the site.

For every project under families/website/ (or the ones named), this
  1. renders it with manim8 under knapen_site and under knapen_site_dark (same scene, same timing),
  2. encodes each render for the web (H.264, at most 1280 px wide, no sound, index first) and takes
     the last frame as the still shown before the clip plays,
  3. writes them to out/clips/<id>.mp4 and .webp, and out/clips/dark/<id>.mp4 and .webp,
     with out/clips.yml holding the size and length of each clip.
With --site <path to the tknapen.github.io checkout> the files are also copied into its assets/clips/,
where the site picks the dark render up by itself. The entry in the site's _data/clips.yml (title,
caption) is written by hand: the caption is the text alternative.

Usage:
  python3 tools/render.py                         # all projects, draft profile
  python3 tools/render.py divisive_normalization --profile final --site ../tknapen.github.io

Needs manim8 on PATH (see README), ffmpeg with libx264 and libwebp, and the family environment
(`manim8 family env sync families/website`). The clip id is the project id with hyphens.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILY = ROOT / "families" / "website"
STYLES = {"light": "knapen_site", "dark": "knapen_site_dark"}
MAX_WIDTH = 1280


def newest_version(style):
    versions = sorted((p.name for p in (ROOT / "styles" / style).iterdir() if (p / "style.yaml").exists()),
                      key=lambda v: [int(x) for x in v.split(".")])
    return versions[-1]


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def render(project, style, profile, env):
    """Returns the manifest of the job (manim8 prints it alone on stdout with --json)."""
    cmd = ["manim8", "render", str(project), "--profile", profile, "--style", style, "--json"]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-3000:])
        raise SystemExit(f"render failed: {project.name} under {style}")
    manifest = json.loads(done.stdout)
    if manifest.get("result", {}).get("status") != "succeeded":
        raise SystemExit(f"render did not succeed: {project.name} under {style}")
    for w in manifest["result"].get("warnings") or []:
        print(f"  warning ({style}): {w}")
    return manifest


def encode(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale='min({MAX_WIDTH},iw)':-2:flags=lanczos,scale=trunc(iw/2)*2:trunc(ih/2)*2"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-vf", scale, "-c:v", "libx264", "-profile:v", "high",
         "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "23", "-movflags", "+faststart", str(dst)])
    still = dst.with_suffix(".webp")
    run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.2", "-i", str(dst), "-update", "1", "-frames:v", "1",
         "-c:v", "libwebp", "-quality", "88", str(still)])
    probe = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
                 "-of", "json", str(dst)], capture_output=True, text=True)
    info = json.loads(probe.stdout)
    return {"width": info["streams"][0]["width"], "height": info["streams"][0]["height"],
            "seconds": round(float(info["format"]["duration"]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("projects", nargs="*", help="project ids under families/website (default: all)")
    ap.add_argument("--profile", default="draft", choices=["storyboard", "draft", "final"])
    ap.add_argument("--site", type=Path, help="checkout of tknapen.github.io to copy the clips into")
    ap.add_argument("--light-only", action="store_true", help="skip the dark render")
    args = ap.parse_args()

    env = dict(os.environ, FONTCONFIG_FILE=str(ROOT / "fonts" / "fonts.conf"))
    projects = [FAMILY / p for p in args.projects] or sorted(p.parent for p in FAMILY.glob("*/animation.yaml"))
    out = ROOT / "out" / "clips"
    index = {}
    for project in projects:
        if not (project / "storyboard.yaml").exists():
            raise SystemExit(f"no storyboard in {project}")
        clip_id = project.name.replace("_", "-")
        print(f"{clip_id}")
        run(["manim8", "codegen", str(project)], env=env, capture_output=True)
        for mode, style in STYLES.items():
            if mode == "dark" and args.light_only:
                continue
            style_ref = f"{style}@{newest_version(style)}"
            manifest = render(project, style_ref, args.profile, env)
            video = project / manifest["result"]["videos"][0]["path"]
            dst = (out / "dark" if mode == "dark" else out) / f"{clip_id}.mp4"
            info = encode(video, dst)
            print(f"  {mode}: {style_ref}, {info['width']}x{info['height']}, {info['seconds']} s, "
                  f"{dst.stat().st_size // 1024} kB{' (cached render)' if manifest.get('cache_hit') else ''}")
            if mode == "light":
                index[clip_id] = dict(info, profile=args.profile, revision=manifest["job"]["revision_id"])
            if args.site:
                target = args.site / "assets" / "clips" / ("dark" if mode == "dark" else "")
                target.mkdir(parents=True, exist_ok=True)
                for f in (dst, dst.with_suffix(".webp")):
                    shutil.copy2(f, target / f.name)
    with open(ROOT / "out" / "clips.yml", "w") as f:
        f.write("# Written by tools/render.py: size and length of each clip, for the site's _data/clips.yml.\n")
        for clip_id, info in index.items():
            f.write(f"- id: {clip_id}\n  width: {info['width']}\n  height: {info['height']}\n  seconds: {info['seconds']}\n"
                    f"  # {info['profile']} render of {info['revision']}\n")
    print(f"wrote {out.relative_to(ROOT)}/ and out/clips.yml" + (f"; copied to {args.site}/assets/clips/" if args.site else ""))


if __name__ == "__main__":
    main()
