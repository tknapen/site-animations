#!/usr/bin/env python3
"""Render every checkpoint of the website's clips under the light and the dark site style, for review.

For each project named (default: all under families/website/), this runs `manim8 snapshot --all` under
knapen_site and knapen_site_dark and writes
  out/checkpoints/<project>/<checkpoint>.light.png and .dark.png   the stills
  out/checkpoints/<project>.png                                    one sheet: a row per checkpoint, light | dark
Checkpoints are the real animation flow fast-forwarded to the named times of the storyboard, so they show
what the video will show. Look at thin strokes and small text first, under both styles, before a final render.

Usage:
  python3 tools/checkpoints.py                       # all projects
  python3 tools/checkpoints.py grid_and_fit --profile draft

Needs manim8 on PATH, the family environment and Pillow (`pip install pillow`).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
FAMILY = ROOT / "families" / "website"
STYLES = {"light": "knapen_site", "dark": "knapen_site_dark"}


def newest_version(style):
    versions = sorted((p.name for p in (ROOT / "styles" / style).iterdir() if (p / "style.yaml").exists()),
                      key=lambda v: [int(x) for x in v.split(".")])
    return versions[-1]


def snapshot(project, style, profile, env):
    cmd = ["manim8", "snapshot", str(project), "--all", "--profile", profile, "--style", style, "--no-cache", "--json"]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-4000:])
        raise SystemExit(f"snapshot failed: {project.name} under {style}")
    manifest = json.loads(done.stdout)
    for w in manifest["result"].get("warnings") or []:
        print(f"  warning ({style}): {w}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("projects", nargs="*", help="project ids under families/website (default: all)")
    ap.add_argument("--profile", default="snapshot", help="manim8 quality profile (snapshot: 960 px wide; draft: full size)")
    args = ap.parse_args()
    env = dict(os.environ, FONTCONFIG_FILE=str(ROOT / "fonts" / "fonts.conf"))
    projects = [FAMILY / p for p in args.projects] or sorted(p.parent for p in FAMILY.glob("*/storyboard.yaml"))
    for project in projects:
        subprocess.run(["manim8", "codegen", str(project)], env=env, capture_output=True, check=True)
        out = ROOT / "out" / "checkpoints" / project.name
        out.mkdir(parents=True, exist_ok=True)
        stills = {}
        for mode, style in STYLES.items():
            manifest = snapshot(project, f"{style}@{newest_version(style)}", args.profile, env)
            for cp in manifest["result"]["checkpoints"]:
                name = cp.get("snapshot_id") or cp.get("id")
                src = project / (cp.get("png_path") or cp.get("path") or cp["png"])
                dst = out / f"{name}.{mode}.png"
                shutil.copyfile(src, dst)
                stills.setdefault(name, {})[mode] = dst
                for w in cp.get("layout_warnings") or []:
                    print(f"  layout ({mode}, {name}): {w}")
        order = list(stills)
        first = Image.open(stills[order[0]]["light"])
        w, h = first.size
        pad, label = 12, 34
        sheet = Image.new("RGB", (2 * w + 3 * pad, len(order) * (h + label + pad) + pad), "#808080")
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate(order):
            y = pad + i * (h + label + pad)
            draw.text((pad, y + 8), f"{project.name}: {name}", fill="white")
            for j, mode in enumerate(STYLES):
                sheet.paste(Image.open(stills[name][mode]).convert("RGB"), (pad + j * (w + pad), y + label))
        sheet.save(out.with_suffix(".png"))
        print(f"{project.name}: {len(order)} checkpoints, sheet at {out.with_suffix('.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
