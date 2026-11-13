# site-animations

Animations made for the Knapen Lab website, https://tknapen.github.io. Each clip is a
[manim8](https://github.com/tknapen/manim8) project: a specification and a storyboard in YAML, rendered by
Manim. Every clip is rendered twice from the same scene, in a light and in a dark style that use the
website's own colours and typefaces, so the site can show the one that fits the visitor's colour scheme.

The website's repository holds only the finished files (`assets/clips/<id>.mp4` and `assets/clips/dark/<id>.mp4`,
each with a still). This repository holds what they are made from.

## Layout

```text
workspace.yaml            marks this folder as a manim8 workspace
styles/knapen_site/       the light style: white page, graphite ink, Public Sans and Space Grotesk,
                          the site's four theme colours as signal colours; larger text than a lecture style,
                          because a clip is shown about 900 px wide in a page
styles/knapen_site_dark/  the same sheet on the site's dark page colour (#121316); only colours differ,
                          so one scene renders under both with the same timing
styles/default/, styles/tomas_scientific/   manim8's own styles (the site styles inherit default@0.5.0)
fonts/                    Public Sans and Space Grotesk (SIL Open Font License) and the fontconfig file
                          the renders use, so the result does not depend on the machine's fonts
families/website/         one manim8 project per clip, and the pinned render environment (uv.lock)
tools/render.py           renders every clip in both styles and prepares the files for the site
out/                      the prepared files (not committed)
```

## Clips

| Project | Shows | Status |
|---|---|---|
| `divisive_normalization` | The DN pRF model of Aqil, Knapen and Dumoulin (2021): the activation constant b brings out surround suppression in the pRF profile; the normalization constant d sets the compression of the response function. | draft, awaiting scientific sign-off |

## Set up

Python 3.12 or later, [uv](https://docs.astral.sh/uv/), ffmpeg, and the Cairo and Pango headers
(`sudo apt-get install -y libcairo2-dev libpango1.0-dev pkg-config` on Debian or Ubuntu;
`brew install cairo pango pkg-config ffmpeg` on macOS).

```bash
uv tool install "manim8 @ git+https://github.com/tknapen/manim8@5737567c6f66a80416129338a670c81180d6b731"
```

```bash
manim8 family env sync families/website
```

The family takes Manim and manim8's render side from the same commit; `families/website/uv.lock` pins them.

## Render

```bash
python3 tools/render.py
```

This renders every project under `knapen_site` and `knapen_site_dark` (draft profile), encodes each for the
web and writes `out/clips/<id>.mp4`, `out/clips/dark/<id>.mp4`, their stills, and `out/clips.yml` with each
clip's size and length. One clip in both styles takes under a minute.

To render one clip at final quality and copy it into a checkout of the website:

```bash
python3 tools/render.py divisive_normalization --profile final --site ../tknapen.github.io
```

The site then needs an entry for the clip in its `_data/clips.yml` (title and caption, with the size and
length from `out/clips.yml`) and `{% include clip.html id="divisive-normalization" %}` on the page where it
belongs. The dark render is picked up by itself.

## Add a clip

```bash
manim8 project init my_clip --family website --workspace .
```

Then follow manim8's loop: a brief in `references/brief.md`, the specification and storyboard as proposals
(`manim8 context`, `manim8 propose validate`, `manim8 propose apply`), `manim8 validate`, `manim8 codegen`,
`manim8 snapshot --all` for the checkpoint stills, and `python3 tools/render.py my_clip` for both styles.
Set `FONTCONFIG_FILE=fonts/fonts.conf` when calling manim8 by hand, as `tools/render.py` does, so the site's
typefaces are found. Scientific claims go into the proposals with their source; a clip goes onto the website
only after its author has signed off on them.

A change to the look is a new style version (`styles/knapen_site/0.2.0/`), never an edit of a published one;
`tools/render.py` uses the newest version of each style.

## Licences

Code under the MIT licence. The fonts in `fonts/` are under the SIL Open Font License (see the OFL files there).
