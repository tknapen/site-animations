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
families/website/components/   drawing code shared by clips (custom manim8 builders)
sim/                      simulations that compute the curves some clips draw (prfmodel)
tools/checkpoints.py      renders every checkpoint of a clip in both styles, as stills and one review sheet
tools/render.py           renders every clip in both styles and prepares the files for the site
out/                      the prepared files (not committed)
```

## Clips

| Project | Shows | Status |
|---|---|---|
| `dn_bar_sweep` | The DN pRF model of Aqil, Knapen and Dumoulin (2021) with both pools drawn on the pRF. A bar crosses the visual field; numerator (activation pool), denominator (normalization pool) and response are traced against the bar's position. Curves simulated with prfmodel. | signed off by the author on 13 November 2026; final render in the website repository |
| `dn_size_tuning` | The same model with a disc growing around the pRF centre: the response peaks and then falls (oversaturation). The normalization constant d and the size ratio σ₂/σ₁ then vary one at a time. Curves simulated with prfmodel. | signed off by the author on 13 November 2026; final render in the website repository |
| `stimulus_to_bold` | One cursor through a 300 s bar-mapping run: the bar over a fixed pRF in the visual field, the stimulus strip, the predicted neural response and the predicted BOLD response. Simulated with prfmodel. | checkpoints signed off by the owner on 13 November 2026; final render in the website repository |
| `model_as_chain` | The modelling loop (parameters and stimulus into a model, its prediction on the data) opens its model box into seven stages, four of them with fitted parameters; a pulse runs from stimulus to BOLD. Synthetic data. | checkpoints signed off by the owner on 13 November 2026; final render in the website repository |
| `grid_and_fit` | Fitting a Gaussian pRF to one measured voxel (prfmodel's example dataset 7t-aot-visual): candidates on a coarse grid with their predictions and R², the best one kept, then refinement to the optimum. | checkpoints signed off by the owner on 13 November 2026; final render in the website repository |
| `fields_to_map` | 690 simulated voxels fitted one at a time: a dot at each fitted pRF centre and a square on a sheet of cortex, coloured by distance and then by direction from fixation. | checkpoints signed off by the owner on 13 November 2026; final render in the website repository |
| `divisive_normalization` | An earlier draft from the equation alone: the activation constant b and the pRF profile, the normalization constant d and the response function. No pools, no stimulus. | draft, not on the website |

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

## Review, then render

```bash
python3 tools/checkpoints.py grid_and_fit
```

This renders every checkpoint of the storyboard under both styles and writes the stills to
`out/checkpoints/<project>/` and one sheet, a row per checkpoint with light beside dark, to
`out/checkpoints/<project>.png`. Checkpoints are the animation itself fast-forwarded to the named times.
Look at them under both styles before a final render: thin strokes and small text first.

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

## Data and custom builders

The two `dn_` clips draw curves that are computed, not typed in. `sim/simulate_dn.py` runs prfmodel's
`DivNormGaussian2DPRFModel` (no impulse response, no scaling) for a bar stepping across the field and for a
disc growing around the pRF centre, and writes the numerator, the denominator and the response to each
project's `data/` folder as CSV, with the parameters in `data/parameters.json`. The CSV files are committed,
so rendering does not need prfmodel; rerun the simulation only to change parameters or stimuli.

```bash
uv venv --python 3.12 sim/.venv
```

```bash
uv pip install --python sim/.venv/bin/python -r sim/requirements.txt
```

```bash
KERAS_BACKEND=jax sim/.venv/bin/python sim/simulate_dn.py
```

The four pRF clips (`stimulus_to_bold`, `model_as_chain`, `grid_and_fit`, `fields_to_map`) were hand-written
Manim scenes for lectures. Each came here as a seed (specification, data, the original scene's source and
last frame, kept in `references/`) and was rebuilt as a manim8 project. Their builders are `prf_run.py`,
`model_chain.py`, `grid_fit.py` and `field_map.py` in `families/website/components/`, on the shared helpers
of `site_kit.py`. A builder draws and the storyboard times: every moving quantity is a parameter that a
`vary_parameter` action drives, and colours are palette roles, so the same scene renders under both styles.

`families/website/components/dn_views.py` draws the `dn_` clips: the visual field with the pRF and its two pools on
the left, the traces on the right. It is a manim8 `custom` builder that returns a parametric component, so
`vary_parameter` actions move the stimulus, d and the size ratio. manim8 (at the pinned commit) accepts such
an action on a custom object only if the object has a data binding whose transforms name parameters; each
project therefore has a `data/controls.csv` of unit values that the binding scales by those parameters. It
holds no data and can go once `custom` can declare parameters itself.

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
