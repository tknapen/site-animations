#!/usr/bin/env python3
"""Simulate the divisive normalization pRF model with prfmodel for the website's DN clips.

Writes the curves the clips draw, as CSV files in the projects' data/ folders:

  families/website/dn_bar_sweep/data/bar_sweep.csv
      the response to a bar stepping across the visual field: for each bar centre, the numerator
      a (G1 . S) + b, the denominator c (G2 . S) + d, and the response numerator / denominator - b / d
  families/website/dn_size_tuning/data/disc_growth.csv
      the response to a disc of growing radius centred on the pRF, for the reference parameters
      (sweep = base), for a range of the normalization constant d (sweep = d) and for a range of the
      size ratio sigma_2 / sigma_1 (sweep = ratio)

The response comes from prfmodel's DivNormGaussian2DPRFModel without an impulse response or scaling
(the predicted neural response). Numerator and denominator are computed from the same model's two
Gaussian tuning profiles, encoded with the stimulus by prfmodel's Gaussian2DPRFModel, and the script
checks that numerator / denominator - b / d reproduces the DN model's own output.

The stimulus design is each grid cell's fraction inside the stimulus, multiplied by the cell's area, so
G . S is the integral of the volume-normalized Gaussian over the stimulus (1 for a full-field stimulus)
and the curves do not step as a stimulus edge moves across cells.

Parameters are illustrative, not fitted to data; they are written to each project's data/parameters.json.

Usage (from the repository root):
  uv venv --python 3.12 sim/.venv && uv pip install --python sim/.venv/bin/python -r sim/requirements.txt
  KERAS_BACKEND=jax sim/.venv/bin/python sim/simulate_dn.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd
from prfmodel.models.prf import DivNormGaussian2DPRFModel, Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus

ROOT = Path(__file__).resolve().parent.parent
FAMILY = ROOT / "families" / "website"

# Illustrative parameters (degrees of visual angle; amplitudes and constants in arbitrary units).
BASE = {
    "mu_x": 1.5, "mu_y": 1.0,
    "sigma_activation": 1.0, "sigma_normalization": 4.0,
    "amplitude_activation": 0.6, "baseline_activation": 0.05,
    "amplitude_normalization": 1.0, "baseline_normalization": 0.25,
}
APERTURE_RADIUS = 8.0      # the stimulus aperture, centred on fixation
BAR_WIDTH = 2.0
EXTENT, N = 10.0, 401      # the simulation grid: [-10, 10] deg in 0.05 deg steps


def grid():
    axis = np.linspace(-EXTENT, EXTENT, N)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return xx, yy, np.stack([yy, xx], axis=-1)   # prfmodel: grid[..., 0] is y, grid[..., 1] is x


CELL = 2 * EXTENT / (N - 1)
CELL_AREA = CELL**2


def stimulus(frames, g):
    # prfmodel's encoder sums pRF x design over the grid without the cell area. Multiplying the design by
    # the cell area makes G . S the integral of the (volume-normalized) Gaussian over the stimulus, so a
    # full-field stimulus gives 1 and the parameters do not depend on the grid resolution.
    return PRFStimulus(design=np.asarray(frames, dtype="float32") * CELL_AREA, grid=g.astype("float32"), dimension_labels=["y", "x"], frame_duration=1.0)


def simulate(frames, g, params):
    """Numerator, denominator and response for every frame, for one parameter set."""
    stim = stimulus(frames, g)
    p = pd.DataFrame([params])
    dn = DivNormGaussian2DPRFModel(impulse_model=None, scaling_model=None)
    response = np.asarray(dn(stim, p))[0]
    gauss = Gaussian2DPRFModel(impulse_model=None, scaling_model=None)
    def encoded(sigma):
        q = pd.DataFrame([{"mu_x": params["mu_x"], "mu_y": params["mu_y"], "sigma": sigma}])
        return np.asarray(gauss(stim, q))[0]
    numerator = params["amplitude_activation"] * encoded(params["sigma_activation"]) + params["baseline_activation"]
    denominator = params["amplitude_normalization"] * encoded(params["sigma_normalization"]) + params["baseline_normalization"]
    check = numerator / denominator - params["baseline_activation"] / params["baseline_normalization"]
    assert np.allclose(check, response, atol=1e-4), float(np.abs(check - response).max())
    return numerator, denominator, response


def inside_disc(distance, radius):
    """Fraction of each grid cell inside a disc, from the cell centre's distance to the disc's centre (a linear
    ramp one cell wide across the edge), so the curves do not step as an edge moves across cells."""
    return np.clip(0.5 + (radius - distance) / CELL, 0.0, 1.0)


def main():
    xx, yy, g = grid()
    aperture = inside_disc(np.hypot(xx, yy), APERTURE_RADIUS)

    # Bar sweep: a vertical bar of BAR_WIDTH deg, its centre stepping from left to right.
    centres = np.round(np.arange(-9.0, 9.0 + 1e-9, 0.1), 3)
    def bar(c):  # the fraction of each cell's width inside the bar
        return np.clip((np.minimum(xx + CELL / 2, c + BAR_WIDTH / 2) - np.maximum(xx - CELL / 2, c - BAR_WIDTH / 2)) / CELL, 0.0, 1.0)
    frames = [bar(c) * aperture for c in centres]
    num, den, resp = simulate(frames, g, BASE)
    out = FAMILY / "dn_bar_sweep" / "data"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bar_centre": centres, "numerator": num, "denominator": den, "response": resp}).round(5).to_csv(out / "bar_sweep.csv", index=False)
    (out / "parameters.json").write_text(json.dumps(dict(BASE, aperture_radius=APERTURE_RADIUS, bar_width=BAR_WIDTH, engine="prfmodel 0.2.0, DivNormGaussian2DPRFModel"), indent=2) + "\n")
    print(f"bar sweep: {len(centres)} positions, response {resp.min():.3f} to {resp.max():.3f}")

    # Disc growth: a disc centred on the pRF, radius from 0 to 6 deg.
    radii = np.round(np.arange(0.0, 6.0 + 1e-9, 0.05), 3)
    distance = np.hypot(xx - BASE["mu_x"], yy - BASE["mu_y"])
    frames = [(inside_disc(distance, r) if r > 0 else np.zeros_like(xx)) * aperture for r in radii]
    rows = []
    def add(sweep, value, params):
        num, den, resp = simulate(frames, g, params)
        for r, n_, d_, y in zip(radii, num, den, resp):
            rows.append({"sweep": sweep, "value": round(float(value), 4), "radius": r, "numerator": n_, "denominator": d_, "response": y})
        return resp
    resp = add("base", 0, BASE)
    peak = radii[int(np.argmax(resp))]
    print(f"disc growth: peak {resp.max():.3f} at {peak} deg, {100 * (resp[-1] / resp.max() - 1):.0f}% by {radii[-1]} deg")
    for d in np.round(np.linspace(0.25, 1.5, 26), 4):
        add("d", d, dict(BASE, baseline_normalization=float(d)))
    for ratio in np.round(np.linspace(1.0, 4.0, 31), 4):
        add("ratio", ratio, dict(BASE, sigma_normalization=float(ratio) * BASE["sigma_activation"]))
    out = FAMILY / "dn_size_tuning" / "data"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(5).to_csv(out / "disc_growth.csv", index=False)
    (out / "parameters.json").write_text(json.dumps(dict(BASE, aperture_radius=APERTURE_RADIUS, engine="prfmodel 0.2.0, DivNormGaussian2DPRFModel"), indent=2) + "\n")
    print(f"wrote {len(rows)} rows of size-response curves")


if __name__ == "__main__":
    main()
