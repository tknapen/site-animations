#!/usr/bin/env python3
"""Simulate a connective field (CF) fit with prfmodel for the website clip `connective_field`.

The source is the model sheet of V1 of the clip `fields_to_map`: 690 simulated voxels, each with a fitted
Gaussian pRF. The script
  1. predicts every source voxel's BOLD response to the bar run of the clip `stimulus_to_bold` with
     prfmodel's Gaussian2DPRFModel (its default two-gamma haemodynamic response),
  2. makes a target voxel "elsewhere in the brain" with prfmodel's GaussianCFModel: a Gaussian-weighted sum
     of the source responses around one source voxel, plus Gaussian noise,
  3. fits a CF to that target as the lab does: a grid of centres and sizes, each with amplitude and baseline
     by least squares, then a local refinement of the centre (over neighbouring voxels) and the size,
and writes what the clip draws to families/website/connective_field/data/:
  voxels.csv        the source sheet (a copy of the fields_to_map table)
  candidates.csv    the candidates in the order the clip shows them, with their R squared and the visual
                    field position of the centre voxel's pRF
  timecourses.csv   the target time course and one prediction per candidate
  parameters.json   the simulated connective field, the noise and the seed

Distances on the sheet are in voxels (one grid step of the sheet). Everything is simulated; nothing is a
measurement.

Usage (from the repository root, with the environment of sim/requirements.txt):
  KERAS_BACKEND=jax sim/.venv/bin/python sim/simulate_cf.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd
from prfmodel.models.cf import GaussianCFModel
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import CFStimulus, PRFStimulus

ROOT = Path(__file__).resolve().parent.parent
FAMILY = ROOT / "families" / "website"
OUT = FAMILY / "connective_field" / "data"
SEED = 20261113
SHEET_STEP = 0.085          # grid step of the sheet: one voxel
TRUE_SIGMA = 2.0            # size of the simulated connective field, in voxels
TRUE_FIELD = (3.0, 35.0)    # the centre voxel is the one whose pRF is nearest this eccentricity (deg) and polar angle (deg)
NOISE = 0.45                # noise SD as a fraction of the target signal's SD
APERTURE, BAR_WIDTH = 8.0, 2.0
EXTENT, N = 10.0, 201


def bar_design(run):
    axis = np.linspace(-EXTENT, EXTENT, N)
    cell = axis[1] - axis[0]
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    aperture = np.clip(0.5 + (APERTURE - np.hypot(xx, yy)) / cell, 0.0, 1.0)
    frames = []
    for centre, direction in zip(run["bar_centre_deg"], run["bar_direction_deg"]):
        if np.isnan(centre):
            frames.append(np.zeros_like(xx))
            continue
        along = xx * np.cos(np.radians(direction)) + yy * np.sin(np.radians(direction))
        inside = np.clip((np.minimum(along + cell / 2, centre + BAR_WIDTH / 2) - np.maximum(along - cell / 2, centre - BAR_WIDTH / 2)) / cell, 0.0, 1.0)
        frames.append(inside * aperture)
    design = np.asarray(frames, dtype="float32") * cell**2     # so that G . S is the integral over the stimulus
    return PRFStimulus(design=design, grid=np.stack([yy, xx], axis=-1).astype("float32"), dimension_labels=["y", "x"], frame_duration=1.0)


def scaled(prediction, target):
    """Amplitude and baseline by least squares; returns the scaled prediction and R squared."""
    A = np.column_stack([prediction, np.ones_like(prediction)])
    beta, *_ = np.linalg.lstsq(A, target, rcond=None)
    fit = A @ beta
    return fit, max(0.0, 1 - np.sum((target - fit) ** 2) / np.sum((target - target.mean()) ** 2))


def main():
    rng = np.random.default_rng(SEED)
    voxels = pd.read_csv(FAMILY / "fields_to_map" / "data" / "voxels.csv").sort_values("order").reset_index(drop=True)
    run = pd.read_csv(FAMILY / "stimulus_to_bold" / "data" / "run.csv")
    n = len(voxels)

    # 1. Source responses: every voxel's predicted BOLD for the bar run.
    prf = Gaussian2DPRFModel()
    params = pd.DataFrame({"mu_x": voxels["x_fit_deg"], "mu_y": voxels["y_fit_deg"], "sigma": voxels["sigma_fit_deg"]})
    # prfmodel's default haemodynamic response (Glover two-gamma), without its temporal derivative.
    defaults = dict(prf.models["impulse_model"].default_parameters, weight_deriv=0.0, baseline=0.0, amplitude=1.0)
    for name in prf.parameter_names:
        if name not in params:
            params[name] = defaults[name]
    assert not params.isna().any().any(), params.columns[params.isna().any()].tolist()
    sources = np.asarray(prf(bar_design(run), params), dtype=float)            # (voxels, frames)

    # 2. The target voxel: a Gaussian-weighted sum of the sources around one voxel, plus noise.
    col = np.round((voxels["sheet_u"] - voxels["sheet_u"].min()) / SHEET_STEP).astype(int).to_numpy()
    row = np.round((voxels["sheet_v"] - voxels["sheet_v"].min()) / SHEET_STEP).astype(int).to_numpy()
    position = np.column_stack([col, row]).astype(float)
    distance = np.linalg.norm(position[:, None, :] - position[None, :, :], axis=-1)   # in voxels
    stimulus = CFStimulus(distance_matrix=distance.astype("float32"), source_response=sources.astype("float32"))
    cf = GaussianCFModel()

    def predict(centres, sigmas):
        p = pd.DataFrame({"center_index": np.asarray(centres, float), "sigma": np.asarray(sigmas, float), "baseline": 0.0, "amplitude": 1.0})
        return np.asarray(cf(stimulus, p), dtype=float)

    want = TRUE_FIELD[0] * np.array([np.cos(np.radians(TRUE_FIELD[1])), np.sin(np.radians(TRUE_FIELD[1]))])
    true_centre = int(np.argmin(np.hypot(voxels["x_fit_deg"] - want[0], voxels["y_fit_deg"] - want[1])))
    signal = predict([true_centre], [TRUE_SIGMA])[0]
    signal = signal / signal.max()
    target = signal + rng.normal(0.0, NOISE * signal.std(), signal.shape)

    # 3. The fit. Grid: every third voxel in both directions, three sizes.
    lattice = [i for i in range(n) if col[i] % 3 == 1 and row[i] % 3 == 1]
    grid = [(c, s) for c in lattice for s in (1.0, 2.0, 4.0)]
    predictions = predict([c for c, _ in grid], [s for _, s in grid])
    results = [(c, s, *scaled(p, target)) for (c, s), p in zip(grid, predictions)]
    best = max(range(len(results)), key=lambda k: results[k][3])

    def evaluate(centre, sigma):
        fit, r2 = scaled(predict([centre], [sigma])[0], target)
        return centre, sigma, fit, r2

    def best_sigma(centre):
        sigmas = np.linspace(0.6, 6.0, 55)
        fits = [scaled(p, target) for p in predict([centre] * len(sigmas), sigmas)]
        k = int(np.argmax([r2 for _, r2 in fits]))
        return centre, float(sigmas[k]), fits[k][0], fits[k][1]

    path = [best_sigma(results[best][0])]
    while len(path) < 12:
        centre = path[-1][0]
        neighbours = [i for i in range(n) if 0 < max(abs(col[i] - col[centre]), abs(row[i] - row[centre])) <= 1]
        step = max((best_sigma(i) for i in neighbours), key=lambda r: r[3])
        if step[3] <= path[-1][3] + 1e-6:
            break
        path.append(step)

    order = rng.permutation([k for k in range(len(results)) if k != best])
    poor = [k for k in order if results[k][3] < 0.15][:3]
    fast = [k for k in order if k not in poor][:36]
    shown = [("slow", results[k]) for k in poor] + [("fast", results[k]) for k in fast] + [("grid_best", results[best])] + [("refine", r) for r in path]

    OUT.mkdir(parents=True, exist_ok=True)
    rows, table = [], {"t_s": run["t_s"].to_numpy(), "target": np.round(target, 4)}
    for seq, (phase, (centre, sigma, fit, r2)) in enumerate(shown):
        v = voxels.iloc[centre]
        rows.append({"seq": seq, "phase": phase, "center_index": centre, "sheet_u": v["sheet_u"], "sheet_v": v["sheet_v"],
                     "sigma_voxels": round(float(sigma), 3), "r2": round(float(r2), 3), "x_deg": v["x_fit_deg"], "y_deg": v["y_fit_deg"],
                     "eccentricity_deg": v["eccentricity_deg"], "polar_angle_deg": v["polar_angle_deg"]})
        table[f"p{seq:02d}"] = np.round(fit, 4)
    pd.DataFrame(rows).to_csv(OUT / "candidates.csv", index=False)
    pd.DataFrame(table).to_csv(OUT / "timecourses.csv", index=False)
    voxels.to_csv(OUT / "voxels.csv", index=False)
    v = voxels.iloc[true_centre]
    (OUT / "parameters.json").write_text(json.dumps({
        "engine": "prfmodel 0.2.0: Gaussian2DPRFModel for the sources, GaussianCFModel for the target and the fit",
        "true_center_index": true_centre, "true_sigma_voxels": TRUE_SIGMA,
        "true_center_prf": {"x_deg": float(v["x_fit_deg"]), "y_deg": float(v["y_fit_deg"]), "eccentricity_deg": float(v["eccentricity_deg"]), "polar_angle_deg": float(v["polar_angle_deg"])},
        "noise_sd_fraction_of_signal_sd": NOISE, "seed": SEED, "sheet_step": SHEET_STEP,
        "grid": {"centres": len(lattice), "sizes_voxels": [1.0, 2.0, 4.0]}}, indent=2) + "\n")
    last = rows[-1]
    print(f"sources {sources.shape}, true centre {true_centre} (ecc {v['eccentricity_deg']:.2f}, angle {v['polar_angle_deg']:.1f}), sigma {TRUE_SIGMA}")
    print(f"grid: {len(grid)} candidates, best R2 {results[best][3]:.3f} at centre {results[best][0]}, sigma {results[best][1]}")
    print(f"refinement: {len(path)} steps to centre {last['center_index']}, sigma {last['sigma_voxels']}, R2 {last['r2']}")
    print(pd.DataFrame(rows)["phase"].value_counts().to_dict(), [r["r2"] for r in rows[:3]])


if __name__ == "__main__":
    main()
