"""Custom manim8 builder for `connective_field`: fitting a connective field to one target voxel.

Left: the source map, a sheet of V1 with one square per voxel, coloured by the eccentricity of each voxel's
pRF. The candidate connective field is a ring at one standard deviation around its centre voxel, and the
squares are inked by the weight the candidate gives them. Right: the target voxel's time course with the
candidate's prediction (the weighted sum of the source voxels' time courses) and an R squared readout. Below
it, the visual field position the target inherits: the pRF centre of the candidate's centre voxel.

Everything drawn comes from the project's data, written by sim/simulate_cf.py with prfmodel:
`voxels.csv` (the sheet), `candidates.csv` (the candidates in order), `timecourses.csv` (the target and one
prediction per candidate) and `parameters.json` (the connective field the target was simulated with).

Parameters that `vary_parameter` drives:
  step    the candidate shown, as a position in the sequence; between two whole numbers the centre, size,
          R squared and prediction are blended (eased)
  refine  0 to 1: the heading changes from 'grid search' to 'refinement' and the kept best leaves
  result  0 to 1: the connective field the target was simulated with appears as a dashed ring, with its label
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from manim import DOWN, LEFT, PI, RIGHT, UP, Arc, VGroup

from scianim_bridge.components.base import ParametricComponent
from site_kit import Kit, Panel, circle_points, clip01, corners, dashed_runs, dashes, declared, project_root, read_csv, set_ink, smooth

ECCENTRICITY_RAMP = ("emphasis", "primary_signal", "theme_perception")


def build(scene: Any, object_id: str, tokens: Any, arguments: dict[str, Any], *, style_role: str | None = None,
          data: dict[str, Any] | None = None, equation: dict[str, Any] | None = None) -> ParametricComponent:
    kit = Kit(scene, tokens)
    root = project_root(scene)
    voxels = sorted(read_csv(root / "data/voxels.csv"), key=lambda r: int(r["order"]))
    candidates = sorted(read_csv(root / "data/candidates.csv"), key=lambda r: int(r["seq"]))
    table = read_csv(root / "data/timecourses.csv")
    truth = json.loads((root / "data/parameters.json").read_text())
    grid_step = float(truth["sheet_step"])
    extent = float(arguments.get("field_radius_deg", 6.5))
    signal = style_role or "primary_signal"

    u = np.array([float(r["sheet_u"]) for r in voxels])
    v = np.array([float(r["sheet_v"]) for r in voxels])
    ecc = np.array([float(r["eccentricity_deg"]) for r in voxels])
    n = len(voxels)
    ts = np.array([float(r["t_s"]) for r in table])
    t_end = float(ts[-1] + (ts[1] - ts[0]))
    target = np.array([float(r["target"]) for r in table])
    predictions = np.array([[float(r[f"p{k:02d}"]) for r in table] for k in range(len(candidates))])
    centre_index = [int(c["center_index"]) for c in candidates]
    size = np.array([float(c["sigma_voxels"]) for c in candidates])
    r2 = np.array([float(c["r2"]) for c in candidates])
    field_x = np.array([float(c["x_deg"]) for c in candidates])
    field_y = np.array([float(c["y_deg"]) for c in candidates])
    best_so_far = [int(np.argmax(r2[: k + 1])) for k in range(len(candidates))]

    # --- layout (frame 16 by 8) -------------------------------------------------------------------------------
    sheet_height = 4.9
    cell = sheet_height / ((v.max() - v.min()) / grid_step + 1)
    sheet_left, sheet_bottom = -7.0, -2.25
    at = np.column_stack([sheet_left + ((u - u.min()) / grid_step + 0.5) * cell,
                          sheet_bottom + ((v - v.min()) / grid_step + 0.5) * cell, np.zeros(n)])
    sheet_mid = sheet_left + ((u.max() - u.min()) / grid_step + 1) * cell / 2
    panel = Panel(kit, -1.55, -0.5, 8.55, 3.0, (0, t_end), (-0.5, 1.75))
    inset_centre, inset_radius = np.array([-1.55, -2.55, 0.0]), 0.92

    group = VGroup()
    group.add(kit.text(arguments.get("sheet_label", "source map: a sheet of V1, one square per voxel"), "caption", "foreground")
              .move_to([sheet_left, 3.38, 0], aligned_edge=LEFT),
              kit.text(arguments.get("sheet_key", "simulated; color: distance from fixation"), "caption", "secondary_text")
              .move_to([sheet_left, 2.98, 0], aligned_edge=LEFT))
    colors = [kit.ramp(ECCENTRICITY_RAMP, e / extent) for e in ecc]
    half = cell * 0.94 / 2
    unit_square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], float) * half
    squares = []
    for i in range(n):
        square = kit.shape(colors[i])
        corners(square, list(at[i] + unit_square))
        squares.append(square)
    group.add(*squares)
    headings = [kit.text(text, "annotation", "secondary_text").move_to([sheet_mid, -2.72, 0]) for text in ("grid search", "refinement")]
    group.add(*headings)

    group.add(panel.y_axis((0, 1), "BOLD (a.u.)"), panel.x_axis((0, 100, 200, 300), "time (s)"),
              kit.line(panel.pt(0, 0), panel.pt(t_end, 0), "grid", 2.0))
    data_line = kit.path("foreground", kit.curve_px * 0.6)
    corners(data_line, panel.polyline(ts, target))
    group.add(data_line,
              kit.text(arguments.get("target_label", "target voxel elsewhere in the brain (simulated)"), "annotation", "foreground")
              .move_to([panel.left + kit.units(10), panel.top + kit.units(44), 0], aligned_edge=LEFT),
              kit.text(arguments.get("prediction_label", "prediction: weighted sum of the V1 voxels under the patch"), "caption", signal)
              .move_to([panel.left + kit.units(18), panel.pt(0, 1.6)[1], 0], aligned_edge=LEFT))

    arc = Arc(radius=inset_radius, start_angle=-PI / 2, angle=PI, color=kit.color("muted"), stroke_width=kit.stroke(1.8))
    arc.move_arc_center_to(inset_centre)
    arm = kit.units(9)
    group.add(arc, kit.line(inset_centre + inset_radius * DOWN, inset_centre + inset_radius * UP, "muted", 1.8),
              kit.line(inset_centre + LEFT * arm, inset_centre + RIGHT * arm, "foreground", 2.2),
              kit.line(inset_centre + DOWN * arm, inset_centre + UP * arm, "foreground", 2.2),
              kit.text(arguments.get("field_label", "visual field position the target inherits"), "caption", "foreground")
              .move_to([inset_centre[0] + inset_radius + 0.45, -2.2, 0], aligned_edge=LEFT))

    kept = kit.path("foreground", kit.curve_px * 0.6)
    ring = kit.path("foreground", kit.curve_px * 1.15)
    dot = kit.dot("foreground", 6)
    simulated = kit.path("comparison", kit.curve_px * 0.9)
    true_at = at[int(truth["true_center_index"])]
    dashes(simulated, dashed_runs(circle_points(true_at, float(truth["true_sigma_voxels"]) * cell, 97), kit.units(13), kit.units(9)))
    simulated_label = kit.text(arguments.get("simulated_label", "dashed: the connective field the target was simulated with"), "caption", "comparison")
    simulated_label.move_to([inset_centre[0] + inset_radius + 0.45, -3.38, 0], aligned_edge=LEFT)
    prediction = kit.path(signal, kit.curve_px * 0.95)
    field_dot = kit.dot(signal, 9)
    readout, size_line, field_line = VGroup(), VGroup(), VGroup()
    group.add(kept, prediction, ring, dot, simulated, simulated_label, field_dot, readout, size_line, field_line)
    weights_state: list[float | None] = [None] * n

    def rebuild(values: dict[str, float]) -> list[str]:
        step = min(max(float(values["step"]), 0.0), len(candidates) - 1.0)
        refine = clip01(values.get("refine", 0.0))
        result = clip01(values.get("result", 0.0))
        lower = min(int(np.floor(step + 1e-9)), len(candidates) - 1)
        upper = min(lower + 1, len(candidates) - 1)
        w = smooth(step - lower)
        centre = (1 - w) * at[centre_index[lower]] + w * at[centre_index[upper]]
        sigma = (1 - w) * size[lower] + w * size[upper]
        fit_r2 = (1 - w) * r2[lower] + w * r2[upper]
        x = (1 - w) * field_x[lower] + w * field_x[upper]
        y = (1 - w) * field_y[lower] + w * field_y[upper]
        # The squares carry the candidate's weights: full ink at its centre, a third of it far away.
        distance = np.hypot(at[:, 0] - centre[0], at[:, 1] - centre[1]) / cell
        ink = 0.32 + 0.68 * np.exp(-0.5 * (distance / sigma) ** 2)
        for i in range(n):
            level = round(float(ink[i]), 3)
            if weights_state[i] != level:
                weights_state[i] = level
                squares[i].set_fill(colors[i], opacity=level)
        corners(ring, circle_points(centre, sigma * cell, 73))
        dot.move_to(centre)
        corners(prediction, panel.polyline(ts, (1 - w) * predictions[lower] + w * predictions[upper]))
        best = best_so_far[lower]
        dashes(kept, dashed_runs(circle_points(at[centre_index[best]], size[best] * cell, 97), kit.units(11), kit.units(9)))
        set_ink(kept, 1 - refine)
        set_ink(headings[0], 1 - refine)
        set_ink(headings[1], refine)
        set_ink(simulated, result)
        set_ink(simulated_label, result)
        field_dot.move_to(inset_centre + inset_radius / extent * np.array([x, y, 0.0]))
        text = kit.show(readout, f"R² = {fit_r2:.2f}", "body", signal)
        text.move_to([panel.right, panel.top + kit.units(48), 0], aligned_edge=RIGHT)
        text = kit.show(size_line, f"size {sigma:.1f} voxels", "annotation", "foreground")
        text.move_to([sheet_mid, -3.3, 0])
        angle = float(np.degrees(np.arctan2(y, x)))
        text = kit.show(field_line, f"eccentricity {float(np.hypot(x, y)):.1f}°   polar angle {angle:+.0f}°".replace("-", "−"), "annotation", signal)
        text.move_to([inset_centre[0] + inset_radius + 0.45, -2.7, 0], aligned_edge=LEFT)
        return []

    parameters = {"step": 0.0, "refine": 0.0, "result": 0.0}
    parameters.update(declared(data))
    rebuild(parameters)
    return ParametricComponent(object_id, "custom", group, text=None,
                               font_px=tokens.smallest_text_px("tick_label", "axis_label", "annotation", "caption"),
                               text_like=False, parameters=parameters, rebuild=rebuild)
