"""Custom manim8 builder for `grid_and_fit`: grid search and refinement of a Gaussian pRF on one voxel.

Left: the visual field with fixation and a scale bar; the candidate as a dot with a one-sigma circle; the best
candidate so far as a dashed circle; a heading for the stage above and the candidate's x, y and sigma below.
Right: the measured time course with the candidate's prediction over it and an R squared readout.

Everything drawn comes from the project's data: `candidates.csv` (the candidates in order: seq, phase, x, y,
sigma, r2) and `timecourses.csv` (the measurement, and one prediction per candidate in columns p00, p01, ...).
The time of sample k is k times `sample_seconds` (0.9 s): the table's own `t_s` column is truncated to whole
seconds and is not used.

Parameters that `vary_parameter` drives:
  step    the candidate shown, as a position in the sequence: at a whole number the candidate of that seq at
          its own values; between two, the centre, size, R squared and prediction blended (eased, so a
          constant-speed sweep of `step` hops from candidate to candidate)
  refine  0 to 1: the heading changes from 'grid search' to 'refinement' and the kept best leaves
  result  0 to 1: the line with the fitted values appears
"""

from __future__ import annotations

from typing import Any

import numpy as np
from manim import DOWN, LEFT, RIGHT, VGroup

from scianim_bridge.components.base import ParametricComponent
from site_kit import Field, Kit, Panel, circle_points, clip01, corners, dashed_runs, dashes, declared, project_root, read_csv, set_ink, smooth


def build(scene: Any, object_id: str, tokens: Any, arguments: dict[str, Any], *, style_role: str | None = None,
          data: dict[str, Any] | None = None, equation: dict[str, Any] | None = None) -> ParametricComponent:
    kit = Kit(scene, tokens)
    root = project_root(scene)
    candidates = sorted(read_csv(root / arguments.get("candidates", "data/candidates.csv")), key=lambda r: int(r["seq"]))
    table = read_csv(root / arguments.get("timecourses", "data/timecourses.csv"))
    step_seconds = float(arguments.get("sample_seconds", 0.9))
    ts = np.arange(len(table)) * step_seconds
    t_end = float(ts[-1] + step_seconds)
    measured = np.array([float(r["measured"]) for r in table])
    predictions = np.array([[float(r[f"p{k:02d}"]) for r in table] for k in range(len(candidates))])
    cx = np.array([float(c["x_deg"]) for c in candidates])
    cy = np.array([float(c["y_deg"]) for c in candidates])
    cs = np.array([float(c["sigma_deg"]) for c in candidates])
    r2 = np.array([float(c["r2"]) for c in candidates])
    best_so_far = [int(np.argmax(r2[: k + 1])) for k in range(len(candidates))]
    signal = style_role or "primary_signal"

    # --- layout (frame 16 by 8) -------------------------------------------------------------------------------
    field = Field(kit, (-5.0, 0.2), 2.15, float(arguments.get("field_radius_deg", 4.0)))
    panel = Panel(kit, -0.75, -1.7, 7.75, 4.0, (0, t_end), (float(measured.min()) - 0.5, float(measured.max()) + 0.5))

    group = VGroup()
    group.add(field.outline("muted", 1.8), field.fixation())
    bar_deg = float(arguments.get("scale_bar_deg", 2.0))
    bar_y = field.centre[1] - field.radius - kit.units(30)
    scale_bar = kit.line([field.centre[0] - bar_deg * field.per_degree / 2, bar_y, 0],
                         [field.centre[0] + bar_deg * field.per_degree / 2, bar_y, 0], "secondary_text", 2.4)
    scale_text = kit.text(f"{bar_deg:g}°", "tick_label", "secondary_text")
    scale_text.next_to(scale_bar, DOWN, buff=kit.units(8))
    group.add(scale_bar, scale_text)
    heading_y = field.centre[1] + field.radius + kit.units(62)
    headings = [kit.text(text, "annotation", "secondary_text") for text in ("grid search", "refinement")]
    for heading in headings:
        heading.move_to([field.centre[0], heading_y, 0])
    group.add(*headings)

    group.add(panel.y_axis((0, 5), "BOLD (% signal change)"), panel.x_axis((0, 100, 200, 300), "time (s)"))
    data_line = kit.path("foreground", kit.curve_px * 0.6)
    corners(data_line, panel.polyline(ts, measured))
    label = kit.text(arguments.get("measured_label", "measured voxel"), "annotation", "foreground")
    label.move_to([panel.left + kit.units(10), panel.top + kit.units(44), 0], aligned_edge=LEFT)
    group.add(data_line, label)

    kept = kit.path("muted", kit.curve_px * 0.7)
    ring = kit.path(signal, kit.curve_px * 1.1)
    dot = kit.dot(signal, 7)
    prediction = kit.path(signal, kit.curve_px * 0.95)
    readout, values_line = VGroup(), VGroup()
    fit = candidates[-1]
    final = kit.text(
        f"fit: x {float(fit['x_deg']):.2f}°, y {float(fit['y_deg']):.2f}°, σ {float(fit['sigma_deg']):.2f}°, R² {float(fit['r2']):.2f}",
        "annotation", "foreground")
    final.move_to([panel.left + panel.width / 2, -3.3, 0])
    group.add(kept, prediction, ring, dot, readout, values_line, final)

    def rebuild(values: dict[str, float]) -> list[str]:
        step = min(max(float(values["step"]), 0.0), len(candidates) - 1.0)
        refine = clip01(values.get("refine", 0.0))
        result = clip01(values.get("result", 0.0))
        lower = min(int(np.floor(step + 1e-9)), len(candidates) - 1)
        upper = min(lower + 1, len(candidates) - 1)
        w = smooth(step - lower)
        x = (1 - w) * cx[lower] + w * cx[upper]
        y = (1 - w) * cy[lower] + w * cy[upper]
        sigma = (1 - w) * cs[lower] + w * cs[upper]
        fit_r2 = (1 - w) * r2[lower] + w * r2[upper]
        corners(ring, field.ring(x, y, sigma))
        dot.move_to(field.pt(x, y))
        corners(prediction, panel.polyline(ts, (1 - w) * predictions[lower] + w * predictions[upper]))
        best = best_so_far[lower]
        outline = circle_points(field.pt(cx[best], cy[best]), cs[best] * field.per_degree, 97)
        dashes(kept, dashed_runs(outline, kit.units(12), kit.units(9)))
        set_ink(kept, 1 - refine)
        set_ink(headings[0], 1 - refine)
        set_ink(headings[1], refine)
        set_ink(final, result)
        text = kit.show(readout, f"R² = {fit_r2:.2f}", "body", signal)
        text.move_to([panel.right, panel.top + kit.units(48), 0], aligned_edge=RIGHT)
        text = kit.show(values_line, f"x {x:+.2f}°   y {y:+.2f}°   σ {sigma:.2f}°".replace("-", "−"), "annotation", signal)
        text.move_to([field.centre[0], -3.3, 0])
        return []

    parameters = {"step": 0.0, "refine": 0.0, "result": 0.0}
    parameters.update(declared(data))
    rebuild(parameters)
    return ParametricComponent(object_id, "custom", group, text=None,
                               font_px=tokens.smallest_text_px("tick_label", "axis_label", "annotation"),
                               text_like=False, parameters=parameters, rebuild=rebuild)
