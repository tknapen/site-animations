"""Custom manim8 builder for the divisive normalization (DN) clips: a visual field beside a plot.

Left: the stimulus aperture with fixation, the pRF as a Gaussian wash, a solid circle at one standard
deviation of the activation pool (sigma_1), a dashed circle at one standard deviation of the normalization
pool (sigma_2), and the stimulus (a bar stepping across the field, or a disc growing around the pRF centre).

Right: numerator, denominator and response of the DN model as a function of the stimulus variable, drawn up
to the current stimulus position, with a cursor.

The curves are not computed here. They are read from the CSV files that `sim/simulate_dn.py` writes with
prfmodel; this builder only draws them and interpolates between simulated parameter values.

Storyboard use:

    kind: custom
    data:                              # declares the parameters, see below
      source: controls
      encoding: {x: position, y: annotate}
      transforms:
        - {op: scale, column: position, by: {parameter: position, default: 0}}
        - {op: scale, column: d, by: {parameter: d, default: 0.25}}
    params:
      builder: dn_views:build
      arguments:
        mode: disc                     # bar | disc
        table: data/disc_growth.csv    # relative to the project
        parameters: data/parameters.json
        x_label: disc radius (deg)
        x_range: [0, 6, 1]             # low, high, tick step
        y_range: [0, 2, 0.5]
        data_label: simulated with prfmodel, illustrative parameters

Parameters that `vary_parameter` can drive: `position` (bar centre or disc radius, in degrees; the bar may start and end beyond the plotted range, so that
it enters and leaves the field), and in disc
mode `d` (the normalization constant), `ratio` (sigma_2 / sigma_1) and `annotate` (0 to 1, the ink of the
peak marker and the percentage drop).

manim8's storyboard contract accepts a `vary_parameter` action on a custom object only when the object has a
data binding whose transforms name parameters. The object therefore binds `data/controls.csv`, a table of
unit values with one column per parameter, and scales each column by its parameter; the defaults given there
are the starting values. The table carries no data. It stands in for a `parameters` field that `custom` does
not have (manim8 at 5737567).

Every drawn state is a function of these values alone, so a checkpoint
reached by fast-forward equals the same moment rendered frame by frame.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Circle, Dot, Line, Text, VGroup, VMobject

from scianim_bridge.components.base import ParametricComponent
from scianim_bridge.components.plots import _axes_with_labels

WASH_LAYERS = 28
WASH_EXTENT_SD = 2.8
WASH_PEAK = 0.62
CIRCLE_DASHES = 44
TRACE_DASH = (3, 2)  # samples drawn, samples skipped


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _columns(rows: list[dict[str, str]], x_name: str) -> dict[str, np.ndarray]:
    return {
        "x": np.array([float(r[x_name]) for r in rows]),
        "numerator": np.array([float(r["numerator"]) for r in rows]),
        "denominator": np.array([float(r["denominator"]) for r in rows]),
        "response": np.array([float(r["response"]) for r in rows]),
    }


class _Curves:
    """The simulated curves, and for the disc the interpolation between simulated values of d and the ratio."""

    def __init__(self, mode: str, rows: list[dict[str, str]], base_d: float, base_ratio: float) -> None:
        self.mode, self.base_d, self.base_ratio = mode, base_d, base_ratio
        if mode == "bar":
            self.base = _columns(rows, "bar_centre")
            self.sweeps: dict[str, tuple[np.ndarray, list[dict[str, np.ndarray]]]] = {}
            return
        self.base = _columns([r for r in rows if r["sweep"] == "base"], "radius")
        self.sweeps = {}
        for name in ("d", "ratio"):
            values = sorted({float(r["value"]) for r in rows if r["sweep"] == name})
            tables = [_columns([r for r in rows if r["sweep"] == name and float(r["value"]) == v], "radius") for v in values]
            self.sweeps[name] = (np.array(values), tables)

    def range_of(self, name: str) -> tuple[float, float]:
        values = self.sweeps[name][0]
        return float(values[0]), float(values[-1])

    def at(self, d: float, ratio: float) -> tuple[dict[str, np.ndarray], list[str]]:
        warnings: list[str] = []
        if self.mode == "bar":
            return self.base, warnings
        ratio_moved = abs(ratio - self.base_ratio) > 1e-9
        d_moved = abs(d - self.base_d) > 1e-9
        if ratio_moved and d_moved:
            warnings.append(
                "dn_views: d and the size ratio differ from their reference values at the same time; the curves "
                "were simulated for one of them at a time, so d is taken at its reference value"
            )
        name, value = ("ratio", ratio) if ratio_moved else ("d", d)
        values, tables = self.sweeps[name]
        low, high = float(values[0]), float(values[-1])
        if value < low - 1e-9 or value > high + 1e-9:
            warnings.append(f"dn_views: {name} = {value:g} is outside the simulated range [{low:g}, {high:g}]; clipped")
        value = min(max(value, low), high)
        upper = int(np.searchsorted(values, value, side="left"))
        upper = min(max(upper, 1), len(values) - 1)
        lower = upper - 1
        weight = (value - values[lower]) / (values[upper] - values[lower])
        a, b = tables[lower], tables[upper]
        return {key: (1 - weight) * a[key] + weight * b[key] for key in a}, warnings


def build(
    scene: Any,
    object_id: str,
    tokens: Any,
    arguments: dict[str, Any],
    *,
    style_role: str | None = None,
    data: dict[str, Any] | None = None,
    equation: dict[str, Any] | None = None,
) -> ParametricComponent:
    mode = arguments.get("mode", "disc")
    if mode not in ("bar", "disc"):
        raise ValueError(f"object '{object_id}': mode must be 'bar' or 'disc', not '{mode}'")
    root = scene.ctx.project_root if getattr(scene, "ctx", None) is not None else Path.cwd()
    model = json.loads((Path(root) / arguments.get("parameters", "data/parameters.json")).read_text())
    rows = _read_table(Path(root) / arguments["table"])
    mu = np.array([model["mu_x"], model["mu_y"]])
    sigma_1 = float(model["sigma_activation"])
    base_ratio = float(model["sigma_normalization"]) / sigma_1
    base_d = float(model["baseline_normalization"])
    aperture = float(model["aperture_radius"])
    bar_width = float(model.get("bar_width", 0.0))
    curves = _Curves(mode, rows, base_d, base_ratio)

    x_lo, x_hi, x_step = (float(v) for v in arguments["x_range"])
    y_lo, y_hi, y_step = (float(v) for v in arguments["y_range"])
    # The parameters and their starting values are the ones the object's data binding declares (see the module docstring).
    declared = {
        t["by"]["parameter"]: float(t["by"]["default"])
        for t in (data or {}).get("transforms", [])
        if isinstance(t.get("by"), dict)
    }
    start = declared.get("position", x_lo)

    signal = tokens.color(style_role or "primary_signal")
    grey = tokens.color("secondary_text")
    muted = tokens.color("muted")
    foreground = tokens.color("foreground")
    emphasis = tokens.color("emphasis")
    annotation = tokens.typeface("annotation")
    caption = tokens.typeface("caption")
    curve_px = tokens.t.geometry.curve_stroke_px_at_1080p
    ends = tokens.line_kwargs()

    # --- layout: the object fills the lower part of the safe area; the field left, the plot right -----------
    sx0, sy0, sx1, sy1 = tokens.safe_rect()
    height = (sy1 - sy0) * float(arguments.get("height_fraction", 0.8))
    header = tokens.units(70)          # the legend row above the plot
    footer = tokens.units(158)         # ticks, axis label and the data caption under the plot
    left_footer = tokens.units(64)     # the parameter readouts under the field
    radius = (height - left_footer) / 2
    field_centre = np.array([sx0 + radius, sy0 + left_footer + radius, 0.0])
    per_degree = radius / (aperture * 1.04)
    plot_height = height - header - footer
    plot_left = sx0 + 2 * radius + tokens.units(190)
    plot_width = sx1 - plot_left - tokens.units(12)

    def field_point(x: float, y: float) -> np.ndarray:
        return field_centre + per_degree * np.array([x, y, 0.0])

    group = VGroup()

    # --- the visual field ------------------------------------------------------------------------------------
    field = Circle(radius=aperture * per_degree, color=muted, stroke_width=tokens.stroke_width(1.6))
    field.set_fill(tokens.color("grid"), opacity=0.55).move_to(field_centre)
    stimulus = VMobject(stroke_width=0)
    stimulus.set_fill(muted, opacity=0.42)
    wash = VGroup()
    targets = [WASH_PEAK * math.exp(-0.5 * (WASH_EXTENT_SD * (i + 1) / WASH_LAYERS) ** 2) for i in range(WASH_LAYERS)]
    for i in reversed(range(WASH_LAYERS)):
        outer = targets[i + 1] if i + 1 < WASH_LAYERS else 0.0
        alpha = 1 - (1 - targets[i]) / (1 - outer)  # the layers stack to a Gaussian without seams
        layer = Circle(radius=WASH_EXTENT_SD * sigma_1 * per_degree * (i + 1) / WASH_LAYERS, stroke_width=0)
        layer.set_fill(signal, opacity=alpha).move_to(field_point(*mu))
        wash.add(layer)
    arm = tokens.units(11)
    fixation = VGroup(
        Line(field_centre + LEFT * arm, field_centre + RIGHT * arm, color=foreground, stroke_width=tokens.stroke_width(2.4), **ends),
        Line(field_centre + DOWN * arm, field_centre + UP * arm, color=foreground, stroke_width=tokens.stroke_width(2.4), **ends),
    )
    activation = Circle(radius=sigma_1 * per_degree, color=signal, stroke_width=tokens.stroke_width(curve_px), **ends)
    activation.move_to(field_point(*mu))
    normalization = VMobject(color=grey, stroke_width=tokens.stroke_width(curve_px * 0.8), **ends)
    label_1 = Text("σ₁", **tokens.text_kwargs(annotation, signal))
    label_2 = Text("σ₂", **tokens.text_kwargs(annotation, grey))
    gap = tokens.units(10)
    label_1.next_to(field_point(mu[0] + sigma_1 * math.cos(-0.9), mu[1] + sigma_1 * math.sin(-0.9)), DOWN + RIGHT, buff=gap * 0.4)
    label_2_points = label_2.points.copy()
    label_2_centre = label_2.get_center().copy()
    group.add(field, stimulus, wash, normalization, activation, fixation, label_1, label_2)

    # --- the plot --------------------------------------------------------------------------------------------
    axes, labels = _axes_with_labels(
        tokens, (x_lo, x_hi, x_step), (y_lo, y_hi, y_step), plot_width, plot_height,
        arguments.get("x_label", "stimulus"), arguments.get("y_label", "model output (a.u.)"),
    )
    plot = VGroup(axes, labels)
    plot.shift(np.array([plot_left, sy0 + footer, 0.0]) - axes.c2p(x_lo, y_lo))
    group.add(plot)
    if y_lo < 0 < y_hi:
        group.add(Line(axes.c2p(x_lo, 0), axes.c2p(x_hi, 0), color=muted, stroke_width=tokens.stroke_width(1.6), **ends))
    cursor = Line(axes.c2p(x_lo, y_lo), axes.c2p(x_lo, y_hi), color=muted, stroke_width=tokens.stroke_width(1.6), **ends)
    reference = VMobject(color=muted, stroke_width=tokens.stroke_width(curve_px * 0.55), **ends)
    numerator = VMobject(color=signal, stroke_width=tokens.stroke_width(curve_px * 0.7), **ends)
    numerator.set_stroke(opacity=0.62)
    denominator = VMobject(color=grey, stroke_width=tokens.stroke_width(curve_px * 0.8), **ends)
    response = VMobject(color=signal, stroke_width=tokens.stroke_width(curve_px * 1.35), **ends)
    dot_radius = tokens.units(tokens.t.geometry.marker_radius_px_at_1080p)
    dots = VGroup(
        Dot(radius=dot_radius * 0.8, **tokens.dot_kwargs(signal)),
        Dot(radius=dot_radius * 0.8, **tokens.dot_kwargs(grey)),
        Dot(radius=dot_radius, **tokens.dot_kwargs(signal)),
    )
    peak_line = VMobject(color=emphasis, stroke_width=tokens.stroke_width(curve_px * 0.6), **ends)
    peak_dot = Dot(radius=dot_radius, **tokens.dot_kwargs(emphasis))
    drop_text = VGroup()
    group.add(cursor, reference, numerator, denominator, response, dots, peak_line, peak_dot, drop_text)

    # Legend: a row above the plot.
    legend = VGroup()
    swatch = tokens.units(20)
    for name, color, width, opacity, dashed in (
        ("numerator", signal, curve_px * 0.7, 0.62, False),
        ("denominator", grey, curve_px * 0.8, 1.0, True),
        ("response", signal, curve_px * 1.35, 1.0, False),
    ):
        if dashed:
            mark = VGroup(*[
                Line(LEFT * swatch + RIGHT * swatch * 2 * (k / 2.5), LEFT * swatch + RIGHT * swatch * 2 * ((k + 0.55) / 2.5),
                     color=color, stroke_width=tokens.stroke_width(width), **ends)
                for k in range(3)
            ])
        else:
            mark = Line(LEFT * swatch, RIGHT * swatch, color=color, stroke_width=tokens.stroke_width(width), **ends)
            mark.set_stroke(opacity=opacity)
        entry = VGroup(mark, Text(name, **tokens.text_kwargs(annotation, foreground))).arrange(RIGHT, buff=gap)
        legend.add(entry)
    legend.arrange(RIGHT, buff=tokens.units(40))
    legend.move_to(axes.c2p(x_lo, y_hi) + UP * (header * 0.62), aligned_edge=LEFT)
    group.add(legend)
    if arguments.get("data_label"):
        note = Text(arguments["data_label"], **tokens.text_kwargs(caption, grey))
        note.next_to(labels, DOWN, buff=tokens.baseline * 0.8).align_to(labels, RIGHT)
        group.add(note)

    # Parameter readouts under the field (disc mode): the value that differs from its reference takes the emphasis ink.
    readouts = VGroup()
    group.add(readouts)
    shown: dict[str, tuple] = {}

    def show_text(holder: VGroup, key: str, string: str, color: Any, opacity: float, face: Any = annotation) -> Text:
        """A fresh `Text` in `holder` when what it shows changed (never `become`, which would carry history)."""
        state = (string, str(color), round(opacity, 3))
        if shown.get(key) != state or not holder.submobjects:
            for old in list(holder.submobjects):
                holder.remove(old)
                scene.retire(old)
            fresh = Text(string, **tokens.text_kwargs(face, color))
            fresh.set_opacity(opacity)
            holder.add(fresh)
            scene.fix_in_frame(fresh)
            shown[key] = state
        return holder.submobjects[0]

    readout_d, readout_ratio = VGroup(), VGroup()
    if mode == "disc":
        readouts.add(readout_d, readout_ratio)

    def corners(mobject: VMobject, points: list[np.ndarray]) -> None:
        if len(points) < 2:
            points = [points[0], points[0] + RIGHT * 1e-4]
        mobject.set_points_as_corners(points)

    def dashes(mobject: VMobject, runs: list[list[np.ndarray]]) -> None:
        mobject.clear_points()
        for run in runs:
            if len(run) < 2:
                continue
            mobject.start_new_path(run[0])
            mobject.add_points_as_corners(run[1:])
        if not mobject.has_points():
            origin = runs[0][0] if runs and runs[0] else axes.c2p(x_lo, y_lo)
            mobject.set_points_as_corners([origin, origin + RIGHT * 1e-4])

    def rebuild(values: dict[str, float]) -> list[str]:
        stimulus_at = float(values["position"])  # the bar may start and end outside the plotted range
        position = min(max(stimulus_at, x_lo), x_hi)
        d = float(values.get("d", base_d))
        ratio = float(values.get("ratio", base_ratio))
        annotate = min(max(float(values.get("annotate", 0.0)), 0.0), 1.0)
        table, warnings = curves.at(d, ratio)
        xs = table["x"]

        # The field: the normalization pool's circle and its label, the stimulus.
        sigma_2 = ratio * sigma_1
        centre = field_point(*mu)
        runs = []
        for k in range(CIRCLE_DASHES):
            angles = np.linspace(2 * math.pi * k / CIRCLE_DASHES, 2 * math.pi * (k + 0.55) / CIRCLE_DASHES, 5)
            runs.append([centre + sigma_2 * per_degree * np.array([math.cos(t), math.sin(t), 0.0]) for t in angles])
        dashes(normalization, runs)
        anchor = centre + (sigma_2 * per_degree + gap * 2.2) * np.array([math.cos(0.75), math.sin(0.75), 0.0])
        label_2.points = label_2_points.copy()
        for member, original in zip(label_2.family_members_with_points(), label_2_family_points, strict=True):
            member.points = original.copy()
        label_2.shift(anchor - label_2_centre)
        if mode == "bar":
            left, right = max(stimulus_at - bar_width / 2, -aperture), min(stimulus_at + bar_width / 2, aperture)
            if right - left <= 1e-6:
                edge = field_point(math.copysign(aperture, stimulus_at), 0.0)
                corners(stimulus, [edge, edge + UP * 1e-4, edge + RIGHT * 1e-4])
            else:
                across = np.linspace(left, right, 24)
                upper = [field_point(x, math.sqrt(max(aperture**2 - x**2, 0.0))) for x in across]
                lower = [field_point(x, -math.sqrt(max(aperture**2 - x**2, 0.0))) for x in across[::-1]]
                corners(stimulus, upper + lower + [upper[0]])
        else:
            extent = max(position, 1e-3)
            if np.hypot(*mu) + extent > aperture + 1e-6:
                warnings.append(f"dn_views: a disc of {extent:g} deg around the pRF leaves the aperture; it is drawn unclipped")
            ring = np.linspace(0, 2 * math.pi, 73)
            corners(stimulus, [field_point(mu[0] + extent * math.cos(t), mu[1] + extent * math.sin(t)) for t in ring])

        # The plot: every trace up to the stimulus position.
        count = int(np.searchsorted(xs, position, side="right"))
        ends_at = {}
        traces = {}
        for name in ("numerator", "denominator", "response"):
            ys = table[name]
            y_end = float(np.interp(position, xs, ys))
            pairs = [(float(x), float(y)) for x, y in zip(xs[:count], ys[:count], strict=True) if x >= x_lo - 1e-9]
            if not pairs or position - pairs[-1][0] > 1e-9:
                pairs.append((position, y_end))
            traces[name] = [axes.c2p(x, min(max(y, y_lo), y_hi)) for x, y in pairs]
            ends_at[name] = axes.c2p(position, min(max(y_end, y_lo), y_hi))
            if min(ys) < y_lo - 1e-9 or max(ys) > y_hi + 1e-9:
                warnings.append(f"dn_views: the {name} leaves the plot's range [{y_lo:g}, {y_hi:g}] and is clipped")
        corners(numerator, traces["numerator"])
        corners(response, traces["response"])
        on, off = TRACE_DASH
        points = traces["denominator"]
        dashes(denominator, [points[i : i + on + 1] for i in range(0, len(points), on + off)])
        base_ys = curves.base["response"]
        moved = abs(d - base_d) > 1e-9 or abs(ratio - base_ratio) > 1e-9
        ghost = [axes.c2p(float(x), float(y)) for x, y in zip(curves.base["x"], base_ys, strict=True) if x_lo - 1e-9 <= x <= position + 1e-9]
        corners(reference, ghost if ghost else [axes.c2p(x_lo, y_lo)])
        reference.set_stroke(opacity=1.0 if moved else 0.0)
        # The cursor and the dots lead the traces and leave as the stimulus reaches the end of the axis.
        lead = min(1.0, max(0.0, (x_hi - position) / (0.04 * (x_hi - x_lo))))
        cursor.put_start_and_end_on(axes.c2p(position, y_lo), axes.c2p(position, y_hi))
        cursor.set_stroke(opacity=lead)
        for dot, name in zip(dots, ("numerator", "denominator", "response"), strict=True):
            dot.move_to(ends_at[name])
            dot.set_fill(opacity=lead)
            dot.set_stroke(opacity=lead)

        # Disc mode: the peak of the response, the drop from the peak to the largest disc, the parameter values.
        if mode == "disc":
            ys = table["response"]
            top = int(np.argmax(ys))
            peak_x, peak_y, last_y = float(xs[top]), float(ys[top]), float(ys[-1])
            foot, head = axes.c2p(peak_x, y_lo), axes.c2p(peak_x, min(peak_y, y_hi))
            steps = max(1, round(float(np.linalg.norm(head - foot)) / tokens.units(18)))
            dashes(peak_line, [[foot + (head - foot) * (k / steps), foot + (head - foot) * ((k + 0.55) / steps)] for k in range(steps)])
            drop = 100 * (last_y / peak_y - 1) if peak_y > 0 else 0.0
            # On a plateau the maximum is not a peak: the marker leaves as the drop goes to zero.
            marked = annotate * min(1.0, max(0.0, abs(drop) / 3.0))
            peak_line.set_stroke(opacity=marked)
            peak_dot.move_to(head)
            peak_dot.set_fill(opacity=marked)
            peak_dot.set_stroke(opacity=marked)
            string = f"−{abs(drop):.0f}% by {x_hi:g}°" if drop < -0.5 else f"no drop by {x_hi:g}°"
            text = show_text(drop_text, "drop", string, emphasis, annotate)
            text.move_to(axes.c2p(x_hi, last_y) + DOWN * tokens.units(34), aligned_edge=RIGHT)
            text_d = show_text(readout_d, "d", f"d = {d:.2f}", emphasis if abs(d - base_d) > 1e-9 else grey, 1.0)
            text_ratio = show_text(
                readout_ratio, "ratio", f"σ₂ / σ₁ = {ratio:.1f}", emphasis if abs(ratio - base_ratio) > 1e-9 else grey, 1.0
            )
            baseline = sy0 + left_footer * 0.3
            text_d.move_to(np.array([field_centre[0] - radius * 0.62, baseline, 0.0]), aligned_edge=LEFT)
            text_ratio.move_to(np.array([field_centre[0] + radius * 0.02, baseline, 0.0]), aligned_edge=LEFT)
        else:
            peak_line.set_stroke(opacity=0.0)
            peak_dot.set_fill(opacity=0.0)
            peak_dot.set_stroke(opacity=0.0)
        return warnings

    label_2_family_points = [member.points.copy() for member in label_2.family_members_with_points()]
    parameters = {"position": start}
    if mode == "disc":
        parameters.update({"d": base_d, "ratio": base_ratio, "annotate": 0.0})
    parameters.update(declared)
    rebuild(parameters)
    return ParametricComponent(
        object_id,
        "custom",
        group,
        text=None,
        font_px=tokens.smallest_text_px("tick_label", "axis_label", "annotation", "caption"),
        text_like=False,
        parameters=parameters,
        rebuild=rebuild,
    )
