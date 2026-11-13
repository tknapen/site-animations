"""Custom manim8 builder for `stimulus_to_bold`: one cursor through a bar-mapping run.

Left: the visual field with fixation, the voxel's pRF as a one-sigma circle and the bar at the cursor's time.
Right, on one time axis: the stimulus as a strip (a band per sweep, an arrow for its direction), the predicted
neural response and the predicted BOLD response, each drawn up to the cursor, a dot riding each trace.

Everything drawn comes from the project's `data/run.csv` (one row per second: bar centre and direction,
empty in blanks; neural; bold). Parameters that `vary_parameter` drives: `t` (seconds of the run) and
`cursor` (the ink of the cursor and its dots, 1 to 0). See `site_kit.declared` for how they are declared.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from manim import DOWN, LEFT, PI, RIGHT, VGroup

from scianim_bridge.components.base import ParametricComponent
from site_kit import Field, Kit, Panel, clip01, corners, declared, project_root, read_csv, set_ink


def build(scene: Any, object_id: str, tokens: Any, arguments: dict[str, Any], *, style_role: str | None = None,
          data: dict[str, Any] | None = None, equation: dict[str, Any] | None = None) -> ParametricComponent:
    kit = Kit(scene, tokens)
    rows = read_csv(project_root(scene) / arguments.get("table", "data/run.csv"))
    ts = np.array([float(r["t_s"]) for r in rows])
    centre = [float(r["bar_centre_deg"]) if r["bar_centre_deg"] != "" else None for r in rows]
    direction = [float(r["bar_direction_deg"]) if r["bar_direction_deg"] != "" else None for r in rows]
    neural = np.array([float(r["neural"]) for r in rows])
    bold = np.array([float(r["bold"]) for r in rows])
    t_end = float(ts[-1] + 1)
    prf = arguments.get("prf", {"x": 2.5, "y": 1.5, "sigma": 1.0})
    bar_width = float(arguments.get("bar_width_deg", 2.0))
    signal = style_role or "primary_signal"

    def same_sweep(k: int) -> bool:
        return (k + 1 < len(ts) and centre[k] is not None and centre[k + 1] is not None
                and direction[k + 1] == direction[k] and centre[k + 1] > centre[k])

    sweeps, k = [], 0
    while k < len(ts):
        if direction[k] is None:
            k += 1
            continue
        j = k
        while same_sweep(j):
            j += 1
        sweeps.append((float(ts[k]), float(ts[j] + 1), direction[k]))
        k = j + 1

    def bar_at(t: float) -> tuple[float, float] | None:
        k = int(np.clip(np.floor(t), 0, len(ts) - 1))
        if centre[k] is None:
            return None
        c = centre[k]
        if same_sweep(k):
            c += (centre[k + 1] - c) * (t - k)
        return c, direction[k]

    # --- layout (frame 16 by 8) -------------------------------------------------------------------------------
    field = Field(kit, (-5.55, -0.15), 1.5, float(arguments.get("field_radius_deg", 8.0)))
    left, width = -2.1, 9.15
    strip = Panel(kit, left, 2.72, width, 0.72, (0, t_end), (0, 1))
    neural_panel = Panel(kit, left, 0.45, width, 1.95, (0, t_end), (-0.05, 0.75))
    bold_panel = Panel(kit, left, -2.55, width, 2.6, (0, t_end), (-0.3, 0.9))

    group = VGroup()
    bar = kit.shape("neutral")
    outline = kit.path("foreground", kit.curve_px * 0.85)
    corners(outline, field.ring(prf["x"], prf["y"], prf["sigma"]))
    group.add(field.outline("muted", 1.8), bar, outline, field.fixation())

    bands = VGroup()
    for start, end, angle in sweeps:
        band = kit.shape("neutral")
        a, b = strip.pt(start, 0), strip.pt(end, 1)
        corners(band, [a, np.array([b[0], a[1], 0]), b, np.array([a[0], b[1], 0]), a])
        middle = (a + b) / 2
        u = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0.0])
        bands.add(band, kit.arrow(middle - u * 0.25, middle + u * 0.25, "foreground", 3.2, tip=0.15))
    group.add(bands)
    label = kit.text("stimulus", "annotation", "foreground")
    label.next_to(strip.pt(0, 0.5), LEFT, buff=kit.units(34))
    group.add(label)
    for panel, lines in ((neural_panel, arguments.get("neural_label", ["predicted", "neural"])),
                         (bold_panel, arguments.get("bold_label", ["predicted", "BOLD"]))):
        heading = VGroup(*[kit.text(line, "annotation", "foreground") for line in lines]).arrange(DOWN, buff=kit.units(10))
        heading.rotate(PI / 2).move_to(np.array([panel.left - kit.units(150), panel.bottom + panel.height / 2, 0.0]))
        group.add(heading, kit.line(panel.pt(0, 0), panel.pt(t_end, 0), "grid", 2.0), panel.y_axis((0, 0.5)))
    group.add(bold_panel.x_axis((0, 100, 200, 300), "time (s)"))

    cursor = kit.line(strip.pt(0, -0.1), bold_panel.pt(0, -0.3), "muted", 2.0, opacity=0.7)
    neural_trace = kit.path(signal, kit.curve_px * 0.95)
    bold_trace = kit.path(signal, kit.curve_px * 1.3)
    dots = VGroup(kit.dot(signal, 8), kit.dot(signal, 8))
    group.add(neural_trace, bold_trace, cursor, dots)

    def rebuild(values: dict[str, float]) -> list[str]:
        t = min(max(float(values["t"]), 0.0), t_end - 1e-3)
        ink = clip01(values.get("cursor", 1.0))
        now = bar_at(t)
        points = field.bar(now[0], bar_width, now[1]) if now else None
        if points is None:
            edge = field.pt(0, 0)
            points = [edge, edge + RIGHT * 1e-4, edge + RIGHT * 2e-4]
        corners(bar, points)
        corners(neural_trace, neural_panel.upto(ts, neural, t))
        corners(bold_trace, bold_panel.upto(ts, bold, t))
        cursor.put_start_and_end_on(strip.pt(t, -0.1), bold_panel.pt(t, -0.3))
        dots[0].move_to(neural_panel.pt(t, float(np.interp(t, ts, neural))))
        dots[1].move_to(bold_panel.pt(t, float(np.interp(t, ts, bold))))
        set_ink(cursor, ink)
        set_ink(dots, ink)
        return []

    parameters = {"t": 0.0, "cursor": 1.0}
    parameters.update(declared(data))
    rebuild(parameters)
    return ParametricComponent(object_id, "custom", group, text=None,
                               font_px=tokens.smallest_text_px("tick_label", "axis_label", "annotation"),
                               text_like=False, parameters=parameters, rebuild=rebuild)
