"""Custom manim8 builder for `fields_to_map`: fitted pRF centres in the visual field and on a sheet of cortex.

Left: the right half of the visual field (semicircle, vertical meridian, fixation) with one dot per voxel at
its fitted pRF centre. Right: a sheet of cortex, one square per voxel on a regular grid. A square and its dot
always share a colour: first the centre's distance from fixation, then its direction from fixation. A line of
text names the colouring.

Everything drawn comes from the project's `data/voxels.csv` (order, sheet_u, sheet_v, x_fit_deg, y_fit_deg,
eccentricity_deg, polar_angle_deg). Colours are ramps between palette roles of the active style.

Parameters that `vary_parameter` drives:
  arrive   0 to 1: the voxels appear in their `order`, staggered at an even pace, each growing from small
  angle    0 to 1: every square and dot changes from the eccentricity colour to the polar-angle colour, and
           the line of text with it
"""

from __future__ import annotations

from typing import Any

import numpy as np
from manim import DOWN, UP, Arc, PI, VGroup

from scianim_bridge.components.base import ParametricComponent
from site_kit import Kit, circle_points, clip01, corners, declared, project_root, read_csv, set_ink, smooth

ECCENTRICITY_RAMP = ("emphasis", "primary_signal", "theme_perception")
ANGLE_RAMP = ("comparison", "theme_perception", "primary_signal")


def build(scene: Any, object_id: str, tokens: Any, arguments: dict[str, Any], *, style_role: str | None = None,
          data: dict[str, Any] | None = None, equation: dict[str, Any] | None = None) -> ParametricComponent:
    kit = Kit(scene, tokens)
    rows = sorted(read_csv(project_root(scene) / arguments.get("table", "data/voxels.csv")), key=lambda r: int(r["order"]))
    n = len(rows)
    u = np.array([float(r["sheet_u"]) for r in rows])
    v = np.array([float(r["sheet_v"]) for r in rows])
    fx = np.array([float(r["x_fit_deg"]) for r in rows])
    fy = np.array([float(r["y_fit_deg"]) for r in rows])
    ecc = np.array([float(r["eccentricity_deg"]) for r in rows])
    ang = np.array([float(r["polar_angle_deg"]) for r in rows])
    extent = float(arguments.get("field_radius_deg", 6.5))
    grid_step = float(arguments.get("sheet_step", 0.085))
    lag = float(arguments.get("lag_ratio", 0.012))  # each voxel starts this fraction of its own fade after the last
    ecc_roles = tuple(arguments.get("eccentricity_ramp", ECCENTRICITY_RAMP))
    ang_roles = tuple(arguments.get("angle_ramp", ANGLE_RAMP))

    # --- layout (frame 16 by 8) -------------------------------------------------------------------------------
    centre = np.array([-6.3, -0.2, 0.0])
    radius = 2.7
    per_degree = radius / extent
    sheet_height = 5.4
    scale = sheet_height / (v.max() - v.min() + grid_step)
    sheet_x0 = 3.9 - (u.max() - u.min()) * scale / 2
    sheet_y0 = -0.2 - (v.max() + v.min()) / 2 * scale
    square_at = [np.array([sheet_x0 + (a - u.min()) * scale, sheet_y0 + b * scale, 0.0]) for a, b in zip(u, v, strict=True)]
    dot_at = [centre + per_degree * np.array([a, b, 0.0]) for a, b in zip(fx, fy, strict=True)]
    half = grid_step * scale * 0.94 / 2
    dot_radius = kit.units(6.5)
    unit_square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], float)
    unit_circle = np.array(circle_points(np.zeros(3), 1.0, 17))

    group = VGroup()
    arc = Arc(radius=radius, start_angle=-PI / 2, angle=PI, color=kit.color("muted"), stroke_width=kit.stroke(1.8))
    arc.move_arc_center_to(centre)
    arm = kit.units(11)
    group.add(arc, kit.line(centre + radius * DOWN, centre + radius * UP, "muted", 1.8),
              kit.line(centre + np.array([-arm, 0, 0]), centre + np.array([arm, 0, 0]), "foreground", 2.4),
              kit.line(centre + arm * DOWN, centre + arm * UP, "foreground", 2.4))
    for text, x in ((arguments.get("field_label", "the visual field: fitted pRF centers"), -4.75),
                    (arguments.get("sheet_label", "a sheet of cortex: one square per voxel"), 3.9)):
        group.add(kit.text(text, "caption", "secondary_text").move_to([x, 3.25, 0]))
    keys = [kit.text(text, "annotation", "foreground").move_to([-0.9, -3.3, 0])
            for text in (arguments.get("key_eccentricity", "color: distance from fixation"),
                         arguments.get("key_angle", "color: direction from fixation"))]
    group.add(*keys)

    ecc_colors = [kit.ramp(ecc_roles, e / extent) for e in ecc]
    ang_colors = [kit.ramp(ang_roles, (90 - a) / 180) for a in ang]
    squares = [kit.shape(ecc_colors[i]) for i in range(n)]
    dots = [kit.shape(ecc_colors[i]) for i in range(n)]
    group.add(*squares, *dots)
    fade = 1 / (1 + (n - 1) * lag)  # the share of the whole arrival that one voxel's own fade takes
    state: list[tuple | None] = [None] * n

    def rebuild(values: dict[str, float]) -> list[str]:
        arrive = clip01(values["arrive"])
        blend = smooth(clip01(values.get("angle", 0.0)))
        for i in range(n):
            shown = smooth((arrive - i * lag * fade) / fade)
            key = (round(shown, 4), round(blend, 4))
            if state[i] == key:
                continue
            state[i] = key
            size = 0.3 + 0.7 * shown
            color = ecc_colors[i].interpolate(ang_colors[i], blend)
            corners(squares[i], list(square_at[i] + unit_square * half * size))
            corners(dots[i], list(dot_at[i] + unit_circle * dot_radius * size))
            squares[i].set_fill(color, opacity=shown)
            dots[i].set_fill(color, opacity=shown)
        set_ink(keys[0], 1 - blend)
        set_ink(keys[1], blend)
        return []

    parameters = {"arrive": 0.0, "angle": 0.0}
    parameters.update(declared(data))
    rebuild(parameters)
    return ParametricComponent(object_id, "custom", group, text=None,
                               font_px=tokens.smallest_text_px("caption", "annotation"),
                               text_like=False, parameters=parameters, rebuild=rebuild)
