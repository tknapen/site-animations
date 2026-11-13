"""Custom manim8 builder for `model_as_chain`: the modelling loop opens its model box into a chain of stages.

The loop (boxes 'parameters θ' and 'stimulus' above a 'model' box with a gear, a panel with the prediction on
the data below) shrinks to the top of the frame and stays. Its model box opens through a light funnel onto a
row of seven boxes: stimulus S, features Φ(S), spatial pooling, transformation f, neural dynamics, HRF, BOLD.
A dashed outline holds boxes two to six. θ tags sit on pooling, transformation, dynamics and HRF. A pulse runs
from the stimulus box to the BOLD box, and the prediction is drawn over the data there.

Data: the project's `data/panel.csv` (t_s, prediction, data), drawn in the loop's panel and in the BOLD box.
The glyphs inside the boxes (a bar in an aperture, a Gaussian, a haemodynamic response) are pictograms, not
data; the HRF glyph is the usual difference of two gamma functions (peaks at 5 and 15 s, ratio 1/6).

Parameters that `vary_parameter` drives, each a stage running from 0 to 1, in this order:
  travel      dots run from the two top boxes into the model box and the gear turns a quarter
  drop        a dot runs from the model box to the panel
  shrink      the loop shrinks and moves to the top
  open        the model box gets a heavier outline, the funnel appears, the dashed outline is drawn
  boxes       the seven boxes appear one after another
  links       the arrows between them grow one after another
  tags        the θ tags come down from the loop's parameter box
  stems       the tags are joined to their boxes
  pulse       a dot runs from the stimulus box to the BOLD box; a stage darkens while the dot is in it
  prediction  the prediction is drawn over the data in the BOLD box and the pulse leaves
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from manim import DOWN, ITALIC, LEFT, RIGHT, UP, Arrow, Circle, RoundedRectangle, VGroup

from scianim_bridge.components.base import ParametricComponent
from site_kit import Field, Kit, Panel, Pose, clip01, corners, dashed_runs, dashes, declared, project_root, read_csv, set_ink, smooth

FITTED = (2, 3, 4, 5)  # boxes with a θ tag: pooling, transformation, neural dynamics, HRF
ROW_Y, BOX_H = -2.3, 1.6
WIDTHS = [1.7, 2.08, 2.08, 2.08, 2.08, 2.08, 1.7]
GAP = 0.24
LOOP_SCALE, LOOP_Y = 0.45, 2.0


def staggered(progress: float, index: int, count: int, lag: float) -> float:
    """Manim's LaggedStart: item `index` of `count`, each starting `lag` of its own length after the last."""
    share = 1 / (1 + (count - 1) * lag)
    return smooth((progress - index * lag * share) / share)


def hrf_curve() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 24, 97)
    h = t**5 * np.exp(-t) / math.gamma(6) - t**15 * np.exp(-t) / math.gamma(16) / 6
    return t / 24, h / h.max()


def build(scene: Any, object_id: str, tokens: Any, arguments: dict[str, Any], *, style_role: str | None = None,
          data: dict[str, Any] | None = None, equation: dict[str, Any] | None = None) -> ParametricComponent:
    kit = Kit(scene, tokens)
    rows = read_csv(project_root(scene) / arguments.get("table", "data/panel.csv"))
    ts = np.array([float(r["t_s"]) for r in rows])
    pred = np.array([float(r["prediction"]) for r in rows])
    noisy = np.array([float(r["data"]) for r in rows])
    signal = style_role or "primary_signal"
    box_fill, active_fill = kit.color("neutral"), kit.mix("neutral", "muted", 0.5)

    def rbox(width: float, height: float, centre: Any, radius: float = 0.12, px: float = 3.0) -> RoundedRectangle:
        box = RoundedRectangle(width=width, height=height, corner_radius=radius, color=kit.color("foreground"),
                               stroke_width=kit.stroke(px))
        box.set_fill(box_fill, opacity=1.0)
        return box.move_to(np.array([centre[0], centre[1], 0.0]))

    def words(*lines: str, role: str = "caption", italic: tuple[int, ...] = ()) -> VGroup:
        return VGroup(*[kit.text(s, role, "foreground", **({"slant": ITALIC} if i in italic else {}))
                        for i, s in enumerate(lines)]).arrange(DOWN, buff=kit.units(9))

    def bold_panel(width: float, height: float, centre: Any) -> tuple[VGroup, Any, Panel]:
        panel = Panel(kit, centre[0] - width / 2, centre[1] - height / 2, width, height, (0, float(ts[-1])), (-0.5, 1.35))
        measured = kit.path("foreground", 2.2)
        corners(measured, panel.polyline(ts, noisy))
        predicted = kit.path(signal, 4.0)
        corners(predicted, panel.polyline(ts, pred))
        return VGroup(kit.line(panel.pt(0, 0), panel.pt(ts[-1], 0), "muted", 1.6), measured), predicted, panel

    # --- the loop, built at full size -------------------------------------------------------------------------
    theta_box = VGroup(rbox(3.2, 0.95, (-2.15, 2.95)))
    theta_box.add(VGroup(kit.text("parameters", "body"), kit.text("θ", "body", slant=ITALIC)).arrange(RIGHT, buff=0.16).move_to(theta_box[0]))
    stim_box = VGroup(rbox(3.2, 0.95, (2.15, 2.95)))
    stim_box.add(kit.text("stimulus", "body").move_to(stim_box[0]))
    model_rect = rbox(5.0, 1.25, (0, 1.0))
    model_text = kit.text("model", "body", px=46)
    gear_centre = np.array([0.85, 1.0, 0.0])
    model_text.move_to(np.array([-0.4, 1.0, 0.0]))
    teeth, gear_r = 8, 0.32
    gear_base = np.array([
        gear_centre + (gear_r if i % 4 in (1, 2) else gear_r * 0.74)
        * np.array([math.cos(2 * math.pi * (i + 0.5) / (4 * teeth)), math.sin(2 * math.pi * (i + 0.5) / (4 * teeth)), 0.0])
        for i in list(range(4 * teeth)) + [0]
    ])
    gear = kit.path("foreground", 2.5)
    gear.set_fill(box_fill, opacity=1.0)
    corners(gear, list(gear_base))
    hub = Circle(radius=gear_r * 0.3, color=kit.color("foreground"), stroke_width=kit.stroke(2.5)).move_to(gear_centre)
    hub.set_fill(kit.color("background"), opacity=1.0)
    loop_panel, loop_pred, _ = bold_panel(4.8, 1.75, (0, -2.35))
    arrow_kw = dict(buff=0.06, stroke_width=kit.stroke(3.5), color=kit.color("foreground"), tip_length=0.18,
                    max_tip_length_to_length_ratio=0.45)
    a1 = Arrow(theta_box[0].get_bottom(), model_rect.get_top() + LEFT * 1.0, path_arc=0.4, **arrow_kw)
    a2 = Arrow(stim_box[0].get_bottom(), model_rect.get_top() + RIGHT * 1.0, path_arc=-0.4, **arrow_kw)
    a3 = Arrow(model_rect.get_bottom(), loop_panel.get_top() + UP * 0.12, **arrow_kw)
    routes = [[np.array(a.point_from_proportion(p)) for p in np.linspace(0, 1, 25)] for a in (a1, a2)]
    route_down = [np.array(model_rect.get_bottom()), np.array(loop_panel.get_top())]
    loop = VGroup(theta_box, stim_box, model_rect, model_text, gear, hub, a1, a2, a3, loop_panel, loop_pred)
    loop_pose = Pose(loop)
    gear_slot = next(i for i, m in enumerate(loop_pose.members) if m is gear)
    loop_small = np.array([0.0, LOOP_Y, 0.0])

    def loop_point(point: np.ndarray, scale: float, centre: np.ndarray) -> np.ndarray:
        return centre + (np.asarray(point, float) - loop_pose.centre) * scale

    # --- the row of seven boxes --------------------------------------------------------------------------------
    total = sum(WIDTHS) + GAP * 6
    xs, x = [], -total / 2
    for width in WIDTHS:
        xs.append(x + width / 2)
        x += width + GAP
    modules, rects = [], []
    bold_pred = None
    for k, (cx, width) in enumerate(zip(xs, WIDTHS, strict=True)):
        c = np.array([cx, ROW_Y, 0.0])
        rect = rbox(width, BOX_H, c)
        if k == 0:
            icon = Field(kit, c + UP * 0.25, 0.42, 8.0)
            bar = kit.shape("muted", 0.55)
            corners(bar, icon.bar(2.0, 2.0, 0.0))
            disc = icon.outline("foreground", 2.0, fill="background")
            content = [disc, bar, icon.fixation(6, 1.8),
                       VGroup(kit.text("stimulus", "caption"), kit.text("S", "caption", slant=ITALIC)).arrange(RIGHT, buff=0.1).move_to(c + DOWN * 0.5)]
        elif k == 1:
            content = [words("features", "Φ(S)", italic=(1,)).move_to(c)]
        elif k == 2:
            wash, layers, extent, peak = VGroup(), 22, 0.39, 0.9
            targets = [peak * math.exp(-0.5 * (2.6 * (i + 1) / layers) ** 2) for i in range(layers)]
            for i in reversed(range(layers)):
                outer = targets[i + 1] if i + 1 < layers else 0.0
                layer = Circle(radius=extent * (i + 1) / layers, stroke_width=0).move_to(c + UP * 0.28)
                layer.set_fill(kit.color(signal), opacity=1 - (1 - targets[i]) / (1 - outer))
                wash.add(layer)
            content = [wash, words("spatial", "pooling").move_to(c + DOWN * 0.42)]
        elif k == 3:
            content = [words("transformation", "f", italic=(1,)).move_to(c)]
        elif k == 4:
            content = [words("neural", "dynamics").move_to(c)]
        elif k == 5:
            hx, hy = hrf_curve()
            glyph = Panel(kit, c[0] - 0.6, c[1] + 0.22 - 0.135, 1.2, 0.45, (0, 1), (-0.3, 1))
            curve = kit.path("foreground", 3.0)
            corners(curve, glyph.polyline(hx, hy))
            content = [kit.line(glyph.pt(0, 0), glyph.pt(1, 0), "muted", 1.6), curve, kit.text("HRF", "caption").move_to(c + DOWN * 0.42)]
        else:
            small, bold_pred, bold_axes = bold_panel(1.4, 0.8, c + UP * 0.2)
            content = [small, bold_pred, kit.text("BOLD", "caption").move_to(c + DOWN * 0.5)]
        modules.append(VGroup(rect, *content))
        rects.append(rect)
    bold_points = bold_axes.polyline(ts, pred)
    module_poses = [Pose(m) for m in modules]

    hull_rect = RoundedRectangle(width=xs[5] - xs[1] + WIDTHS[1] + 0.26, height=2.1, corner_radius=0.2)
    hull_rect.move_to([(xs[1] + xs[5]) / 2, ROW_Y, 0])
    hull_runs = dashed_runs([np.array(hull_rect.point_from_proportion(p)) for p in np.linspace(0, 1, 400)], 0.16, 0.11)
    hull = kit.path("muted", 2.5)
    funnel = kit.shape("grid")
    links = [(kit.path("foreground", 3.5), kit.shape("foreground"), rects[k].get_right() + RIGHT * 0.03, rects[k + 1].get_left() + LEFT * 0.03)
             for k in range(6)]
    tags, tag_poses, stems = [], [], []
    for k in FITTED:
        target = np.array([xs[k], ROW_Y + 1.5, 0.0])
        frame = RoundedRectangle(width=0.62, height=0.58, corner_radius=0.1, color=kit.color("foreground"), stroke_width=kit.stroke(2.5))
        frame.set_fill(kit.color("background"), opacity=1.0).move_to(target)
        tag = VGroup(frame, kit.text("θ", "body", slant=ITALIC).move_to(target))
        tags.append(tag)
        tag_poses.append(Pose(tag))
        stems.append((kit.path("foreground", 2.5), np.array(frame.get_bottom()), np.array(rects[k].get_top())))
    travellers = [kit.dot("foreground", 10) for _ in range(3)]
    pulse = kit.dot("foreground", 13)
    pulse_from, pulse_to = np.array(rects[0].get_right()), np.array(rects[6].get_left())

    # Drawn in this order: the pulse runs over the boxes' fills and under their labels and glyphs.
    group = VGroup(funnel, hull, loop, *rects, *[part for shaft, head, _, _ in links for part in (shaft, head)],
                   *[stem for stem, _, _ in stems], pulse, *[VGroup(*module.submobjects[1:]) for module in modules],
                   *tags, *travellers)

    def rebuild(values: dict[str, float]) -> list[str]:
        p = {name: clip01(values.get(name, 0.0)) for name in
             ("travel", "drop", "shrink", "open", "boxes", "links", "tags", "stems", "pulse", "prediction")}
        # The loop: the gear's quarter turn, then the whole loop scaled and moved.
        turn = -math.pi / 2 * smooth(p["travel"])
        rotation = np.array([[math.cos(turn), -math.sin(turn), 0], [math.sin(turn), math.cos(turn), 0], [0, 0, 1]])
        loop_pose.points[gear_slot] = gear_centre + (loop_pose_gear - gear_centre) @ rotation.T
        s = smooth(p["shrink"])
        scale = 1 + (LOOP_SCALE - 1) * s
        centre = loop_pose.centre + (loop_small - loop_pose.centre) * s
        loop_pose.set(scale, centre)
        model_rect.set_stroke(width=kit.stroke(3.0 + 2.0 * smooth(p["open"])))
        for dot, route, stage in ((travellers[0], routes[0], "travel"), (travellers[1], routes[1], "travel"), (travellers[2], route_down, "drop")):
            along = smooth(p[stage]) * (len(route) - 1)
            i = min(int(along), len(route) - 2)
            dot.move_to(loop_point(route[i] + (route[i + 1] - route[i]) * (along - i), scale, centre))
            set_ink(dot, 1.0 if 0.0 < p[stage] < 1.0 else 0.0)
        # The funnel and the dashed outline.
        low_left, low_right = (loop_point(model_rect_corners[0], scale, centre), loop_point(model_rect_corners[1], scale, centre))
        corners(funnel, [low_left, low_right, hull_corners[1], hull_corners[0], low_left])
        set_ink(funnel, smooth(p["open"]))
        shown = int(round(smooth(p["open"]) * len(hull_runs)))
        dashes(hull, hull_runs[:shown], fallback=hull_runs[0][0])
        set_ink(hull, 1.0 if shown else 0.0)
        # The boxes, the arrows between them, the tags and their stems.
        pulse_at = pulse_from + (pulse_to - pulse_from) * p["pulse"]
        for k, (module, pose) in enumerate(zip(modules, module_poses, strict=True)):
            a = staggered(p["boxes"], k, 7, 0.14)
            pose.set(0.85 + 0.15 * a)
            if 1 <= k <= 5:
                near = clip01((1.15 - abs(pulse_at[0] - xs[k])) / 0.3) if 0.0 < p["pulse"] < 1.0 else 0.0
                rects[k].set_fill(box_fill.interpolate(active_fill, near))
            set_ink(module, a)
        for k, (shaft, head, start, end) in enumerate(links):
            a = staggered(p["links"], k, 6, 0.12)
            tip_at = start + (end - start) * max(a, 1e-3)
            size = 0.15 * max(a, 1e-3)
            corners(shaft, [start, tip_at - RIGHT * size * 0.8])
            corners(head, [tip_at, tip_at - RIGHT * size + UP * size * 0.42, tip_at - RIGHT * size + DOWN * size * 0.42, tip_at])
            set_ink(shaft, 1.0 if a > 0 else 0.0)
            set_ink(head, 1.0 if a > 0 else 0.0)
        a = smooth(p["tags"])
        origin = loop_point(theta_centre, scale, centre)
        for tag, pose in zip(tags, tag_poses, strict=True):
            pose.set(0.6 + 0.4 * a, origin + (pose.centre - origin) * a)
            set_ink(tag, clip01(a * 1.6))
        for stem, top, bottom in stems:
            corners(stem, [top, top + (bottom - top) * max(smooth(p["stems"]), 1e-3)])
            set_ink(stem, 1.0 if p["stems"] > 0 else 0.0)
        # The pulse, and the prediction in the BOLD box.
        pulse.move_to(pulse_at)
        set_ink(pulse, clip01(p["pulse"] * 40) * (1 - smooth(p["prediction"])))
        count = int(round(p["prediction"] * (len(bold_points) - 1)))
        corners(bold_pred, bold_points[: count + 1])
        set_ink(bold_pred, (1.0 if count > 0 else 0.0) * staggered(p["boxes"], 6, 7, 0.14))
        return []

    loop_pose_gear = loop_pose.points[gear_slot].copy()
    model_rect_corners = (np.array(model_rect.get_corner(DOWN + LEFT)), np.array(model_rect.get_corner(DOWN + RIGHT)))
    hull_corners = (np.array(hull_rect.get_corner(UP + LEFT)), np.array(hull_rect.get_corner(UP + RIGHT)))
    theta_centre = np.array(theta_box[0].get_center())
    parameters = {name: 0.0 for name in ("travel", "drop", "shrink", "open", "boxes", "links", "tags", "stems", "pulse", "prediction")}
    parameters.update(declared(data))
    rebuild(parameters)
    return ParametricComponent(object_id, "custom", group, text=None,
                               font_px=tokens.smallest_text_px("caption", "annotation"),
                               text_like=False, parameters=parameters, rebuild=rebuild)
