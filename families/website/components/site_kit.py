"""Drawing helpers shared by the website's custom manim8 builders.

Everything takes its colours, strokes and type from the style tokens by role, so one builder renders under
the light and the dark site style. Lengths are frame units unless a name says `px` (pixels at 1080p, the
unit of the style sheet). Builders keep every drawn state a function of their parameter values alone: the
helpers here set points and ink absolutely and never accumulate a change.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from manim import DOWN, LEFT, PI, RIGHT, UP, Circle, Line, Polygon, Text, VGroup, VMobject


def smooth(x: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3 - 2 * x)


def clip01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def declared(data: dict[str, Any] | None) -> dict[str, float]:
    """The parameters a custom object's data binding declares, with their starting values.

    manim8 accepts a `vary_parameter` action on a custom object only when the object has a data binding whose
    transforms name parameters, so each project binds `data/controls.csv` (unit values, one column per
    parameter) and scales each column by its parameter. The table carries no data."""
    return {
        t["by"]["parameter"]: float(t["by"]["default"])
        for t in (data or {}).get("transforms", [])
        if isinstance(t.get("by"), dict)
    }


def project_root(scene: Any) -> Path:
    return Path(scene.ctx.project_root) if getattr(scene, "ctx", None) is not None else Path.cwd()


def corners(mobject: VMobject, points: list[np.ndarray]) -> None:
    """Set a path through `points`; a single point becomes a path too short to see."""
    if len(points) < 2:
        points = [points[0], points[0] + RIGHT * 1e-4]
    mobject.set_points_as_corners(points)


def dashes(mobject: VMobject, runs: list[list[np.ndarray]], fallback: np.ndarray | None = None) -> None:
    """Set several separate strokes in one mobject."""
    mobject.clear_points()
    for run in runs:
        if len(run) < 2:
            continue
        mobject.start_new_path(run[0])
        mobject.add_points_as_corners(run[1:])
    if not mobject.has_points():
        origin = runs[0][0] if runs and runs[0] else (fallback if fallback is not None else np.zeros(3))
        mobject.set_points_as_corners([origin, origin + RIGHT * 1e-4])


def dashed_runs(points: list[np.ndarray], on: float, off: float) -> list[list[np.ndarray]]:
    """Cut a polyline into dashes `on` long with gaps `off` long, measured along the line."""
    runs: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    travelled, drawing, left = 0.0, True, on
    for a, b in zip(points[:-1], points[1:], strict=True):
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = float(np.linalg.norm(b - a))
        done = 0.0
        if drawing and not current:
            current = [a]
        while length - done > left:
            done += left
            p = a + (b - a) * (done / length)
            if drawing:
                current.append(p)
                runs.append(current)
                current = []
            else:
                current = [p]
            drawing = not drawing
            left = on if drawing else off
        left -= length - done
        if drawing:
            current.append(b)
        travelled += length
    if drawing and len(current) > 1:
        runs.append(current)
    return runs


def circle_points(centre: np.ndarray, radius: float, count: int = 73) -> list[np.ndarray]:
    return [centre + radius * np.array([math.cos(t), math.sin(t), 0.0]) for t in np.linspace(0, 2 * math.pi, count)]


class Kit:
    """Tokens by role, and text that can change without carrying history."""

    def __init__(self, scene: Any, tokens: Any) -> None:
        self.scene, self.tokens = scene, tokens
        self._shown: dict[int, tuple] = {}

    def color(self, role: str) -> Any:
        return self.tokens.color(role)

    def mix(self, role_a: str, role_b: str, t: float) -> Any:
        return self.tokens.color(role_a).interpolate(self.tokens.color(role_b), clip01(t))

    def ramp(self, roles: tuple[str, ...], t: float) -> Any:
        t = clip01(t) * (len(roles) - 1)
        i = min(int(t), len(roles) - 2)
        return self.mix(roles[i], roles[i + 1], t - i)

    def stroke(self, px: float) -> float:
        return self.tokens.stroke_width(px)

    def units(self, px: float) -> float:
        return self.tokens.units(px)

    @property
    def ends(self) -> dict[str, Any]:
        return self.tokens.line_kwargs()

    @property
    def curve_px(self) -> float:
        return self.tokens.t.geometry.curve_stroke_px_at_1080p

    @property
    def axis_px(self) -> float:
        return self.tokens.t.geometry.axis_stroke_px_at_1080p

    def text(self, string: str, role: str = "annotation", color: Any = "foreground", px: float | None = None, **kw: Any) -> Text:
        ink = self.color(color) if isinstance(color, str) else color
        return Text(string, **self.tokens.text_kwargs(self.tokens.typeface(role), ink, px), **kw)

    def line(self, a: Any, b: Any, color: Any = "foreground", px: float | None = None, opacity: float = 1.0) -> Line:
        ink = self.color(color) if isinstance(color, str) else color
        line = Line(a, b, color=ink, stroke_width=self.stroke(px if px is not None else self.axis_px), **self.ends)
        if opacity != 1.0:
            line.set_stroke(opacity=opacity)
        return line

    def path(self, color: Any = "foreground", px: float | None = None, opacity: float = 1.0) -> VMobject:
        ink = self.color(color) if isinstance(color, str) else color
        mobject = VMobject(color=ink, stroke_width=self.stroke(px if px is not None else self.curve_px), **self.ends)
        mobject.set_fill(opacity=0)
        if opacity != 1.0:
            mobject.set_stroke(opacity=opacity)
        return mobject

    def shape(self, fill: Any, opacity: float = 1.0) -> VMobject:
        ink = self.color(fill) if isinstance(fill, str) else fill
        mobject = VMobject(stroke_width=0)
        mobject.set_fill(ink, opacity=opacity)
        return mobject

    def dot(self, color: Any = "primary_signal", radius_px: float | None = None) -> Circle:
        ink = self.color(color) if isinstance(color, str) else color
        radius = self.units(radius_px if radius_px is not None else self.tokens.t.geometry.marker_radius_px_at_1080p)
        dot = Circle(radius=radius, stroke_width=0)
        dot.set_fill(ink, opacity=1.0)
        return dot

    def arrow(self, a: Any, b: Any, color: Any = "foreground", px: float = 3.0, tip: float = 0.16) -> VGroup:
        """A straight arrow as a line and a filled tip (drawn here so its parts can be posed and faded)."""
        ink = self.color(color) if isinstance(color, str) else color
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        n = np.array([-u[1], u[0], 0.0])
        shaft = Line(a, b - u * tip * 0.8, color=ink, stroke_width=self.stroke(px))
        head = Polygon(b, b - u * tip + n * tip * 0.42, b - u * tip - n * tip * 0.42, stroke_width=0)
        head.set_fill(ink, opacity=1.0)
        return VGroup(shaft, head)

    def show(self, holder: VGroup, string: str, role: str = "annotation", color: Any = "foreground",
             opacity: float = 1.0, px: float | None = None) -> Text:
        """Show `string` in `holder` as a fresh `Text` whenever what it shows changes (never `become`, which
        would carry the outlines of every earlier value); the caller places the returned text."""
        ink = self.color(color) if isinstance(color, str) else color
        state = (string, role, str(ink), round(opacity, 3), px)
        if self._shown.get(id(holder)) != state or not holder.submobjects:
            for old in list(holder.submobjects):
                holder.remove(old)
                self.scene.retire(old)
            fresh = self.text(string, role, ink, px)
            fresh.set_opacity(opacity)
            holder.add(fresh)
            self.scene.fix_in_frame(fresh)
            self._shown[id(holder)] = state
        return holder.submobjects[0]


def set_ink(mobject: Any, opacity: float) -> None:
    """Scale the full ink of a static mobject (captured the first time) by `opacity`, family included."""
    for member in mobject.get_family():
        if not isinstance(member, VMobject) or not member.has_points():
            continue
        if not hasattr(member, "_full_ink"):
            member._full_ink = (member.get_fill_opacity(), member.get_stroke_opacity())
        fill, stroke = member._full_ink
        member.set_fill(opacity=fill * opacity, family=False)
        member.set_stroke(opacity=stroke * opacity, family=False)


class Pose:
    """A static group that can be scaled and moved as a whole, always from its points as built."""

    def __init__(self, group: Any) -> None:
        self.group = group
        self.members = [m for m in group.get_family() if isinstance(m, VMobject) and m.has_points()]
        self.points = [m.points.copy() for m in self.members]
        self.centre = np.asarray(group.get_center(), float).copy()

    def set(self, scale: float = 1.0, centre: np.ndarray | None = None, about: np.ndarray | None = None) -> None:
        """Scale about `about` (the built centre by default), then move the built centre to `centre`."""
        pivot = self.centre if about is None else np.asarray(about, float)
        target = self.centre if centre is None else np.asarray(centre, float)
        shift = target - (pivot + (self.centre - pivot) * scale)
        for member, base in zip(self.members, self.points, strict=True):
            member.points = pivot + (base - pivot) * scale + shift


class Field:
    """A circular visual field: degrees to frame units, with its furniture."""

    def __init__(self, kit: Kit, centre: Any, radius: float, extent_deg: float) -> None:
        self.kit, self.centre, self.radius, self.extent = kit, np.array([centre[0], centre[1], 0.0]), radius, extent_deg
        self.per_degree = radius / extent_deg

    def pt(self, x: float, y: float) -> np.ndarray:
        return self.centre + self.per_degree * np.array([x, y, 0.0])

    def outline(self, color: str = "muted", px: float = 1.8, fill: str | None = None, fill_opacity: float = 1.0) -> Circle:
        circle = Circle(radius=self.radius, color=self.kit.color(color), stroke_width=self.kit.stroke(px))
        circle.move_to(self.centre)
        if fill:
            circle.set_fill(self.kit.color(fill), opacity=fill_opacity)
        return circle

    def fixation(self, arm_px: float = 11, px: float = 2.4, color: str = "foreground") -> VGroup:
        arm = self.kit.units(arm_px)
        return VGroup(
            self.kit.line(self.centre + LEFT * arm, self.centre + RIGHT * arm, color, px),
            self.kit.line(self.centre + DOWN * arm, self.centre + UP * arm, color, px),
        )

    def ring(self, x: float, y: float, sigma: float, count: int = 73) -> list[np.ndarray]:
        return circle_points(self.pt(x, y), sigma * self.per_degree, count)

    def bar(self, centre_deg: float, width_deg: float, direction_deg: float) -> list[np.ndarray] | None:
        """Corner points of a bar clipped to the field: `centre_deg` along the direction of motion
        (counter-clockwise from rightward), the bar perpendicular to it. None when it lies outside."""
        low = max(centre_deg - width_deg / 2, -self.extent)
        high = min(centre_deg + width_deg / 2, self.extent)
        if high - low <= 1e-6:
            return None
        angle = math.radians(direction_deg)
        u = np.array([math.cos(angle), math.sin(angle), 0.0])
        v = np.array([-math.sin(angle), math.cos(angle), 0.0])
        along = np.linspace(low, high, 20)
        reach = [math.sqrt(max(self.extent**2 - s**2, 0.0)) for s in along]
        upper = [self.centre + self.per_degree * (s * u + q * v) for s, q in zip(along, reach, strict=True)]
        lower = [self.centre + self.per_degree * (s * u - q * v) for s, q in zip(along[::-1], reach[::-1], strict=True)]
        return upper + lower + [upper[0]]


class Panel:
    """A plot box: data coordinates to frame units, with despined axes drawn on request."""

    def __init__(self, kit: Kit, left: float, bottom: float, width: float, height: float,
                 x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
        self.kit = kit
        self.left, self.bottom, self.width, self.height = left, bottom, width, height
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def top(self) -> float:
        return self.bottom + self.height

    def pt(self, x: float, y: float) -> np.ndarray:
        return np.array([
            self.left + (x - self.x0) / (self.x1 - self.x0) * self.width,
            self.bottom + (y - self.y0) / (self.y1 - self.y0) * self.height,
            0.0,
        ])

    def polyline(self, xs: Any, ys: Any) -> list[np.ndarray]:
        return [self.pt(float(x), min(max(float(y), self.y0), self.y1)) for x, y in zip(xs, ys, strict=True)]

    def upto(self, xs: np.ndarray, ys: np.ndarray, x_end: float) -> list[np.ndarray]:
        """The polyline from the first sample to `x_end`, the last point interpolated."""
        count = int(np.searchsorted(xs, x_end, side="right"))
        points = self.polyline(xs[:count], ys[:count])
        if count == 0 or x_end - xs[count - 1] > 1e-9:
            points.append(self.pt(x_end, min(max(float(np.interp(x_end, xs, ys)), self.y0), self.y1)))
        return points

    def y_axis(self, ticks: tuple[float, ...] = (), label: str | None = None, color: str = "muted",
               label_color: str = "secondary_text", label_gap_px: float = 16) -> VGroup:
        kit = self.kit
        tick = kit.units(kit.tokens.t.geometry.tick_length_px_at_1080p)
        group = VGroup(kit.line(self.pt(self.x0, self.y0), self.pt(self.x0, self.y1), color))
        numbers = VGroup()
        for value in ticks:
            foot = self.pt(self.x0, value)
            group.add(kit.line(foot, foot + LEFT * tick, color))
            number = kit.text(f"{value:g}", "tick_label", label_color)
            number.next_to(foot + LEFT * tick, LEFT, buff=kit.units(8))
            numbers.add(number)
        group.add(numbers)
        if label:
            text = kit.text(label, "axis_label", "foreground").rotate(PI / 2)
            anchor = numbers if len(numbers) else group
            text.next_to(anchor, LEFT, buff=kit.units(label_gap_px))
            text.set_y(self.bottom + self.height / 2)
            group.add(text)
        return group

    def x_axis(self, ticks: tuple[float, ...] = (), label: str | None = None, color: str = "muted",
               label_color: str = "secondary_text") -> VGroup:
        kit = self.kit
        tick = kit.units(kit.tokens.t.geometry.tick_length_px_at_1080p)
        group = VGroup(kit.line(self.pt(self.x0, self.y0), self.pt(self.x1, self.y0), color))
        numbers = VGroup()
        for value in ticks:
            foot = self.pt(value, self.y0)
            group.add(kit.line(foot, foot + DOWN * tick, color))
            number = kit.text(f"{value:g}", "tick_label", label_color)
            number.next_to(foot + DOWN * tick, DOWN, buff=kit.units(8))
            numbers.add(number)
        group.add(numbers)
        if label:
            text = kit.text(label, "axis_label", "foreground")
            text.next_to(numbers if len(numbers) else group, DOWN, buff=kit.units(12))
            text.set_x(self.left + self.width / 2)
            group.add(text)
        return group
