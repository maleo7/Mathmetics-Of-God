#!/usr/bin/env python3
"""
Conic Sections Visualizer  (3D cone slice  +  2D curve, side by side)
---------------------------------------------------------------------
Pick a conic (circle, ellipse, parabola, hyperbola), type in its defining
parameters, and the program draws two panels in one figure:

  LEFT  (3D) : a double-napped cone, the cutting plane at exactly the angle
               that produces your conic, and the intersection curve
               highlighted in crimson.
  RIGHT (2D) : the same curve drawn from its standard equation, with the
               centre/vertex, foci, the a and b lengths, the asymptotes
               (hyperbola) and the directrix (parabola) marked.

THE MATHEMATICS
===============
Double cone (apex at the origin, axis along z):

        x^2 + y^2 = t^2 * z^2          with  t = tan(alpha)

alpha is the half-angle of the cone.  Because z is squared, both nappes
(z > 0 and z < 0) come for free.  A horizontal slice at height z is a
circle of radius t*|z|.

Cutting plane, tilted about the x-axis:

        z = m*y + d                    m = slope,  d = height at y = 0

Substituting the plane into the cone and completing the square gives the
single master equation that this whole program is built on:

        x^2 + A*(y - y0)^2 = t^2*d^2 / A
                 with  A  = 1 - t^2*m^2
                 and   y0 = t^2*m*d / A

The sign of A decides the conic (this is the whole classification!):

        m = 0            ->  A = 1      ->  CIRCLE     (horizontal plane)
        0 < t|m| < 1     ->  A > 0      ->  ELLIPSE    (gentle tilt)
        t|m| = 1         ->  A = 0      ->  PARABOLA   (parallel to a
                                                        slant/generator line)
        t|m| > 1         ->  A < 0      ->  HYPERBOLA  (steep enough to
                                                        reach both nappes)

Careful: (x, y) above are the *horizontal projections* of the curve.  The
true in-plane coordinate along the direction of tilt is stretched by the
slope of the plane,

        v = y * sqrt(1 + m^2)

and it is that in-plane length which must equal the semi-axis the user
asked for.  Every formula below already includes that factor.

Usage:
    python conic_sections_visualizer.py             # interactive
    python conic_sections_visualizer.py --demo      # self-check, no windows
    python conic_sections_visualizer.py --no-plot   # console output only
    python conic_sections_visualizer.py --save      # write conic_<type>.png

Dependencies:
    pip install numpy matplotlib
"""

import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
# Default cone half-angle.  45 degrees (t = tan 45 = 1) gives the classic
# "ice-cream cone" picture where the slant lines have slope 1.
DEFAULT_HALF_ANGLE_DEG = 45.0

CONE_FACE = "#7fb3d5"      # translucent cone surface
CONE_WIRE = "#2e6da4"      # wireframe rings / generator lines
PLANE_FACE = "#f4d03f"     # translucent cutting plane
CURVE_COLOR = "crimson"    # the intersection curve (and the 2D curve)
FOCUS_COLOR = "darkgreen"
GUIDE_COLOR = "#7d3c98"

# Any point of the intersection curve must satisfy x^2 + y^2 - t^2 z^2 = 0.
CONE_TOL = 1e-9


class QuitRequested(Exception):
    """Raised by the input helpers when the user types 'quit'."""


# ==========================================================================
# SECTION 1 - Geometry: derive the cutting plane from the user's parameters
# ==========================================================================
def circle_geometry(r: float, half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG) -> dict:
    """
    CIRCLE: a horizontal plane, m = 0, so z = d is constant.

    The cone's horizontal cross-section at height d has radius t*d, so to
    get radius r we simply take

        d = r / t

    Parametric curve (the ordinary circle parametrisation):
        x = r*cos(theta),  y = r*sin(theta),  z = d
    """
    t = math.tan(math.radians(half_angle_deg))
    m = 0.0
    d = r / t

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full_like(theta, d)

    return _pack(
        kind="circle",
        t=t,
        m=m,
        d=d,
        curves=[(x, y, z)],
        height_factor=1.6,
        params={"r": r},
    )


def ellipse_geometry(a_in: float, b_in: float,
                     half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG) -> dict:
    """
    ELLIPSE: tilt the plane, but keep it shallower than the cone's slant.

    From the master equation with A = 1 - t^2 m^2 > 0 the two semi-axes of
    the section are

        semi-axis across the tilt (along x) :  b = t*d / sqrt(A)
        semi-axis along the tilt (in-plane) :  a = t*d * sqrt(1+m^2) / A

    so their ratio k = a/b = sqrt(1+m^2) / sqrt(A) depends only on m.
    Solving k^2 * (1 - t^2 m^2) = 1 + m^2 for m gives

        m^2 = (k^2 - 1) / (1 + k^2 * t^2)        and then   d = b*sqrt(A)/t

    k = 1 returns m = 0, i.e. the circle - the formulas degrade gracefully.

    Parametric curve (projected onto the ground, then lifted by the plane):
        x = (t*d/sqrt(A)) * cos(theta)
        y = y0 + (t*d/A)  * sin(theta)
        z = m*y + d
    """
    major, minor = max(a_in, b_in), min(a_in, b_in)
    t = math.tan(math.radians(half_angle_deg))

    k = major / minor                       # eccentricity-like axis ratio >= 1
    m = math.sqrt((k * k - 1.0) / (1.0 + k * k * t * t))
    A = 1.0 - t * t * m * m                 # > 0 for an ellipse
    d = minor * math.sqrt(A) / t
    y0 = t * t * m * d / A                  # centre of the projected ellipse

    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    x = (t * d / math.sqrt(A)) * np.cos(theta)
    y = y0 + (t * d / A) * np.sin(theta)
    z = m * y + d

    return _pack(
        kind="ellipse",
        t=t,
        m=m,
        d=d,
        curves=[(x, y, z)],
        height_factor=1.3,
        params={"a": a_in, "b": b_in, "major": major, "minor": minor},
        note="In 3D the major axis lies along the plane's direction of steepest tilt.",
    )


def parabola_geometry(a: float, half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG) -> dict:
    """
    PARABOLA: the plane is exactly parallel to a generator (slant) line,
    which is the borderline case A = 0, i.e.

        m = 1 / t

    With A = 0 the master equation loses its y^2 term and becomes linear
    in y, which is precisely why a parabola appears:

        x^2 - 2*t*d*y - t^2*d^2 = 0     =>     y = (x^2 - t^2 d^2) / (2*t*d)

    Measured inside the tilted plane (v = y*sqrt(1+m^2)) this reads
    v = [sqrt(1+t^2) / (2 t^2 d)] * u^2 + const, so matching the requested
    coefficient a of y = a*x^2 gives

        d = sqrt(1 + t^2) / (2 * t^2 * a)

    A negative a makes d negative: the plane drops below the apex and the
    section lands on the lower nappe, opening downwards. Exactly right.
    """
    t = math.tan(math.radians(half_angle_deg))
    m = 1.0 / t                                   # parallel to the slant line
    d = math.sqrt(1.0 + t * t) / (2.0 * t * t * a)

    # Sweep far enough that the branch is clearly visible inside the cone.
    # An ODD sample count keeps x = 0 in the array, so the vertex itself is
    # a genuine point of the curve rather than something we step over.
    s_max = 3.0 * abs(d)
    x = np.linspace(-s_max, s_max, 501)
    y = (x * x - t * t * d * d) / (2.0 * t * d)
    z = m * y + d

    return _pack(
        kind="parabola",
        t=t,
        m=m,
        d=d,
        curves=[(x, y, z)],
        height_factor=1.2,
        params={"a": a},
        note="The plane is parallel to a slant line of the cone, so the section never closes.",
    )


def hyperbola_geometry(a: float, b: float,
                       half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG) -> dict:
    """
    HYPERBOLA: tilt the plane past the slant so it slices BOTH nappes,
    i.e. A < 0.  Writing B = -A = t^2 m^2 - 1 > 0 the master equation
    rearranges to

        (y - y0)^2 / (t d / B)^2  -  x^2 / (t d / sqrt(B))^2 = 1

    so the semi-axes of the section are

        transverse (in-plane) :  a = t*d*sqrt(1+m^2) / B
        conjugate  (along x)  :  b = t*d / sqrt(B)

    with ratio k = a/b = sqrt(1+m^2)/sqrt(B).  Solving for m:

        m^2 = (1 + k^2) / (k^2 * t^2 - 1)        and then  d = a*B / (t*sqrt(1+m^2))

    That formula needs k^2 t^2 > 1, i.e. tan(alpha) > b/a.  On the default
    45-degree cone only a > b is reachable (a = b would need a vertical
    plane, m -> infinity).  So when a/b is small we simply open the cone
    wider - a perfectly legitimate cone, just blunter.

    Parametric curve, one branch per nappe (cosh/sinh is to the hyperbola
    what cos/sin is to the ellipse, since cosh^2 - sinh^2 = 1):
        x = (t*d/sqrt(B)) * sinh(s)
        y = y0 +- (t*d/B) * cosh(s)
        z = m*y + d
    """
    k = a / b
    # Widen the cone if the requested axis ratio cannot be cut from a 45-deg one.
    t_needed = 1.3 / k
    t = max(math.tan(math.radians(half_angle_deg)), t_needed)
    widened = t > math.tan(math.radians(half_angle_deg)) + 1e-12

    m = math.sqrt((1.0 + k * k) / (k * k * t * t - 1.0))
    B = t * t * m * m - 1.0                  # = -A > 0 for a hyperbola
    d = a * B / (t * math.sqrt(1.0 + m * m))
    y0 = -t * t * m * d / B                  # centre of the projected hyperbola

    s = np.linspace(-1.4, 1.4, 401)   # odd count => s = 0 (the vertex) is sampled
    curves = []
    for branch in (+1.0, -1.0):              # one branch on each nappe
        x = (t * d / math.sqrt(B)) * np.sinh(s)
        y = y0 + branch * (t * d / B) * np.cosh(s)
        z = m * y + d
        curves.append((x, y, z))

    note = "The plane is steeper than the cone's slant, so it cuts both nappes: two branches."
    if widened:
        note += (f"\nCone half-angle opened to {math.degrees(math.atan(t)):.1f}"
                 f"\u00b0 so that a/b = {k:.3g} is an achievable section.")

    return _pack(
        kind="hyperbola",
        t=t,
        m=m,
        d=d,
        curves=curves,
        height_factor=1.12,
        params={"a": a, "b": b},
        note=note,
    )


def _pack(kind, t, m, d, curves, height_factor, params, note="") -> dict:
    """Bundle a section's geometry and work out a sensible drawing box."""
    z_all = np.concatenate([c[2] for c in curves])
    height = max(float(np.max(np.abs(z_all))) * height_factor, 1e-6)
    return {
        "kind": kind,
        "t": t,
        "half_angle_deg": math.degrees(math.atan(t)),
        "m": m,
        "d": d,
        "plane_angle_deg": math.degrees(math.atan(abs(m))),
        "curves": curves,
        "height": height,
        "radius": t * height,          # cone radius at the top/bottom rim
        "params": params,
        "note": note,
    }


def cone_residual(geom: dict) -> float:
    """
    Sanity check: every point of the intersection curve must sit on the cone,
    x^2 + y^2 - t^2 z^2 = 0.  Returns the largest absolute residual.
    """
    t = geom["t"]
    worst = 0.0
    for x, y, z in geom["curves"]:
        scale = np.maximum(1.0, x * x + y * y)          # relative comparison
        worst = max(worst, float(np.max(np.abs(x * x + y * y - t * t * z * z) / scale)))
    return worst


# ==========================================================================
# SECTION 2 - The 3D panel: cone, cutting plane, intersection curve
# ==========================================================================
def draw_double_cone(ax, t: float, height: float) -> None:
    """
    Draw both nappes of x^2 + y^2 = t^2 z^2 as one translucent surface.

    Parametrisation: sweep z from -H to +H and theta around the axis; the
    ring at height z has radius t*|z| (which collapses to the apex at z = 0).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 121)
    z_line = np.linspace(-height, height, 121)
    theta_grid, z_grid = np.meshgrid(theta, z_line)
    r_grid = t * np.abs(z_grid)

    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)

    ax.plot_surface(x, y, z_grid, color=CONE_FACE, alpha=0.20,
                    linewidth=0, antialiased=True, shade=True)
    # Rings + generator lines give the eye something to hold on to.
    ax.plot_wireframe(x, y, z_grid, rstride=10, cstride=15,
                      color=CONE_WIRE, linewidth=0.6, alpha=0.45)


def draw_cutting_plane(ax, m: float, d: float, extent: float, height: float) -> None:
    """
    Draw the plane z = m*y + d, clipped to the cone's bounding box.

    Points whose z falls outside [-H, H] are set to NaN so matplotlib simply
    does not draw them - that keeps a steep plane from shooting off-screen.
    """
    span = np.linspace(-extent, extent, 60)
    x_grid, y_grid = np.meshgrid(span, span)
    z_grid = m * y_grid + d
    z_grid = np.where(np.abs(z_grid) <= height, z_grid, np.nan)

    ax.plot_surface(x_grid, y_grid, z_grid, color=PLANE_FACE, alpha=0.30,
                    linewidth=0, antialiased=True, shade=False)


def draw_3d_panel(ax, geom: dict) -> None:
    """Assemble the left-hand panel: cone + plane + highlighted section."""
    height = geom["height"]
    extent = geom["radius"]

    draw_double_cone(ax, geom["t"], height)
    draw_cutting_plane(ax, geom["m"], geom["d"], extent * 1.05, height)

    for i, (x, y, z) in enumerate(geom["curves"]):
        ax.plot(x, y, z, color=CURVE_COLOR, linewidth=3.5, zorder=10,
                label="intersection curve" if i == 0 else None)

    ax.scatter([0], [0], [0], color="black", s=18)           # the apex
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-height, height)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-58)
    try:                                                     # matplotlib >= 3.3
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:                                   # pragma: no cover
        pass

    plane_eq = f"z = {geom['m']:.3g}\u00b7y + {geom['d']:.3g}"
    if abs(geom["m"]) < 1e-12:
        plane_eq = f"z = {geom['d']:.3g}   (horizontal)"
    ax.set_title(
        f"3D: plane slicing a double cone\n"
        f"cone half-angle {geom['half_angle_deg']:.1f}\u00b0   |   "
        f"plane {plane_eq}",
        fontsize=10, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8)


# ==========================================================================
# SECTION 3 - The 2D panel: the curve from its standard equation
# ==========================================================================
def style_2d_axes(ax, title: str, limit_x: float, limit_y: float) -> None:
    """Common cosmetics: axes through the origin, grid, equal aspect."""
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.grid(True, linestyle="--", alpha=0.40)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-limit_x, limit_x)
    ax.set_ylim(-limit_y, limit_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)


def mark_point(ax, x, y, label, color, offset=(8, 8), marker="o") -> None:
    """Plot a labelled key point (centre, vertex, focus, ...)."""
    ax.plot([x], [y], marker, color=color, markersize=8, zorder=6)
    ax.annotate(label, xy=(x, y), xytext=offset, textcoords="offset points",
                color=color, fontsize=9, fontweight="bold", zorder=6)


def annotate_length(ax, p0, p1, label, color, va="bottom") -> None:
    """Draw a double-headed arrow between two points and label it (the a / b legs)."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=2.0))
    ax.text((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0, f" {label}",
            color=color, fontsize=11, fontweight="bold", ha="center", va=va,
            zorder=7)


def draw_circle_2d(ax, r: float) -> None:
    """x^2 + y^2 = r^2, parametrised as (r cos t, r sin t)."""
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot(r * np.cos(theta), r * np.sin(theta), color=CURVE_COLOR,
            linewidth=2.5, label=f"radius r = {r:g}")

    annotate_length(ax, (0.0, 0.0), (r * math.cos(math.pi / 4), r * math.sin(math.pi / 4)),
                    f"r = {r:g}", GUIDE_COLOR)
    mark_point(ax, 0.0, 0.0, "centre / focus (0, 0)", FOCUS_COLOR, offset=(10, -16))

    lim = 1.35 * r
    style_2d_axes(ax, f"2D: $x^2 + y^2 = {r:g}^2$\n(eccentricity e = 0)", lim, lim)
    ax.legend(loc="upper right", fontsize=9)


def draw_ellipse_2d(ax, a: float, b: float) -> None:
    """
    x^2/a^2 + y^2/b^2 = 1, parametrised as (a cos t, b sin t).

    c^2 = |a^2 - b^2| and the foci sit on the longer axis, so the picture
    stays correct whether the user typed a > b or a < b.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot(a * np.cos(theta), b * np.sin(theta), color=CURVE_COLOR,
            linewidth=2.5, label="ellipse")

    c = math.sqrt(abs(a * a - b * b))
    horizontal = a >= b                       # is the major axis along x?
    foci = [(c, 0.0), (-c, 0.0)] if horizontal else [(0.0, c), (0.0, -c)]
    ecc = c / max(a, b)

    annotate_length(ax, (0.0, 0.0), (a, 0.0), f"a = {a:g}", GUIDE_COLOR)
    annotate_length(ax, (0.0, 0.0), (0.0, b), f"b = {b:g}", "teal")
    mark_point(ax, 0.0, 0.0, "centre (0, 0)", "black", offset=(-30, -18))
    # Nudge each focus label back towards the centre so it cannot run off
    # the edge of the axes when the focus is close to the frame.
    for fx, fy in foci:
        dx = -46 if fx > 0 else 6
        dy = 10 if horizontal else (10 if fy > 0 else -20)
        mark_point(ax, fx, fy, f"F({fx:.2f}, {fy:.2f})", FOCUS_COLOR, offset=(dx, dy))

    lim = 1.35 * max(a, b)
    style_2d_axes(
        ax,
        f"2D: $x^2/{a:g}^2 + y^2/{b:g}^2 = 1$\n"
        f"c = {c:.3f},  eccentricity e = {ecc:.3f}",
        lim, lim,
    )
    ax.legend(loc="upper right", fontsize=9)


def draw_parabola_2d(ax, a: float) -> None:
    """
    y = a*x^2, i.e. x^2 = (1/a)*y.  Comparing with x^2 = 4p*y gives the
    focal distance p = 1/(4a): focus at (0, p), directrix y = -p.
    """
    x_max = 3.0 / math.sqrt(abs(a))           # keeps |y| around 9 whatever a is
    x = np.linspace(-x_max, x_max, 601)       # odd => the vertex x = 0 is included
    y = a * x * x
    ax.plot(x, y, color=CURVE_COLOR, linewidth=2.5, label=f"y = {a:g}x\u00b2")

    p = 1.0 / (4.0 * a)                       # signed focal distance
    ax.axhline(-p, color=GUIDE_COLOR, linestyle="--", linewidth=1.5,
               label=f"directrix y = {-p:.3f}")
    mark_point(ax, 0.0, 0.0, "vertex (0, 0)", "black", offset=(12, -18))
    mark_point(ax, 0.0, p, f"focus (0, {p:.3f})", FOCUS_COLOR, offset=(12, 6))
    # The p arrow sits on the y-axis between vertex and focus, so push its
    # label to the left to keep clear of the focus label on the right.
    ax.annotate("", xy=(0.0, p), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="<->", color="teal", linewidth=2.0))
    ax.text(0.0, p / 2.0, "p ", color="teal", fontsize=11, fontweight="bold",
            ha="right", va="center", zorder=7)

    y_span = max(abs(a) * x_max * x_max, abs(p) * 2.0) * 1.15
    style_2d_axes(ax, f"2D: $y = {a:g}\\,x^2$\n(eccentricity e = 1)",
                  x_max * 1.15, y_span)
    ax.legend(loc="upper right", fontsize=9)


def draw_hyperbola_2d(ax, a: float, b: float) -> None:
    """
    x^2/a^2 - y^2/b^2 = 1, parametrised as (+-a cosh s, b sinh s).

    c^2 = a^2 + b^2 puts the foci at (+-c, 0); the asymptotes y = +-(b/a)x
    are the diagonals of the 'fundamental rectangle' spanned by a and b.
    """
    s = np.linspace(-2.0, 2.0, 501)           # odd => the vertices s = 0 are included
    for sign in (+1.0, -1.0):
        ax.plot(sign * a * np.cosh(s), b * np.sinh(s), color=CURVE_COLOR,
                linewidth=2.5, label="hyperbola" if sign > 0 else None)

    lim_x = a * math.cosh(2.0) * 1.15
    lim_y = b * math.sinh(2.0) * 1.15

    # Asymptotes and the rectangle that generates them.
    guide_x = np.array([-lim_x, lim_x])
    ax.plot(guide_x, (b / a) * guide_x, color=GUIDE_COLOR, linestyle="--",
            linewidth=1.4, label=f"asymptotes y = \u00b1{b / a:.3g}x")
    ax.plot(guide_x, -(b / a) * guide_x, color=GUIDE_COLOR, linestyle="--",
            linewidth=1.4)
    ax.add_patch(plt.Rectangle((-a, -b), 2 * a, 2 * b, fill=False,
                               edgecolor="teal", linestyle=":", linewidth=1.3))

    c = math.hypot(a, b)
    # Everything of interest sits on the x-axis, so the labels have to be
    # fanned out vertically (and away from each other) to stay readable.
    annotate_length(ax, (0.0, 0.0), (a, 0.0), f"a = {a:g}", GUIDE_COLOR)
    annotate_length(ax, (a, 0.0), (a, b), f"b = {b:g}", "teal")
    mark_point(ax, 0.0, 0.0, "centre", "black", offset=(-16, 12))
    for vx in (a, -a):
        mark_point(ax, vx, 0.0, f"vertex ({vx:g}, 0)", "black", offset=(-34, -24))
    for fx in (c, -c):
        mark_point(ax, fx, 0.0, f"F({fx:.2f}, 0)", FOCUS_COLOR, offset=(-24, 16))

    style_2d_axes(
        ax,
        f"2D: $x^2/{a:g}^2 - y^2/{b:g}^2 = 1$\n"
        f"c = {c:.3f},  eccentricity e = {c / a:.3f}",
        lim_x, lim_y,
    )
    ax.legend(loc="upper right", fontsize=9)


# ==========================================================================
# SECTION 4 - Figure assembly and console reporting
# ==========================================================================
def make_figure():
    """
    Side-by-side layout.  subplots() cannot mix projections directly, so the
    left axes is created normally, removed, and replaced by a 3D axes in the
    very same grid slot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.5))
    ax1.remove()
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    return fig, ax1, ax2


def show_figure(fig, geom: dict, show: bool, save_path: str | None = None) -> None:
    """Add the shared title/footnote, then save and/or display the figure."""
    fig.suptitle(f"Conic section: {geom['kind'].upper()}", fontsize=15, fontweight="bold")
    if geom["note"]:
        fig.text(0.5, 0.015, geom["note"], ha="center", va="bottom",
                 fontsize=9, style="italic", color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"  Figure written to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_report(geom: dict, feature_lines: list[str]) -> None:
    """Echo the derived cone/plane data plus the conic's own key features."""
    print("\n" + "=" * 72)
    print(f"  {geom['kind'].upper()}")
    print("=" * 72)
    print(f"  Cone            : x\u00b2 + y\u00b2 = t\u00b2z\u00b2  with t = tan(alpha) = {geom['t']:.4f}")
    print(f"  Cone half-angle : {geom['half_angle_deg']:.2f}\u00b0")
    print(f"  Cutting plane   : z = {geom['m']:.4f}\u00b7y + {geom['d']:.4f}")
    print(f"  Plane tilt      : {geom['plane_angle_deg']:.2f}\u00b0 from horizontal")
    a_param = 1.0 - geom["t"] ** 2 * geom["m"] ** 2
    print(f"  Classifier A    : 1 - t\u00b2m\u00b2 = {a_param:+.4f}   "
          f"({'A > 0' if a_param > 1e-9 else ('A = 0' if abs(a_param) <= 1e-9 else 'A < 0')})")
    print("  " + "-" * 68)
    for line in feature_lines:
        print(f"  {line}")
    print(f"  Curve lies on the cone (max relative residual): {cone_residual(geom):.2e}")
    print("=" * 72)


# --------------------------------------------------------------------------
# One public function per conic type
# --------------------------------------------------------------------------
def plot_circle(r: float, show: bool = True, save_path: str | None = None) -> dict:
    """Build and display the circle: 3D horizontal slice + 2D circle."""
    geom = circle_geometry(r)
    print_report(geom, [
        f"Equation        : x\u00b2 + y\u00b2 = {r:g}\u00b2",
        f"Centre          : (0, 0)      Radius: {r:g}",
        "Foci            : both at the centre (0, 0)",
        "Eccentricity    : 0",
    ])

    fig, ax1, ax2 = make_figure()
    draw_3d_panel(ax1, geom)
    draw_circle_2d(ax2, r)
    show_figure(fig, geom, show, save_path)
    return geom


def plot_ellipse(a: float, b: float, show: bool = True,
                 save_path: str | None = None) -> dict:
    """Build and display the ellipse: 3D gentle slice + 2D ellipse."""
    geom = ellipse_geometry(a, b)
    c = math.sqrt(abs(a * a - b * b))
    axis = "x-axis" if a >= b else "y-axis"
    print_report(geom, [
        f"Equation        : x\u00b2/{a:g}\u00b2 + y\u00b2/{b:g}\u00b2 = 1",
        f"Semi-axes       : a = {a:g}, b = {b:g}   (major axis along the {axis})",
        f"Centre          : (0, 0)      c = sqrt|a\u00b2-b\u00b2| = {c:.4f}",
        f"Foci            : ({c:.4f}, 0) and ({-c:.4f}, 0)" if a >= b
        else f"Foci            : (0, {c:.4f}) and (0, {-c:.4f})",
        f"Eccentricity    : {c / max(a, b):.4f}",
    ])

    fig, ax1, ax2 = make_figure()
    draw_3d_panel(ax1, geom)
    draw_ellipse_2d(ax2, a, b)
    show_figure(fig, geom, show, save_path)
    return geom


def plot_parabola(a: float, show: bool = True, save_path: str | None = None) -> dict:
    """Build and display the parabola: 3D slant-parallel slice + 2D parabola."""
    geom = parabola_geometry(a)
    p = 1.0 / (4.0 * a)
    print_report(geom, [
        f"Equation        : y = {a:g}x\u00b2     (x\u00b2 = 4p\u00b7y with p = {p:.4f})",
        "Vertex          : (0, 0)",
        f"Focus           : (0, {p:.4f})",
        f"Directrix       : y = {-p:.4f}",
        f"Opens           : {'upward' if a > 0 else 'downward'}",
        "Eccentricity    : 1",
    ])

    fig, ax1, ax2 = make_figure()
    draw_3d_panel(ax1, geom)
    draw_parabola_2d(ax2, a)
    show_figure(fig, geom, show, save_path)
    return geom


def plot_hyperbola(a: float, b: float, show: bool = True,
                   save_path: str | None = None) -> dict:
    """Build and display the hyperbola: 3D steep slice + 2D hyperbola."""
    geom = hyperbola_geometry(a, b)
    c = math.hypot(a, b)
    print_report(geom, [
        f"Equation        : x\u00b2/{a:g}\u00b2 - y\u00b2/{b:g}\u00b2 = 1",
        f"Semi-axes       : transverse a = {a:g}, conjugate b = {b:g}",
        f"Centre          : (0, 0)      Vertices: (\u00b1{a:g}, 0)",
        f"Foci            : (\u00b1{c:.4f}, 0)",
        f"Asymptotes      : y = \u00b1({b:g}/{a:g})x = \u00b1{b / a:.4f}x",
        f"Eccentricity    : {c / a:.4f}",
    ])

    fig, ax1, ax2 = make_figure()
    draw_3d_panel(ax1, geom)
    draw_hyperbola_2d(ax2, a, b)
    show_figure(fig, geom, show, save_path)
    return geom


# ==========================================================================
# SECTION 5 - Input handling
# ==========================================================================
CONIC_CHOICES = {
    "1": "circle", "c": "circle", "circle": "circle",
    "2": "ellipse", "e": "ellipse", "ellipse": "ellipse",
    "3": "parabola", "p": "parabola", "parabola": "parabola",
    "4": "hyperbola", "h": "hyperbola", "hyperbola": "hyperbola",
}


def _read(prompt: str) -> str:
    """input() that turns 'quit' into a QuitRequested exception."""
    try:
        raw = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise QuitRequested from None
    if raw.lower() in ("q", "quit", "exit"):
        raise QuitRequested
    return raw


def ask_conic_type() -> str:
    """Prompt until a recognised conic name/number is entered."""
    print("\nWhich conic section?")
    print("  1) circle      2) ellipse      3) parabola      4) hyperbola")
    while True:
        choice = _read("Choice [1-4]: ").lower()
        if choice in CONIC_CHOICES:
            return CONIC_CHOICES[choice]
        print("  Please type 1-4, or a name like 'ellipse'.")


def ask_positive(label: str, default: float) -> float:
    """Prompt until a strictly positive number is entered (blank = default)."""
    while True:
        raw = _read(f"{label} [default {default:g}]: ")
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number, e.g. 3 or 2.5")
            continue
        if value <= 0:
            print("  The value must be positive.")
            continue
        return value


def ask_nonzero(label: str, default: float) -> float:
    """Prompt until a non-zero number is entered (blank = default)."""
    while True:
        raw = _read(f"{label} [default {default:g}]: ")
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number, e.g. 0.5 or -2")
            continue
        if abs(value) < 1e-12:
            print("  The value must not be zero.")
            continue
        return value


# ==========================================================================
# SECTION 6 - Demo / self-check and entry point
# ==========================================================================
def run_demo() -> int:
    """
    Derive all four sections over a spread of parameters and confirm that
    every generated point really lies on the cone.  No windows are opened,
    so this also works on a headless machine.
    """
    cases = [
        ("circle", circle_geometry, (3.0,)),
        ("circle", circle_geometry, (0.5,)),
        ("ellipse", ellipse_geometry, (4.0, 2.0)),
        ("ellipse", ellipse_geometry, (2.0, 2.0)),      # degenerates to a circle
        ("ellipse", ellipse_geometry, (1.0, 5.0)),      # a < b: axes get swapped
        ("parabola", parabola_geometry, (1.0,)),
        ("parabola", parabola_geometry, (0.25,)),
        ("parabola", parabola_geometry, (-2.0,)),       # opens downward
        ("hyperbola", hyperbola_geometry, (3.0, 1.0)),
        ("hyperbola", hyperbola_geometry, (2.0, 2.0)),  # forces a wider cone
        ("hyperbola", hyperbola_geometry, (1.0, 4.0)),  # forces a much wider cone
    ]

    header = f"{'conic':<11}{'params':<14}{'alpha':>8}{'slope m':>10}{'d':>10}{'A':>10}{'residual':>12}"
    print(header)
    print("-" * len(header))

    failures = 0
    for name, builder, args in cases:
        geom = builder(*args)
        residual = cone_residual(geom)
        A = 1.0 - geom["t"] ** 2 * geom["m"] ** 2
        params = ", ".join(f"{v:g}" for v in args)
        print(f"{name:<11}{params:<14}{geom['half_angle_deg']:>7.2f}\u00b0"
              f"{geom['m']:>10.4f}{geom['d']:>10.4f}{A:>+10.4f}{residual:>12.2e}")
        if residual > CONE_TOL:
            failures += 1
            print(f"    !! points drift off the cone (residual {residual:.3e})")

    print("-" * len(header))
    print("Every section lies exactly on its cone."
          if failures == 0 else f"{failures} case(s) failed.")
    return 0 if failures == 0 else 1


def main() -> int:
    args = set(sys.argv[1:])
    if "--demo" in args:
        return run_demo()
    show_plot = "--no-plot" not in args
    # --save also writes the figure to conic_<type>.png next to the script.
    saving = "--save" in args

    print("=" * 72)
    print("  CONIC SECTIONS: 3D CONE SLICE  +  2D CURVE")
    print("=" * 72)
    print("Choose a conic and give its parameters. You get one figure with the")
    print("cone-and-plane construction on the left and the standard curve on")
    print("the right.  Type 'quit' at any prompt (or press Ctrl-C) to stop.")

    while True:
        try:
            kind = ask_conic_type()

            out = f"conic_{kind}.png" if saving else None

            if kind == "circle":
                r = ask_positive("Radius r", 3.0)
                plot_circle(r, show=show_plot, save_path=out)

            elif kind == "ellipse":
                a = ask_positive("Semi-axis a (along x)", 4.0)
                b = ask_positive("Semi-axis b (along y)", 2.0)
                plot_ellipse(a, b, show=show_plot, save_path=out)

            elif kind == "parabola":
                a = ask_nonzero("Coefficient a in y = a\u00b7x\u00b2", 0.5)
                plot_parabola(a, show=show_plot, save_path=out)

            else:  # hyperbola
                a = ask_positive("Transverse semi-axis a (along x)", 3.0)
                b = ask_positive("Conjugate semi-axis b (along y)", 2.0)
                plot_hyperbola(a, b, show=show_plot, save_path=out)

            again = _read("\nAnother conic? (Y/n): ").lower()
            if again.startswith("n"):
                raise QuitRequested

        except QuitRequested:
            print("\nGoodbye!")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())