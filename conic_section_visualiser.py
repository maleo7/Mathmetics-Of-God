#!/usr/bin/env python3
"""
Conic Sections Visualiser  (live 3D cone slicer  +  2D standard-form companion)
-------------------------------------------------------------------------------
A single persistent window holds a wireframe double-napped cone, a translucent
cutting plane that can be driven continuously through it, and the intersection
curve drawn as a glowing outline that stays glued to the cone's surface.

  CENTRE (3D) : the double cone, the cutting plane, and the live section.
  RIGHT  (2D) : the same curve drawn from its standard equation, with the
                centre/vertices, foci, the a and b lengths, the asymptotes
                (hyperbola) and the directrix (parabola) marked.
  CARDS       : a small schematic card of the current shape and a rounded
                equation card, both tinted with the conic's accent colour.

Everything updates in place: pick a conic with the radio buttons, drag the
parameter sliders or the cone half-angle, or press Play to sweep the plane
from horizontal, through increasing tilt, across the exact parabola angle and
on into hyperbola territory.  The classification label, the equation card and
the accent colour follow the geometry frame by frame.

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

MOVING THROUGH THE PARABOLA
===========================
The ellipse and hyperbola closed forms both carry A in a denominator, so both
blow up as A -> 0.  The sweep is therefore driven by the plane's tilt ANGLE and
`section_from_plane` branches on the regime it lands in, instead of
interpolating one closed form straight through the singularity:

        A > +eps   ->  ellipse form
        |A| <= eps ->  m is snapped to exactly +-1/t and the exact parabola
                       form  y = (x^2 - t^2 d^2) / (2 t^2 m d)  is used
        A < -eps   ->  hyperbola form (both branches, one per nappe)

Snapping m (rather than merely swapping formula) keeps the plane that is drawn
and the curve that is drawn the *same* plane, so the residual check below stays
at machine precision right through the crossing.

Usage:
    python conic_section_visualiser.py            # live interactive scene
    python conic_section_visualiser.py --demo     # self-check, no windows
    python conic_section_visualiser.py --no-plot  # console report only
    python conic_section_visualiser.py --save     # render the sweep to
                                                  # conic_section_visualiser.mp4
                                                  # (.gif fallback without ffmpeg)

    Optional:  --type {circle,ellipse,parabola,hyperbola}   starting conic
               --alpha DEG                                  cone half-angle
               --frames N                                   sweep frame count

Interaction:
    radio buttons  : choose the conic type
    sliders        : that type's parameters, plus the cone half-angle
    Play / Pause   : run the continuous circle -> ellipse -> parabola ->
                     hyperbola sweep (loops)
    Auto-rotate    : slow idle spin of the camera
    Reset          : back to that type's default parameters
    mouse drag     : free camera rotation (matplotlib 3D default)

Dependencies:
    pip install numpy matplotlib
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

import matplotlib

# Pick a non-interactive backend BEFORE pyplot is imported for any of the
# headless modes, so the script runs unchanged over SSH / in CI.
if {"--demo", "--no-plot", "--save"} & set(sys.argv[1:]):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
# Default cone half-angle.  45 degrees (t = tan 45 = 1) gives the classic
# "ice-cream cone" picture where the slant lines have slope 1.
DEFAULT_HALF_ANGLE_DEG = 45.0
MIN_HALF_ANGLE_DEG = 15.0
MAX_HALF_ANGLE_DEG = 80.0

# Any point of the intersection curve must satisfy x^2 + y^2 - t^2 z^2 = 0.
CONE_TOL = 1e-9

# Half-width of the "this counts as a parabola" band in A = 1 - t^2 m^2.
PARABOLA_EPS = 1e-3
# Below this the plane passes through the apex and the section degenerates.
DEGENERATE_D = 1e-12

# Fixed drawing box used while sweeping, so the camera never jumps.
SWEEP_HEIGHT = 3.2
SWEEP_D_FRACTION = 0.55          # plane offset d, as a fraction of the box
SWEEP_MAX_EXTRA_TILT_DEG = 24.0  # how far past the parabola angle to lean
SWEEP_MAX_TILT_DEG = 84.0        # ...but never past this (m would explode)
SWEEP_SECONDS = 9.0              # nominal wall-clock length of one loop

CURVE_SAMPLES = 481              # ODD, so vertices at the parameter origin land on the curve
MAX_SEGMENTS = 6                 # pre-allocated glowing polyline slots
GLOW_LAYERS = ((9.0, 0.07), (6.0, 0.13), (3.4, 0.45), (1.7, 1.00))

# --- dark palette ---------------------------------------------------------
BG = "#080b12"
PANEL_BG = "#0d121c"
PANE = (0.04, 0.055, 0.085, 1.0)
FG = "#e8eefb"
MUTED = "#7f8da6"
GRID = "#243044"
CONE_WIRE = "#41597d"
CONE_FACE = "#121b29"
APEX = "#f2f6ff"
WIDGET_BG = "#161f2e"
WIDGET_HOVER = "#22304a"

# One entry per conic; drives the plane, the glow, the 2D curve, the schematic
# card, the equation-card border and the headline all at once.
CONIC_STYLE: dict[str, dict[str, str]] = {
    "circle": {"accent": "#22d3ee", "guide": "#3f7f92", "title": "CIRCLE",
               "blurb": "horizontal plane  \u2022  A = 1"},
    "ellipse": {"accent": "#2dd4bf", "guide": "#3f8f84", "title": "ELLIPSE",
                "blurb": "gentle tilt  \u2022  A > 0"},
    "parabola": {"accent": "#a78bfa", "guide": "#6f5da8", "title": "PARABOLA",
                 "blurb": "parallel to a slant line  \u2022  A = 0"},
    "hyperbola": {"accent": "#fb7185", "guide": "#a4545f", "title": "HYPERBOLA",
                  "blurb": "steeper than the slant  \u2022  A < 0"},
}
FOCUS_COLOR = "#ffd166"

# Slider layouts and defaults, one row per conic type.
CONIC_PARAMS: dict[str, list[dict]] = {
    "circle": [{"key": "r", "label": "radius r", "min": 0.35, "max": 6.0, "init": 3.0}],
    "ellipse": [{"key": "a", "label": "semi-axis a", "min": 0.35, "max": 6.0, "init": 4.0},
                {"key": "b", "label": "semi-axis b", "min": 0.35, "max": 6.0, "init": 2.0}],
    "parabola": [{"key": "a", "label": "coefficient a", "min": -2.0, "max": 2.0, "init": 0.5}],
    "hyperbola": [{"key": "a", "label": "transverse a", "min": 0.35, "max": 6.0, "init": 3.0},
                  {"key": "b", "label": "conjugate b", "min": 0.35, "max": 6.0, "init": 2.0}],
}
CONIC_ORDER = ("circle", "ellipse", "parabola", "hyperbola")


# ==========================================================================
# SECTION 1 - Geometry
# ==========================================================================
# 1a. Parameters -> cutting plane (carried over unchanged from the static tool)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# 1b. The inverse direction: cutting plane -> section  (new, drives the sweep)
# --------------------------------------------------------------------------
def _pack_fixed(kind: str, t: float, m: float, d: float,
                curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                height: float, note: str = "") -> dict:
    """
    Same dict shape as :func:`_pack`, but with the drawing box supplied by the
    caller instead of being fitted around the curve.

    While sweeping, the box has to stay put or the camera would lurch every
    frame; and near A = 0 the section grows without bound, so a fitted box
    would zoom out to nothing.  The curve is clipped to this box instead.
    """
    return {
        "kind": kind,
        "t": t,
        "half_angle_deg": math.degrees(math.atan(t)),
        "m": m,
        "d": d,
        "plane_angle_deg": math.degrees(math.atan(abs(m))),
        "curves": curves,
        "height": height,
        "radius": t * height,
        "params": {},
        "note": note,
    }


def classify(t: float, m: float, eps: float = PARABOLA_EPS) -> str:
    """
    Name the conic produced by a plane of slope m on a cone of steepness t,
    straight from the sign of A = 1 - t^2 m^2 (see the module docstring).
    """
    if abs(m) < 1e-12:
        return "circle"
    A = 1.0 - t * t * m * m
    if A > eps:
        return "ellipse"
    if A < -eps:
        return "hyperbola"
    return "parabola"


def _split_segments(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    keep: np.ndarray, min_points: int = 2
                    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Cut a sampled curve into the contiguous runs that stay inside the box.

    Points are dropped rather than blanked with NaN, which keeps every stored
    array a genuine set of on-cone points - so :func:`cone_residual` can be
    reused verbatim - while still breaking the polyline where the curve
    leaves the drawing box.
    """
    if bool(keep.all()):
        return [(x, y, z)]
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [idx.size]))
    out = []
    for s, e in zip(starts, ends):
        run = idx[s:e]
        if run.size >= min_points:
            out.append((x[run], y[run], z[run]))
    return out


def _apex_section(t: float, m: float, height: float) -> tuple[str, list]:
    """
    The degenerate sections, i.e. what is left when d = 0 and the plane runs
    straight through the apex.  A conic "of zero size":

        A > 0  ->  a single point (the apex)
        A = 0  ->  one generator line (the plane contains the slant it is
                   parallel to)
        A < 0  ->  two generator lines, x = +-sqrt(B)*y, the degenerate
                   hyperbola whose branches have collapsed onto its asymptotes
    """
    A = 1.0 - t * t * m * m
    if A > PARABOLA_EPS:
        zero = np.zeros(2)
        return "circle" if abs(m) < 1e-12 else "ellipse", [(zero, zero.copy(), zero.copy())]

    if abs(A) <= PARABOLA_EPS:
        y_max = t * height
        y = np.linspace(-y_max, y_max, 2)
        return "parabola", [(np.zeros_like(y), y, m * y)]

    B = -A
    y_max = height / abs(m)
    y = np.linspace(-y_max, y_max, 2)
    return "hyperbola", [(math.sqrt(B) * y, y, m * y),
                         (-math.sqrt(B) * y, y, m * y)]


def section_from_plane(t: float, m: float, d: float, height: float,
                       samples: int = CURVE_SAMPLES) -> dict:
    """
    Intersect the cone x^2 + y^2 = t^2 z^2 with the plane z = m*y + d and
    return the section, clipped to the box |z| <= height.

    This is the inverse of the four ``*_geometry`` builders above: they go
    parameters -> plane, this one goes plane -> curve, which is what a
    continuously moving plane needs.

    The regime is chosen from A = 1 - t^2 m^2 and each regime uses its own
    closed form, so nothing is ever interpolated through the A -> 0
    singularity:

    * ``A > eps`` - ELLIPSE (or CIRCLE when m = 0)::

          x = (t*d/sqrt(A)) * cos(theta)
          y = y0 + (t*d/A)  * sin(theta),        y0 = t^2*m*d/A

    * ``|A| <= eps`` - PARABOLA.  m is snapped to exactly +-1/t first, so the
      plane really is parallel to a generator, and then the y^2 term of the
      master equation is genuinely absent::

          x^2 - 2*t^2*m*d*y - t^2*d^2 = 0   =>   y = (x^2 - t^2 d^2)/(2 t^2 m d)

      Substituting back gives z = (x^2 + t^2 d^2) / (2 t^2 d) regardless of the
      sign of m, so |z| <= H bounds the sweep of x by
      x^2 <= t^2 |d| (2H - |d|).

    * ``A < -eps`` - HYPERBOLA, with B = -A > 0 and one branch per nappe::

          x = (t*d/sqrt(B)) * sinh(s)
          y = y0 +- (t*d/B) * cosh(s),           y0 = -t^2*m*d/B

      Here z = c0 +- (m*t*d/B)*cosh(s) is monotone in cosh(s), so the visible
      range of s follows from a single arccosh instead of guesswork.

    Because the curve lies on the cone, x^2 + y^2 = t^2 z^2, clipping in z
    also bounds x and y - one test is enough.
    """
    A = 1.0 - t * t * m * m

    # Inside the band, commit to the parabola: move the plane onto the exact
    # slant-parallel slope so plane and curve stay consistent with each other.
    if abs(A) <= PARABOLA_EPS:
        m = math.copysign(1.0 / t, m if m != 0.0 else 1.0)
        A = 0.0

    if abs(d) < DEGENERATE_D:
        kind, curves = _apex_section(t, m, height)
        return _pack_fixed(kind, t, m, 0.0, curves, height,
                           note="The plane passes through the apex: a degenerate section.")

    if A > 0.0:                                   # ----- circle / ellipse -----
        kind = "circle" if abs(m) < 1e-12 else "ellipse"
        sqrt_a = math.sqrt(A)
        y0 = t * t * m * d / A
        theta = np.linspace(0.0, 2.0 * np.pi, samples)
        x = (t * d / sqrt_a) * np.cos(theta)
        y = y0 + (t * d / A) * np.sin(theta)
        z = m * y + d

    elif A == 0.0:                                # -------- parabola ----------
        kind = "parabola"
        x2_max = t * t * abs(d) * (2.0 * height - abs(d))
        x_max = math.sqrt(x2_max) if x2_max > 0.0 else abs(t * d) * 1e-3
        # ODD sample count => x = 0, the vertex, is a real point of the curve.
        n = samples if samples % 2 else samples + 1
        x = np.linspace(-x_max, x_max, n)
        y = (x * x - t * t * d * d) / (2.0 * t * t * m * d)
        z = m * y + d

    else:                                         # ------- hyperbola ----------
        kind = "hyperbola"
        B = -A
        y0 = -t * t * m * d / B
        c0 = m * y0 + d
        k = m * t * d / B
        n = samples if samples % 2 else samples + 1

        curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for branch in (+1.0, -1.0):               # one branch on each nappe
            slope = branch * k
            if abs(slope) < 1e-15:
                continue
            # z = c0 + slope*cosh(s) is monotone in cosh(s): one bound suffices.
            limit = (height - c0) / slope if slope > 0.0 else (-height - c0) / slope
            cosh_max = min(max(limit, 1.0), math.cosh(6.0))
            s_max = math.acosh(cosh_max)
            s = np.linspace(-s_max, s_max, n)     # odd => the vertex s = 0 is sampled
            x = (t * d / math.sqrt(B)) * np.sinh(s)
            y = y0 + branch * (t * d / B) * np.cosh(s)
            z = m * y + d
            curves.extend(_split_segments(x, y, z, np.abs(z) <= height * (1.0 + 1e-9)))
        return _pack_fixed(kind, t, m, d, curves, height)

    curves = _split_segments(x, y, z, np.abs(z) <= height * (1.0 + 1e-9))
    return _pack_fixed(kind, t, m, d, curves, height)


def standard_form(t: float, m: float, d: float) -> dict:
    """
    Read the standard-form parameters of the section straight off the plane.

    This simply runs each ``*_geometry`` derivation backwards:

        circle    :  r = t|d|
        ellipse   :  b = t|d|/sqrt(A),   a = t|d|*sqrt(1+m^2)/A
        parabola  :  d = sqrt(1+t^2)/(2 t^2 a)   =>   a = sqrt(1+t^2)/(2 t^2 d)
        hyperbola :  a = t|d|*sqrt(1+m^2)/B,     b = t|d|/sqrt(B)

    (the sqrt(1+m^2) being the in-plane stretch discussed in the module
    docstring).  ``a`` is always the semi-axis measured along the plane's
    direction of steepest tilt, so the 2D companion plots it along its own
    x-axis.

    Returns the kind, the numeric parameters, a mathtext equation for the
    card, and the plain-text feature lines used by the console report.
    """
    kind = classify(t, m)
    ad = abs(d)

    if kind == "circle":
        r = t * ad
        return {"kind": kind, "r": r,
                "equation": rf"$x^{{2}} + y^{{2}} = {r:.2f}^{{2}}$",
                "plain": f"x\u00b2 + y\u00b2 = {r:.3g}\u00b2"}

    if kind == "ellipse":
        A = 1.0 - t * t * m * m
        b = t * ad / math.sqrt(A)                       # across the tilt
        a = t * ad * math.sqrt(1.0 + m * m) / A         # along the tilt
        return {"kind": kind, "a": a, "b": b,
                "equation": rf"$\frac{{x^{{2}}}}{{{a:.2f}^{{2}}}} + "
                            rf"\frac{{y^{{2}}}}{{{b:.2f}^{{2}}}} = 1$",
                "plain": f"x\u00b2/{a:.3g}\u00b2 + y\u00b2/{b:.3g}\u00b2 = 1"}

    if kind == "parabola":
        a = math.sqrt(1.0 + t * t) / (2.0 * t * t * d)  # signed: d < 0 opens downward
        return {"kind": kind, "a": a,
                "equation": rf"$y = {a:.3f}\,x^{{2}}$",
                "plain": f"y = {a:.3g}x\u00b2"}

    B = t * t * m * m - 1.0
    a = t * ad * math.sqrt(1.0 + m * m) / B
    b = t * ad / math.sqrt(B)
    return {"kind": kind, "a": a, "b": b,
            "equation": rf"$\frac{{x^{{2}}}}{{{a:.2f}^{{2}}}} - "
                        rf"\frac{{y^{{2}}}}{{{b:.2f}^{{2}}}} = 1$",
            "plain": f"x\u00b2/{a:.3g}\u00b2 - y\u00b2/{b:.3g}\u00b2 = 1"}


# --------------------------------------------------------------------------
# 1c. The sweep: one continuous journey through all four conics
# --------------------------------------------------------------------------
def _smoothstep(s: float) -> float:
    """Ease-in/ease-out on [0, 1]; keeps the plane from starting or stopping abruptly."""
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def sweep_plane(u: float, t: float, height: float = SWEEP_HEIGHT) -> tuple[float, float]:
    """
    Map sweep progress ``u`` in [0, 1] to a cutting plane ``(m, d)``.

    A pure function, so the Play button, the ``--save`` export and the
    ``--demo`` self-check all follow byte-for-byte the same trajectory.

    The journey mirrors the classic demonstration:

        u < 0.22   slide DOWN, horizontal (m = 0), d: +d0 -> -d0.  The circle
                   shrinks smoothly to a single point at the apex and reopens
                   on the lower nappe.
        u < 0.34   slide back UP to +d0: the circle grows again.
        u < 0.62   TILT from horizontal to the parabola angle.  A falls from 1
                   towards 0 and the circle stretches into ever longer ellipses.
        u < 0.70   HOLD at the parabola: m is set to exactly 1/t, so t|m| = 1
                   and A = 0.  The section opens for the first time.
        u <= 1.00  keep LEANING past the slant into hyperbola territory, where
                   the plane reaches both nappes and two branches appear.

    Because the tilt is driven by the ANGLE phi (m = tan phi) and the parabola
    is an explicitly held phase, the sweep lands exactly on t|m| = 1 rather
    than skipping over it between frames.
    """
    u = float(min(max(u, 0.0), 1.0))
    d0 = SWEEP_D_FRACTION * height
    phi_par = math.atan2(1.0, t)        # the tilt at which t*tan(phi) = 1
    phi_max = min(math.radians(SWEEP_MAX_TILT_DEG),
                  phi_par + math.radians(SWEEP_MAX_EXTRA_TILT_DEG))

    if u < 0.22:                                     # descend through the apex
        return 0.0, d0 * (1.0 - 2.0 * _smoothstep(u / 0.22))
    if u < 0.34:                                     # come back up
        return 0.0, d0 * (-1.0 + 2.0 * _smoothstep((u - 0.22) / 0.12))
    if u < 0.62:                                     # tilt: circle -> ellipse
        return math.tan(_smoothstep((u - 0.34) / 0.28) * phi_par), d0
    if u < 0.70:                                     # hold exactly on the parabola
        return 1.0 / t, d0
    s = _smoothstep((u - 0.70) / 0.30)               # lean on into the hyperbola
    return math.tan(phi_par + s * (phi_max - phi_par)), d0


def sweep_section(u: float, t: float, height: float = SWEEP_HEIGHT) -> dict:
    """Convenience: the full section produced at sweep position ``u``."""
    m, d = sweep_plane(u, t, height)
    return section_from_plane(t, m, d, height)


def plane_quad(m: float, d: float, radius: float, height: float
               ) -> list[tuple[float, float, float]] | None:
    """
    The four corners of the visible part of z = m*y + d.

    The plane is clipped in y to whatever keeps |z| <= height, so a steep
    plane stays inside the box instead of shooting off-screen.  Returns None
    when the plane misses the box entirely.
    """
    if abs(m) < 1e-12:
        if abs(d) > height:
            return None
        y_lo, y_hi = -radius, radius
    else:
        y_a, y_b = (-height - d) / m, (height - d) / m
        y_lo, y_hi = max(min(y_a, y_b), -radius), min(max(y_a, y_b), radius)
        if y_hi <= y_lo:
            return None
    return [(-radius, y_lo, m * y_lo + d), (radius, y_lo, m * y_lo + d),
            (radius, y_hi, m * y_hi + d), (-radius, y_hi, m * y_hi + d)]


# ==========================================================================
# SECTION 2 - The 3D scene: cone, cutting plane, glowing intersection curve
# ==========================================================================
def style_3d_axes(ax) -> None:
    """Dark panes, faint grid, muted ticks - so the glowing curve carries the eye."""
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_pane_color(PANE)
        except Exception:                                    # pragma: no cover
            pass
        axis.line.set_color(GRID)
        try:                                                 # private but stable
            axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
        except Exception:                                    # pragma: no cover
            pass
    ax.tick_params(colors=MUTED, labelsize=7)
    for label in (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel):
        label("")


def build_double_cone(ax, t: float, height: float) -> list:
    """
    Draw both nappes of x^2 + y^2 = t^2 z^2 as one thin wireframe shell.

    Parametrisation: sweep z from -H to +H and theta around the axis; the
    ring at height z has radius t*|z| (which collapses to the apex at z = 0).

    Returns the artists so the caller can remove them when the half-angle
    changes - which is the only time the cone has to be rebuilt at all.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 121)
    z_line = np.linspace(-height, height, 121)
    theta_grid, z_grid = np.meshgrid(theta, z_line)
    r_grid = t * np.abs(z_grid)

    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)

    shell = ax.plot_surface(x, y, z_grid, color=CONE_FACE, alpha=0.35,
                            linewidth=0, antialiased=True, shade=False, zorder=1)
    # Rings + generator lines give the eye something to hold on to.
    wire = ax.plot_wireframe(x, y, z_grid, rstride=10, cstride=8,
                             color=CONE_WIRE, linewidth=0.55, alpha=0.75, zorder=2)
    apex = ax.scatter([0], [0], [0], color=APEX, s=22, depthshade=False, zorder=5)
    return [shell, wire, apex]


class Curve3D:
    """
    A glowing polyline with a fixed artist budget.

    matplotlib has no real glow, so each segment is drawn several times: wide
    and nearly transparent underneath, thin and opaque on top.  All the
    artists are created once and then only ever fed new data, which is what
    keeps the sweep smooth - no artist is created or destroyed per frame.
    """

    def __init__(self, ax, max_segments: int = MAX_SEGMENTS) -> None:
        self.lines: list[list] = []
        for _ in range(max_segments):
            layers = []
            for width, alpha in GLOW_LAYERS:
                (line,) = ax.plot(np.empty(0), np.empty(0), np.empty(0),
                                  linewidth=width, alpha=alpha, solid_capstyle="round",
                                  zorder=20)
                layers.append(line)
            self.lines.append(layers)

    def update(self, curves: list, colour: str) -> None:
        """Point the artists at the current segments; blank any spare slots."""
        empty = np.empty(0)
        for slot, layers in enumerate(self.lines):
            if slot < len(curves):
                x, y, z = curves[slot]
            else:
                x = y = z = empty
            for line in layers:
                line.set_data_3d(x, y, z)
                line.set_color(colour)

    @property
    def artists(self) -> list:
        return [line for layers in self.lines for line in layers]


def box_aspect_for(t: float) -> tuple[float, float, float]:
    """
    Keep the cone's opening angle honest on screen.

    The box spans 2*radius across and 2*height up, so a faithful aspect is
    (1, 1, height/radius) = (1, 1, 1/t).  Very sharp or very blunt cones would
    make that unusably tall or flat, so the vertical is clamped - a mild,
    deliberate distortion at the extremes of the half-angle slider only.
    """
    return (1.0, 1.0, float(min(max(1.0 / t, 0.62), 1.8)))


# ==========================================================================
# SECTION 3 - The 2D companion, the schematic card and the equation card
# ==========================================================================
MIN_2D_LIMIT = 0.05          # keeps set_xlim(-0, 0) from ever happening


def style_2d_axes(ax, title: str, limit_x: float, limit_y: float, accent: str) -> None:
    """Common cosmetics: axes through the origin, grid, equal aspect - dark themed."""
    ax.set_facecolor(PANEL_BG)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    ax.grid(True, linestyle="--", alpha=0.18, color=GRID)
    ax.set_xlim(-limit_x, limit_x)
    ax.set_ylim(-limit_y, limit_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8, color=accent)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def mark_point(ax, x, y, label, color, offset=(8, 8), marker="o") -> None:
    """Plot a labelled key point (centre, vertex, focus, ...)."""
    ax.plot([x], [y], marker, color=color, markersize=5, zorder=6)
    ax.annotate(label, xy=(x, y), xytext=offset, textcoords="offset points",
                color=color, fontsize=7, fontweight="bold", zorder=6)


def annotate_length(ax, p0, p1, label, color, va="bottom") -> None:
    """Draw a double-headed arrow between two points and label it (the a / b legs)."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.4))
    ax.text((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0, f" {label}",
            color=color, fontsize=8, fontweight="bold", ha="center", va=va,
            zorder=7)


def _fit(value: float, cap: float | None) -> tuple[float, bool]:
    """Clamp a view limit; report whether the curve still fits inside it."""
    value = max(value, MIN_2D_LIMIT)
    if cap is None or value <= cap:
        return value, True
    return cap, False


def draw_circle_2d(ax, r: float, accent: str, guide: str, cap: float | None = None) -> None:
    """x^2 + y^2 = r^2, parametrised as (r cos t, r sin t)."""
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot(r * np.cos(theta), r * np.sin(theta), color=accent, linewidth=2.0)

    lim, detailed = _fit(1.35 * r, cap)
    if detailed:
        annotate_length(ax, (0.0, 0.0),
                        (r * math.cos(math.pi / 4), r * math.sin(math.pi / 4)),
                        f"r = {r:.2f}", guide)
        mark_point(ax, 0.0, 0.0, "centre / focus", FOCUS_COLOR, offset=(8, -14))
    style_2d_axes(ax, f"$x^2 + y^2 = {r:.2f}^2$    (e = 0)", lim, lim, accent)


def draw_ellipse_2d(ax, a: float, b: float, accent: str, guide: str,
                    cap: float | None = None) -> None:
    """
    x^2/a^2 + y^2/b^2 = 1, parametrised as (a cos t, b sin t).

    c^2 = |a^2 - b^2| and the foci sit on the longer axis, so the picture
    stays correct whether a > b or a < b.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot(a * np.cos(theta), b * np.sin(theta), color=accent, linewidth=2.0)

    c = math.sqrt(abs(a * a - b * b))
    horizontal = a >= b                       # is the major axis along x?
    foci = [(c, 0.0), (-c, 0.0)] if horizontal else [(0.0, c), (0.0, -c)]
    ecc = c / max(a, b) if max(a, b) > 0 else 0.0

    lim, detailed = _fit(1.35 * max(a, b), cap)
    if detailed:
        # The foci also sit on the x-axis and label upwards, so the 'a' label
        # is hung BELOW the axis to keep the two clear of each other.
        annotate_length(ax, (0.0, 0.0), (a, 0.0), f"a = {a:.2f}", guide, va="top")
        annotate_length(ax, (0.0, 0.0), (0.0, b), f"b = {b:.2f}", guide)
        mark_point(ax, 0.0, 0.0, "centre", FG, offset=(-26, -14))
        # Nudge each focus label back towards the centre so it cannot run off
        # the edge of the axes when the focus is close to the frame.
        for fx, fy in foci:
            dx = -34 if fx > 0 else 6
            dy = 8 if horizontal else (8 if fy > 0 else -16)
            mark_point(ax, fx, fy, f"F({fx:.2f}, {fy:.2f})", FOCUS_COLOR, offset=(dx, dy))
    style_2d_axes(ax,
                  f"$x^2/{a:.2f}^2 + y^2/{b:.2f}^2 = 1$    (e = {ecc:.3f})",
                  lim, lim, accent)


def draw_parabola_2d(ax, a: float, accent: str, guide: str,
                     cap: float | None = None) -> None:
    """
    y = a*x^2, i.e. x^2 = (1/a)*y.  Comparing with x^2 = 4p*y gives the
    focal distance p = 1/(4a): focus at (0, p), directrix y = -p.
    """
    x_max = 3.0 / math.sqrt(abs(a))           # keeps |y| around 9 whatever a is
    x = np.linspace(-x_max, x_max, 601)       # odd => the vertex x = 0 is included
    ax.plot(x, a * x * x, color=accent, linewidth=2.0)

    p = 1.0 / (4.0 * a)                       # signed focal distance
    lim_x, detailed = _fit(x_max * 1.15, cap)
    lim_y, _ = _fit(max(abs(a) * x_max * x_max, abs(p) * 2.0) * 1.15, cap)

    if detailed:
        ax.axhline(-p, color=guide, linestyle="--", linewidth=1.2)
        ax.text(-lim_x * 0.96, -p, f" directrix y = {-p:.3f}", color=guide,
                fontsize=7, va="bottom")
        mark_point(ax, 0.0, 0.0, "vertex", FG, offset=(10, -14))
        mark_point(ax, 0.0, p, f"focus (0, {p:.3f})", FOCUS_COLOR, offset=(10, 4))
        # The p arrow sits on the y-axis between vertex and focus, so push its
        # label to the left to keep clear of the focus label on the right.
        ax.annotate("", xy=(0.0, p), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="<->", color=guide, linewidth=1.4))
        ax.text(0.0, p / 2.0, "p ", color=guide, fontsize=8, fontweight="bold",
                ha="right", va="center", zorder=7)
    style_2d_axes(ax, f"$y = {a:.3f}\\,x^2$    (e = 1)", lim_x, lim_y, accent)


def draw_hyperbola_2d(ax, a: float, b: float, accent: str, guide: str,
                      cap: float | None = None) -> None:
    """
    x^2/a^2 - y^2/b^2 = 1, parametrised as (+-a cosh s, b sinh s).

    c^2 = a^2 + b^2 puts the foci at (+-c, 0); the asymptotes y = +-(b/a)x
    are the diagonals of the 'fundamental rectangle' spanned by a and b.
    """
    s = np.linspace(-2.0, 2.0, 501)           # odd => the vertices s = 0 are included
    for sign in (+1.0, -1.0):
        ax.plot(sign * a * np.cosh(s), b * np.sinh(s), color=accent, linewidth=2.0)

    lim_x, detailed = _fit(a * math.cosh(2.0) * 1.15, cap)
    lim_y, _ = _fit(b * math.sinh(2.0) * 1.15, cap)

    # Asymptotes and the rectangle that generates them.
    guide_x = np.array([-lim_x, lim_x])
    ax.plot(guide_x, (b / a) * guide_x, color=guide, linestyle="--", linewidth=1.1)
    ax.plot(guide_x, -(b / a) * guide_x, color=guide, linestyle="--", linewidth=1.1)

    c = math.hypot(a, b)
    if detailed:
        ax.add_patch(plt.Rectangle((-a, -b), 2 * a, 2 * b, fill=False,
                                   edgecolor=guide, linestyle=":", linewidth=1.0))
        # Everything of interest sits on the x-axis, so the labels have to be
        # fanned out vertically (and away from each other) to stay readable.
        annotate_length(ax, (0.0, 0.0), (a, 0.0), f"a = {a:.2f}", guide)
        annotate_length(ax, (a, 0.0), (a, b), f"b = {b:.2f}", guide)
        mark_point(ax, 0.0, 0.0, "centre", FG, offset=(-14, 10))
        for vx in (a, -a):
            mark_point(ax, vx, 0.0, "vertex", FG, offset=(-16, -18))
        for fx in (c, -c):
            mark_point(ax, fx, 0.0, f"F({fx:.2f}, 0)", FOCUS_COLOR, offset=(-20, 12))
        ax.text(lim_x * 0.97, (b / a) * lim_x * 0.9, f"y = \u00b1{b / a:.3g}x",
                color=guide, fontsize=7, ha="right")
    style_2d_axes(ax,
                  f"$x^2/{a:.2f}^2 - y^2/{b:.2f}^2 = 1$    (e = {c / a:.3f})",
                  lim_x, lim_y, accent)


def draw_standard_curve(ax, std: dict, cap: float | None = None) -> None:
    """Redraw the 2D companion for whichever conic ``std`` describes."""
    style = CONIC_STYLE[std["kind"]]
    accent, guide = style["accent"], style["guide"]
    ax.clear()
    if std["kind"] == "circle":
        draw_circle_2d(ax, max(std["r"], 1e-6), accent, guide, cap)
    elif std["kind"] == "ellipse":
        draw_ellipse_2d(ax, std["a"], std["b"], accent, guide, cap)
    elif std["kind"] == "parabola":
        draw_parabola_2d(ax, std["a"], accent, guide, cap)
    else:
        draw_hyperbola_2d(ax, std["a"], std["b"], accent, guide, cap)


def draw_schematic(ax, kind: str) -> None:
    """
    The little inset card: the current conic reduced to its bare silhouette,
    with no axes, no numbers and no scale - just the shape, in the accent
    colour, so the eye can tell at a glance what the plane is producing.
    """
    accent = CONIC_STYLE[kind]["accent"]
    ax.clear()
    # The rounded plate behind this axes supplies the card's frame and fill,
    # so the axes itself stays completely transparent - otherwise the square
    # spines would cut across the rounded corners.
    ax.set_facecolor("none")
    ax.patch.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if kind == "circle":
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color=accent, linewidth=2.2)
    elif kind == "ellipse":
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(1.35 * np.cos(th), 0.8 * np.sin(th), color=accent, linewidth=2.2)
    elif kind == "parabola":
        x = np.linspace(-1.15, 1.15, 200)
        ax.plot(x, 0.95 * x * x - 0.85, color=accent, linewidth=2.2)
    else:
        s = np.linspace(-1.25, 1.25, 200)
        for sign in (+1.0, -1.0):
            ax.plot(sign * 0.55 * np.cosh(s), 0.62 * np.sinh(s),
                    color=accent, linewidth=2.2)
        g = np.array([-1.5, 1.5])
        ax.plot(g, 1.13 * g, color=accent, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.plot(g, -1.13 * g, color=accent, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(CONIC_STYLE[kind]["title"], color=accent, fontsize=9,
                 fontweight="bold", pad=4)


# ==========================================================================
# SECTION 4 - The live scene: widgets, in-place updates, animation
# ==========================================================================
class ConicScene:
    """
    One persistent figure holding the whole tool.

    Two ways of driving the same picture share every artist:

    * PARAMETER MODE - the radio buttons and sliders call the matching
      ``*_geometry`` builder, exactly as the static tool did, and the drawing
      box is fitted around the resulting curve.
    * SWEEP MODE - Play (or ``--save``) walks :func:`sweep_plane` and calls
      :func:`section_from_plane` instead, inside a box that is deliberately
      held still so the camera never lurches.

    Nothing is ever closed and reopened: the cone is rebuilt only when the
    half-angle changes, the plane is a single polygon whose vertices are
    replaced, and the intersection curve is a fixed pool of line artists that
    are only ever handed new data.
    """

    def __init__(self, kind: str = "circle",
                 half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG,
                 interactive: bool = True,
                 figsize: tuple[float, float] = (16.0, 9.0)) -> None:
        self.kind = kind
        self.half_angle_deg = half_angle_deg
        self.interactive = interactive
        self.values: dict[str, float] = {}

        self.playing = False
        self.rotating = False
        self.u = 0.0
        self.u_step = 1.0 / max(SWEEP_SECONDS * 25.0, 1.0)
        self.elev, self.azim = 16.0, -62.0

        self._cone_artists: list = []
        self._cone_key: tuple[float, float] | None = None
        self._anim: FuncAnimation | None = None

        self._build_figure(figsize)
        self._build_widgets()
        self.select_kind(kind, reset=True)

    # -- figure scaffolding ------------------------------------------------
    def _build_figure(self, figsize: tuple[float, float]) -> None:
        """Lay out the panels, cards and (optionally) the widget rail."""
        self.fig = plt.figure(figsize=figsize, facecolor=BG)
        self.fig.canvas.manager.set_window_title("Conic Sections \u2014 live cone slicer")

        left = 0.145 if self.interactive else 0.03
        width = 0.45 if self.interactive else 0.55
        self.ax3d = self.fig.add_axes((left, 0.16, width, 0.75), projection="3d")
        style_3d_axes(self.ax3d)
        self.ax3d.view_init(elev=self.elev, azim=self.azim)

        self.ax2d = self.fig.add_axes((0.645, 0.20, 0.30, 0.46))
        self.ax_inset = self.fig.add_axes((0.845, 0.735, 0.135, 0.165))

        # Rounded backing plate for the schematic card.  Figure-level patches
        # are drawn AFTER the axes whenever their zorder ties, so the plate has
        # to be pushed explicitly behind the inset or it would paint over it.
        self.card_patch = FancyBboxPatch(
            (0.832, 0.705), 0.161, 0.235, transform=self.fig.transFigure,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=PANEL_BG, edgecolor=CONIC_STYLE[self.kind]["accent"],
            linewidth=1.3, zorder=-1)
        self.fig.add_artist(self.card_patch)
        self.ax_inset.set_zorder(2)

        self.title_text = self.fig.text(
            0.5, 0.965, "", ha="center", va="top", fontsize=17, fontweight="bold")
        self.subtitle_text = self.fig.text(
            0.5, 0.928, "", ha="center", va="top", fontsize=9.5, color=MUTED)
        self.status_text = self.fig.text(
            0.012, 0.30, "", ha="left", va="top", fontsize=8, color=MUTED,
            family="monospace", linespacing=1.6)
        # The note sits in the gap between the 3D panel (bottom 0.16) and the
        # slider rail (top 0.12), and grows UPWARDS into that gap, so the
        # two-line hyperbola note still clears the sliders underneath it.
        self.note_text = self.fig.text(
            0.37, 0.126, "", ha="center", va="bottom", fontsize=8,
            style="italic", color=MUTED)

        # The rounded equation card.
        self.equation_text = self.fig.text(
            0.795, 0.093, "", ha="center", va="center", fontsize=17, color=FG,
            bbox=dict(boxstyle="round,pad=0.62,rounding_size=0.35",
                      facecolor=PANEL_BG, edgecolor=CONIC_STYLE[self.kind]["accent"],
                      linewidth=2.0))

        self.curve3d = Curve3D(self.ax3d)
        self.plane = Poly3DCollection([[(0, 0, 0), (0, 0, 0), (0, 0, 0)]],
                                      zorder=6)
        self.plane.set_facecolor(to_rgba(CONIC_STYLE[self.kind]["accent"], 0.22))
        self.plane.set_edgecolor(to_rgba(CONIC_STYLE[self.kind]["accent"], 0.85))
        self.plane.set_linewidth(1.3)
        self.ax3d.add_collection3d(self.plane)

    def _build_widgets(self) -> None:
        """Radio selector, action buttons and the (rebuildable) slider rail."""
        self.sliders: list[Slider] = []
        self.slider_axes: list = []
        if not self.interactive:
            self.radio = None
            return

        ax_radio = self.fig.add_axes((0.012, 0.60, 0.115, 0.26), facecolor=WIDGET_BG)
        ax_radio.set_title("conic type", color=FG, fontsize=9,
                           fontweight="bold", pad=6)
        labels = [k.capitalize() for k in CONIC_ORDER]
        active = CONIC_ORDER.index(self.kind)
        try:                                   # matplotlib >= 3.7 styling hooks
            self.radio = RadioButtons(
                ax_radio, labels, active=active,
                label_props={"color": [FG] * len(labels), "fontsize": [10] * len(labels)},
                radio_props={"s": [70] * len(labels),
                             "edgecolor": [MUTED] * len(labels)})
        except TypeError:                      # pragma: no cover - older matplotlib
            self.radio = RadioButtons(ax_radio, labels, active=active)
            for text in self.radio.labels:
                text.set_color(FG)
        for spine in ax_radio.spines.values():
            spine.set_color(GRID)
        self.radio.on_clicked(self._on_radio)

        self.btn_play = self._make_button((0.012, 0.525, 0.115, 0.052), "\u25b6  Play")
        self.btn_rotate = self._make_button((0.012, 0.462, 0.115, 0.052), "Auto-rotate")
        self.btn_reset = self._make_button((0.012, 0.399, 0.115, 0.052), "Reset")
        self.btn_play.on_clicked(self._on_play)
        self.btn_rotate.on_clicked(self._on_rotate)
        self.btn_reset.on_clicked(self._on_reset)

        # Two parameter slots (some conics use one) plus the half-angle.
        self.slider_axes = [
            self.fig.add_axes((0.22, 0.098, 0.30, 0.022), facecolor=WIDGET_BG),
            self.fig.add_axes((0.22, 0.062, 0.30, 0.022), facecolor=WIDGET_BG),
        ]
        self.ax_alpha = self.fig.add_axes((0.22, 0.020, 0.30, 0.022), facecolor=WIDGET_BG)
        self.slider_alpha = self._make_slider(
            self.ax_alpha, "cone half-angle \u03b1",
            MIN_HALF_ANGLE_DEG, MAX_HALF_ANGLE_DEG, self.half_angle_deg, "#9fb3d1")
        self.slider_alpha.on_changed(self._on_alpha)

    def _make_button(self, rect: tuple[float, float, float, float], label: str) -> Button:
        ax = self.fig.add_axes(rect)
        button = Button(ax, label, color=WIDGET_BG, hovercolor=WIDGET_HOVER)
        button.label.set_color(FG)
        button.label.set_fontsize(9.5)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        return button

    def _make_slider(self, ax, label: str, vmin: float, vmax: float,
                     vinit: float, accent: str) -> Slider:
        try:                                   # matplotlib >= 3.5 styling hooks
            slider = Slider(ax, label, vmin, vmax, valinit=vinit, color=accent,
                            track_color=WIDGET_BG,
                            handle_style={"facecolor": accent, "edgecolor": FG,
                                          "size": 9})
        except TypeError:                      # pragma: no cover - older matplotlib
            slider = Slider(ax, label, vmin, vmax, valinit=vinit, color=accent)
        slider.label.set_color(FG)
        slider.label.set_fontsize(9)
        slider.valtext.set_color(accent)
        slider.valtext.set_fontsize(9)
        return slider

    def _rebuild_param_sliders(self) -> None:
        """Swap the parameter sliders over to the newly selected conic."""
        if not self.interactive:
            return
        accent = CONIC_STYLE[self.kind]["accent"]
        specs = CONIC_PARAMS[self.kind]
        self.sliders = []
        for ax, spec in zip(self.slider_axes, specs):
            ax.clear()
            ax.set_visible(True)
            ax.set_facecolor(WIDGET_BG)
            slider = self._make_slider(ax, spec["label"], spec["min"], spec["max"],
                                       self.values[spec["key"]], accent)
            slider.on_changed(self._on_param)
            self.sliders.append(slider)
        for ax in self.slider_axes[len(specs):]:
            ax.clear()
            ax.set_visible(False)

    # -- geometry plumbing -------------------------------------------------
    def _geometry_from_values(self) -> dict:
        """Run the appropriate ``*_geometry`` builder on the current slider values."""
        alpha = self.half_angle_deg
        if self.kind == "circle":
            return circle_geometry(self.values["r"], alpha)
        if self.kind == "ellipse":
            return ellipse_geometry(self.values["a"], self.values["b"], alpha)
        if self.kind == "parabola":
            a = self.values["a"]
            # y = a x^2 is meaningless at a = 0 (d would be infinite): hold off it.
            if abs(a) < 0.02:
                a = math.copysign(0.02, a if a != 0.0 else 1.0)
            return parabola_geometry(a, alpha)
        return hyperbola_geometry(self.values["a"], self.values["b"], alpha)

    def _ensure_cone(self, t: float, height: float) -> None:
        """Rebuild the cone only when its shape or the box has actually changed."""
        key = (round(t, 9), round(height, 9))
        if key == self._cone_key:
            return
        for artist in self._cone_artists:
            artist.remove()
        self._cone_artists = build_double_cone(self.ax3d, t, height)
        self._cone_key = key

    def render(self, geom: dict, cap_2d: float | None = None) -> None:
        """
        Push one section into every artist: cone, plane, glow, 2D panel, cards.

        Works identically for a parameter-built section and a swept one, which
        is why Play, the sliders and the exporter all funnel through here.
        """
        kind = geom["kind"]
        style = CONIC_STYLE[kind]
        accent = style["accent"]
        t, m, d = geom["t"], geom["m"], geom["d"]
        height, radius = geom["height"], geom["radius"]

        self._ensure_cone(t, height)

        quad = plane_quad(m, d, radius * 1.06, height)
        self.plane.set_verts([quad] if quad else [[(0, 0, 0), (0, 0, 0), (0, 0, 0)]])
        self.plane.set_facecolor(to_rgba(accent, 0.20))
        self.plane.set_edgecolor(to_rgba(accent, 0.85))

        self.curve3d.update(geom["curves"], accent)

        self.ax3d.set_xlim(-radius, radius)
        self.ax3d.set_ylim(-radius, radius)
        self.ax3d.set_zlim(-height, height)
        try:                                                 # matplotlib >= 3.3
            self.ax3d.set_box_aspect(box_aspect_for(t))
        except AttributeError:                               # pragma: no cover
            pass

        std = standard_form(t, m, d)
        draw_standard_curve(self.ax2d, std, cap_2d)
        draw_schematic(self.ax_inset, kind)

        self.equation_text.set_text(std["equation"])
        self.equation_text.get_bbox_patch().set_edgecolor(accent)
        self.card_patch.set_edgecolor(accent)

        self.title_text.set_text(style["title"])
        self.title_text.set_color(accent)
        self.subtitle_text.set_text(style["blurb"])

        A = 1.0 - t * t * m * m
        plane_eq = (f"z = {d:+.3f}   (horizontal)" if abs(m) < 1e-12
                    else f"z = {m:+.3f}\u00b7y {d:+.3f}")
        self.status_text.set_text(
            f"cone   x\u00b2+y\u00b2 = t\u00b2z\u00b2\n"
            f"t      = {t:.4f}\n"
            f"alpha  = {geom['half_angle_deg']:.2f}\u00b0\n"
            f"plane  {plane_eq}\n"
            f"tilt   = {geom['plane_angle_deg']:.2f}\u00b0\n"
            f"t|m|   = {t * abs(m):.4f}\n"
            f"A      = {A:+.4f}\n"
            f"resid  = {cone_residual(geom):.1e}")
        self.note_text.set_text(geom.get("note", ""))

    def render_parameters(self) -> None:
        """Draw whatever the sliders currently say (parameter mode)."""
        geom = self._geometry_from_values()
        # hyperbola_geometry may open the cone wider than requested; mirror that
        # back into the slider so the display never contradicts the picture.
        if self.interactive and abs(geom["half_angle_deg"] - self.half_angle_deg) > 1e-6:
            self.half_angle_deg = geom["half_angle_deg"]
            self.slider_alpha.eventson = False
            self.slider_alpha.set_val(min(geom["half_angle_deg"], MAX_HALF_ANGLE_DEG))
            self.slider_alpha.eventson = True
        self.render(geom)

    def render_sweep(self, u: float) -> None:
        """Draw the sweep at position ``u`` (sweep mode)."""
        t = math.tan(math.radians(self.half_angle_deg))
        m, d = sweep_plane(u, t, SWEEP_HEIGHT)
        geom = section_from_plane(t, m, d, SWEEP_HEIGHT)
        geom["note"] = ("sweeping: the plane tilts continuously, and the section "
                        "changes name the instant A changes sign.")
        self.render(geom, cap_2d=2.6 * SWEEP_HEIGHT)

    # -- widget callbacks --------------------------------------------------
    def select_kind(self, kind: str, reset: bool = False) -> None:
        """Switch conic type, reloading that type's sliders and defaults."""
        self.kind = kind
        # Keep the radio in step when the type is changed in code (--type, or
        # Reset), with events off so this does not loop straight back in here.
        if self.radio is not None and self.radio.value_selected.lower() != kind:
            self.radio.eventson = False
            self.radio.set_active(CONIC_ORDER.index(kind))
            self.radio.eventson = True

        if reset or not self.values:
            self.values = {s["key"]: s["init"] for s in CONIC_PARAMS[kind]}
        else:
            for spec in CONIC_PARAMS[kind]:
                self.values.setdefault(spec["key"], spec["init"])
            self.values = {s["key"]: self.values.get(s["key"], s["init"])
                           for s in CONIC_PARAMS[kind]}
        self._rebuild_param_sliders()
        self._stop_playing()
        self.render_parameters()
        self._refresh()

    def _on_radio(self, label: str) -> None:
        self.select_kind(label.lower(), reset=True)

    def _on_param(self, _value: float) -> None:
        specs = CONIC_PARAMS[self.kind]
        for spec, slider in zip(specs, self.sliders):
            self.values[spec["key"]] = float(slider.val)
        self._stop_playing()
        self.render_parameters()
        self._refresh()

    def _on_alpha(self, value: float) -> None:
        self.half_angle_deg = float(value)
        if self.playing:
            self.render_sweep(self.u)
        else:
            self.render_parameters()
        self._refresh()

    def _on_play(self, _event) -> None:
        self.playing = not self.playing
        self.btn_play.label.set_text("\u23f8  Pause" if self.playing else "\u25b6  Play")
        if self.playing:
            self.render_sweep(self.u)
        else:
            self.render_parameters()
        self._sync_timer()
        self._refresh()

    def _on_rotate(self, _event) -> None:
        self.rotating = not self.rotating
        self.btn_rotate.label.set_text("Stop rotation" if self.rotating else "Auto-rotate")
        self._sync_timer()
        self._refresh()

    def _on_reset(self, _event) -> None:
        self.half_angle_deg = DEFAULT_HALF_ANGLE_DEG
        self.slider_alpha.eventson = False
        self.slider_alpha.set_val(DEFAULT_HALF_ANGLE_DEG)
        self.slider_alpha.eventson = True
        self.elev, self.azim = 16.0, -62.0
        self.ax3d.view_init(elev=self.elev, azim=self.azim)
        self.u = 0.0
        self.select_kind(self.kind, reset=True)

    def _stop_playing(self) -> None:
        if self.playing:
            self.playing = False
            if self.interactive:
                self.btn_play.label.set_text("\u25b6  Play")
            self._sync_timer()

    # -- animation ---------------------------------------------------------
    def _tick(self, _frame: int) -> list:
        """
        The single shared heartbeat.

        Play and auto-rotate deliberately share one timer: two independent
        ``FuncAnimation`` objects on one figure end up fighting over the draw
        and the motion stutters.
        """
        if self.playing:
            self.u += self.u_step
            if self.u > 1.0:
                self.u -= 1.0                     # the sweep loops
            self.render_sweep(self.u)
        if self.rotating:
            self.azim = (self.azim + 0.45) % 360.0
            self.ax3d.view_init(elev=self.elev, azim=self.azim)
        return self.curve3d.artists

    def _sync_timer(self) -> None:
        """Only run the timer when something is actually moving."""
        if self._anim is None:
            return
        if self.playing or self.rotating:
            self._anim.resume()
        else:
            self._anim.pause()

    def _refresh(self) -> None:
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Open the window and hand control to matplotlib's event loop."""
        self._anim = FuncAnimation(self.fig, self._tick, interval=40,
                                   blit=False, cache_frame_data=False,
                                   save_count=1)
        self._anim.pause()                       # idle until Play / Auto-rotate
        plt.show()


# --------------------------------------------------------------------------
# Sweep export
# --------------------------------------------------------------------------
def save_sweep(path_stem: str = "conic_section_visualiser",
               half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG,
               frames: int = 220) -> str:
    """
    Render the whole sweep to a movie.

    Tries ``FFMpegWriter`` for an .mp4 and falls back to ``PillowWriter`` for a
    .gif when ffmpeg is not installed.  Availability is checked up front
    rather than by catching an error part-way through, which would otherwise
    leave a truncated file behind.
    """
    scene = ConicScene(kind="circle", half_angle_deg=half_angle_deg,
                       interactive=False, figsize=(12.8, 7.2))
    values = np.linspace(0.0, 1.0, frames)

    def animate(u: float):
        scene.render_sweep(float(u))
        return scene.curve3d.artists

    anim = FuncAnimation(scene.fig, animate, frames=values,
                         blit=False, cache_frame_data=False)

    if FFMpegWriter.isAvailable():
        path, writer, dpi = f"{path_stem}.mp4", FFMpegWriter(fps=30, bitrate=3600), 100
    else:
        print("  ffmpeg not found - falling back to an animated GIF.")
        print("  (install ffmpeg for a smaller, smoother .mp4)")
        path, writer, dpi = f"{path_stem}.gif", PillowWriter(fps=20), 68

    print(f"  Rendering {frames} frames to {path} ...")
    anim.save(path, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": BG},
              progress_callback=lambda i, n: (
                  print(f"    frame {i + 1}/{n}", end="\r", flush=True)))
    print(f"\n  Animation written to {path}")
    plt.close(scene.fig)
    return path


# ==========================================================================
# SECTION 5 - Console reporting, demo / self-check
# ==========================================================================
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


def feature_lines(kind: str, params: dict) -> list[str]:
    """The standard-form facts for one conic, as printed by the console report."""
    if kind == "circle":
        r = params["r"]
        return [f"Equation        : x\u00b2 + y\u00b2 = {r:g}\u00b2",
                f"Centre          : (0, 0)      Radius: {r:g}",
                "Foci            : both at the centre (0, 0)",
                "Eccentricity    : 0"]

    if kind == "ellipse":
        a, b = params["a"], params["b"]
        c = math.sqrt(abs(a * a - b * b))
        axis = "x-axis" if a >= b else "y-axis"
        foci = (f"Foci            : ({c:.4f}, 0) and ({-c:.4f}, 0)" if a >= b
                else f"Foci            : (0, {c:.4f}) and (0, {-c:.4f})")
        return [f"Equation        : x\u00b2/{a:g}\u00b2 + y\u00b2/{b:g}\u00b2 = 1",
                f"Semi-axes       : a = {a:g}, b = {b:g}   (major axis along the {axis})",
                f"Centre          : (0, 0)      c = sqrt|a\u00b2-b\u00b2| = {c:.4f}",
                foci,
                f"Eccentricity    : {c / max(a, b):.4f}"]

    if kind == "parabola":
        a = params["a"]
        p = 1.0 / (4.0 * a)
        return [f"Equation        : y = {a:g}x\u00b2     (x\u00b2 = 4p\u00b7y with p = {p:.4f})",
                "Vertex          : (0, 0)",
                f"Focus           : (0, {p:.4f})",
                f"Directrix       : y = {-p:.4f}",
                f"Opens           : {'upward' if a > 0 else 'downward'}",
                "Eccentricity    : 1"]

    a, b = params["a"], params["b"]
    c = math.hypot(a, b)
    return [f"Equation        : x\u00b2/{a:g}\u00b2 - y\u00b2/{b:g}\u00b2 = 1",
            f"Semi-axes       : transverse a = {a:g}, conjugate b = {b:g}",
            f"Centre          : (0, 0)      Vertices: (\u00b1{a:g}, 0)",
            f"Foci            : (\u00b1{c:.4f}, 0)",
            f"Asymptotes      : y = \u00b1({b:g}/{a:g})x = \u00b1{b / a:.4f}x",
            f"Eccentricity    : {c / a:.4f}"]


def _default_geometry(kind: str, half_angle_deg: float) -> tuple[dict, dict]:
    """Build one conic from its slider defaults; returns (geometry, params)."""
    params = {s["key"]: s["init"] for s in CONIC_PARAMS[kind]}
    if kind == "circle":
        return circle_geometry(params["r"], half_angle_deg), params
    if kind == "ellipse":
        return ellipse_geometry(params["a"], params["b"], half_angle_deg), params
    if kind == "parabola":
        return parabola_geometry(params["a"], half_angle_deg), params
    return hyperbola_geometry(params["a"], params["b"], half_angle_deg), params


def print_sweep_table(half_angle_deg: float, samples: int = 21) -> None:
    """Walk the sweep and show where each conic gives way to the next."""
    t = math.tan(math.radians(half_angle_deg))
    print("\n" + "=" * 72)
    print("  SWEEP: one continuous tilt through all four conics")
    print("=" * 72)
    header = (f"{'u':>6}{'slope m':>11}{'d':>9}{'tilt':>9}"
              f"{'t|m|':>9}{'A':>10}  {'conic':<10}{'residual':>11}")
    print(header)
    print("-" * len(header))
    for u in np.linspace(0.0, 1.0, samples):
        m, d = sweep_plane(float(u), t, SWEEP_HEIGHT)
        geom = section_from_plane(t, m, d, SWEEP_HEIGHT)
        A = 1.0 - t * t * geom["m"] ** 2
        print(f"{u:>6.3f}{geom['m']:>11.4f}{geom['d']:>9.3f}"
              f"{geom['plane_angle_deg']:>8.2f}\u00b0{t * abs(geom['m']):>9.4f}"
              f"{A:>+10.4f}  {geom['kind']:<10}{cone_residual(geom):>11.2e}")
    print("=" * 72)


def run_console(half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG) -> int:
    """``--no-plot``: the full console report, no GUI at all."""
    print("=" * 72)
    print("  CONIC SECTIONS: 3D CONE SLICE  +  2D CURVE   (console report)")
    print("=" * 72)
    for kind in CONIC_ORDER:
        geom, params = _default_geometry(kind, half_angle_deg)
        print_report(geom, feature_lines(kind, params))
    print_sweep_table(half_angle_deg)
    return 0


def run_demo() -> int:
    """
    Derive all four sections over a spread of parameters and confirm that
    every generated point really lies on the cone.  No windows are opened,
    so this also works on a headless machine.

    The sweep is checked the same way: because it is driven by angle and
    branches on the regime instead of interpolating through A = 0, every
    frame - including the exact parabola - must sit on the cone to machine
    precision too.
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

    header = (f"{'conic':<11}{'params':<14}{'alpha':>8}{'slope m':>10}"
              f"{'d':>10}{'A':>10}{'residual':>12}")
    print("PART 1 - parameter-driven sections")
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

    # ---- Part 2: the sweep, across several cone angles -------------------
    print("\nPART 2 - swept sections (plane driven by tilt angle)")
    header2 = f"{'alpha':>8}{'frames':>8}  {'kinds seen':<44}{'worst residual':>16}"
    print(header2)
    print("-" * len(header2))

    for alpha in (25.0, 45.0, 65.0):
        t = math.tan(math.radians(alpha))
        seen: list[str] = []
        worst = 0.0
        for u in np.linspace(0.0, 1.0, 61):
            geom = sweep_section(float(u), t, SWEEP_HEIGHT)
            worst = max(worst, cone_residual(geom))
            if not seen or seen[-1] != geom["kind"]:
                seen.append(geom["kind"])
        order = " -> ".join(seen)
        print(f"{alpha:>7.1f}\u00b0{61:>8}  {order:<44}{worst:>16.2e}")
        if worst > CONE_TOL:
            failures += 1
            print(f"    !! swept points drift off the cone (residual {worst:.3e})")
        for expected in CONIC_ORDER:
            if expected not in seen:
                failures += 1
                print(f"    !! the sweep never produced a {expected}")

    # ---- Part 3: the parabola crossing itself ----------------------------
    print("\nPART 3 - the A -> 0 crossing")
    t = math.tan(math.radians(DEFAULT_HALF_ANGLE_DEG))
    geom = section_from_plane(t, 1.0 / t, SWEEP_D_FRACTION * SWEEP_HEIGHT, SWEEP_HEIGHT)
    A = 1.0 - t * t * geom["m"] ** 2
    print(f"  exact slant-parallel plane : m = {geom['m']:.6f}, t|m| = {t * abs(geom['m']):.6f}")
    print(f"  classifier A               : {A:+.3e}")
    print(f"  section                    : {geom['kind']}")
    print(f"  residual                   : {cone_residual(geom):.2e}")
    if geom["kind"] != "parabola" or cone_residual(geom) > CONE_TOL:
        failures += 1
        print("    !! the exact parabola frame is wrong")

    print("\n" + "-" * 72)
    print("Every section lies exactly on its cone."
          if failures == 0 else f"{failures} case(s) failed.")
    return 0 if failures == 0 else 1


# ==========================================================================
# SECTION 6 - CLI entry point
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live 3D cone slicer: watch a tilting plane turn a circle "
                    "into an ellipse, a parabola and a hyperbola.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true",
                        help="headless self-check: every point must lie on the cone")
    parser.add_argument("--no-plot", action="store_true", dest="no_plot",
                        help="console report only, no GUI")
    parser.add_argument("--save", action="store_true",
                        help="render the sweep to conic_section_visualiser.mp4 "
                             "(.gif if ffmpeg is unavailable)")
    parser.add_argument("--type", choices=CONIC_ORDER, default="circle",
                        help="conic to start on (default: circle)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_HALF_ANGLE_DEG,
                        metavar="DEG", help="cone half-angle in degrees (default: 45)")
    parser.add_argument("--frames", type=int, default=220, metavar="N",
                        help="number of frames for --save (default: 220)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    alpha = float(min(max(args.alpha, MIN_HALF_ANGLE_DEG), MAX_HALF_ANGLE_DEG))

    if args.demo:
        return run_demo()

    if args.no_plot:
        return run_console(alpha)

    if args.save:
        print("=" * 72)
        print("  CONIC SECTIONS - rendering the sweep")
        print("=" * 72)
        save_sweep(half_angle_deg=alpha, frames=max(args.frames, 2))
        return 0

    print("=" * 72)
    print("  CONIC SECTIONS: LIVE CONE SLICER")
    print("=" * 72)
    print("  radio buttons : choose circle / ellipse / parabola / hyperbola")
    print("  sliders       : that conic's parameters, plus the cone half-angle")
    print("  Play          : sweep the plane through all four conics (loops)")
    print("  Auto-rotate   : slow idle spin;  drag with the mouse to look around")
    print("  Reset         : back to the defaults")
    print("  Close the window to quit.")
    ConicScene(kind=args.type, half_angle_deg=alpha).show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
