#!/usr/bin/env python3
"""
Derivative Curve Tracer  (the top graph draws the bottom graph)
------------------------------------------------------------------------------
Type a function f(x) and a range.  A dot then walks along the curve carrying
its tangent line with it, and everything that dot learns about the steepness
gets written down in the panel underneath -- one stroke per step -- until the
whole of f'(x) has been drawn by hand.

  TOP PANEL     f(x) in magenta, complete from the start.  A dot slides along
                it with a short tangent segment through it, an arc marking the
                angle theta that the tangent makes with the horizontal, and a
                live readout of f'(x) = tan(theta).
  BOTTOM PANEL  empty at frame one.  Each step stamps the single point
                (x, f'(x)) in amber, so the derivative curve accumulates
                stroke by stroke instead of simply appearing.
  BETWEEN THEM  a thin dashed line joining the two dots, because the whole
                point is that they are the *same* x.
  CONTROLS      Pause, then drag the x slider (or step with the arrow keys) to
                park the dot on any point and read its slope, angle and
                condition -- increasing, decreasing or stationary.

THE MATHEMATICS
===============
The derivative of f at a point is the slope of the tangent line there:

        f'(x)  =  slope of the tangent at x  =  tan(theta)

where theta is the angle between the tangent and the x-axis.  That is one
number per x.  Feed every x in an interval through that sentence and the
numbers you get back form a *function* -- the derivative function f'(x):

        f'  :  x  |-->  slope of f at x

This is the step that is easy to state and hard to feel.  f'(x) is not a new
curve pulled out of a rulebook; it is a *transcript* of f's steepness.  Where f
climbs, f' is positive.  Where f falls, f' is negative.  Where f levels off at
a peak or a trough, the tangent goes flat, theta goes to zero, and f' crosses
the axis.  The steeper f gets, the further from zero f' goes.

So the animation is arranged to make the transcript visible as it is written:

        tangent tilts up      ->  theta > 0  ->  amber trace above the axis
        tangent goes flat     ->  theta = 0  ->  amber trace touches the axis
        tangent tilts down    ->  theta < 0  ->  amber trace below the axis

Watch the top panel's peaks and troughs line up with the bottom panel's zeros
and you have understood what a derivative *is*.

WHY THE ANGLE IS DRAWN IN PIXELS
================================
theta = arctan(f'(x)) is an honest angle only when one unit across the page is
one unit up the page.  These panels are not square and the y-range is chosen
to fit the function, so a slope of 1 usually does *not* look like 45 degrees on
screen.  Drawing the arc naively in data coordinates would therefore put a
"45 degree" label on an arc that plainly is not 45 degrees, and the picture
would be quietly lying.  Instead the tangent segment and the arc are both
built in pixel space -- measured with the axes' own data-to-display scaling --
so the arc always agrees with the angle the eye actually sees, and the printed
number always agrees with arctan(f'(x)).

Usage:
    python derivative_curve_tracer.py                      # prompts, then animates
    python derivative_curve_tracer.py --func "x**2"
    python derivative_curve_tracer.py --func "sin(x)" --xmin -6.283 --xmax 6.283
    python derivative_curve_tracer.py --func "exp(-x**2)" --frames 320 --once
    python derivative_curve_tracer.py --func "x**3 - 3*x" --save
    python derivative_curve_tracer.py --func "x**2" --no-widgets   # clean window
    python derivative_curve_tracer.py --demo               # headless self-check
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from dataclasses import dataclass
from typing import Callable

import matplotlib

# Pick the non-GUI backend *before* pyplot is imported, but only for the
# headless modes, so the script runs unchanged over SSH / in CI.
if {"--demo", "--no-plot", "--save"} & set(sys.argv[1:]):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import Button, Slider
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

# ==========================================================================
# SECTION 0 - Constants and shared vocabulary
# ==========================================================================

X = sp.Symbol("x", real=True)          # the one and only variable we allow

DEFAULT_XMIN = -3.0                    # sensible default window if none given
DEFAULT_XMAX = 3.0
DEFAULT_FRAMES = 200                   # one smooth sweep
DEFAULT_INTERVAL = 25                  # milliseconds between frames
HOLD_FRAMES = 52                       # rest on the finished curve before looping
CURVE_SAMPLES = 1600                   # resolution of the static f(x) curve

TANGENT_PX = 74.0                      # half-length of the tangent stub, pixels
ARC_PX = 34.0                          # radius of the angle arc, pixels
MIN_FRAMES = 8                         # below this there is nothing to watch
MAX_FRAMES = 5000                      # above this it is a rendering job, not a demo
SIMPLIFY_OPS = 60                      # don't call simplify() on monsters
JUMP_FACTOR = 0.5                      # f' jump that counts as "not differentiable"
BIG_STEP = 10                          # frames moved by shift+arrow
FLAT_TOL = 1e-3                        # |f'| below this fraction of max|f'| reads as flat

# Colours, kept in one place so the figure stays coherent.
BG = "#ffffff"
C_CURVE = "#e5308c"      # f(x) -- magenta
C_DERIV = "#f2a20c"      # f'(x) -- amber, the curve being written
C_TANGENT = "#3949ab"    # the tangent segment
C_ARC = "#00897b"        # the angle arc and its label
C_LINK = "#7a7a7a"       # the dashed line joining the two dots
C_GRID = "#c9c9c9"
C_WIDGET = "#ececec"     # widget faces
C_WIDGET_HOVER = "#d6d6d6"

# ==========================================================================
# SECTION 1 - Parsing: turn user text into a sympy expression, safely
# ==========================================================================


class TracerError(Exception):
    """Raised for anything the user could plausibly have typed wrong."""


# Only these names are meaningful inside f(x).  Anything else becomes a stray
# symbol and is rejected with a readable message, so a typo like 'sinn(x)'
# reports "unknown name" rather than exploding somewhere deep in sympy.
_ALLOWED_NAMES: dict[str, object] = {
    "x": X,
    # trigonometry
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    # hyperbolics
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    # growth and decay
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    # odds and ends
    "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
    "floor": sp.floor, "ceiling": sp.ceiling,
    # constants
    "pi": sp.pi, "Pi": sp.pi, "PI": sp.pi, "E": sp.E, "e": sp.E,
}

# The machinery sympy's own parser emits (Symbol(...), Integer(...), ...).
# '__builtins__' is deliberately emptied so parsing cannot reach Python itself.
_PARSE_GLOBALS: dict[str, object] = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
}


def preprocess(text: str) -> str:
    """
    Make ordinary handwritten maths acceptable to a Python parser.

        x^2 + 3x      ->  x**2 + 3*x
        2sin(x)       ->  2*sin(x)
        (x+1)(x-1)    ->  (x+1)*(x-1)

    The negative lookahead protects scientific notation, so '1e-3' survives
    intact instead of being mangled into '1*e-3'.
    """
    expr = text.strip()
    expr = expr.replace("\u2212", "-")   # unicode minus, pasted from the web
    expr = expr.replace("^", "**")       # '^' means power here, not xor

    # digit followed by a letter or '(' -> implicit multiplication
    expr = re.sub(r"(\d)(?![eE][-+]?\d)\s*([A-Za-z_(])", r"\1*\2", expr)
    # ')' followed by a name, number or '(' -> implicit multiplication.
    # '**' and other operators are untouched because they are not in the class.
    expr = re.sub(r"\)\s*([A-Za-z_0-9(])", r")*\1", expr)
    return expr


def parse_function(text: str) -> sp.Expr:
    """Parse the user's text into a sympy expression in the single symbol x."""
    if not text or not text.strip():
        raise TracerError("Please enter a function of x, e.g. x**2 or sin(x)")

    cleaned = preprocess(text)
    try:
        expr = parse_expr(
            cleaned,
            local_dict=dict(_ALLOWED_NAMES),
            global_dict=dict(_PARSE_GLOBALS),
            transformations=standard_transformations,
            evaluate=True,
        )
    except Exception as exc:  # noqa: BLE001 - any parse failure is user error
        raise TracerError(
            f"Could not read '{text.strip()}' as a function of x ({exc})."
        ) from exc

    if not isinstance(expr, sp.Expr) or isinstance(expr, sp.logic.boolalg.Boolean):
        raise TracerError(
            "That is not an expression in x. Enter just the right-hand side, "
            "e.g. 'x**2 + 3*x' rather than 'y = x**2 + 3*x'."
        )

    unknown = sorted(s.name for s in expr.free_symbols if s.name != "x")
    if unknown:
        raise TracerError(
            f"Unknown name(s): {', '.join(unknown)}. "
            "The function may only use x and the standard functions "
            "(sin, cos, tan, exp, log, sqrt, ...)."
        )

    # Re-bind whatever 'x' we got onto our own real-valued symbol, so later
    # simplifications know x is real.
    return expr.subs({s: X for s in expr.free_symbols})


def parse_number(text: str, what: str) -> float:
    """
    Read a number that may itself be a small expression, so the user can type
    'pi', '-2*pi' or 'sqrt(2)' for a range endpoint instead of a decimal.
    """
    try:
        value = float(sp.N(parse_expr(
            preprocess(text),
            local_dict=dict(_ALLOWED_NAMES),
            global_dict=dict(_PARSE_GLOBALS),
        )))
    except Exception as exc:  # noqa: BLE001
        raise TracerError(f"'{text.strip()}' is not a number I can use for {what}.") from exc
    if not math.isfinite(value):
        raise TracerError(f"{what} must be a finite number.")
    return value


def expr_text(expr: sp.Expr) -> str:
    """A compact, plain-text rendering, e.g. 'x^2 + 3*x'. Safe in any font."""
    return sp.sstr(expr).replace("**", "^")


# ==========================================================================
# SECTION 2 - Symbolic differentiation, then compiled numerics
# ==========================================================================


def differentiate(f: sp.Expr) -> sp.Expr:
    """
    The first derivative, symbolically -- the whole point of using sympy.

    simplify() is what turns d/dx[sin(x)*cos(x)] from
    '-sin(x)^2 + cos(x)^2' into 'cos(2*x)', which is worth having in the
    panel title.  It is skipped on large expressions because it can take
    longer than the animation itself and the tidied form would be unreadable
    anyway.
    """
    try:
        f_prime = sp.diff(f, X)
    except Exception as exc:  # noqa: BLE001
        raise TracerError(f"Could not differentiate f(x) ({exc}).") from exc

    if sp.count_ops(f_prime) <= SIMPLIFY_OPS:
        try:
            f_prime = sp.simplify(f_prime)
        except Exception:  # noqa: BLE001 - tidier output is strictly optional
            pass
    return f_prime


def make_numeric(expr: sp.Expr, label: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Compile the expression into a fast numpy function via lambdify.

    The wrapper never raises: anything outside the domain (log of a negative,
    a division by zero, an overflow) comes back as NaN.  matplotlib simply
    leaves a gap in the curve there, which is exactly the honest picture --
    sqrt(x) really has no tangent to the left of the origin, and the trace
    should show that rather than inventing one.
    """
    try:
        compiled = sp.lambdify(X, expr, modules=["numpy"])
    except Exception as exc:  # noqa: BLE001
        raise TracerError(f"Could not evaluate {label} numerically ({exc}).") from exc

    def elementwise(values: np.ndarray) -> np.ndarray:
        """Fallback path for expressions that refuse to vectorise."""
        out = np.full(values.shape, np.nan, dtype=float)
        flat_in, flat_out = values.ravel(), out.ravel()
        for i, v in enumerate(flat_in):
            try:
                flat_out[i] = float(compiled(v))
            except Exception:  # noqa: BLE001
                flat_out[i] = np.nan
        return out

    def evaluate(values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        with np.errstate(all="ignore"):          # silence numpy's domain warnings
            try:
                raw = compiled(arr)
            except Exception:  # noqa: BLE001
                raw = elementwise(arr)
        raw = np.asarray(raw)
        if np.iscomplexobj(raw):
            # A complex result means we stepped off the real domain -> NaN.
            raw = np.where(np.abs(raw.imag) < 1e-12, raw.real, np.nan)
        # broadcast_to covers constant expressions: lambdify turns f' = 1 into
        # a function that returns the scalar 1 whatever you hand it.
        out = np.broadcast_to(np.asarray(raw, dtype=float), arr.shape).astype(float).copy()
        out[~np.isfinite(out)] = np.nan
        return out

    return evaluate


# ==========================================================================
# SECTION 3 - Sampling: everything the animation needs, computed up front
# ==========================================================================


@dataclass
class Track:
    """
    The whole sweep, precomputed.

    Computing f and f' for every frame in two vectorised calls before the
    animation starts buys two things: frames cost almost nothing to draw, and
    the axis limits can be fixed in advance so the view never lurches
    mid-sweep (which would wreck the illusion that one panel is drawing the
    other).
    """
    source: str
    f: sp.Expr
    f_prime: sp.Expr
    x_min: float
    x_max: float
    xs: np.ndarray            # frame x-positions
    fs: np.ndarray            # f at those positions  (NaN where undefined)
    ds: np.ndarray            # f' at those positions (NaN where undefined)
    thetas: np.ndarray        # arctan(f'), radians   (NaN where undefined)
    curve_x: np.ndarray       # dense samples for the static top curve
    curve_y: np.ndarray
    top_ylim: tuple[float, float]
    bottom_ylim: tuple[float, float]
    notes: list[str]

    @property
    def frames(self) -> int:
        return int(self.xs.size)


def _nice_ylim(values: np.ndarray, extras: tuple[float, ...] = ()) -> tuple[float, float]:
    """
    Choose y-limits from the middle 96% of the sampled values.

    Clipping the tails is what keeps a nearby asymptote -- tan(x), or 1/x
    near the origin -- from squashing the whole interesting part of the curve
    into a flat line at the middle of the panel.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    lo = float(np.percentile(finite, 2.0))
    hi = float(np.percentile(finite, 98.0))
    for value in extras:
        if math.isfinite(value):
            lo, hi = min(lo, value), max(hi, value)

    if hi - lo < 1e-9:                      # a constant function
        centre = 0.5 * (lo + hi)
        lo, hi = centre - 1.0, centre + 1.0
    pad = 0.16 * (hi - lo)
    return lo - pad, hi + pad


def _find_jumps(xs: np.ndarray, ds: np.ndarray) -> list[float]:
    """
    Flag places where f' leaps between neighbouring samples.

    A genuine derivative is continuous almost everywhere, so a big jump means
    one of two interesting things: a corner in f (|x| at the origin, where f'
    flips from -1 to +1 and no tangent exists), or a pole (tan(x), where the
    slope runs off to infinity).  Either way it is worth a note rather than a
    refusal -- the break in the amber trace is the most instructive thing on
    screen.
    """
    finite = ds[np.isfinite(ds)]
    if finite.size < 3:
        return []

    spread = float(np.percentile(finite, 98.0) - np.percentile(finite, 2.0))
    if spread <= 0:
        return []
    threshold = max(JUMP_FACTOR * spread, 1e-9)

    hits: list[float] = []
    for i in range(ds.size - 1):
        a, b = ds[i], ds[i + 1]
        if math.isfinite(a) and math.isfinite(b) and abs(b - a) > threshold:
            hits.append(0.5 * (xs[i] + xs[i + 1]))
    return hits


def analyse(source: str, x_min: float = DEFAULT_XMIN, x_max: float = DEFAULT_XMAX,
            frames: int = DEFAULT_FRAMES) -> Track:
    """Parse, validate, differentiate, and sample the whole sweep."""
    # --- validate the window -------------------------------------------
    if not math.isfinite(x_min) or not math.isfinite(x_max):
        raise TracerError("The x-range endpoints must be finite numbers.")
    if x_min == x_max:
        raise TracerError(
            f"x_min and x_max are both {x_min:g}, so there is no range to "
            "sweep. Give two different endpoints, e.g. -3 and 3."
        )
    if x_min > x_max:
        x_min, x_max = x_max, x_min          # a helpful silent fix
    if not MIN_FRAMES <= frames <= MAX_FRAMES:
        raise TracerError(
            f"frames must be between {MIN_FRAMES} and {MAX_FRAMES} "
            f"(got {frames}); {DEFAULT_FRAMES} gives a smooth sweep."
        )

    # --- symbolic side --------------------------------------------------
    f = parse_function(source)
    f_prime = differentiate(f)

    # --- numeric side ---------------------------------------------------
    f_num = make_numeric(f, "f(x)")
    d_num = make_numeric(f_prime, "f'(x)")

    xs = np.linspace(x_min, x_max, frames)
    fs = f_num(xs)
    ds = d_num(xs)

    curve_x = np.linspace(x_min, x_max, CURVE_SAMPLES)
    curve_y = f_num(curve_x)

    # --- is there anything to look at? ----------------------------------
    if not np.any(np.isfinite(curve_y)):
        raise TracerError(
            f"f(x) = {expr_text(f)} has no real values anywhere on "
            f"[{x_min:g}, {x_max:g}], so there is nothing to plot. "
            "Try a range inside its domain (log and sqrt need x > 0)."
        )
    if not np.any(np.isfinite(ds)):
        raise TracerError(
            f"f'(x) = {expr_text(f_prime)} could not be evaluated anywhere on "
            f"[{x_min:g}, {x_max:g}], so the derivative panel would stay "
            "empty. Try a range inside the domain of f."
        )

    # --- axis limits, fixed now so the view never jumps -----------------
    top_ylim = _nice_ylim(curve_y, extras=tuple(fs[np.isfinite(fs)][:1]))
    bottom_ylim = _nice_ylim(ds, extras=(0.0,))   # always show the f' = 0 axis

    # --- advisory notes (never fatal) -----------------------------------
    notes: list[str] = []

    missing = int(np.count_nonzero(~np.isfinite(fs)))
    if missing:
        notes.append(
            f"f(x) is undefined at {missing} of the {frames} sampled positions; "
            "the dot and the trace go blank there, which is the honest picture."
        )

    gaps = int(np.count_nonzero(~np.isfinite(ds))) - missing
    if gaps > 0:
        notes.append(
            f"f'(x) is undefined at {gaps} position(s) where f(x) itself is "
            "fine -- typically a vertical tangent or a domain edge."
        )

    jumps = _find_jumps(xs, ds)
    if jumps:
        where = ", ".join(f"x = {v:.4g}" for v in jumps[:3])
        more = f" (and {len(jumps) - 3} more)" if len(jumps) > 3 else ""
        notes.append(
            f"f'(x) jumps sharply near {where}{more}. f(x) is not "
            "differentiable there -- a corner or a pole -- so the amber trace "
            "leaps instead of flowing."
        )

    return Track(source=source.strip(), f=f, f_prime=f_prime,
                 x_min=x_min, x_max=x_max, xs=xs, fs=fs, ds=ds,
                 thetas=np.arctan(ds), curve_x=curve_x, curve_y=curve_y,
                 top_ylim=top_ylim, bottom_ylim=bottom_ylim, notes=notes)


# ==========================================================================
# SECTION 4 - Console report
# ==========================================================================


def _cell(value: float, width: int, spec: str = ".6g") -> str:
    """Format a number for the table, or say so when it does not exist."""
    if not math.isfinite(value):
        return f"{'undefined':>{width}}"
    return f"{format(value, spec):>{width}}"


def print_report(track: Track, rows: int = 9) -> None:
    """
    Print the same story the figure tells, in numbers.

    Having the table on screen next to the animation means every claim the
    picture makes can be checked by eye -- especially the sign rule, which is
    the part worth internalising.
    """
    print()
    print(f"  f(x)   = {expr_text(track.f)}")
    print(f"  f'(x)  = {expr_text(track.f_prime)}"
          f"      (the symbolic derivative)")
    print(f"  range  = [{track.x_min:g}, {track.x_max:g}]"
          f"   in {track.frames} frames")
    print()
    print("  SLOPE, ANGLE, AND THE POINT THAT GETS PLOTTED BELOW")
    print("  " + "-" * 68)
    print(f"    {'x':>10}  {'f(x)':>14}  {'f\'(x) = tan(theta)':>20}  {'theta':>10}")
    print(f"    {'-' * 10}  {'-' * 14}  {'-' * 20}  {'-' * 10}")

    # Sample evenly across the sweep, including both endpoints.
    picks = np.unique(np.linspace(0, track.frames - 1, rows).astype(int))
    for i in picks:
        theta = track.thetas[i]
        angle = (f"{math.degrees(theta):>9.3f}\u00b0"
                 if math.isfinite(theta) else f"{'--':>10}")
        print(f"    {track.xs[i]:>10.4g}  {_cell(track.fs[i], 14)}"
              f"  {_cell(track.ds[i], 20)}  {angle}")
    print(f"    {'-' * 10}  {'-' * 14}  {'-' * 20}  {'-' * 10}")
    print()
    print("    Where f(x) rises, f'(x) sits above zero; where f(x) falls, below.")
    print("    At a peak or a trough the tangent is flat, so theta = 0 and the")
    print("    amber curve crosses the axis exactly there.")

    for note in track.notes:
        print()
        print(f"  [Note] {note}")
    print()


# ==========================================================================
# SECTION 5 - The figure: two panels, one writing the other
# ==========================================================================


def _px_per_data(ax) -> tuple[float, float]:
    """
    How many screen pixels one data unit spans, horizontally and vertically.

    Measured by transforming the corners of the current view rather than
    asking for the axes bbox, so it needs no renderer and works before the
    first draw (which matters for the headless self-check).
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    span_x = max(abs(x1 - x0), 1e-12)
    span_y = max(abs(y1 - y0), 1e-12)
    return abs(p1[0] - p0[0]) / span_x, abs(p1[1] - p0[1]) / span_y


class TracerScene:
    """
    Builds the two-panel figure and moves the handful of live artists.

    The static things -- the f(x) curve, the axes, the titles -- are created
    once.  Every frame then touches only the dot, the tangent, the arc, the
    growing trace and the connector, so the animation stays smooth even at a
    few hundred frames.

    The scene -- not the FuncAnimation -- owns the frame index.  That matters:
    the timer, the slider, the buttons and the arrow keys all want to move the
    playhead, and if matplotlib kept its own private counter they would fight.
    One 'cursor' attribute, and every control simply sets or nudges it.
    """

    def __init__(self, track: Track, interactive: bool = False,
                 loop: bool = True):
        self.track = track
        self.interactive = interactive

        # --- playback state, the single source of truth ------------------
        # cursor is always a *sweep frame* index in [0, frames-1]. The rest at
        # the end of a pass is counted separately, so 'which x am I looking at'
        # never has to be decoded out of a padded sequence.
        self.cursor = 0
        self.playing = True
        self.loop = loop
        self.hold = HOLD_FRAMES
        self._hold_left = HOLD_FRAMES
        self._syncing = False           # guards slider <-> cursor feedback

        # How flat counts as flat, judged against the size of f' on *this*
        # sweep. A fixed epsilon would call a gentle slope "stationary" on a
        # steep function and never fire at all on a shallow one.
        finite = track.ds[np.isfinite(track.ds)]
        scale = float(np.percentile(np.abs(finite), 98.0)) if finite.size else 1.0
        self.flat_eps = max(FLAT_TOL * scale, 1e-12)

        self._build_figure()
        self._draw_static()
        self._create_live_artists()
        if interactive:
            self._build_widgets()
        self.render(0)

    # ---------------------------------------------------------------- setup

    def _build_figure(self) -> None:
        t = self.track
        # Interactive windows get taller and keep a strip free at the bottom
        # for the controls, so adding them does not squeeze the two panels.
        height = 9.6 if self.interactive else 9.0
        bottom = 0.19 if self.interactive else 0.09

        # sharex is what makes the connector genuinely vertical: both panels
        # are guaranteed to put the same x at the same screen position.
        self.fig, (self.ax_top, self.ax_bot) = plt.subplots(
            2, 1, figsize=(11.5, height), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.30},
        )
        self.fig.patch.set_facecolor(BG)
        self.fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=bottom)

        try:
            self.fig.canvas.manager.set_window_title(
                f"f(x) = {expr_text(t.f)}   ->   f'(x) = {expr_text(t.f_prime)}"
            )
        except Exception:  # noqa: BLE001 - headless backends have no manager
            pass

        self.fig.suptitle(
            "The tangent's slope in the top panel is the height plotted in the "
            "bottom panel",
            fontsize=12.5, fontweight="bold", y=0.965,
        )

    def _style(self, ax, ylabel: str, ylim: tuple[float, float]) -> None:
        """Shared cosmetics, so the two panels read as one picture."""
        t = self.track
        ax.set_facecolor("#fcfcfc")
        ax.grid(True, linestyle="--", alpha=0.45, color=C_GRID, zorder=0)
        ax.axhline(0, color="#333333", linewidth=1.0, zorder=1)
        if t.x_min <= 0 <= t.x_max:
            ax.axvline(0, color="#333333", linewidth=1.0, zorder=1)
        ax.set_xlim(t.x_min, t.x_max)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=11.5)

    def _draw_static(self) -> None:
        t = self.track

        # ---- top panel: the function, drawn in full from the start -------
        self._style(self.ax_top, "f(x)", t.top_ylim)
        self.ax_top.plot(t.curve_x, t.curve_y, color=C_CURVE, linewidth=2.5,
                         zorder=3, label=f"f(x) = {expr_text(t.f)}")
        self.ax_top.set_title(f"f(x) = {expr_text(t.f)}", fontsize=12.5,
                              fontweight="bold", color=C_CURVE, pad=9)
        self.ax_top.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

        # ---- bottom panel: deliberately empty ---------------------------
        # Its limits are already fixed (from the precomputed f' values) so the
        # trace draws into a stable frame instead of rescaling as it goes.
        self._style(self.ax_bot, "f'(x)", t.bottom_ylim)
        self.ax_bot.set_title(f"f'(x) = {expr_text(t.f_prime)}", fontsize=12.5,
                              fontweight="bold", color="#b8760a", pad=9)
        self.ax_bot.set_xlabel("x", fontsize=11.5)

        # sharex hides the top panel's tick labels by default. Put them back
        # and label the axis properly: each panel should be readable on its
        # own, and the matching tick marks let the eye confirm the two panels
        # really do share an x-scale rather than taking the connector's word
        # for it.
        self.ax_top.tick_params(labelbottom=True)
        self.ax_top.set_xlabel("x", fontsize=11.5)

    def _create_live_artists(self) -> None:
        """Every artist the animation moves, created empty, once."""
        t = self.track

        # ---- top panel ---------------------------------------------------
        self.tangent, = self.ax_top.plot([], [], color=C_TANGENT, linewidth=2.6,
                                         solid_capstyle="round", zorder=5,
                                         label="tangent")
        self.h_ref, = self.ax_top.plot([], [], color=C_ARC, linewidth=1.3,
                                       linestyle=(0, (4, 3)), zorder=5)
        self.arc, = self.ax_top.plot([], [], color=C_ARC, linewidth=2.0, zorder=6)
        self.arc_label = self.ax_top.text(
            0, 0, "", color=C_ARC, fontsize=10, fontweight="bold",
            ha="center", va="center", zorder=8,
        )
        self.dot_top, = self.ax_top.plot([], [], "o", color=C_CURVE, markersize=11,
                                         markeredgecolor="white",
                                         markeredgewidth=1.6, zorder=7)

        # ---- bottom panel: the trace grows one point per frame -----------
        # A single Line2D fed a lengthening slice of preallocated arrays.
        # NaNs inside the slice break the line exactly where f' is undefined.
        self.trace_x = np.full(t.frames, np.nan)
        self.trace_y = np.full(t.frames, np.nan)
        self.trace, = self.ax_bot.plot([], [], color=C_DERIV, linewidth=2.8,
                                       solid_capstyle="round", zorder=4,
                                       label=f"f'(x) = {expr_text(t.f_prime)}")
        self.dot_bot, = self.ax_bot.plot([], [], "o", color=C_DERIV, markersize=11,
                                         markeredgecolor="white",
                                         markeredgewidth=1.6, zorder=6)
        self.ax_bot.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

        # ---- faint droplines inside each panel --------------------------
        self.drop_top, = self.ax_top.plot([], [], color=C_LINK, linewidth=1.0,
                                          linestyle=":", alpha=0.85, zorder=2)
        self.drop_bot, = self.ax_bot.plot([], [], color=C_LINK, linewidth=1.0,
                                          linestyle=":", alpha=0.85, zorder=2)

        # ---- the connector between the panels ---------------------------
        # A ConnectionPatch lives on the figure, not inside either axes, so it
        # is free to cross the gap between them without being clipped.
        self.link = ConnectionPatch(
            xyA=(t.x_min, 0.0), coordsA=self.ax_top.transData,
            xyB=(t.x_min, 0.0), coordsB=self.ax_bot.transData,
            color=C_LINK, linewidth=1.2, linestyle=(0, (5, 4)),
            alpha=0.95, zorder=10,
        )
        self.link.set_clip_on(False)
        self.fig.add_artist(self.link)

        # ---- the live numeric readout -----------------------------------
        self.card = self.ax_top.text(
            0.015, 0.975, "", transform=self.ax_top.transAxes,
            va="top", ha="left", fontsize=10.5, family="monospace", zorder=11,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#fffdf3",
                      edgecolor=C_TANGENT, alpha=0.95),
        )

    # -------------------------------------------------------------- widgets

    def _build_widgets(self) -> None:
        """
        The controls: pause, scrub, reset, and keyboard stepping.

        A Pause button alone would be frustrating -- a 200-frame sweep at 25 ms
        goes by in five seconds, so catching one particular x by reflex is
        hopeless.  The slider is the control that actually answers "show me
        this point", and it is deliberately labelled and stepped in *x* rather
        than in frame numbers, because x is what you care about.
        """
        t = self.track

        # One slider notch per animation frame.  With the step matched exactly,
        # slider position and frame index are two names for the same thing and
        # cannot drift apart.
        self.x_step = (t.x_max - t.x_min) / max(1, t.frames - 1)
        slider_ax = self.fig.add_axes([0.10, 0.088, 0.62, 0.032])
        try:                                    # matplotlib >= 3.5 styling hooks
            self.slider = Slider(
                slider_ax, "scrub:  x  ", t.x_min, t.x_max,
                valinit=t.x_min, valstep=self.x_step, color=C_DERIV,
                track_color=C_WIDGET, initcolor="none",
            )
        except TypeError:  # pragma: no cover - older matplotlib
            self.slider = Slider(slider_ax, "scrub:  x  ", t.x_min, t.x_max,
                                 valinit=t.x_min, valstep=self.x_step,
                                 color=C_DERIV)
        self.slider.on_changed(self._on_scrub)

        self.play_button = self._make_button([0.755, 0.083, 0.098, 0.042], "Pause")
        self.play_button.on_clicked(self._on_play_clicked)
        self.reset_button = self._make_button([0.865, 0.083, 0.098, 0.042], "Reset")
        self.reset_button.on_clicked(self._on_reset_clicked)

        self.fig.text(
            0.10, 0.036,
            "space = play/pause     \u2190 \u2192 = step one frame     "
            "shift+\u2190 \u2192 = step " f"{BIG_STEP}" "     home/end = ends"
            "        (dragging the slider pauses)",
            fontsize=9.5, color="#555555",
        )

        # matplotlib binds left/right to its view-history navigation, so an
        # arrow key would step the sweep *and* undo a zoom.  Take just those
        # two bindings away for the life of this figure, and put them back on
        # close -- rcParams are global, and a plotting script should not leave
        # the user's keyboard rearranged.
        self._keymap_taken: list[tuple[str, str]] = []
        for param, key in (("keymap.back", "left"), ("keymap.forward", "right")):
            if key in plt.rcParams[param]:
                plt.rcParams[param] = [k for k in plt.rcParams[param] if k != key]
                self._keymap_taken.append((param, key))

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _make_button(self, rect: list[float], label: str) -> Button:
        button = Button(self.fig.add_axes(rect), label, color=C_WIDGET,
                        hovercolor=C_WIDGET_HOVER)
        button.label.set_fontsize(10.5)
        return button

    # ------------------------------------------------------------- playback

    def set_cursor(self, index: int, from_slider: bool = False) -> None:
        """
        Move the playhead to a sweep frame and redraw.

        The slider is kept in step here, guarded by _syncing: without the flag,
        set_val() would fire _on_scrub, which would call back into here, and
        the two would ping-pong.
        """
        index = int(np.clip(index, 0, self.track.frames - 1))
        self.cursor = index
        self.render(index)

        if not from_slider and getattr(self, "slider", None) is not None:
            self._syncing = True
            try:
                self.slider.set_val(self.track.xs[index])
            finally:
                self._syncing = False

    def advance(self) -> None:
        """
        One tick of the clock: the only thing that moves the playhead by itself.

        While paused it does nothing at all, which is what lets you study a
        point for as long as you like.
        """
        if not self.playing:
            return

        last = self.track.frames - 1
        if self.cursor < last:
            self.set_cursor(self.cursor + 1)
            self._hold_left = self.hold
            return

        # On the final frame: rest on the completed curve for a moment. Without
        # this the derivative would vanish the instant it was finished, which is
        # precisely when it is worth looking at.
        if self._hold_left > 0:
            self._hold_left -= 1
            return

        if self.loop:
            self.set_cursor(0)
            self._hold_left = self.hold
        else:
            # --once: stop, but stay scrubbable rather than frozen, so the
            # sweep can still be replayed or inspected by hand.
            self.set_playing(False)

    def set_playing(self, playing: bool) -> None:
        """Set the play state and make the button label say what it will do."""
        self.playing = bool(playing)
        if getattr(self, "play_button", None) is not None:
            self.play_button.label.set_text("Pause" if self.playing else "Play")
            self.fig.canvas.draw_idle()

    def step(self, delta: int) -> None:
        """Nudge by delta frames, clamping at the ends. Pauses first."""
        self.set_playing(False)
        self.set_cursor(self.cursor + delta)

    # ------------------------------------------------------------ callbacks

    def _on_scrub(self, value: float) -> None:
        if self._syncing:
            return
        # Dragging is a request to look at one place, so stop the clock rather
        # than letting it fight the mouse.
        self.set_playing(False)
        index = int(round((float(value) - self.track.x_min) / self.x_step))
        self.set_cursor(index, from_slider=True)

    def _on_play_clicked(self, _event) -> None:
        # Pressing Play while parked on the finished curve should replay it,
        # not sit there looking broken.
        if not self.playing and self.cursor >= self.track.frames - 1:
            self.set_cursor(0)
        self._hold_left = self.hold
        self.set_playing(not self.playing)

    def _on_reset_clicked(self, _event) -> None:
        self.set_playing(False)
        self.set_cursor(0)

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == " " or key == "space":
            self._on_play_clicked(event)
        elif key == "left":
            self.step(-1)
        elif key == "right":
            self.step(1)
        elif key == "shift+left":
            self.step(-BIG_STEP)
        elif key == "shift+right":
            self.step(BIG_STEP)
        elif key == "home":
            self.step(-self.track.frames)
        elif key == "end":
            self.step(self.track.frames)

    def release_keys(self) -> None:
        """
        Give the arrow keys back to matplotlib's navigation.

        Safe to call twice, because it empties the list of what it borrowed.
        """
        for param, key in getattr(self, "_keymap_taken", []):
            if key not in plt.rcParams[param]:
                plt.rcParams[param] = list(plt.rcParams[param]) + [key]
        self._keymap_taken = []

    def _on_close(self, _event) -> None:
        self.release_keys()

    # --------------------------------------------------------------- frames

    def _hide_top_marks(self) -> None:
        """f(x) is undefined here: there is no dot and no tangent to draw."""
        for artist in (self.tangent, self.h_ref, self.arc,
                       self.dot_top, self.drop_top):
            artist.set_data([], [])
        self.arc_label.set_text("")

    def _draw_tangent_and_angle(self, x: float, y: float, slope: float) -> None:
        """
        Draw the tangent stub, the horizontal reference, and the angle arc.

        All three are laid out in *pixel* space and converted back to data
        units by dividing componentwise -- legitimate because the data-to-
        display transform here is axis-aligned with no rotation.  The payoff
        is that the arc on screen matches arctan(slope) no matter how
        unequally the two axes happen to be scaled.
        """
        ax = self.ax_top
        sx, sy = _px_per_data(ax)

        # The tangent direction, in pixels.  Using the pixel slope (rather
        # than the data slope) is what keeps the drawn angle honest.
        dxp, dyp = sx, slope * sy
        norm = math.hypot(dxp, dyp)
        if norm <= 0 or not math.isfinite(norm):
            self._hide_top_marks()
            return
        ux, uy = dxp / norm, dyp / norm          # unit vector, pixel space

        # A fixed pixel length means a near-vertical tangent cannot shoot off
        # the top of the panel the way a fixed *data* length would.
        dx = TANGENT_PX * ux / sx
        dy = TANGENT_PX * uy / sy
        self.tangent.set_data([x - dx, x + dx], [y - dy, y + dy])

        # The horizontal arm the angle is measured from, slightly longer than
        # the arc so the arc visibly lands on it.
        ref = ARC_PX * 1.5 / sx
        self.h_ref.set_data([x, x + ref], [y, y])

        # The arc itself: pixel-space circle from the horizontal round to the
        # tangent, mapped back to data units.
        theta_px = math.atan2(uy, ux)
        ts = np.linspace(0.0, theta_px, 48)
        self.arc.set_data(x + ARC_PX * np.cos(ts) / sx,
                          y + ARC_PX * np.sin(ts) / sy)

        # Label the true angle (from the data slope, not the pixel slope) on
        # the arc's bisector, far enough out that the text clears both the arc
        # and the tangent line instead of sitting on top of them.
        mid = 0.5 * theta_px
        label_r = ARC_PX + 46.0
        self.arc_label.set_position((x + label_r * math.cos(mid) / sx,
                                     y + label_r * math.sin(mid) / sy))
        self.arc_label.set_text(f"\u03b8 = {math.degrees(math.atan(slope)):.1f}\u00b0")

    def render(self, i: int) -> None:
        """Put the whole scene into the state it should be in at frame i."""
        t = self.track
        i = int(np.clip(i, 0, t.frames - 1))
        x, y, slope, theta = t.xs[i], t.fs[i], t.ds[i], t.thetas[i]

        # ---- bottom panel: extend the trace ------------------------------
        # Rewriting the slice (rather than appending) means frame 0 of a
        # repeat pass automatically wipes the previous sweep clean.
        self.trace_x[:i + 1] = t.xs[:i + 1]
        self.trace_y[:i + 1] = t.ds[:i + 1]
        self.trace_x[i + 1:] = np.nan
        self.trace_y[i + 1:] = np.nan
        self.trace.set_data(self.trace_x[:i + 1], self.trace_y[:i + 1])

        if math.isfinite(slope):
            self.dot_bot.set_data([x], [slope])
            self.drop_bot.set_data([x, x], [self.ax_bot.get_ylim()[0], slope])
        else:
            self.dot_bot.set_data([], [])
            self.drop_bot.set_data([], [])

        # ---- top panel: dot, tangent, arc --------------------------------
        if math.isfinite(y):
            self.dot_top.set_data([x], [y])
            self.drop_top.set_data([x, x], [self.ax_top.get_ylim()[0], y])
            if math.isfinite(slope):
                self._draw_tangent_and_angle(x, y, float(slope))
            else:
                for artist in (self.tangent, self.h_ref, self.arc):
                    artist.set_data([], [])
                self.arc_label.set_text("")
        else:
            self._hide_top_marks()

        # ---- the connector: same x, top dot to bottom dot ----------------
        if math.isfinite(y) and math.isfinite(slope):
            self.link.xy1 = (x, y)
            self.link.xy2 = (x, slope)
            self.link.set_visible(True)
        else:
            self.link.set_visible(False)

        # ---- the readout -------------------------------------------------
        self.card.set_text(self._card_text(x, y, slope, theta))

    def _condition(self, slope: float) -> str:
        """
        What the slope says about f here -- the thing you pause to check.

        'Stationary' is judged against the size of f' on this sweep rather than
        against exact zero, so floating-point dust either side of a turning
        point still reads as flat instead of flickering increasing/decreasing.
        """
        if not math.isfinite(slope):
            return "no tangent  (f' undefined)"
        if abs(slope) <= self.flat_eps:
            return "stationary  (f' = 0)"
        if slope > 0:
            return "increasing  (f' > 0)"
        return "decreasing  (f' < 0)"

    def _card_text(self, x: float, y: float, slope: float, theta: float) -> str:
        """The live numbers, with f'(x) = tan(theta) spelled out."""
        def num(value: float, spec: str = "9.3f") -> str:
            return format(value, spec) if math.isfinite(value) else f"{'undefined':>9}"

        angle = (f"{math.degrees(theta):9.3f}\u00b0"
                 if math.isfinite(theta) else f"{'undefined':>9}")
        return "\n".join([
            f"x            = {num(x)}",
            f"f(x)         = {num(y)}",
            f"f'(x) = tan \u03b8 = {num(slope)}",
            f"\u03b8            = {angle}",
            f"f is         = {self._condition(slope)}",
        ])

    def show(self) -> None:
        plt.show()


# ==========================================================================
# SECTION 6 - The animation
# ==========================================================================


def frame_sequence(frames: int, hold: int = HOLD_FRAMES) -> list[int]:
    """
    The sweep, then a rest on the final frame.

    Repeating the last index is what stops the animation snapping back to an
    empty panel the instant the derivative is finished: the completed curve
    sits there for about a second first, which is when it is actually worth
    looking at.
    """
    return list(range(frames)) + [frames - 1] * max(0, hold)


def animate(scene: TracerScene, interval: int = DEFAULT_INTERVAL,
            loop: bool = True) -> FuncAnimation:
    """
    Drive the scene with a clock, and hand the animation back (keep a reference!).

    The FuncAnimation here is deliberately a *timer*, not a playlist: it counts
    forever and ignores the index it is handed, because the scene owns the
    playhead.  Handing matplotlib a finite frame list instead would give it a
    second, private idea of where the sweep had got to, and scrubbing would be
    undone the moment the timer ticked again.
    """
    scene.loop = loop

    def step(_i):
        scene.advance()
        # blit is off, so the return value is decorative -- but returning the
        # touched artists keeps the door open for enabling it later.
        return (scene.trace, scene.dot_top, scene.dot_bot,
                scene.tangent, scene.arc, scene.link)

    return FuncAnimation(
        scene.fig, step, frames=itertools.count(), interval=max(1, int(interval)),
        blit=False, repeat=False, cache_frame_data=False, save_count=1,
    )


def save_animation(track: Track, interval: int, path_stem: str) -> int:
    """
    Render the sweep to a video file.

    Tries FFMpegWriter for an .mp4 and falls back to PillowWriter for a .gif
    when ffmpeg is not installed, so --save always produces *something*.
    """
    scene = TracerScene(track)
    sequence = frame_sequence(track.frames)

    def step(i: int):
        scene.render(i)
        return ()

    anim = FuncAnimation(scene.fig, step, frames=sequence,
                         interval=max(1, int(interval)), blit=False,
                         repeat=False, cache_frame_data=False)

    if FFMpegWriter.isAvailable():
        path, writer, dpi = f"{path_stem}.mp4", FFMpegWriter(fps=30, bitrate=3600), 100
    else:
        print("  (install ffmpeg for a smaller, smoother .mp4)")
        path, writer, dpi = f"{path_stem}.gif", PillowWriter(fps=25), 70

    total = len(sequence)
    print(f"  Rendering {total} frames to {path} ...")
    anim.save(path, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": BG},
              progress_callback=lambda i, n: (
                  print(f"\r    frame {i + 1}/{n or total}", end="", flush=True)))
    print(f"\n  Wrote {path}\n")
    plt.close(scene.fig)
    return 0


# ==========================================================================
# SECTION 7 - Running it
# ==========================================================================


def run_once(source: str, x_min: float, x_max: float, frames: int,
             interval: int = DEFAULT_INTERVAL, loop: bool = True,
             show_plot: bool = True, save: bool = False,
             widgets: bool = True) -> int:
    """Analyse one function, print the report, then animate it."""
    track = analyse(source, x_min, x_max, frames)
    print_report(track)

    if save:
        return save_animation(track, interval, "derivative_curve_tracer")
    if not show_plot:
        return 0

    scene = TracerScene(track, interactive=widgets, loop=loop)
    # The FuncAnimation must stay referenced or Python will garbage-collect it
    # and the figure will sit there perfectly still.
    anim = animate(scene, interval=interval, loop=loop)
    if loop:
        print("  Watch the amber curve get written, then the sweep replays.")
    else:
        print("  The sweep runs once and rests on the finished curve.")
    if widgets:
        print("  Pause and drag the x slider (or use the arrow keys) to park on")
        print("  any point and read its slope, angle and condition.")
    print("  Close the window to continue.\n")
    scene.show()
    del anim
    return 0


def _ask(prompt: str) -> str:
    """input() that treats EOF/Ctrl-C and 'quit' as a request to stop."""
    try:
        answer = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nGoodbye!")
    if answer.lower() in ("quit", "exit", "q"):
        raise SystemExit("Goodbye!")
    return answer


def repl(interval: int, loop: bool, show_plot: bool,
         widgets: bool = True) -> int:
    """Prompt for f(x), the range and the frame count, over and over."""
    while True:
        source = _ask("f(x)> ")
        if not source:
            continue

        range_text = _ask(f"x-range  (two numbers, blank = "
                          f"{DEFAULT_XMIN:g} {DEFAULT_XMAX:g})> ")
        if not range_text:
            x_min, x_max = DEFAULT_XMIN, DEFAULT_XMAX
        else:
            parts = range_text.replace(",", " ").split()
            if len(parts) != 2:
                print("  [Error] Give exactly two numbers, e.g. '-3 3' "
                      "or '0 6.283'.\n")
                continue
            try:
                x_min = parse_number(parts[0], "x_min")
                x_max = parse_number(parts[1], "x_max")
            except TracerError as exc:
                print(f"  [Error] {exc}\n")
                continue

        frames_text = _ask(f"frames   (blank = {DEFAULT_FRAMES})> ")
        if not frames_text:
            frames = DEFAULT_FRAMES
        else:
            try:
                frames = int(float(frames_text))
            except ValueError:
                print(f"  [Error] '{frames_text}' is not a frame count.\n")
                continue

        try:
            run_once(source, x_min, x_max, frames, interval=interval,
                     loop=loop, show_plot=show_plot, widgets=widgets)
        except TracerError as exc:
            print(f"  [Error] {exc}\n")
        except Exception as exc:  # noqa: BLE001 - never dump a traceback
            print(f"  [Error] {exc}\n")


# ==========================================================================
# SECTION 8 - Headless self-check
# ==========================================================================


def run_demo() -> int:
    """Verify the derivative, the sampling and the figure, without a window."""
    print("=" * 72)
    print("  DERIVATIVE CURVE TRACER - SELF CHECK")
    print("=" * 72)

    failures = 0

    # --- the symbolic derivative must be the right function ----------------
    cases = [
        ("x**2", "2*x"),
        ("x^2", "2*x"),                     # handwritten form parses the same
        ("sin(x)", "cos(x)"),
        ("exp(-x**2)", "-2*x*exp(-x^2)"),
        ("x**3 - 3*x", "3*x^2 - 3"),
        ("2sin(x) + x^2", "2*x + 2*cos(x)"),
        ("log(x)", "1/x"),
        ("tanh(x)", "1 - tanh(x)^2"),
    ]
    print("\n  Symbolic f'(x), and agreement with a central difference")
    print("  " + "-" * 68)
    for source, expected in cases:
        try:
            track = analyse(source, 0.3, 2.3, 60)
        except TracerError as exc:
            print(f"  FAIL  {source!r}: unexpected error: {exc}")
            failures += 1
            continue

        got = expr_text(track.f_prime)
        # Compare as mathematics, not as text: 'cos(x)' and 'cos(x)*1' agree.
        symbol_ok = sp.simplify(track.f_prime - parse_function(expected)) == 0

        # And compare against a numeric derivative of f, which is an entirely
        # independent route to the same numbers.
        f_num = make_numeric(track.f, "f")
        h = 1e-6
        probe = track.xs[2:-2]
        central = (f_num(probe + h) - f_num(probe - h)) / (2 * h)
        gap = np.nanmax(np.abs(central - track.ds[2:-2]))
        numeric_ok = bool(gap < 1e-5 * max(1.0, float(np.nanmax(np.abs(central)))))

        # theta must be consistent with the slope it came from.
        theta_ok = bool(np.allclose(np.tan(track.thetas), track.ds,
                                    rtol=1e-9, atol=1e-9, equal_nan=True))

        ok = symbol_ok and numeric_ok and theta_ok
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  d/dx[{source:<14}] = {got:<22} "
              f"max|symbolic - numeric| = {gap:.2e}")

    # --- the trace must build up, not appear all at once -------------------
    print("\n  The bottom panel fills in one point per frame")
    print("  " + "-" * 68)
    track = analyse("x**2", -3.0, 3.0, 40)
    scene = TracerScene(track)
    lengths = []
    for i in (0, 1, 10, 25, 39):
        scene.render(i)
        lengths.append(len(scene.trace.get_xdata()))
    grows = lengths == [1, 2, 11, 26, 40]
    failures += 0 if grows else 1
    print(f"  {'ok  ' if grows else 'FAIL'}  trace length by frame "
          f"[0, 1, 10, 25, 39] -> {lengths}")

    # a replay must wipe the previous sweep rather than draw over it
    scene.render(0)
    reset = len(scene.trace.get_xdata()) == 1
    failures += 0 if reset else 1
    print(f"  {'ok  ' if reset else 'FAIL'}  looping back to frame 0 clears "
          f"the trace ({len(scene.trace.get_xdata())} point)")

    # --- both dots must sit at the same x, and the connector join them -----
    scene.render(17)
    x_top = float(scene.dot_top.get_xdata()[0])
    x_bot = float(scene.dot_bot.get_xdata()[0])
    aligned = abs(x_top - x_bot) < 1e-12 and abs(scene.link.xy1[0] - x_top) < 1e-12
    failures += 0 if aligned else 1
    print(f"  {'ok  ' if aligned else 'FAIL'}  top dot x = {x_top:.6g}, "
          f"bottom dot x = {x_bot:.6g}, connector shares it")

    # --- the drawn angle must match arctan(f'(x)) on screen ---------------
    print("\n  The arc's on-screen angle equals arctan(f'(x))")
    print("  " + "-" * 68)
    scene.fig.canvas.draw()                 # settle the transforms
    worst = 0.0
    for i in (5, 12, 20, 33):
        scene.render(i)
        xs_t, ys_t = scene.tangent.get_data()
        sx, sy = _px_per_data(scene.ax_top)
        # slope of the drawn tangent, back in data units
        drawn = ((ys_t[1] - ys_t[0]) / (xs_t[1] - xs_t[0]))
        worst = max(worst, abs(drawn - track.ds[i]))
        # the arc must start on the horizontal and end on the tangent
        ax_, ay_ = scene.arc.get_data()
        end_px = ((ax_[-1] - track.xs[i]) * sx, (ay_[-1] - track.fs[i]) * sy)
        arc_angle = math.degrees(math.atan2(end_px[1], end_px[0]))
        tan_px = ((xs_t[1] - track.xs[i]) * sx, (ys_t[1] - track.fs[i]) * sy)
        tan_angle = math.degrees(math.atan2(tan_px[1], tan_px[0]))
        agree = abs(arc_angle - tan_angle) < 0.5
        failures += 0 if agree else 1
        print(f"  {'ok  ' if agree else 'FAIL'}  x = {track.xs[i]:>7.4g}  "
              f"arc ends at {arc_angle:>7.2f}\u00b0, tangent points at "
              f"{tan_angle:>7.2f}\u00b0")
    slope_ok = worst < 1e-9
    failures += 0 if slope_ok else 1
    print(f"  {'ok  ' if slope_ok else 'FAIL'}  drawn tangent slope matches "
          f"f'(x) to {worst:.2e}")
    plt.close(scene.fig)

    # --- gaps and corners must be survived, and noted ---------------------
    print("\n  Awkward functions are traced with gaps, not crashes")
    print("  " + "-" * 68)
    for source, why in [("Abs(x)", "corner at 0"),
                        ("sqrt(x)", "domain edge / vertical tangent"),
                        ("tan(x)", "poles"),
                        ("1/x", "pole at 0")]:
        try:
            tr = analyse(source, -3.0, 3.0, 120)
            sc = TracerScene(tr)
            for i in (0, tr.frames // 2, tr.frames - 1):
                sc.render(i)
            sc.fig.canvas.draw()
            plt.close(sc.fig)
            note = tr.notes[0][:44] + "..." if tr.notes else "no note"
            print(f"  ok    {source:<9} ({why:<30}) -> {note}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {source}: {type(exc).__name__}: {exc}")
            failures += 1

    # --- bad input must be refused, gracefully ---------------------------
    print("\n  Bad input must raise a clear TracerError, not crash")
    print("  " + "-" * 68)
    refusals = [
        ("x**2 +", -3.0, 3.0, 200, "syntax error"),
        ("y + 1", -3.0, 3.0, 200, "unknown symbol"),
        ("", -3.0, 3.0, 200, "empty input"),
        ("x**2", 2.0, 2.0, 200, "empty range"),
        ("x**2", -3.0, 3.0, 2, "too few frames"),
        ("x**2", -3.0, 3.0, 99999, "absurd frame count"),
        ("log(x)", -5.0, -1.0, 200, "nowhere real on the range"),
        ("x**2", float("nan"), 3.0, 200, "non-finite endpoint"),
    ]
    for source, lo, hi, n, why in refusals:
        try:
            analyse(source, lo, hi, n)
        except TracerError as exc:
            print(f"  ok    {source!r:<10} [{lo:g}, {hi:g}] n={n:<6} -> "
                  f"{str(exc)[:38]}...")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {source!r}: wrong exception type: "
                  f"{type(exc).__name__}: {exc}")
            failures += 1
            continue
        print(f"  FAIL  {source!r} should have been refused ({why})")
        failures += 1

    # a reversed range is silently corrected rather than refused
    fixed = analyse("x**2", 3.0, -3.0, 40)
    ok = fixed.x_min == -3.0 and fixed.x_max == 3.0
    failures += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'}  reversed range 3..-3 corrected to "
          f"{fixed.x_min:g}..{fixed.x_max:g}")

    # --- the animation itself must run -----------------------------------
    print("\n  Animation")
    print("  " + "-" * 68)
    seq = frame_sequence(50)
    held = len(seq) == 50 + HOLD_FRAMES and seq[-1] == 49 and seq[49] == 49
    failures += 0 if held else 1
    print(f"  {'ok  ' if held else 'FAIL'}  sweep of 50 frames + {HOLD_FRAMES} "
          f"holding frames = {len(seq)} total (rests before looping)")

    for loop in (True, False):
        try:
            tr = analyse("sin(x)", -6.283, 6.283, 30)
            sc = TracerScene(tr, loop=loop)
            anim = animate(sc, interval=10, loop=loop)
            # Draw first, then reset: the animation's initial draw fires the
            # step function once by itself, so counting ticks from an assumed
            # zero without flushing that would be off by one.
            sc.fig.canvas.draw()
            sc.set_cursor(0)
            sc._hold_left = sc.hold
            sc.set_playing(True)
            # Exactly enough ticks to walk to the last frame, sit out the rest,
            # and take one more step -- the step that must wrap (loop) or stop.
            for _ in range((tr.frames - 1) + sc.hold + 1):
                sc.advance()
            if loop:
                right = sc.cursor == 0 and sc.playing
                why = f"wrapped to frame {sc.cursor}, still playing"
            else:
                right = sc.cursor == tr.frames - 1 and not sc.playing
                why = (f"rested on frame {sc.cursor} of {tr.frames - 1} "
                       f"then paused itself")
            failures += 0 if right else 1
            print(f"  {'ok  ' if right else 'FAIL'}  clock ran a full pass "
                  f"(loop={loop}): {why}")
            del anim
            plt.close(sc.fig)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  animation (loop={loop}): {type(exc).__name__}: {exc}")
            failures += 1

    # --- pause and scrub: the controls that let you inspect one point -----
    print("\n  Pause, scrub and step")
    print("  " + "-" * 68)
    # 100 frames over [-3, 3] puts a sample exactly on x = 1, where
    # f'(x) = 3x^2 - 3 vanishes -- so the stationary case can be checked
    # against a real zero rather than something merely close to one.
    tr = analyse("x**3 - 3*x", -3.0, 3.0, 100)
    sc = TracerScene(tr, interactive=True)

    # pausing must stop the clock without moving the playhead
    sc.set_cursor(40)
    sc._on_play_clicked(None)
    stopped = (not sc.playing and sc.cursor == 40
               and sc.play_button.label.get_text() == "Play")
    for _ in range(25):
        sc.advance()                        # ticks must be ignored while paused
    stopped = stopped and sc.cursor == 40
    failures += 0 if stopped else 1
    print(f"  {'ok  ' if stopped else 'FAIL'}  paused at frame 40, 25 clock "
          f"ticks later still at frame {sc.cursor}, button says "
          f"{sc.play_button.label.get_text()!r}")

    # scrubbing must land on a nearest sampled x, and the trace must follow.
    # "A" nearest, not "the": a target can fall exactly halfway between two
    # samples, and either neighbour is then an equally correct answer.
    scrub_ok = True
    for target in (-3.0, -1.4142, 0.0, 1.0, 2.75, 3.0):
        sc.slider.set_val(target)           # as a mouse drag would
        gaps = np.abs(tr.xs - target)
        landed = abs(gaps[sc.cursor] - gaps.min()) < 1e-12
        drawn = len(sc.trace.get_xdata()) == sc.cursor + 1
        scrub_ok = scrub_ok and landed and drawn
        if not (landed and drawn):
            print(f"  FAIL  scrub to x = {target:g}: landed on frame "
                  f"{sc.cursor} at x = {tr.xs[sc.cursor]:g}, trace "
                  f"{len(sc.trace.get_xdata())} points")
    failures += 0 if scrub_ok else 1
    if scrub_ok:
        print("  ok    scrubbing to 6 x-values landed on the nearest frame "
              "each time, forwards and backwards")

    # scrubbing backwards must shorten the trace, not leave a stale tail
    sc.slider.set_val(tr.xs[80])
    long_trace = len(sc.trace.get_xdata())
    sc.slider.set_val(tr.xs[12])
    shrank = len(sc.trace.get_xdata()) == 13 and long_trace == 81
    failures += 0 if shrank else 1
    print(f"  {'ok  ' if shrank else 'FAIL'}  dragging back from frame 80 to "
          f"12 shortened the trace {long_trace} -> "
          f"{len(sc.trace.get_xdata())} points")

    # the arrow keys must step, and clamp at both ends rather than wrap
    class _Key:                             # Agg has no keyboard; fake the event
        def __init__(self, key): self.key = key

    sc.set_cursor(50)
    sc._on_key(_Key("right"))
    sc._on_key(_Key("right"))
    sc._on_key(_Key("shift+right"))
    stepped = sc.cursor == 50 + 2 + BIG_STEP
    sc.set_cursor(0)
    sc._on_key(_Key("left"))                # must not wrap round to the end
    at_start = sc.cursor == 0
    sc.set_cursor(tr.frames - 1)
    sc._on_key(_Key("right"))               # must not run off the end
    at_end = sc.cursor == tr.frames - 1
    sc._on_key(_Key("home"))
    home_ok = sc.cursor == 0
    sc._on_key(_Key("end"))
    end_ok = sc.cursor == tr.frames - 1
    keys_ok = stepped and at_start and at_end and home_ok and end_ok
    failures += 0 if keys_ok else 1
    print(f"  {'ok  ' if keys_ok else 'FAIL'}  arrows step (+1,+1,+{BIG_STEP}) "
          f"and clamp at both ends; home/end jump to 0 and {tr.frames - 1}")

    # the slider must stay in step when the clock (not the mouse) moves things
    sc.set_cursor(0)
    sc.set_playing(True)
    for _ in range(30):
        sc.advance()
    synced = abs(sc.slider.val - tr.xs[sc.cursor]) < 0.5 * sc.x_step
    failures += 0 if synced else 1
    print(f"  {'ok  ' if synced else 'FAIL'}  after 30 ticks the slider reads "
          f"x = {sc.slider.val:.4g} and the dot sits at "
          f"x = {tr.xs[sc.cursor]:.4g}")

    # the card must name the condition, and get it right at a turning point
    turning = int(np.argmin(np.abs(tr.xs - 1.0)))       # f'(1) = 0 exactly
    middle = int(np.argmin(np.abs(tr.xs - 0.0)))        # steepest descent
    sc.set_cursor(turning)
    flat = "stationary" in sc.card.get_text()
    sc.set_cursor(tr.frames - 1)
    rising = "increasing" in sc.card.get_text()
    sc.set_cursor(middle)
    falling = "decreasing" in sc.card.get_text()
    told = flat and rising and falling
    failures += 0 if told else 1
    print(f"  {'ok  ' if told else 'FAIL'}  the card reads stationary at "
          f"x = {tr.xs[turning]:.3g} (f' = {tr.ds[turning]:.2g}), increasing "
          f"at x = {tr.x_max:g}, decreasing at x = {tr.xs[middle]:.3g}")

    # Reset must rewind to the start, paused
    sc._on_reset_clicked(None)
    rewound = sc.cursor == 0 and not sc.playing
    failures += 0 if rewound else 1
    print(f"  {'ok  ' if rewound else 'FAIL'}  Reset rewinds to frame "
          f"{sc.cursor}, paused")

    # closing must hand the arrow keys back to matplotlib's navigation.
    # Agg never emits close_event, so the handler is called directly here --
    # what is being checked is that the restore itself is correct and that the
    # scene really did borrow the bindings in the first place.
    taken = list(sc._keymap_taken)
    borrowed = bool(taken) and not any(
        key in plt.rcParams[param] for param, key in taken)
    sc._on_close(None)
    restored = borrowed and all(key in plt.rcParams[param] for param, key in taken)
    sc.release_keys()                       # must be safe to call twice
    plt.close(sc.fig)
    failures += 0 if restored else 1
    print(f"  {'ok  ' if restored else 'FAIL'}  {len(taken)} key binding(s) "
          f"borrowed from matplotlib's navigation, then given back on close")

    # and the non-interactive scene -- the one --save uses -- has no widgets
    plain = TracerScene(analyse("x**2", -2.0, 2.0, 20))
    bare = (getattr(plain, "slider", None) is None
            and getattr(plain, "play_button", None) is None)
    plain.render(9)                         # still perfectly renderable
    failures += 0 if bare else 1
    plt.close(plain.fig)
    print(f"  {'ok  ' if bare else 'FAIL'}  a non-interactive scene builds no "
          f"widgets, so --save and --demo are untouched")

    print("\n" + "=" * 72)
    if failures:
        print(f"  {failures} check(s) FAILED")
    else:
        print("  All checks passed.")
    print("=" * 72)
    return 1 if failures else 0


# ==========================================================================
# SECTION 9 - Command line
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Animate the first derivative: a dot walks along f(x) "
                    "carrying its tangent, and the panel below plots the "
                    "slope it reports, stroke by stroke.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1],
    )
    parser.add_argument("--func", "-f", metavar="EXPR",
                        help='the function of x, e.g. "x**2" or "sin(x)"')
    parser.add_argument("--xmin", type=float, default=DEFAULT_XMIN, metavar="A",
                        help=f"left end of the sweep (default: {DEFAULT_XMIN:g})")
    parser.add_argument("--xmax", type=float, default=DEFAULT_XMAX, metavar="B",
                        help=f"right end of the sweep (default: {DEFAULT_XMAX:g})")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, metavar="N",
                        help=f"steps in the sweep (default: {DEFAULT_FRAMES})")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        metavar="MS",
                        help="milliseconds per frame; lower is faster "
                             f"(default: {DEFAULT_INTERVAL})")
    parser.add_argument("--once", action="store_true",
                        help="run the sweep once and rest on the finished "
                             "curve instead of looping")
    parser.add_argument("--save", action="store_true",
                        help="render to derivative_curve_tracer.mp4 (or .gif) "
                             "instead of opening a window")
    parser.add_argument("--no-plot", action="store_true", dest="no_plot",
                        help="console report only, no window")
    parser.add_argument("--no-widgets", action="store_true", dest="no_widgets",
                        help="hide the pause/scrub controls for a clean, "
                             "screenshot-friendly window")
    parser.add_argument("--demo", action="store_true",
                        help="headless self-check, no windows")
    return parser


def banner() -> None:
    print("=" * 72)
    print("  DERIVATIVE CURVE TRACER")
    print("=" * 72)
    print("  f'(x) = tan(theta), the slope of the tangent -- one number per x.")
    print("  Collect those numbers as you slide along f(x) and you have drawn")
    print("  the derivative function.")
    print()
    print("  Enter a function of x, e.g.:")
    print("    x**2              sin(x)          exp(-x**2)")
    print("    x^3 - 3x          tanh(x)         x*sin(x)")
    print("  Type 'quit' at any prompt to stop.")
    print()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.demo:
        return run_demo()

    banner()

    if args.func is not None:
        try:
            return run_once(args.func, args.xmin, args.xmax, args.frames,
                            interval=args.interval, loop=not args.once,
                            show_plot=not args.no_plot, save=args.save,
                            widgets=not args.no_widgets)
        except TracerError as exc:
            print(f"  [Error] {exc}\n")
            return 1
        except Exception as exc:  # noqa: BLE001 - never dump a traceback
            print(f"  [Error] {exc}\n")
            return 1

    return repl(interval=args.interval, loop=not args.once,
                show_plot=not args.no_plot, widgets=not args.no_widgets)


if __name__ == "__main__":
    raise SystemExit(main())
