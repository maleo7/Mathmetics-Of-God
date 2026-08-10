#!/usr/bin/env python3
"""
Derivative Limit Visualiser  (the limit definition, drawn and dragged)
------------------------------------------------------------------------------
Type a function f(x) and a point a.  The program computes the derivative two
completely different ways, prints them side by side, and then *draws* the
reason they agree.

  CONSOLE : the exact symbolic derivative f'(x) = d/dx f(x) evaluated at a,
            the limit statement with the real numbers substituted in, and a
            convergence table showing the difference quotient crawling toward
            that exact answer as h shrinks.
  FIGURE  : the curve, the point (a, f(a)), the true tangent line, and secant
            lines for the biggest h values in the table -- plus a slider that
            drags h continuously toward 0 so you can watch a secant line fall
            onto the tangent in real time.

THE MATHEMATICS
===============
The derivative is defined as the limit of the slope of a secant line:

                             f(a + h) - f(a)
        f'(a)  =   lim       ---------------
                   h -> 0           h

Each individual h gives a *secant*: the straight line through the two points
(a, f(a)) and (a+h, f(a+h)).  Its slope is exactly the fraction above -- a
rise over a run, nothing more.  The limit says: slide the second point toward
the first, and those secant slopes settle on one number.  That number is the
slope of the *tangent*, the line that best matches the curve at a.

So there are three objects and they are all the same thing:

        the limit          (an algebraic statement)
        the tangent slope  (a geometric statement)
        f'(a)              (the symbolic derivative, evaluated)

This program computes all three independently and shows they coincide.

WHY THE TABLE MATTERS
=====================
The forward difference quotient [f(a+h) - f(a)] / h has error proportional to
h, so each 10x shrink of h buys roughly one extra correct digit -- you can
watch that happen column by column.  The central difference

        [f(a+h) - f(a-h)] / (2h)

has error proportional to h^2 and is printed alongside to make the point that
*how* you approach the limit changes how fast you get there, even though both
land on the same value.

WHEN THE LIMIT DOES NOT EXIST
=============================
A derivative fails to exist exactly when the secant slopes disagree depending
on which side you approach from, or blow up.  Before trusting sympy, the
program probes both sides:

        left  = [f(a) - f(a-h)] / h
        right = [f(a+h) - f(a)] / h        with a tiny h

If those two disagree, there is a corner at a and f'(a) genuinely does not
exist -- so the program says so instead of drawing a meaningless tangent.
(This matters: sympy will happily hand you d/dx|x| = sign(x) and evaluate it
to 0 at the origin, which is not a derivative, it is an artefact.)

Usage:
    python derivative_limit_visualiser.py                       # prompts + live slider
    python derivative_limit_visualiser.py --func "sin(x)" --at 0
    python derivative_limit_visualiser.py --func "x**2" --at 3 --h 2 --static
    python derivative_limit_visualiser.py --func "exp(x)" --at 1 --no-plot
    python derivative_limit_visualiser.py --demo                # headless self-check
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.widgets import Button, Slider
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

# ==========================================================================
# SECTION 0 - Constants and shared vocabulary
# ==========================================================================

X = sp.Symbol("x", real=True)          # the one and only variable we allow
H = sp.Symbol("h", positive=True)      # the step size, used symbolically

DEFAULT_H = 1.0                        # starting step size if the user gives none
TABLE_ROWS = 6                         # how many 10x shrinks to tabulate
MIN_H_EXPONENT = -6.0                  # slider bottoms out at h = 1e-6
PROBE_H = 1e-6                         # step used for the two-sided corner probe
CORNER_TOL = 1e-3                      # relative gap that counts as a corner
MISMATCH_TOL = 1e-3                    # symbolic vs numeric disagreement warning
CURVE_SAMPLES = 1400                   # resolution of the plotted curve

# Colours, kept in one place so the figure stays coherent.
C_CURVE = "#1f6fb4"      # the function itself
C_POINT = "#c62828"      # the point (a, f(a))
C_TANGENT = "#c62828"    # the true tangent line
C_LIVE = "#7b2fbe"       # the slider-driven secant (must not clash with below)
C_SECANT = "#ef7d00"     # first frozen reference secant (largest h)
C_SECANT_2 = "#2e9e5b"   # second frozen reference secant
C_GUIDE = "#8a8a8a"      # rise/run construction lines

# ==========================================================================
# SECTION 1 - Parsing: turn user text into a sympy expression, safely
# ==========================================================================


class DerivativeError(Exception):
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
    "floor": sp.floor, "ceiling": sp.ceiling, "factorial": sp.factorial,
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
    "factorial": sp.factorial,
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
        raise DerivativeError("Please enter a function of x, e.g. x**2 + 3*x")

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
        raise DerivativeError(
            f"Could not read '{text.strip()}' as a function of x ({exc})."
        ) from exc

    if not isinstance(expr, sp.Expr) or isinstance(expr, sp.logic.boolalg.Boolean):
        raise DerivativeError(
            "That is not an expression in x. Enter just the right-hand side, "
            "e.g. 'x**2 + 3*x' rather than 'y = x**2 + 3*x'."
        )

    unknown = sorted(s.name for s in expr.free_symbols if s.name != "x")
    if unknown:
        raise DerivativeError(
            f"Unknown name(s): {', '.join(unknown)}. "
            "The function may only use x and the standard functions "
            "(sin, cos, tan, exp, log, sqrt, ...)."
        )

    # Re-bind whatever 'x' we got onto our own real-valued symbol, so later
    # simplifications know x is real.
    return expr.subs({s: X for s in expr.free_symbols})


def expr_text(expr: sp.Expr) -> str:
    """A compact, plain-text rendering, e.g. 'x^2 + 3*x'. Safe in any font."""
    return sp.sstr(expr).replace("**", "^")


# ==========================================================================
# SECTION 2 - Evaluation: exact where it matters, numeric where it is fast
# ==========================================================================


def evaluate_at(expr: sp.Expr, value: float, what: str = "f(x)") -> float:
    """
    Evaluate an expression at a point and insist the answer is a real number.

    This is the gate that catches poles (1/x at 0 -> complex infinity) and
    excursions off the domain (log(x) at -1 -> a complex number).
    """
    try:
        exact = expr.subs(X, sp.Float(value))
        result = complex(sp.N(exact, 20))
    except Exception as exc:  # noqa: BLE001 - zoo/nan/oo all land here
        raise DerivativeError(
            f"{what} is undefined at x = {value:g} "
            f"(the expression blows up or is not a number there)."
        ) from exc

    if abs(result.imag) > 1e-9 * max(1.0, abs(result.real)):
        raise DerivativeError(
            f"{what} is not real at x = {value:g} "
            f"(it evaluates to {result.real:.6g}{result.imag:+.6g}i). "
            "Pick a point inside the real domain of the function."
        )

    real = result.real
    if not math.isfinite(real):
        raise DerivativeError(f"{what} is infinite at x = {value:g}.")
    return real


def make_numeric(expr: sp.Expr) -> Callable[[np.ndarray], np.ndarray]:
    """
    Compile the expression into a fast numpy function.

    The wrapper never raises: anything outside the domain (log of a negative,
    a division by zero, an overflow) comes back as NaN.  matplotlib simply
    leaves a gap in the curve there, which is exactly the honest picture.
    """
    try:
        compiled = sp.lambdify(X, expr, modules=["numpy"])
    except Exception as exc:  # noqa: BLE001
        raise DerivativeError(f"Could not evaluate f(x) numerically ({exc}).") from exc

    def elementwise(values: np.ndarray) -> np.ndarray:
        """Fallback path for expressions that refuse to vectorise."""
        out = np.empty(values.shape, dtype=float)
        for i, v in enumerate(values.ravel()):
            try:
                out.ravel()[i] = float(compiled(v))
            except Exception:  # noqa: BLE001
                out.ravel()[i] = np.nan
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
        out = np.broadcast_to(np.asarray(raw, dtype=float), arr.shape).astype(float)
        out = out.copy()
        out[~np.isfinite(out)] = np.nan
        return out

    return evaluate


def numeric_value(evaluate: Callable, value: float) -> float:
    """Evaluate the compiled function at a single point."""
    return float(evaluate(np.array([float(value)]))[0])


# ==========================================================================
# SECTION 3 - Does the derivative actually exist here?
# ==========================================================================


def assert_differentiable(evaluate: Callable, a: float, f_a: float) -> None:
    """
    Probe the secant slope from the left and from the right with a tiny h.

    For a differentiable point the two agree to many digits.  When they do
    not, the limit definition itself has failed and there is no tangent to
    draw -- a corner (|x| at 0), a jump, or a domain edge.
    """
    f_right = numeric_value(evaluate, a + PROBE_H)
    f_left = numeric_value(evaluate, a - PROBE_H)

    if not math.isfinite(f_right) or not math.isfinite(f_left):
        side = "right" if not math.isfinite(f_right) else "left"
        raise DerivativeError(
            f"f(x) is not defined just to the {side} of x = {a:g}, so the "
            f"limit defining f'({a:g}) cannot be taken from both sides. "
            "This usually means a is on the edge of the domain."
        )

    slope_right = (f_right - f_a) / PROBE_H
    slope_left = (f_a - f_left) / PROBE_H

    scale = max(1.0, abs(slope_left), abs(slope_right))
    if abs(slope_right - slope_left) > CORNER_TOL * scale:
        raise DerivativeError(
            f"f(x) has a corner at x = {a:g}: the secant slope approaches "
            f"{slope_left:.6g} from the left but {slope_right:.6g} from the "
            f"right. The two-sided limit does not exist, so f'({a:g}) does "
            "not exist either."
        )


# ==========================================================================
# SECTION 4 - The convergence table and the substituted limit statement
# ==========================================================================


@dataclass
class Row:
    """One line of the shrinking-h table."""
    h: float
    f_plus: float
    forward: float
    central: float
    error: float


def convergence_table(evaluate: Callable, a: float, f_a: float,
                      h0: float, exact: float, rows: int = TABLE_ROWS) -> list[Row]:
    """Build the h = h0, h0/10, h0/100, ... difference-quotient table."""
    table: list[Row] = []
    for i in range(rows):
        h = h0 * (10.0 ** -i)
        f_plus = numeric_value(evaluate, a + h)
        f_minus = numeric_value(evaluate, a - h)

        forward = (f_plus - f_a) / h if math.isfinite(f_plus) else math.nan
        central = ((f_plus - f_minus) / (2.0 * h)
                   if math.isfinite(f_plus) and math.isfinite(f_minus) else math.nan)
        error = abs(forward - exact) if math.isfinite(forward) else math.nan

        table.append(Row(h=h, f_plus=f_plus, forward=forward,
                         central=central, error=error))
    return table


def point_form(a: float) -> sp.Expr:
    """
    Pick a readable exact stand-in for the point a.

    A tidy fraction is worth having (0.5 -> 1/2, 2.0 -> 2) because it keeps
    the algebra exact.  But nsimplify turns 1.0471975512 into
    1308996939/1250000000, which is exact and completely unreadable -- so
    past a modest denominator we keep the decimal instead.
    """
    exact = sp.nsimplify(sp.Float(a), rational=True)
    if isinstance(exact, sp.Rational) and abs(exact.q) <= 100:
        return exact
    return sp.Float(a)


def limit_story(f: sp.Expr, a: float, slope: float) -> list[str]:
    """
    Write out the limit definition with the user's actual numbers in it, and
    where possible show the algebra collapsing, e.g. for f(x) = x^2 + 3x at 2:

        f'(2) = lim(h->0) [ f(2 + h) - f(2) ] / h
              = lim(h->0) [ ((h + 2)^2 + 3*h + 6) - (10) ] / h
              = lim(h->0) (h + 7)
              = 7
    """
    a_exact = point_form(a)
    a_txt = expr_text(a_exact)

    lines = [f"f'({a_txt}) = lim(h->0) [ f({a_txt} + h) - f({a_txt}) ] / h"]
    pad = " " * (len(f"f'({a_txt})"))

    # Only attempt the symbolic collapse on modest expressions; simplify() can
    # be slow, and a monstrous one-liner would not be readable anyway.
    if sp.count_ops(f) <= 25:
        try:
            f_at_a = sp.simplify(f.subs(X, a_exact))
            f_at_ah = sp.expand(f.subs(X, a_exact + H))
            lines.append(
                f"{pad} = lim(h->0) [ ({expr_text(f_at_ah)}) "
                f"- ({expr_text(f_at_a)}) ] / h"
            )
            collapsed = sp.simplify(sp.expand((f_at_ah - f_at_a) / H))
            if collapsed.has(H):
                # Parenthesise a sum so 'lim(h->0) h + 7' cannot be misread
                # as '(lim h) + 7'.
                shown = expr_text(collapsed)
                if collapsed.is_Add:
                    shown = f"({shown})"
                lines.append(f"{pad} = lim(h->0) {shown}")
        except Exception:  # noqa: BLE001 - the algebra display is a bonus
            pass

    lines.append(f"{pad} = {slope:.10g}")
    return lines


# ==========================================================================
# SECTION 5 - Tie it together
# ==========================================================================


@dataclass
class Report:
    """Everything the console and the figure need, computed once."""
    source: str
    f: sp.Expr
    f_prime: sp.Expr
    a: float
    f_a: float
    slope: float
    h0: float
    table: list[Row]
    evaluate: Callable
    story: list[str] = field(default_factory=list)
    note: str = ""


def analyse(source: str, a: float, h0: float = DEFAULT_H) -> Report:
    """Parse, validate, differentiate symbolically, and tabulate numerically."""
    if not math.isfinite(a):
        raise DerivativeError("The point a must be a finite number.")
    if not math.isfinite(h0) or h0 <= 0:
        raise DerivativeError("The step size h must be a positive, finite number.")

    f = parse_function(source)

    # 1. Is f even defined (and real) at a?
    f_a = evaluate_at(f, a, "f(x)")

    # 2. Compile for the numeric side of the comparison.
    evaluate = make_numeric(f)
    if not math.isfinite(numeric_value(evaluate, a)):
        raise DerivativeError(
            f"f(x) could not be evaluated numerically at x = {a:g}."
        )

    # 3. Does the two-sided limit stand a chance?
    assert_differentiable(evaluate, a, f_a)

    # 4. The exact answer, symbolically.
    try:
        f_prime = sp.diff(f, X)
    except Exception as exc:  # noqa: BLE001
        raise DerivativeError(f"Could not differentiate f(x) ({exc}).") from exc
    try:
        f_prime = sp.simplify(f_prime)
    except Exception:  # noqa: BLE001 - tidier output is optional
        pass

    slope = evaluate_at(f_prime, a, "f'(x)")

    # 5. The same answer, numerically, from the limit definition.
    table = convergence_table(evaluate, a, f_a, h0, slope)

    # 6. Cross-check the two routes and note (do not fail on) any disagreement.
    note = ""
    probe = 1e-5
    left = numeric_value(evaluate, a - probe)
    right = numeric_value(evaluate, a + probe)
    if math.isfinite(left) and math.isfinite(right):
        central = (right - left) / (2.0 * probe)
        if abs(central - slope) > MISMATCH_TOL * max(1.0, abs(slope)):
            note = (
                f"Warning: the numeric slope near x = {a:g} is {central:.6g}, "
                f"which disagrees with the symbolic f'({a:g}) = {slope:.6g}. "
                "Treat this point with suspicion -- f(x) may be badly behaved here."
            )

    return Report(source=source.strip(), f=f, f_prime=f_prime, a=a, f_a=f_a,
                  slope=slope, h0=h0, table=table, evaluate=evaluate,
                  story=limit_story(f, a, slope), note=note)


# ==========================================================================
# SECTION 6 - Console output
# ==========================================================================


def _cell(value: float, width: int, spec: str = ".10g") -> str:
    """Format a number for the table, or say so when it does not exist."""
    if not math.isfinite(value):
        return f"{'undefined':>{width}}"
    return f"{format(value, spec):>{width}}"


def print_report(report: Report) -> None:
    a, slope = report.a, report.slope

    # Line the four labels up on one column, whatever width 'a' happens to be.
    labels = ["f(x)", "f'(x)", f"f({a:g})", f"f'({a:g})"]
    w = max(len(name) for name in labels)

    print()
    print(f"  {labels[0]:<{w}}  = {expr_text(report.f)}")
    print(f"  {labels[1]:<{w}}  = {expr_text(report.f_prime)}"
          f"      (the symbolic derivative)")
    print(f"  {labels[2]:<{w}}  = {report.f_a:.10g}")
    print(f"  {labels[3]:<{w}}  = {slope:.10g}      <-- the answer, exactly")
    print()
    print("  THE LIMIT DEFINITION, WITH YOUR NUMBERS")
    print("  " + "-" * 68)
    for line in report.story:
        print(f"    {line}")
    print()
    print("  SHRINKING h TOWARD 0")
    print("  " + "-" * 68)
    print(f"    {'h':>10}  {'f(a+h)':>16}  {'[f(a+h)-f(a)]/h':>18}"
          f"  {'|error|':>12}  {'central diff':>16}")
    print(f"    {'-' * 10}  {'-' * 16}  {'-' * 18}  {'-' * 12}  {'-' * 16}")
    for row in report.table:
        print(f"    {row.h:>10.4g}  {_cell(row.f_plus, 16)}"
              f"  {_cell(row.forward, 18)}  {_cell(row.error, 12, '.3e')}"
              f"  {_cell(row.central, 16)}")
    print(f"    {'-' * 10}  {'-' * 16}  {'-' * 18}  {'-' * 12}  {'-' * 16}")
    print(f"    {'h -> 0':>10}  {'':>16}  {slope:>18.10g}  {0.0:>12.3e}"
          f"  {slope:>16.10g}")
    print()
    print("    The forward quotient's error falls like h: one more correct digit")
    print("    per row. The central difference's error falls like h^2, so it")
    print("    closes in twice as fast. Both approach the same exact value.")

    if report.note:
        print()
        print(f"  [Note] {report.note}")
    print()


# ==========================================================================
# SECTION 7 - The figure: curve, point, tangent, secants, and a live slider
# ==========================================================================


class DerivativeScene:
    """
    Draws the geometry of the limit definition.

    Static artists (the curve, the point, the tangent) are created once.  The
    live secant, its rise/run construction and the readout are the only things
    the slider touches, so dragging stays smooth.
    """

    def __init__(self, report: Report, interactive: bool = True):
        self.report = report
        self.interactive = interactive
        self.slider = None
        self.play_button = None
        self.reset_button = None
        self.timer = None
        self.playing = False

        self._build_figure()
        self._draw_static()
        if interactive:
            self._build_widgets()
            self._update(self.slider.val)
        else:
            self._draw_live_secant(report.h0)

    # ---------------------------------------------------------------- setup

    def _build_figure(self) -> None:
        r = self.report
        self.fig, self.ax = plt.subplots(figsize=(11.5, 7.6))
        bottom = 0.24 if self.interactive else 0.10
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=bottom)

        try:
            self.fig.canvas.manager.set_window_title(
                f"Derivative of {expr_text(r.f)} at x = {r.a:g}"
            )
        except Exception:  # noqa: BLE001 - headless backends have no manager
            pass

    def _sample_curve(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample f over a window centred on a and wide enough to hold a + h."""
        r = self.report
        span = max(3.0, 2.5 * abs(r.h0))
        xs = np.linspace(r.a - span, r.a + span, CURVE_SAMPLES)
        return xs, r.evaluate(xs), span

    def _draw_static(self) -> None:
        r = self.report
        ax = self.ax
        xs, ys, span = self._sample_curve()
        self.xs, self.span = xs, span

        finite = ys[np.isfinite(ys)]
        if finite.size == 0:
            raise DerivativeError(
                f"f(x) has no finite values near x = {r.a:g}, so there is "
                "nothing to plot."
            )

        ax.axhline(0, color="black", linewidth=1.0, zorder=1)
        ax.axvline(0, color="black", linewidth=1.0, zorder=1)
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)

        # --- the function ------------------------------------------------
        ax.plot(xs, ys, color=C_CURVE, linewidth=2.4, zorder=4,
                label=f"f(x) = {expr_text(r.f)}")

        # --- the true tangent: y = f(a) + f'(a)(x - a) --------------------
        tangent = r.f_a + r.slope * (xs - r.a)
        ax.plot(xs, tangent, color=C_TANGENT, linewidth=2.0, zorder=6,
                label=f"tangent: slope f'({r.a:g}) = {r.slope:.6g}")

        # --- reference secants for the largest h values in the table ------
        for row, colour in zip(r.table[:2], (C_SECANT, C_SECANT_2)):
            if not math.isfinite(row.f_plus) or not math.isfinite(row.forward):
                continue
            secant = r.f_a + row.forward * (xs - r.a)
            ax.plot(xs, secant, color=colour, linewidth=1.6, linestyle="--",
                    alpha=0.85, zorder=5,
                    label=f"secant h = {row.h:g}  (slope {row.forward:.6g})")
            ax.plot(r.a + row.h, row.f_plus, "o", color=colour,
                    markersize=7, zorder=7)

        # --- the point the whole story is about ---------------------------
        ax.plot(r.a, r.f_a, "o", color=C_POINT, markersize=10,
                markeredgecolor="white", markeredgewidth=1.4, zorder=9)
        ax.annotate(f"({r.a:g}, {r.f_a:.4g})",
                    xy=(r.a, r.f_a), xytext=(12, 14),
                    textcoords="offset points", color=C_POINT,
                    fontsize=10, fontweight="bold", zorder=10)

        # --- framing -------------------------------------------------------
        extras = [r.f_a] + [row.f_plus for row in r.table[:2]
                            if math.isfinite(row.f_plus)]
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(*self._nice_ylim(finite, extras))

        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(
            f"f(x) = {expr_text(r.f)}          "
            f"f'({r.a:g}) = {r.slope:.6g}\n"
            "the secant slope [f(a+h) - f(a)] / h becomes the tangent slope as h -> 0",
            fontsize=12, fontweight="bold", pad=14,
        )

        # --- the live secant and its rise/run construction -----------------
        # Deliberately a different colour from the frozen reference secants,
        # so at startup (where it sits on top of the h = h0 one) it is still
        # obvious which line the slider is moving.
        live_label = "live secant (slider)" if self.interactive else "_nolegend_"
        self.secant_line, = ax.plot([], [], color=C_LIVE, linewidth=2.8,
                                    zorder=8, label=live_label)
        self.secant_point, = ax.plot([], [], "o", color=C_LIVE, markersize=9,
                                     markeredgecolor="white", markeredgewidth=1.2,
                                     zorder=9)
        self.run_line, = ax.plot([], [], color=C_GUIDE, linewidth=1.2,
                                 linestyle=":", zorder=7)
        self.rise_line, = ax.plot([], [], color=C_GUIDE, linewidth=1.2,
                                  linestyle=":", zorder=7)

        ax.legend(loc="best", fontsize=9, framealpha=0.92)

        # --- the formula card ---------------------------------------------
        self.card = ax.text(
            0.015, 0.985, self._card_text(r.h0),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace", zorder=11,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#fffdf5",
                      edgecolor=C_TANGENT, alpha=0.94),
        )

    @staticmethod
    def _nice_ylim(finite: np.ndarray, extras: list[float]) -> tuple[float, float]:
        """
        Clip to the middle 96% of the sampled values so a nearby asymptote
        cannot squash the interesting part of the curve into a flat line.
        """
        lo = float(np.percentile(finite, 2.0))
        hi = float(np.percentile(finite, 98.0))
        for value in extras:
            if math.isfinite(value):
                lo, hi = min(lo, value), max(hi, value)
        if hi - lo < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.15 * (hi - lo)
        return lo - pad, hi + pad

    def _card_text(self, h: float) -> str:
        """The limit formula with the live numbers substituted in."""
        r = self.report
        f_plus = numeric_value(r.evaluate, r.a + h)
        quotient = (f_plus - r.f_a) / h if math.isfinite(f_plus) else math.nan
        gap = abs(quotient - r.slope) if math.isfinite(quotient) else math.nan

        lines = [
            f"f'({r.a:g}) = lim  [ f({r.a:g}+h) - f({r.a:g}) ] / h = {r.slope:.8g}",
            f"          h->0",
            "",
            f"h            = {h:.6g}",
            f"secant slope = "
            + (f"{quotient:.8g}" if math.isfinite(quotient) else "undefined"),
            f"error        = "
            + (f"{gap:.3e}" if math.isfinite(gap) else "undefined"),
        ]
        return "\n".join(lines)

    def _draw_live_secant(self, h: float) -> None:
        """Move the secant, its endpoint and the rise/run guides to this h."""
        r = self.report
        f_plus = numeric_value(r.evaluate, r.a + h)

        if not math.isfinite(f_plus) or h == 0:
            for artist in (self.secant_line, self.secant_point,
                           self.run_line, self.rise_line):
                artist.set_data([], [])
            return

        slope = (f_plus - r.f_a) / h
        ys = r.f_a + slope * (self.xs - r.a)
        self.secant_line.set_data(self.xs, ys)
        self.secant_point.set_data([r.a + h], [f_plus])
        # the 'run' along the x direction, then the 'rise' up to the curve
        self.run_line.set_data([r.a, r.a + h], [r.f_a, r.f_a])
        self.rise_line.set_data([r.a + h, r.a + h], [r.f_a, f_plus])

    # -------------------------------------------------------------- widgets

    def _h_from_t(self, t: float) -> float:
        """
        Map slider position t in [0, 1] to h geometrically.

        t = 0 gives the user's starting h; t = 1 gives a microscopic h.  The
        mapping is logarithmic because the interesting behaviour spans orders
        of magnitude, not a linear range.
        """
        top = math.log10(self.report.h0)
        bottom = min(MIN_H_EXPONENT, top - 2.0)
        return 10.0 ** (top + t * (bottom - top))

    def _build_widgets(self) -> None:
        slider_ax = self.fig.add_axes([0.10, 0.115, 0.525, 0.035])
        play_ax = self.fig.add_axes([0.735, 0.108, 0.11, 0.050])
        reset_ax = self.fig.add_axes([0.855, 0.108, 0.11, 0.050])

        self.slider = Slider(slider_ax, "drag:  h \u2192 0   ", 0.0, 1.0,
                             valinit=0.0, color=C_LIVE)
        self.slider.on_changed(self._update)

        self.play_button = Button(play_ax, "Play", color="#e8e8e8",
                                  hovercolor="#d0d0d0")
        self.play_button.on_clicked(self._on_play)

        self.reset_button = Button(reset_ax, "Reset", color="#e8e8e8",
                                   hovercolor="#d0d0d0")
        self.reset_button.on_clicked(self._on_reset)

        self.fig.text(
            0.10, 0.045,
            "Drag the slider (or press Play) to shrink h toward 0 and watch the "
            "purple secant fall onto the red tangent.",
            fontsize=9.5, color="#444444",
        )

        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(self._advance)

    def _update(self, t: float) -> None:
        """Slider callback: recompute the secant and refresh the readouts."""
        h = self._h_from_t(float(t))
        self._draw_live_secant(h)
        self.card.set_text(self._card_text(h))
        if self.slider is not None:
            self.slider.valtext.set_text(f"h = {h:.3e}")
        self.fig.canvas.draw_idle()

    def _advance(self) -> None:
        """Timer tick for the Play sweep."""
        t = float(self.slider.val) + 0.006
        if t >= 1.0:
            self.slider.set_val(1.0)
            self._stop_play()
            return
        self.slider.set_val(t)

    def _stop_play(self) -> None:
        self.playing = False
        if self.timer is not None:
            self.timer.stop()
        self.play_button.label.set_text("Play")
        self.fig.canvas.draw_idle()

    def _on_play(self, _event) -> None:
        if self.playing:
            self._stop_play()
            return
        if float(self.slider.val) >= 0.999:
            self.slider.set_val(0.0)
        self.playing = True
        self.play_button.label.set_text("Pause")
        self.timer.start()

    def _on_reset(self, _event) -> None:
        self._stop_play()
        self.slider.set_val(0.0)

    def show(self) -> None:
        plt.show()


# ==========================================================================
# SECTION 8 - Running it
# ==========================================================================


def run_once(source: str, a: float, h0: float,
             static: bool = False, show_plot: bool = True) -> int:
    """Analyse one function/point pair, print the report, then draw it."""
    report = analyse(source, a, h0)
    print_report(report)

    if not show_plot:
        return 0

    scene = DerivativeScene(report, interactive=not static)
    if static:
        print("  Close the window to continue.\n")
    else:
        print("  Drag the h slider (or press Play) to collapse the secant "
              "onto the tangent.")
        print("  Close the window to continue.\n")
    scene.show()
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


def repl(static: bool, show_plot: bool) -> int:
    """Prompt for f(x), a and h, over and over, never crashing."""
    while True:
        source = _ask("f(x)> ")
        if not source:
            continue

        a_text = _ask(f"a (point to differentiate at)> ")
        try:
            a = float(sp.N(parse_expr(preprocess(a_text),
                                      local_dict=dict(_ALLOWED_NAMES),
                                      global_dict=dict(_PARSE_GLOBALS))))
        except Exception:  # noqa: BLE001
            print(f"  [Error] '{a_text}' is not a number I can use for a.\n")
            continue

        h_text = _ask(f"h (starting step size, blank = {DEFAULT_H})> ")
        if not h_text:
            h0 = DEFAULT_H
        else:
            try:
                h0 = float(h_text)
            except ValueError:
                print(f"  [Error] '{h_text}' is not a valid step size.\n")
                continue

        try:
            run_once(source, a, h0, static=static, show_plot=show_plot)
        except DerivativeError as exc:
            print(f"  [Error] {exc}\n")
        except Exception as exc:  # noqa: BLE001 - never dump a traceback
            print(f"  [Error] {exc}\n")


# ==========================================================================
# SECTION 9 - Headless self-check
# ==========================================================================


def run_demo() -> int:
    """Verify the symbolic and numeric routes agree, and that errors are caught."""
    print("=" * 72)
    print("  DERIVATIVE LIMIT VISUALISER - SELF CHECK")
    print("=" * 72)

    failures = 0

    # --- known derivatives -------------------------------------------------
    cases = [
        ("x**2 + 3*x", 2.0, 7.0),
        ("x^2 + 3x", 2.0, 7.0),          # handwritten form must parse the same
        ("sin(x)", 0.0, 1.0),
        ("cos(x)", math.pi / 3, -math.sin(math.pi / 3)),
        ("exp(x)", 1.0, math.e),
        ("log(x)", 2.0, 0.5),
        ("sqrt(x)", 4.0, 0.25),
        ("x**3 - 2*x", -1.0, 1.0),
        ("2sin(x) + x^2", 1.0, 2 * math.cos(1.0) + 2.0),
    ]
    print("\n  Exact derivative vs the limit definition")
    print("  " + "-" * 68)
    for source, a, expected in cases:
        try:
            report = analyse(source, a, DEFAULT_H)
        except DerivativeError as exc:
            print(f"  FAIL  {source!r} at {a:g}: unexpected error: {exc}")
            failures += 1
            continue

        exact_ok = abs(report.slope - expected) < 1e-9

        # the difference quotient must march steadily toward the exact value
        errors = [row.error for row in report.table if math.isfinite(row.error)]
        shrinking = all(errors[i + 1] <= errors[i] + 1e-12
                        for i in range(len(errors) - 1))
        converged = errors[-1] < 1e-3 * max(1.0, abs(expected))

        status = "ok  " if (exact_ok and shrinking and converged) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"  {status}  f(x) = {source:<16} a = {a:<6.4g} "
              f"f'(a) = {report.slope:<12.8g} "
              f"final |error| = {errors[-1]:.2e}")

    # --- things that must be refused, gracefully ---------------------------
    refusals = [
        ("x**2 +", 1.0, "syntax error"),
        ("y + 1", 1.0, "unknown symbol"),
        ("1/x", 0.0, "pole at the point"),
        ("Abs(x)", 0.0, "corner"),
        ("x**(1/3)", 0.0, "vertical tangent / domain edge"),
        ("log(x)", -1.0, "outside the real domain"),
        ("sqrt(x)", 0.0, "domain edge"),
        ("", 1.0, "empty input"),
    ]
    print("\n  Bad input must raise a clear DerivativeError, not crash")
    print("  " + "-" * 68)
    for source, a, why in refusals:
        try:
            analyse(source, a, DEFAULT_H)
        except DerivativeError as exc:
            print(f"  ok    {source!r:<14} at {a:<5g} -> {str(exc)[:58]}...")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {source!r} at {a:g}: wrong exception type: "
                  f"{type(exc).__name__}: {exc}")
            failures += 1
            continue
        print(f"  FAIL  {source!r} at {a:g}: should have been refused ({why})")
        failures += 1

    # --- a bad step size ---------------------------------------------------
    for bad_h in (0.0, -1.0):
        try:
            analyse("x**2", 1.0, bad_h)
        except DerivativeError:
            print(f"  ok    h = {bad_h:<10g} -> refused")
        else:
            print(f"  FAIL  h = {bad_h:g} should have been refused")
            failures += 1

    # --- the figure must build headlessly, static and interactive ----------
    print("\n  Figure construction")
    print("  " + "-" * 68)
    for interactive in (False, True):
        try:
            report = analyse("x**2 + 3*x", 2.0, 1.0)
            scene = DerivativeScene(report, interactive=interactive)
            if interactive:
                scene.slider.set_val(0.5)     # exercise the slider callback
                scene._advance()              # exercise the Play tick
            scene.fig.canvas.draw()
            plt.close(scene.fig)
            mode = "interactive" if interactive else "static     "
            print(f"  ok    {mode} figure built and rendered")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  figure (interactive={interactive}): "
                  f"{type(exc).__name__}: {exc}")
            failures += 1

    print("\n" + "=" * 72)
    if failures:
        print(f"  {failures} check(s) FAILED")
    else:
        print("  All checks passed.")
    print("=" * 72)
    return 1 if failures else 0


# ==========================================================================
# SECTION 10 - Command line
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualise the limit definition of the derivative: shrink h "
                    "toward 0 and watch a secant line become the tangent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1],
    )
    parser.add_argument("--func", "-f", metavar="EXPR",
                        help='the function of x, e.g. "x**2 + 3*x" or "sin(x)"')
    parser.add_argument("--at", "-a", type=float, metavar="A",
                        help="the point where the derivative is evaluated")
    parser.add_argument("--h", type=float, default=DEFAULT_H, metavar="H",
                        help=f"starting step size (default: {DEFAULT_H})")
    parser.add_argument("--static", action="store_true",
                        help="plain snapshot instead of the live h slider")
    parser.add_argument("--no-plot", action="store_true", dest="no_plot",
                        help="console report only, no window")
    parser.add_argument("--demo", action="store_true",
                        help="headless self-check, no windows")
    return parser


def banner() -> None:
    print("=" * 72)
    print("  DERIVATIVE LIMIT VISUALISER")
    print("=" * 72)
    print("  f'(a) = lim(h->0) [ f(a+h) - f(a) ] / h")
    print()
    print("  Enter a function of x, e.g.:")
    print("    x**2 + 3*x        sin(x)          exp(x)/x")
    print("    x^3 - 2x + 1      log(x)          sqrt(x) + cos(x)")
    print("  Type 'quit' at any prompt to stop.")
    print()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # No window will ever be shown in these modes, so use the file backend.
    if args.demo or args.no_plot:
        matplotlib.use("Agg", force=True)

    if args.demo:
        return run_demo()

    banner()

    if args.func is not None:
        if args.at is None:
            print("  [Error] --at is required when --func is given "
                  '(e.g. --func "x**2" --at 3).')
            return 2
        try:
            return run_once(args.func, args.at, args.h,
                            static=args.static, show_plot=not args.no_plot)
        except DerivativeError as exc:
            print(f"  [Error] {exc}\n")
            return 1
        except Exception as exc:  # noqa: BLE001 - never dump a traceback
            print(f"  [Error] {exc}\n")
            return 1

    return repl(static=args.static, show_plot=not args.no_plot)


if __name__ == "__main__":
    raise SystemExit(main())
