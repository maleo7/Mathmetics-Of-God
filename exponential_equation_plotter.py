#!/usr/bin/env python3
"""
Exponential Equation 2D Visualizer
----------------------------------
Enter an exponential equation in x and y via keyboard input and get a 2D plot
of the resulting exponential curve.

Supported formats (spaces optional, '^' or '**' both work):
  y = 2^x
  y = 3*2^x - 1
  y = 2^x + 3
  y = 0.5*(2^x)
  y = e^x
  y = 2*e^x - 4
  2^x = y

The program parses the equation with sympy, reduces it to the standard
form  y = a * b**x + c, computes key features (y-intercept, horizontal
asymptote, growth/decay behaviour) and plots the curve with matplotlib.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


class EquationParseError(Exception):
    """Raised when the equation cannot be parsed."""


def parse_exponential(equation: str) -> tuple[float, float, float]:
    """
    Parse an exponential equation into the standard form: y = a * b**x + c.

    Returns (a, b, c) with b > 0 and b != 1.
    """
    x, y = sp.symbols("x y")

    # Preprocess the raw text:
    #   - convert '^' to '**' (power operator)
    #   - insert '*' for implicit multiplication (e.g. 2x -> 2*x, 3(x) -> 3*(x))
    eq = equation.replace("^", "**")
    # coefficient directly before x or y  (2x -> 2*x)  -- but not 2**x
    eq = re.sub(r"(\d)(?=[xy])", r"\1*", eq)
    # coefficient directly before an opening parenthesis  (3(x) -> 3*(x))
    eq = re.sub(r"(\d)\(", r"\1*(", eq)

    if "=" not in eq:
        raise EquationParseError("Equation must contain an '=' sign.")

    try:
        left, right = eq.split("=", 1)
        # Treat 'e' as Euler's number, everything else as symbols.
        local = {"x": x, "y": y, "e": sp.E, "E": sp.E}
        lhs = sp.sympify(left, locals=local)
        rhs = sp.sympify(right, locals=local)
    except Exception as e:  # noqa: BLE001
        raise EquationParseError(f"Could not parse equation: {e}") from e

    # Bring everything to one side: lhs - rhs = 0
    expr = sp.expand(lhs - rhs)

    # The equation must be linear in y (a plottable function y = f(x)).
    if expr.coeff(y, 2) != 0:
        raise EquationParseError("Equation must be linear in y (no y**2 term).")

    b_y = sp.simplify(expr.coeff(y, 1))
    if b_y == 0:
        raise EquationParseError("Equation must contain a 'y' term to plot y = f(x).")

    # expr = b_y * y + (terms in x) = 0  =>  y = -(terms in x) / b_y
    x_part = sp.expand(expr - b_y * y)
    rhs_expr = sp.expand(-x_part / b_y)  # this is y as a function of x

    # rhs_expr must contain an exponential term of the form (base ** x).
    a_sym, b_sym, c_sym = _extract_exponential(rhs_expr, x)

    a = float(a_sym)
    b = float(b_sym)
    c = float(c_sym)

    if b <= 0:
        raise EquationParseError("The base of the exponential must be positive.")
    if abs(b - 1.0) < 1e-12:
        raise EquationParseError("Base cannot be 1 (that would make y a constant).")
    if abs(a) < 1e-12:
        raise EquationParseError("The exponential coefficient 'a' cannot be 0.")

    return a, b, c


def _extract_exponential(rhs_expr, x) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    From an expression of the form  a * b**x + c  extract (a, b, c).

    Raises EquationParseError if the expression is not a simple exponential
    in x (e.g. it is a polynomial, or x appears outside an exponent).
    """
    # Split the additive terms:  a*b**x  and the constant c.
    expr = sp.expand(rhs_expr)
    terms = sp.Add.make_args(expr)

    a_sym = None
    b_sym = None
    c_sym = sp.Integer(0)

    for term in terms:
        if not term.has(x):
            # constant term contributes to c
            c_sym += term
            continue

        # term depends on x: it must be  coeff * base**x
        base, coeff = _match_exponential_term(term, x)
        if base is None:
            raise EquationParseError(
                "This is not an exponential equation. The variable x must appear "
                "only in an exponent, e.g. y = 2^x. Use the linear or quadratic "
                "plotter for polynomial equations."
            )
        if a_sym is not None:
            raise EquationParseError(
                "Only a single exponential term (a*b**x) is supported."
            )
        a_sym = coeff
        b_sym = base

    if a_sym is None or b_sym is None:
        raise EquationParseError(
            "No exponential term found. Enter something like y = 2^x + 1."
        )

    return sp.simplify(a_sym), sp.simplify(b_sym), sp.simplify(c_sym)


def _match_exponential_term(term, x):
    """
    Try to interpret `term` as coeff * base**x.

    Returns (base, coeff) or (None, None) if it does not match.
    """
    # Separate the numeric/constant coefficient from the x-dependent factor.
    coeff = sp.Integer(1)
    x_factor = sp.Integer(1)
    for factor in sp.Mul.make_args(term):
        if factor.has(x):
            x_factor *= factor
        else:
            coeff *= factor

    # x_factor should now be of the form base**x  (base independent of x).
    #
    # Note: sympy automatically rewrites  E**x  as the function  exp(x).
    # Normalise both  exp(arg)  and  base**arg  to a common (base, exponent).
    if isinstance(x_factor, sp.exp):
        base, exponent = sp.E, x_factor.args[0]
    elif isinstance(x_factor, sp.Pow):
        base, exponent = x_factor.as_base_exp()
    else:
        return None, None

    # exponent may be of the form  k * x  (a constant multiple of x).
    k = exponent.coeff(x, 1)
    if k != 0 and (exponent - k * x) == 0 and not base.has(x) and not k.has(x):
        # base**(k*x) = (base**k)**x  ->  effective base is base**k
        effective_base = sp.simplify(base ** k)
        return effective_base, coeff

    return None, None




def analyze(a: float, b: float, c: float) -> dict:
    """
    Compute the key features of the curve y = a * b**x + c.
    """
    y_intercept = a * (b ** 0) + c  # = a + c

    if b > 1:
        base_behaviour = "growth" if a > 0 else "decay (reflected)"
    else:  # 0 < b < 1
        base_behaviour = "decay" if a > 0 else "growth (reflected)"

    # Increasing/decreasing depends on both the base and the sign of a.
    if (b > 1 and a > 0) or (b < 1 and a < 0):
        monotonic = "increasing"
    else:
        monotonic = "decreasing"

    return {
        "y_intercept": y_intercept,
        "asymptote": c,  # horizontal asymptote y = c
        "behaviour": base_behaviour,
        "monotonic": monotonic,
    }


def plot_exponential(a: float, b: float, c: float, features: dict) -> None:
    asymptote = features["asymptote"]
    y_int = features["y_intercept"]

    # Choose an x-range that shows the interesting part of the curve.
    x_min, x_max = -6.0, 6.0
    xs = np.linspace(x_min, x_max, 800)
    ys = a * np.power(b, xs) + c

    fig, ax = plt.subplots(figsize=(8, 8))

    # Axis lines and grid
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.5)

    # The exponential curve
    label = f"y = {a:g}\u00b7{b:g}^x + {c:g}"
    ax.plot(xs, ys, color="blue", linewidth=2.5, label=label)

    # Horizontal asymptote  y = c
    ax.axhline(
        asymptote,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=f"asymptote: y = {asymptote:.2f}",
    )

    # y-intercept
    ax.plot(0, y_int, "mo", markersize=8)
    ax.annotate(
        f"y-int (0, {y_int:.2f})",
        xy=(0, y_int),
        xytext=(8, 8),
        textcoords="offset points",
        color="magenta",
        fontsize=9,
    )

    # Frame the view nicely around the curve, but clamp extreme values so a
    # fast-growing exponential does not make the plot unreadable.
    finite_ys = ys[np.isfinite(ys)]
    y_lo = min(finite_ys.min(), asymptote, y_int)
    y_hi = max(finite_ys.max(), asymptote, y_int)
    # limit the vertical span to a sensible window around the asymptote
    span_cap = 50.0
    y_lo = max(y_lo, asymptote - span_cap)
    y_hi = min(y_hi, asymptote + span_cap)
    pad = 0.1 * (y_hi - y_lo if y_hi > y_lo else 1.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Visualization of Exponential Equation")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def print_features(a: float, b: float, c: float, features: dict) -> None:
    print(f"  Parsed as: y = {a:g} * {b:g}^x + {c:g}")
    print(f"  Behaviour:            {features['behaviour']}")
    print(f"  Monotonic:            {features['monotonic']}")
    print(f"  y-intercept:          (0, {features['y_intercept']:.4g})")
    print(f"  Horizontal asymptote: y = {features['asymptote']:.4g}")


def main() -> None:
    print("=" * 55)
    print("  EXPONENTIAL EQUATION 2D VISUALIZER")
    print("=" * 55)
    print("Enter an exponential equation in x and y, e.g.:")
    print("  y = 2^x")
    print("  y = 3*2^x - 1")
    print("  y = 2^x + 3")
    print("  y = e^x")
    print("  2^x = y")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            raw = input("Equation> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            a, b, c = parse_exponential(raw)
            features = analyze(a, b, c)
            print_features(a, b, c, features)
            plot_exponential(a, b, c, features)
        except EquationParseError as e:
            print(f"  [Error] {e}")
        except ValueError as e:
            print(f"  [Error] Could not parse numbers: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  [Error] {e}")


if __name__ == "__main__":
    main()
