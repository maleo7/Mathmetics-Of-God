quit#!/usr/bin/env python3
"""
Quadratic Equation 2D Visualizer
--------------------------------
Enter a quadratic equation in x and y via keyboard input and get a 2D plot
of the resulting parabola.

Supported formats (spaces optional, '^' or '**' both work):
  y = x^2 - 4x + 3
  y = 2x**2 + 3x - 5
  y = -x^2 + 4
  2x^2 + 3x - 1 = y
  y = 0.5x^2

The program parses the equation with sympy, reduces it to the standard
form  y = a*x**2 + b*x + c, computes key features (vertex, axis of
symmetry, roots, y-intercept) and plots the parabola with matplotlib.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


class EquationParseError(Exception):
    """Raised when the equation cannot be parsed."""


def parse_quadratic(equation: str) -> tuple[float, float, float]:
    """
    Parse a quadratic equation into the standard form: y = a*x**2 + b*x + c.

    Returns (a, b, c) with a != 0.
    """
    x, y = sp.symbols("x y")

    # Preprocess the raw text:
    #   - convert '^' to '**' (power operator)
    #   - insert '*' for implicit multiplication (e.g. 2x -> 2*x, 3x**2 -> 3*x**2)
    eq = equation.replace("^", "**")
    eq = re.sub(r"(\d)([xy])", r"\1*\2", eq)
    # handle a coefficient right before an opening parenthesis, e.g. 2(x-1)
    eq = re.sub(r"(\d)\(", r"\1*(", eq)

    if "=" not in eq:
        raise EquationParseError("Equation must contain an '=' sign.")

    try:
        left, right = eq.split("=", 1)
        lhs = sp.sympify(left)
        rhs = sp.sympify(right)
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

    # Collect the x-side coefficients from the remaining expression.
    # expr = b_y * y + (terms in x)  = 0  =>  y = -(terms in x) / b_y
    x_part = sp.expand(expr - b_y * y)
    rhs_expr = sp.expand(-x_part / b_y)  # this is y as a function of x

    # Must be a genuine quadratic in x (no higher powers).
    poly = sp.Poly(rhs_expr, x)
    if poly.degree() > 2:
        raise EquationParseError(
            "This is not a quadratic equation (highest power of x must be 2)."
        )

    a = float(rhs_expr.coeff(x, 2))
    b = float(rhs_expr.coeff(x, 1))
    c = float(rhs_expr.coeff(x, 0))

    if abs(a) < 1e-12:
        raise EquationParseError(
            "This is a linear equation (a = 0). Use the linear plotter instead."
        )

    return a, b, c


def analyze(a: float, b: float, c: float) -> dict:
    """
    Compute the key features of the parabola y = a*x**2 + b*x + c.
    """
    vertex_x = -b / (2 * a)
    vertex_y = a * vertex_x**2 + b * vertex_x + c

    discriminant = b**2 - 4 * a * c
    roots: list[float] = []
    if discriminant > 0:
        sqrt_d = discriminant**0.5
        roots = [(-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)]
    elif abs(discriminant) < 1e-12:
        roots = [-b / (2 * a)]

    return {
        "vertex": (vertex_x, vertex_y),
        "axis_of_symmetry": vertex_x,
        "discriminant": discriminant,
        "roots": roots,
        "y_intercept": c,
        "opens": "upward" if a > 0 else "downward",
    }


def plot_quadratic(a: float, b: float, c: float, features: dict) -> None:
    vertex_x, vertex_y = features["vertex"]
    roots = features["roots"]

    # Auto-center the x-range around the vertex so the parabola is well framed.
    span = 10.0
    x_min, x_max = vertex_x - span, vertex_x + span
    xs = np.linspace(x_min, x_max, 800)
    ys = a * xs**2 + b * xs + c

    fig, ax = plt.subplots(figsize=(8, 8))

    # Axis lines and grid
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.5)

    # The parabola
    label = f"y = {a:g}x\u00b2 + {b:g}x + {c:g}"
    ax.plot(xs, ys, color="blue", linewidth=2.5, label=label)

    # Axis of symmetry
    ax.axvline(
        vertex_x,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"axis: x = {vertex_x:.2f}",
    )

    # Vertex
    ax.plot(vertex_x, vertex_y, "ro", markersize=8)
    ax.annotate(
        f"vertex ({vertex_x:.2f}, {vertex_y:.2f})",
        xy=(vertex_x, vertex_y),
        xytext=(10, 10),
        textcoords="offset points",
        color="red",
        fontsize=9,
    )

    # Roots / x-intercepts
    for r in roots:
        ax.plot(r, 0, "go", markersize=8)
        ax.annotate(
            f"root ({r:.2f}, 0)",
            xy=(r, 0),
            xytext=(5, -15),
            textcoords="offset points",
            color="green",
            fontsize=9,
        )

    # y-intercept
    ax.plot(0, c, "mo", markersize=8)
    ax.annotate(
        f"y-int (0, {c:.2f})",
        xy=(0, c),
        xytext=(8, 8),
        textcoords="offset points",
        color="magenta",
        fontsize=9,
    )

    # Frame the view nicely around the interesting points.
    y_points = [vertex_y, c] + [0.0]
    y_lo = min(y_points) - span
    y_hi = max(y_points) + span
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Visualization of Quadratic Equation")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def print_features(a: float, b: float, c: float, features: dict) -> None:
    vx, vy = features["vertex"]
    print(f"  Parsed as: y = {a:g}x^2 + {b:g}x + {c:g}")
    print(f"  Opens:            {features['opens']}")
    print(f"  Vertex:           ({vx:.4g}, {vy:.4g})")
    print(f"  Axis of symmetry: x = {features['axis_of_symmetry']:.4g}")
    print(f"  y-intercept:      (0, {features['y_intercept']:.4g})")
    print(f"  Discriminant:     {features['discriminant']:.4g}")
    roots = features["roots"]
    if not roots:
        print("  Roots:            none (no real x-intercepts)")
    elif len(roots) == 1:
        print(f"  Root (repeated):  x = {roots[0]:.4g}")
    else:
        print(f"  Roots:            x = {roots[0]:.4g},  x = {roots[1]:.4g}")


def main() -> None:
    print("=" * 55)
    print("  QUADRATIC EQUATION 2D VISUALIZER")
    print("=" * 55)
    print("Enter a quadratic equation in x and y, e.g.:")
    print("  y = x^2 - 4x + 3")
    print("  y = 2x**2 + 3x - 5")
    print("  y = -x^2 + 4")
    print("  2x^2 + 3x - 1 = y")
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
            a, b, c = parse_quadratic(raw)
            features = analyze(a, b, c)
            print_features(a, b, c, features)
            plot_quadratic(a, b, c, features)
        except EquationParseError as e:
            print(f"  [Error] {e}")
        except ValueError as e:
            print(f"  [Error] Could not parse numbers: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  [Error] {e}")


if __name__ == "__main__":
    main()
