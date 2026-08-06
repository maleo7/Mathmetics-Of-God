#!/usr/bin/env python3
"""
Linear Equation 2D Visualizer
-----------------------------
Enter a linear equation in x and y via keyboard input and get a 2D plot.

Supported formats (spaces optional):
  y = 2x + 3
  y = -0.5x - 1
  3x + 2y = 6
  2y - 4x = 8
  x = 5            (vertical line)
  y = 4            (horizontal line)

The program parses the equation with sympy, solves for y (or handles
vertical lines), and plots it on an x-y coordinate plane.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


class EquationParseError(Exception):
    """Raised when the equation cannot be parsed."""


def parse_equation(equation: str) -> tuple[float, float, float]:
    """
    Parse a linear equation into the standard form: A*x + B*y = C.

    Returns (A, B, C).
    """
    x, y = sp.symbols("x y")

    # Preprocess: insert '*' for implicit multiplication (e.g. 2x -> 2*x)
    eq = re.sub(r"(\d)([xy])", r"\1*\2", equation)

    try:
        # Parse both sides of the equation
        left, right = eq.split("=", 1)
        lhs = sp.sympify(left)
        rhs = sp.sympify(right)
    except Exception as e:
        raise EquationParseError(f"Could not parse equation: {e}") from e

    # Bring to standard form: lhs - rhs = 0
    expr = sp.expand(lhs - rhs)

    # Collect coefficients of x and y
    a = sp.simplify(expr.coeff(x, 1))
    b = sp.simplify(expr.coeff(y, 1))
    c = sp.simplify(expr - a * x - b * y)  # constant term

    # Ensure the equation is linear
    if expr.has(x**2) or expr.has(y**2) or expr.has(x * y):
        raise EquationParseError(
            "This is not a linear equation (only x and y to the first power)."
        )

    # Convert to floats
    a = float(a)
    b = float(b)
    c = float(-c)  # since expr = a*x + b*y + c = 0  =>  a*x + b*y = -c

    # Normalize sign for readability
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c

    return a, b, c


def equation_to_function(a: float, b: float, c: float):
    """
    Return a callable y(x) and a flag is_vertical.
    """
    if abs(b) < 1e-9:
        # Vertical line: A*x = C  =>  x = C/A
        if abs(a) < 1e-9:
            raise EquationParseError("Equation has no x or y terms.")
        x_val = c / a
        return None, True, x_val
    # y = (C - A*x) / B
    return (lambda x: (c - a * x) / b), False, None


def plot_equation(a: float, b: float, c: float) -> None:
    func, is_vertical, x_val = equation_to_function(a, b, c)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Axis lines
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)

    # Grid
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot the line
    if is_vertical:
        ax.axvline(x_val, color="blue", linewidth=2.5, label=f"x = {x_val:.2f}")
    else:
        xs = np.linspace(-10, 10, 400)
        ys = func(xs)
        ax.plot(xs, ys, color="blue", linewidth=2.5, label=f"{a:.2f}x + {b:.2f}y = {c:.2f}")

    # Axes labels and limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Visualization of Linear Equation")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


def main() -> None:
    print("=" * 55)
    print("  LINEAR EQUATION 2D VISUALIZER")
    print("=" * 55)
    print("Enter a linear equation in x and y, e.g.:")
    print("  y = 2x + 3")
    print("  3x + 2y = 6")
    print("  x = 5")
    print("  y = 4")
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
            a, b, c = parse_equation(raw)
            print(f"  Parsed as: {a:.2f}x + {b:.2f}y = {c:.2f}")
            plot_equation(a, b, c)
        except EquationParseError as e:
            print(f"  [Error] {e}")
        except ValueError as e:
            print(f"  [Error] Could not parse numbers: {e}")
        except Exception as e:
            print(f"  [Error] {e}")


if __name__ == "__main__":
    main()
