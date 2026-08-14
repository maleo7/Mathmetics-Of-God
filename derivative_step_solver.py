#!/usr/bin/env python3
"""
Step-by-Step Derivative Solver
==============================================================================
Type a function f(x) and watch every differentiation rule applied one at a
time, exactly like a teacher working through it on paper.

    python derivative_step_solver.py                         # REPL mode
    python derivative_step_solver.py --func "x**3 * sin(x)"
    python derivative_step_solver.py --func "x**x" --visualize

Console output: numbered steps via sympy.pretty(), ending in a boxed summary.
Visualization:  a colour-coded "worked solution" document saved as
                derivative_steps.pdf and derivative_steps.png.

Only single-variable real functions of x.  First derivative only.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

# ==========================================================================
# SECTION 0 — Constants and shared vocabulary
# ==========================================================================

X = sp.Symbol("x", real=True)

# Colour palette for rule labels in the visualization.
RULE_COLORS = {
    "Constant":                "#666666",
    "Identity":                "#666666",
    "Sum Rule":                "#2e7d32",
    "Difference Rule":         "#2e7d32",
    "Constant Multiple Rule":  "#558b2f",
    "Power Rule":              "#1565c0",
    "Product Rule":            "#e65100",
    "Quotient Rule":           "#00796b",
    "Chain Rule":              "#7b1fa2",
    "Logarithmic Differentiation": "#c62828",
    "Table Lookup":            "#616161",
    "Simplify":                "#37474f",
    "Fallback":                "#455a64",
}

def _rule_color(name: str) -> str:
    """Return the hex colour for a rule, with a sensible default."""
    for key, color in RULE_COLORS.items():
        if key in name:
            return color
    return "#455a64"


# ==========================================================================
# SECTION 1 — Parsing: turn user text into a sympy expression, safely
# ==========================================================================

class SolverError(Exception):
    """Raised for anything the user could plausibly have typed wrong."""


_ALLOWED_NAMES: dict[str, object] = {
    "x": X,
    # trigonometry
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "acot": sp.acot, "asec": sp.asec, "acsc": sp.acsc,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "arccot": sp.acot, "arcsec": sp.asec, "arccsc": sp.acsc,
    # hyperbolics
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "coth": sp.coth, "sech": sp.sech, "csch": sp.csch,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "acoth": sp.acoth, "asech": sp.asech, "acsch": sp.acsch,
    "arcsinh": sp.asinh, "arccosh": sp.acosh, "arctanh": sp.atanh,
    "arccoth": sp.acoth, "arcsech": sp.asech, "arccsch": sp.acsch,
    # exponential / logarithmic
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    # special
    "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
    # constants
    "pi": sp.pi, "Pi": sp.pi, "PI": sp.pi, "E": sp.E, "e": sp.E,
}

_PARSE_GLOBALS: dict[str, object] = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
}


def preprocess(text: str) -> str:
    """Make handwritten maths acceptable to a Python parser."""
    expr = text.strip()
    expr = expr.replace("\u2212", "-")
    expr = expr.replace("^", "**")
    expr = re.sub(r"(\d)(?![eE][-+]?\d)\s*([A-Za-z_(])", r"\1*\2", expr)
    expr = re.sub(r"\)\s*([A-Za-z_0-9(])", r")*\1", expr)
    return expr


def parse_function(text: str) -> sp.Expr:
    """Parse user text into a sympy expression in the single symbol x."""
    if not text or not text.strip():
        raise SolverError("Please enter a function of x, e.g. x**2 or sin(x)")

    cleaned = preprocess(text)
    try:
        expr = parse_expr(
            cleaned,
            local_dict=dict(_ALLOWED_NAMES),
            global_dict=dict(_PARSE_GLOBALS),
            transformations=standard_transformations,
            evaluate=True,
        )
    except Exception as exc:
        raise SolverError(
            f"Could not read '{text.strip()}' as a function of x ({exc})."
        ) from exc

    if not isinstance(expr, sp.Expr) or isinstance(expr, sp.logic.boolalg.Boolean):
        raise SolverError(
            "That is not an expression in x.  Enter just the right-hand side, "
            "e.g. 'x**2 + 3*x' rather than 'y = x**2 + 3*x'."
        )

    unknown = sorted(s.name for s in expr.free_symbols if s.name != "x")
    if unknown:
        raise SolverError(
            f"Unknown name(s): {', '.join(unknown)}.  "
            "The function may only use x and the standard functions "
            "(sin, cos, tan, exp, log, sqrt, ...)."
        )

    return expr.subs({s: X for s in expr.free_symbols})


def _pretty(expr: sp.Expr) -> str:
    """Compact unicode string for an expression."""
    return sp.pretty(expr, use_unicode=True)


def _latex_safe(expr: sp.Expr) -> str:
    """LaTeX string safe for matplotlib mathtext."""
    s = sp.latex(expr)
    # mathtext doesn't support \operatorname; replace with \mathrm
    s = s.replace(r"\operatorname", r"\mathrm")
    return s


# ==========================================================================
# SECTION 2 — Step data structure
# ==========================================================================

@dataclass
class DiffStep:
    """One step in the differentiation process."""
    step_number: int = 0                   # assigned after collection
    sub_expr: sp.Expr = sp.Integer(0)      # what is being differentiated
    rule_name: str = ""                    # e.g. "Product Rule"
    rule_formula: str = ""                 # e.g. "d/dx[u·v] = u'v + uv'"
    result: sp.Expr = sp.Integer(0)        # immediate result
    explanation: str = ""                  # plain-English sentence
    color: str = "#455a64"                 # hex colour for the PDF label
    children: list[DiffStep] = field(default_factory=list)


def _flatten_steps(root: DiffStep) -> list[DiffStep]:
    """Depth-first flattening of the step tree, then number sequentially."""
    flat: list[DiffStep] = []

    def walk(step: DiffStep) -> None:
        flat.append(step)
        for child in step.children:
            walk(child)

    walk(root)
    for i, step in enumerate(flat, 1):
        step.step_number = i
    return flat


# ==========================================================================
# SECTION 3 — Derivative table (standard derivatives)
# ==========================================================================

# Each entry maps a sympy function class to:
#   (derivative_as_sympy_expr_of_u, "formula string", "function name")
# where u is the argument of the function.

_U = sp.Symbol("u")

# Build the table as a dict: func_class -> (derivative_expr, formula_str, name)
_DERIV_TABLE: dict[type, tuple[sp.Expr, str, str]] = {}


def _register(func_class, deriv_expr, formula: str, name: str):
    _DERIV_TABLE[func_class] = (deriv_expr, formula, name)


# --- Trigonometric ---
_register(sp.sin,  sp.cos(_U),             "d/dx[sin(u)] = cos(u)",              "sin")
_register(sp.cos, -sp.sin(_U),             "d/dx[cos(u)] = −sin(u)",             "cos")
_register(sp.tan,  sp.sec(_U)**2,          "d/dx[tan(u)] = sec²(u)",             "tan")
_register(sp.cot, -sp.csc(_U)**2,          "d/dx[cot(u)] = −csc²(u)",            "cot")
_register(sp.sec,  sp.sec(_U)*sp.tan(_U),  "d/dx[sec(u)] = sec(u)·tan(u)",       "sec")
_register(sp.csc, -sp.csc(_U)*sp.cot(_U),  "d/dx[csc(u)] = −csc(u)·cot(u)",     "csc")

# --- Inverse trigonometric ---
_register(sp.asin,  1/sp.sqrt(1 - _U**2),          "d/dx[arcsin(u)] = 1/√(1−u²)",          "arcsin")
_register(sp.acos, -1/sp.sqrt(1 - _U**2),          "d/dx[arccos(u)] = −1/√(1−u²)",         "arccos")
_register(sp.atan,  1/(1 + _U**2),                 "d/dx[arctan(u)] = 1/(1+u²)",            "arctan")
_register(sp.acot, -1/(1 + _U**2),                 "d/dx[arccot(u)] = −1/(1+u²)",           "arccot")
_register(sp.asec,  1/(sp.Abs(_U)*sp.sqrt(_U**2 - 1)), "d/dx[arcsec(u)] = 1/(|u|√(u²−1))", "arcsec")
_register(sp.acsc, -1/(sp.Abs(_U)*sp.sqrt(_U**2 - 1)), "d/dx[arccsc(u)] = −1/(|u|√(u²−1))","arccsc")

# --- Exponential / logarithmic ---
_register(sp.exp,  sp.exp(_U),             "d/dx[eᵘ] = eᵘ",                     "exp")
_register(sp.log,  1/_U,                   "d/dx[ln(u)] = 1/u",                  "ln")

# --- Hyperbolic ---
_register(sp.sinh,  sp.cosh(_U),           "d/dx[sinh(u)] = cosh(u)",            "sinh")
_register(sp.cosh,  sp.sinh(_U),           "d/dx[cosh(u)] = sinh(u)",            "cosh")
_register(sp.tanh,  sp.sech(_U)**2,        "d/dx[tanh(u)] = sech²(u)",           "tanh")
_register(sp.coth, -sp.csch(_U)**2,        "d/dx[coth(u)] = −csch²(u)",          "coth")
_register(sp.sech, -sp.sech(_U)*sp.tanh(_U), "d/dx[sech(u)] = −sech(u)·tanh(u)", "sech")
_register(sp.csch, -sp.csch(_U)*sp.coth(_U), "d/dx[csch(u)] = −csch(u)·coth(u)", "csch")

# --- Inverse hyperbolic ---
_register(sp.asinh,  1/sp.sqrt(_U**2 + 1),         "d/dx[arcsinh(u)] = 1/√(u²+1)",         "arcsinh")
_register(sp.acosh,  1/sp.sqrt(_U**2 - 1),         "d/dx[arccosh(u)] = 1/√(u²−1)",         "arccosh")
_register(sp.atanh,  1/(1 - _U**2),                "d/dx[arctanh(u)] = 1/(1−u²)",           "arctanh")
_register(sp.acoth,  1/(1 - _U**2),                "d/dx[arccoth(u)] = 1/(1−u²)",           "arccoth")
_register(sp.asech, -1/(_U*sp.sqrt(1 - _U**2)),    "d/dx[arcsech(u)] = −1/(u√(1−u²))",     "arcsech")
_register(sp.acsch, -1/(sp.Abs(_U)*sp.sqrt(1 + _U**2)), "d/dx[arccsch(u)] = −1/(|u|√(1+u²))", "arccsch")

# --- Abs ---
# Handled specially below because of the x=0 note.


def _table_lookup_deriv(func_class, inner: sp.Expr) -> Optional[tuple[sp.Expr, str, str]]:
    """Look up the derivative of func(inner) from the standard table.
    Returns (derivative_expr, formula_string, function_name) or None."""
    entry = _DERIV_TABLE.get(func_class)
    if entry is None:
        return None
    deriv_template, formula, name = entry
    return deriv_template.subs(_U, inner), formula, name


# ==========================================================================
# SECTION 3b — Rule detectors (one function per rule)
# ==========================================================================

def _depends_on_x(expr: sp.Expr) -> bool:
    """True if the expression contains x."""
    return X in expr.free_symbols


def _is_constant(expr: sp.Expr) -> Optional[DiffStep]:
    """d/dx[c] = 0 where c has no x."""
    if _depends_on_x(expr):
        return None
    return DiffStep(
        sub_expr=expr,
        rule_name="Constant",
        rule_formula="d/dx[c] = 0",
        result=sp.Integer(0),
        explanation=f"The derivative of a constant ({_pretty(expr)}) is 0.",
        color=_rule_color("Constant"),
    )


def _is_bare_x(expr: sp.Expr) -> Optional[DiffStep]:
    """d/dx[x] = 1."""
    if expr != X:
        return None
    return DiffStep(
        sub_expr=expr,
        rule_name="Identity",
        rule_formula="d/dx[x] = 1",
        result=sp.Integer(1),
        explanation="The derivative of x with respect to x is 1.",
        color=_rule_color("Identity"),
    )


def _is_sum(expr: sp.Expr) -> Optional[DiffStep]:
    """d/dx[u ± v ± ...] = u' ± v' ± ..."""
    if not isinstance(expr, sp.Add):
        return None

    terms = expr.args
    children = []
    result_terms = []
    for term in terms:
        child = differentiate_step(term)
        children.append(child)
        result_terms.append(child.result)

    result = sp.Add(*result_terms)
    return DiffStep(
        sub_expr=expr,
        rule_name="Sum Rule",
        rule_formula="d/dx[u ± v] = u' ± v'",
        result=result,
        explanation="Differentiate each term separately and combine.",
        color=_rule_color("Sum Rule"),
        children=children,
    )


def _is_constant_multiple(expr: sp.Expr) -> Optional[DiffStep]:
    """d/dx[c·u] = c·u' where c is constant w.r.t. x."""
    if not isinstance(expr, sp.Mul):
        return None

    # Separate into constant and x-dependent factors.
    const_factors = []
    x_factors = []
    for factor in expr.args:
        if _depends_on_x(factor):
            x_factors.append(factor)
        else:
            const_factors.append(factor)

    # Must have exactly one x-dependent factor group and at least one constant.
    if len(x_factors) != 1 or not const_factors:
        return None

    c = sp.Mul(*const_factors)
    u = x_factors[0]
    child = differentiate_step(u)
    result = c * child.result

    return DiffStep(
        sub_expr=expr,
        rule_name="Constant Multiple Rule",
        rule_formula="d/dx[c·u] = c·u'",
        result=result,
        explanation=f"Factor out the constant {_pretty(c)} and differentiate the rest.",
        color=_rule_color("Constant Multiple Rule"),
        children=[child],
    )


def _is_quotient_rule(expr: sp.Expr) -> Optional[DiffStep]:
    """d/dx[u/v] = (u'v − uv') / v²
    Sympy represents u/v as Mul(u, Pow(v, -1))."""
    if not isinstance(expr, sp.Mul):
        return None

    # Look for a factor that is Pow(something_with_x, -1).
    numerator_factors = []
    denominator = None
    for factor in expr.args:
        if (isinstance(factor, sp.Pow)
                and factor.args[1] == sp.Integer(-1)
                and _depends_on_x(factor.args[0])):
            if denominator is not None:
                # Multiple 1/v terms — not a clean quotient, fall through.
                return None
            denominator = factor.args[0]
        else:
            numerator_factors.append(factor)

    if denominator is None:
        return None

    # If all numerator factors are constants and there's just one, it could
    # still be a quotient.  But if the numerator doesn't depend on x, it's
    # really c/v which is a constant multiple of 1/v — handle it that way.
    u = sp.Mul(*numerator_factors) if numerator_factors else sp.Integer(1)
    v = denominator

    if not _depends_on_x(u):
        return None  # let constant-multiple rule handle c/v(x)

    u_step = differentiate_step(u)
    v_step = differentiate_step(v)

    u_prime = u_step.result
    v_prime = v_step.result
    result = (u_prime * v - u * v_prime) / v**2

    return DiffStep(
        sub_expr=expr,
        rule_name="Quotient Rule",
        rule_formula="d/dx[u/v] = (u'v − uv') / v²",
        result=result,
        explanation=(
            f"Identify u = {_pretty(u)} and v = {_pretty(v)}, then apply the quotient rule."
        ),
        color=_rule_color("Quotient Rule"),
        children=[u_step, v_step],
    )


def _is_log_diff(expr: sp.Expr) -> Optional[DiffStep]:
    """Logarithmic Differentiation for u(x)^v(x) where both depend on x.
    Rewrite as exp(v·ln(u)) → u^v · (v'·ln(u) + v·u'/u)."""
    if not isinstance(expr, sp.Pow):
        return None

    base, exponent = expr.args
    if not (_depends_on_x(base) and _depends_on_x(exponent)):
        return None

    u, v = base, exponent
    u_step = differentiate_step(u)
    v_step = differentiate_step(v)

    u_prime = u_step.result
    v_prime = v_step.result

    # d/dx[u^v] = u^v · (v'·ln(u) + v·u'/u)
    result = expr * (v_prime * sp.log(u) + v * u_prime / u)

    return DiffStep(
        sub_expr=expr,
        rule_name="Logarithmic Differentiation",
        rule_formula="d/dx[u^v] = u^v · (v'·ln(u) + v·u'/u)",
        result=result,
        explanation=(
            f"Both base u = {_pretty(u)} and exponent v = {_pretty(v)} depend on x.  "
            f"Rewrite as exp(v·ln(u)) and apply the chain rule to get "
            f"u^v · (v'·ln(u) + v·u'/u)."
        ),
        color=_rule_color("Logarithmic Differentiation"),
        children=[u_step, v_step],
    )


def _is_power_rule(expr: sp.Expr) -> Optional[DiffStep]:
    """d/dx[u^n] = n·u^(n-1)·u'  where n is constant.
    Also handles sqrt(u) = u^(1/2) and u^(1/n)."""
    if not isinstance(expr, sp.Pow):
        return None

    base, exponent = expr.args
    if not _depends_on_x(base):
        # a^x where a is constant — exponential, not power rule.
        return None
    if _depends_on_x(exponent):
        return None  # handled by log diff

    n = exponent
    u = base

    # Determine sub-case for naming.
    if n == sp.Rational(1, 2):
        sub_label = "Square Root"
        formula = "d/dx[√u] = 1/(2√u) · u'"
    elif n.is_Rational and n.p == 1 and n.q > 1:
        sub_label = f"Root (u^(1/{n.q}))"
        formula = f"d/dx[u^(1/{n.q})] = (1/{n.q})·u^(1/{n.q}−1) · u'"
    else:
        sub_label = "Power Rule"
        formula = "d/dx[u^n] = n·u^(n−1) · u'"

    if u == X:
        # Simple case: d/dx[x^n] = n·x^(n-1), no chain rule needed.
        result = n * X**(n - 1)
        return DiffStep(
            sub_expr=expr,
            rule_name=sub_label,
            rule_formula=formula,
            result=result,
            explanation=(
                f"Apply the power rule with n = {_pretty(n)}: "
                f"bring down the exponent and reduce it by 1."
            ),
            color=_rule_color("Power Rule"),
        )
    else:
        # Chain rule is needed: d/dx[u^n] = n·u^(n-1)·u'
        u_step = differentiate_step(u)
        u_prime = u_step.result
        result = n * u**(n - 1) * u_prime
        return DiffStep(
            sub_expr=expr,
            rule_name=f"{sub_label} + Chain Rule",
            rule_formula=formula,
            result=result,
            explanation=(
                f"Apply the power rule with n = {_pretty(n)} to the outer function, "
                f"then multiply by the derivative of the inner function u = {_pretty(u)}."
            ),
            color=_rule_color("Power Rule"),
            children=[u_step],
        )


def _is_exponential_const_base(expr: sp.Expr) -> Optional[DiffStep]:
    """d/dx[a^u] = a^u · ln(a) · u'  where a is a constant."""
    if not isinstance(expr, sp.Pow):
        return None

    base, exponent = expr.args
    if _depends_on_x(base):
        return None
    if not _depends_on_x(exponent):
        return None  # constant^constant is just a constant

    a = base
    u = exponent

    if u == X:
        result = expr * sp.log(a)
        return DiffStep(
            sub_expr=expr,
            rule_name="Exponential Rule (constant base)",
            rule_formula="d/dx[aˣ] = aˣ · ln(a)",
            result=result,
            explanation=(
                f"The base a = {_pretty(a)} is constant, so apply "
                f"d/dx[aˣ] = aˣ · ln(a)."
            ),
            color=_rule_color("Table Lookup"),
        )
    else:
        u_step = differentiate_step(u)
        u_prime = u_step.result
        result = expr * sp.log(a) * u_prime
        return DiffStep(
            sub_expr=expr,
            rule_name="Exponential Rule (constant base) + Chain Rule",
            rule_formula="d/dx[a^u] = a^u · ln(a) · u'",
            result=result,
            explanation=(
                f"The base a = {_pretty(a)} is constant.  Apply the exponential rule, "
                f"then multiply by the derivative of the inner function u = {_pretty(u)}."
            ),
            color=_rule_color("Chain Rule"),
            children=[u_step],
        )


def _is_product_rule(expr: sp.Expr) -> Optional[DiffStep]:
    """d/dx[u·v] = u'v + uv'  (two or more x-dependent factors)."""
    if not isinstance(expr, sp.Mul):
        return None

    # Gather all x-dependent factors.
    x_factors = [f for f in expr.args if _depends_on_x(f)]
    const_factors = [f for f in expr.args if not _depends_on_x(f)]

    if len(x_factors) < 2:
        return None

    # Split into u (first factor) and v (product of the rest).
    u = x_factors[0]
    v = sp.Mul(*x_factors[1:]) if len(x_factors) > 1 else x_factors[1]
    c = sp.Mul(*const_factors) if const_factors else sp.Integer(1)

    u_step = differentiate_step(u)
    v_step = differentiate_step(v)

    u_prime = u_step.result
    v_prime = v_step.result

    result = c * (u_prime * v + u * v_prime)

    explanation = f"Identify u = {_pretty(u)} and v = {_pretty(v)}, then apply the product rule."
    if c != sp.Integer(1):
        explanation = (
            f"Factor out the constant {_pretty(c)}.  "
            f"Then with u = {_pretty(u)} and v = {_pretty(v)}, apply the product rule."
        )

    return DiffStep(
        sub_expr=expr,
        rule_name="Product Rule",
        rule_formula="d/dx[u·v] = u'v + uv'",
        result=result,
        explanation=explanation,
        color=_rule_color("Product Rule"),
        children=[u_step, v_step],
    )


def _is_abs(expr: sp.Expr) -> Optional[DiffStep]:
    """d/dx[|u|] = sign(u) · u', with a note about x = 0."""
    if not (isinstance(expr, sp.Abs) or
            (hasattr(expr, 'func') and expr.func == sp.Abs)):
        return None

    u = expr.args[0]
    if u == X:
        result = sp.sign(X)
        return DiffStep(
            sub_expr=expr,
            rule_name="Table Lookup (Abs)",
            rule_formula="d/dx[|x|] = sign(x)",
            result=result,
            explanation=(
                "d/dx[|x|] = sign(x), which is +1 for x > 0 and −1 for x < 0.  "
                "Note: the derivative is undefined at x = 0."
            ),
            color=_rule_color("Table Lookup"),
        )
    elif _depends_on_x(u):
        u_step = differentiate_step(u)
        u_prime = u_step.result
        result = sp.sign(u) * u_prime
        return DiffStep(
            sub_expr=expr,
            rule_name="Table Lookup (Abs) + Chain Rule",
            rule_formula="d/dx[|u|] = sign(u) · u'",
            result=result,
            explanation=(
                f"Apply d/dx[|u|] = sign(u) · u' with u = {_pretty(u)}.  "
                "Note: the derivative is undefined where u = 0."
            ),
            color=_rule_color("Chain Rule"),
            children=[u_step],
        )
    return None


def _is_chain_rule(expr: sp.Expr) -> Optional[DiffStep]:
    """d/dx[f(g(x))] = f'(g(x)) · g'(x) — for any function in the table.
    When g(x) = x, this degenerates into a simple table lookup."""
    if not hasattr(expr, 'func') or not hasattr(expr, 'args'):
        return None
    if not expr.args:
        return None

    func_class = type(expr.func) if not isinstance(expr.func, type) else expr.func

    # Check if it's a known function in our table.
    lookup = _table_lookup_deriv(func_class, expr.args[0])
    if lookup is None:
        # Try the function object itself (some sympy classes).
        lookup = _table_lookup_deriv(expr.func, expr.args[0])
    if lookup is None:
        return None

    deriv_of_outer, formula, func_name = lookup
    inner = expr.args[0]

    if inner == X:
        # Simple table lookup: d/dx[f(x)] = f'(x).
        return DiffStep(
            sub_expr=expr,
            rule_name=f"Table Lookup ({func_name})",
            rule_formula=formula,
            result=deriv_of_outer,
            explanation=(
                f"From the standard derivative table: {formula}."
            ),
            color=_rule_color("Table Lookup"),
        )
    elif _depends_on_x(inner):
        # Chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x).
        inner_step = differentiate_step(inner)
        inner_prime = inner_step.result
        result = deriv_of_outer * inner_prime

        return DiffStep(
            sub_expr=expr,
            rule_name=f"Chain Rule ({func_name})",
            rule_formula="d/dx[f(g(x))] = f'(g(x)) · g'(x)",
            result=result,
            explanation=(
                f"The outer function is {func_name}(·) and the inner function is "
                f"g(x) = {_pretty(inner)}.  "
                f"By the table, the outer derivative is {formula.split('=')[1].strip()}, "
                f"evaluated at g(x).  Multiply by g'(x)."
            ),
            color=_rule_color("Chain Rule"),
            children=[inner_step],
        )
    else:
        # Inner doesn't depend on x → whole thing is constant.
        return _is_constant(expr)


# ==========================================================================
# SECTION 4 — Recursive dispatcher
# ==========================================================================

# The detectors, tried in priority order.
_RULE_DETECTORS = [
    _is_constant,
    _is_bare_x,
    _is_sum,
    _is_abs,
    _is_constant_multiple,
    _is_quotient_rule,
    _is_log_diff,
    _is_exponential_const_base,
    _is_power_rule,
    _is_product_rule,
    _is_chain_rule,
]


def differentiate_step(expr: sp.Expr) -> DiffStep:
    """Walk the expression tree and detect which rule applies, recursively."""
    for detector in _RULE_DETECTORS:
        result = detector(expr)
        if result is not None:
            return result

    # Fallback: use sympy.diff directly (shouldn't normally happen).
    fallback_result = sp.diff(expr, X)
    return DiffStep(
        sub_expr=expr,
        rule_name="Fallback (sympy.diff)",
        rule_formula="(no matching rule — computed directly)",
        result=fallback_result,
        explanation=(
            f"No specific rule matched for {_pretty(expr)}; "
            "computed using sympy's built-in differentiation."
        ),
        color=_rule_color("Fallback"),
    )


def solve_derivative(func_text: str) -> tuple[sp.Expr, sp.Expr, list[DiffStep], sp.Expr, Optional[str]]:
    """
    Main entry point: parse, differentiate step-by-step, simplify, cross-check.

    Returns:
        (original_expr, raw_derivative, steps, simplified_derivative, warning_or_None)
    """
    f = parse_function(func_text)

    # Step-by-step differentiation.
    root_step = differentiate_step(f)
    steps = _flatten_steps(root_step)

    raw_derivative = root_step.result

    # Simplify as a final step.
    try:
        simplified = sp.simplify(raw_derivative)
    except Exception:
        simplified = raw_derivative

    # Cross-check against sympy.diff.
    warning = None
    try:
        sympy_answer = sp.diff(f, X)
        diff = sp.simplify(simplified - sympy_answer)
        if diff != 0:
            # Try harder.
            diff2 = sp.simplify(sp.trigsimp(simplified - sympy_answer))
            if diff2 != 0:
                warning = (
                    f"⚠ Cross-check warning: our result and sympy.diff() disagree.\n"
                    f"  Our result:   {_pretty(simplified)}\n"
                    f"  sympy.diff(): {_pretty(sympy_answer)}\n"
                    f"  These may be equivalent forms that simplify() could not unify."
                )
    except Exception:
        pass

    return f, raw_derivative, steps, simplified, warning


# ==========================================================================
# SECTION 5 — Console output
# ==========================================================================

_BOX_WIDTH = 72


def _box_line(text: str, width: int = _BOX_WIDTH) -> str:
    """Centre text inside a box line."""
    padding = max(0, width - 4 - len(text))
    left = padding // 2
    right = padding - left
    return f"║ {' ' * left}{text}{' ' * right} ║"


def print_steps(
    func_text: str,
    f: sp.Expr,
    raw_derivative: sp.Expr,
    steps: list[DiffStep],
    simplified: sp.Expr,
    warning: Optional[str],
) -> None:
    """Print the worked solution to the console."""
    print()
    print("=" * _BOX_WIDTH)
    print(f"  Differentiating:  f(x) = {_pretty(f)}")
    print("=" * _BOX_WIDTH)
    print()

    for step in steps:
        header = f"Step {step.step_number}: [{step.rule_name}]"
        print(header)
        print("-" * len(header))
        print(f"  Sub-expression:  d/dx[ {_pretty(step.sub_expr)} ]")
        print(f"  Rule:            {step.rule_formula}")
        # Word-wrap the explanation.
        wrapped = textwrap.fill(step.explanation, width=64, initial_indent="  ",
                                subsequent_indent="                   ")
        print(f"  Explanation:   {wrapped.strip()}")
        print(f"  Result:          {_pretty(step.result)}")
        print()

    # Raw combined derivative.
    print("-" * _BOX_WIDTH)
    print("  Raw derivative (before simplification):")
    print(f"    {_pretty(raw_derivative)}")
    print()

    # Simplify step.
    print(f"  Final Step: [Simplify]")
    print(f"    {_pretty(simplified)}")
    print()

    # Boxed summary.
    top = "╔" + "═" * (_BOX_WIDTH - 2) + "╗"
    bot = "╚" + "═" * (_BOX_WIDTH - 2) + "╝"
    f_line = f"f(x)  = {_pretty(f)}"
    fp_line = f"f'(x) = {_pretty(simplified)}"
    print(top)
    print(_box_line("RESULT"))
    print(_box_line(""))
    print(_box_line(f_line))
    print(_box_line("  →"))
    print(_box_line(fp_line))
    print(_box_line(""))
    print(bot)

    if warning:
        print()
        print(warning)

    print()


# ==========================================================================
# SECTION 6 — Visualization (matplotlib mathtext, PDF/PNG)
# ==========================================================================

def _render_visualization(
    func_text: str,
    f: sp.Expr,
    raw_derivative: sp.Expr,
    steps: list[DiffStep],
    simplified: sp.Expr,
) -> None:
    """Render the worked solution as a colour-coded document using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # --- Layout parameters ---
    ROW_HEIGHT = 0.65          # inches per step row
    HEADER_HEIGHT = 1.4        # inches for the title block
    FOOTER_HEIGHT = 1.8        # inches for the boxed result
    MARGIN_TOP = 0.5
    MARGIN_BOTTOM = 0.4
    PAGE_WIDTH = 11.0
    MAX_PAGE_HEIGHT = 16.0     # maximum height for one page
    ROWS_PER_PAGE = int((MAX_PAGE_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
                         - MARGIN_TOP - MARGIN_BOTTOM) / ROW_HEIGHT)

    # Prepare LaTeX strings.
    f_latex = _latex_safe(f)
    simplified_latex = _latex_safe(simplified)
    raw_latex = _latex_safe(raw_derivative)

    step_data = []
    for step in steps:
        sub_latex = _latex_safe(step.sub_expr)
        res_latex = _latex_safe(step.result)
        step_data.append({
            "number": step.step_number,
            "rule_name": step.rule_name,
            "formula": step.rule_formula,
            "sub_latex": sub_latex,
            "res_latex": res_latex,
            "color": step.color,
            "explanation": step.explanation,
        })

    total_steps = len(step_data) + 1  # +1 for the simplify step
    num_pages = max(1, (total_steps + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)

    # --- Chunk steps into pages ---
    pages: list[list[dict]] = []
    idx = 0
    for page_num in range(num_pages):
        chunk = step_data[idx:idx + ROWS_PER_PAGE]
        idx += ROWS_PER_PAGE
        # On the last page, add the simplify pseudo-step.
        if page_num == num_pages - 1 or idx >= len(step_data):
            chunk.append({
                "number": len(step_data) + 1,
                "rule_name": "Simplify",
                "formula": "Combine and simplify the raw derivative",
                "sub_latex": raw_latex,
                "res_latex": simplified_latex,
                "color": _rule_color("Simplify"),
                "explanation": "",
            })
            pages.append(chunk)
            break
        pages.append(chunk)

    # --- Render pages ---
    all_figs = []

    for page_idx, page_steps in enumerate(pages):
        n_rows = len(page_steps)
        body_height = n_rows * ROW_HEIGHT
        is_last = page_idx == len(pages) - 1
        footer = FOOTER_HEIGHT if is_last else 0.2
        fig_height = HEADER_HEIGHT + body_height + footer + MARGIN_TOP + MARGIN_BOTTOM

        fig = plt.figure(figsize=(PAGE_WIDTH, fig_height))
        fig.patch.set_facecolor("#fafafa")

        # --- Title (only on first page) ---
        if page_idx == 0:
            fig.text(0.5, 1.0 - MARGIN_TOP / fig_height,
                     "Step-by-Step Derivative",
                     fontsize=22, fontweight="bold", ha="center", va="top",
                     color="#212121")
            try:
                fig.text(0.5, 1.0 - (MARGIN_TOP + 0.45) / fig_height,
                         rf"$f(x) = {f_latex}$",
                         fontsize=16, ha="center", va="top", color="#424242")
            except Exception:
                fig.text(0.5, 1.0 - (MARGIN_TOP + 0.45) / fig_height,
                         f"f(x) = {sp.sstr(f)}",
                         fontsize=16, ha="center", va="top", color="#424242")
        else:
            fig.text(0.5, 1.0 - 0.25 / fig_height,
                     f"(continued — page {page_idx + 1})",
                     fontsize=13, ha="center", va="top", color="#757575")

        # --- Step rows ---
        y_start = 1.0 - (HEADER_HEIGHT + MARGIN_TOP) / fig_height

        for i, sd in enumerate(page_steps):
            y = y_start - (i + 0.5) * ROW_HEIGHT / fig_height
            row_top = y_start - i * ROW_HEIGHT / fig_height
            row_bot = y_start - (i + 1) * ROW_HEIGHT / fig_height

            # Alternating background.
            if i % 2 == 0:
                fig.patches.append(plt.Rectangle(
                    (0.02, row_bot), 0.96, ROW_HEIGHT / fig_height,
                    transform=fig.transFigure, facecolor="#f0f0f0",
                    edgecolor="none", zorder=0,
                ))

            # Step number + rule label.
            label = f"Step {sd['number']}:  {sd['rule_name']}"
            fig.text(0.04, y, label,
                     fontsize=10, fontweight="bold", va="center",
                     color=sd["color"])

            # Formula: d/dx[sub_expr] = result.
            try:
                formula_text = rf"$\frac{{d}}{{dx}}\left[{sd['sub_latex']}\right] = {sd['res_latex']}$"
                fig.text(0.38, y, formula_text,
                         fontsize=11, va="center", color="#212121")
            except Exception:
                fig.text(0.38, y, sd["formula"],
                         fontsize=10, va="center", color="#212121")

        # --- Footer: boxed final result (only on last page) ---
        if is_last:
            footer_y = MARGIN_BOTTOM / fig_height + FOOTER_HEIGHT / fig_height / 2

            # Background box.
            box_h = (FOOTER_HEIGHT - 0.4) / fig_height
            box_y = footer_y - box_h / 2
            fig.patches.append(plt.Rectangle(
                (0.06, box_y), 0.88, box_h,
                transform=fig.transFigure,
                facecolor="#e8f5e9", edgecolor="#2e7d32",
                linewidth=2.5, zorder=5, clip_on=False,
            ))

            fig.text(0.5, footer_y + box_h * 0.25,
                     "Final Result", fontsize=15, fontweight="bold",
                     ha="center", va="center", color="#1b5e20", zorder=10)
            try:
                result_text = rf"$f'(x) = {simplified_latex}$"
                fig.text(0.5, footer_y - box_h * 0.15,
                         result_text, fontsize=14, ha="center", va="center",
                         color="#212121", zorder=10)
            except Exception:
                fig.text(0.5, footer_y - box_h * 0.15,
                         f"f'(x) = {sp.sstr(simplified)}",
                         fontsize=14, ha="center", va="center",
                         color="#212121", zorder=10)

        all_figs.append(fig)

    # --- Save as PDF (multi-page) ---
    pdf_path = "derivative_steps.pdf"
    try:
        with PdfPages(pdf_path) as pdf:
            for fig in all_figs:
                pdf.savefig(fig, bbox_inches="tight")
        print(f"  ✓ Saved PDF:  {pdf_path}")
    except Exception as exc:
        print(f"  ✗ Could not save PDF: {exc}")

    # --- Save as PNG (stacked into one tall image) ---
    png_path = "derivative_steps.png"
    try:
        if len(all_figs) == 1:
            all_figs[0].savefig(png_path, dpi=150, bbox_inches="tight",
                                facecolor="#fafafa")
        else:
            # Stack multiple pages vertically into one tall image.
            import io
            from PIL import Image

            images = []
            for fig in all_figs:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                            facecolor="#fafafa")
                buf.seek(0)
                images.append(Image.open(buf))

            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)
            combined = Image.new("RGB", (max_width, total_height), (250, 250, 250))
            y_offset = 0
            for img in images:
                combined.paste(img, (0, y_offset))
                y_offset += img.height
            combined.save(png_path)
        print(f"  ✓ Saved PNG:  {png_path}")
    except ImportError:
        # No PIL — just save the first page.
        try:
            all_figs[0].savefig(png_path, dpi=150, bbox_inches="tight",
                                facecolor="#fafafa")
            print(f"  ✓ Saved PNG (first page only): {png_path}")
        except Exception as exc:
            print(f"  ✗ Could not save PNG: {exc}")
    except Exception as exc:
        print(f"  ✗ Could not save PNG: {exc}")

    # --- Show on screen ---
    print("  Displaying visualization (close the window to continue)...")
    plt.show()


# ==========================================================================
# SECTION 7 — CLI and REPL
# ==========================================================================

def _process_one(func_text: str, visualize: bool = True) -> bool:
    """Differentiate one function.  Returns True on success."""
    try:
        f, raw_derivative, steps, simplified, warning = solve_derivative(func_text)
    except SolverError as exc:
        print(f"\n  Error: {exc}\n")
        return False

    print_steps(func_text, f, raw_derivative, steps, simplified, warning)

    if visualize:
        print("  Generating visualization...")
        try:
            _render_visualization(func_text, f, raw_derivative, steps, simplified)
        except Exception as exc:
            print(f"  ✗ Visualization failed: {exc}")
        print()
    else:
        print("  (visualization skipped — use without --no-visualize to see it)")
        print()

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-by-step derivative solver — shows every rule like a teacher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s --func "x**3 * sin(x)"
              %(prog)s --func "(2*x+1)/(x**2-3)"
              %(prog)s --func "exp(sin(x**2))" --no-visualize
              %(prog)s --func "x**x"
              %(prog)s                                  (interactive REPL)
        """),
    )
    parser.add_argument("--func", "-f", type=str, default=None,
                        help="Function f(x) to differentiate (e.g. 'x**3 * sin(x)').")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip the visual document — console output only.")

    args = parser.parse_args()
    visualize = not args.no_visualize

    # --- One-shot mode ---
    if args.func:
        _process_one(args.func, visualize)
        return

    # --- REPL mode ---
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          Step-by-Step Derivative Solver                        ║")
    print("║  Enter a function f(x) and see every differentiation rule     ║")
    print("║  applied, one step at a time.                                 ║")
    print("║                                                               ║")
    print("║  Examples:  x**3 * sin(x)                                     ║")
    print("║             (2*x+1)/(x**2-3)                                  ║")
    print("║             exp(sin(x**2))                                    ║")
    print("║             x**x                                              ║")
    print("║                                                               ║")
    print("║  Type 'quit' or 'exit' to leave.                              ║")
    print("║  Visualization is ON by default (saves PDF/PNG + shows        ║")
    print("║  the window).  Add --no-viz after the function to skip it.    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            user_input = input("  f(x) = ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye!\n")
            break

        # Check for inline --no-viz flag to opt out of visualization.
        do_viz = visualize
        if "--no-viz" in user_input:
            do_viz = False
            user_input = user_input.replace("--no-viz", "").strip()

        _process_one(user_input, do_viz)


if __name__ == "__main__":
    main()
