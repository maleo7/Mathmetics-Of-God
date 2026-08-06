#!/usr/bin/env python3
"""
Unit Circle Trigonometry Visualizer
-----------------------------------
Models trigonometry with the unit-circle idea.

You enter a radius r and an angle theta (in degrees or radians) and the
program:

  1. Locates the point on the circle:  x = r*cos(theta),  y = r*sin(theta)
  2. Computes all six trig functions individually (sin, cos, tan,
     cosec, sec, cot), reporting "undefined" instead of crashing where a
     function has a pole (tan/sec at 90 deg, cot/cosec at 0 deg, ...).
  3. Verifies the core relationships numerically, printing both sides:
        reciprocal:   csc = 1/sin,  sec = 1/cos,  cot = 1/tan
        quotient:     tan = sin/cos,            cot = cos/sin
        Pythagorean:  sin^2+cos^2 = 1, 1+tan^2 = sec^2, 1+cot^2 = csc^2
  4. Opens a dynamic matplotlib window with Sliders for theta and r that
     shows the rotating radius, its sin/cos projections, and the
     sin/cos/tan waveforms unrolling in real time.

Usage:
    python unit_circle_trigonometry.py            # interactive
    python unit_circle_trigonometry.py --demo     # table of standard angles
    python unit_circle_trigonometry.py --no-plot  # console output only
"""

import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# A trig function is treated as undefined when its denominator (sin or cos)
# is within this distance of zero.  This keeps sin(pi) == 1.22e-16 from being
# mistaken for a genuinely non-zero value.
ZERO_TOL = 1e-10

# Relative tolerance used when comparing the two sides of an identity.
IDENTITY_TOL = 1e-9

# Vertical clipping for the tangent waveform (tan is unbounded).
TAN_CLIP = 6.0

# Order in which identities are reported.
IDENTITY_GROUPS = [
    ("Reciprocal identities", ["csc_reciprocal", "sec_reciprocal", "cot_reciprocal"]),
    ("Quotient identities", ["tan_quotient", "cot_quotient"]),
    ("Pythagorean identities", ["pythagorean_1", "pythagorean_2", "pythagorean_3"]),
]


# --------------------------------------------------------------------------
# Core mathematics
# --------------------------------------------------------------------------
def compute_trig_values(r: float, theta_rad: float) -> dict:
    """
    Compute the point (x, y) on a circle of radius r at angle theta, and all
    six trigonometric functions derived from it.

    The point is       x = r*cos(theta),  y = r*sin(theta)
    and the ratios are sin = y/r, cos = x/r, tan = y/x, csc = r/y,
                       sec = r/x, cot = x/y  (all independent of r).

    Undefined functions are returned as None rather than raising.
    """
    x = r * math.cos(theta_rad)
    y = r * math.sin(theta_rad)

    sin_val = math.sin(theta_rad)
    cos_val = math.cos(theta_rad)

    # tan and sec have poles wherever cos(theta) == 0  (90 deg, 270 deg, ...)
    if abs(cos_val) > ZERO_TOL:
        tan_val = math.tan(theta_rad)
        sec_val = 1.0 / cos_val
    else:
        tan_val = None
        sec_val = None

    # cot and cosec have poles wherever sin(theta) == 0  (0 deg, 180 deg, ...)
    if abs(sin_val) > ZERO_TOL:
        cot_val = 1.0 / math.tan(theta_rad)
        csc_val = 1.0 / sin_val
    else:
        cot_val = None
        csc_val = None

    return {
        "r": r,
        "theta_rad": theta_rad,
        "theta_deg": math.degrees(theta_rad),
        "x": x,
        "y": y,
        "sin": sin_val,
        "cos": cos_val,
        "tan": tan_val,
        "csc": csc_val,
        "sec": sec_val,
        "cot": cot_val,
    }


def _identity(label: str, left, right) -> dict:
    """Build one identity record, comparing both sides with a relative tolerance."""
    if left is None or right is None:
        return {"label": label, "left": None, "right": None, "match": None}
    scale = max(1.0, abs(left), abs(right))
    return {
        "label": label,
        "left": left,
        "right": right,
        "match": abs(left - right) <= IDENTITY_TOL * scale,
    }


def check_identities(values: dict) -> dict:
    """
    Check the reciprocal, quotient and Pythagorean identities for the given
    values, returning both sides of every relationship so they can be shown
    side by side.

    Each entry is {"label", "left", "right", "match"} where match is None if
    the identity is undefined at this angle.
    """
    sin_v, cos_v = values["sin"], values["cos"]
    tan_v, cot_v = values["tan"], values["cot"]
    sec_v, csc_v = values["sec"], values["csc"]

    # Right-hand sides, guarded against division by zero.
    inv_sin = 1.0 / sin_v if abs(sin_v) > ZERO_TOL else None
    inv_cos = 1.0 / cos_v if abs(cos_v) > ZERO_TOL else None
    inv_tan = 1.0 / tan_v if (tan_v is not None and abs(tan_v) > ZERO_TOL) else None
    sin_over_cos = sin_v / cos_v if abs(cos_v) > ZERO_TOL else None
    cos_over_sin = cos_v / sin_v if abs(sin_v) > ZERO_TOL else None

    return {
        # csc = 1/sin, sec = 1/cos, cot = 1/tan
        "csc_reciprocal": _identity("csc(t)      = 1/sin(t)", csc_v, inv_sin),
        "sec_reciprocal": _identity("sec(t)      = 1/cos(t)", sec_v, inv_cos),
        "cot_reciprocal": _identity("cot(t)      = 1/tan(t)", cot_v, inv_tan),
        # tan = sin/cos, cot = cos/sin
        "tan_quotient": _identity("tan(t)      = sin(t)/cos(t)", tan_v, sin_over_cos),
        "cot_quotient": _identity("cot(t)      = cos(t)/sin(t)", cot_v, cos_over_sin),
        # sin^2+cos^2 = 1, 1+tan^2 = sec^2, 1+cot^2 = csc^2
        "pythagorean_1": _identity("sin^2+cos^2 = 1", sin_v**2 + cos_v**2, 1.0),
        "pythagorean_2": _identity(
            "1 + tan^2   = sec^2",
            1.0 + tan_v**2 if tan_v is not None else None,
            sec_v**2 if sec_v is not None else None,
        ),
        "pythagorean_3": _identity(
            "1 + cot^2   = csc^2",
            1.0 + cot_v**2 if cot_v is not None else None,
            csc_v**2 if csc_v is not None else None,
        ),
    }


# --------------------------------------------------------------------------
# Console reporting
# --------------------------------------------------------------------------
def fmt(value, width: int = 12, digits: int = 6) -> str:
    """Format a possibly-undefined number for display."""
    if value is None:
        return "undefined".rjust(width)
    if abs(value) >= 1e6:
        return f"{value:>{width}.4e}"
    return f"{value:>{width}.{digits}f}"


def identity_lines(identities: dict) -> list[str]:
    """Render the identity table as a list of text lines."""
    lines: list[str] = []
    for title, keys in IDENTITY_GROUPS:
        lines.append(f"{title}:")
        for key in keys:
            item = identities[key]
            if item["match"] is None:
                lines.append(f"  {item['label']:<26}  undefined at this angle")
            else:
                mark = "OK" if item["match"] else "MISMATCH"
                lines.append(
                    f"  {item['label']:<26}  {fmt(item['left'])} vs {fmt(item['right'])}  [{mark}]"
                )
        lines.append("")
    return lines


def print_report(values: dict, identities: dict) -> None:
    """Print the point, the six functions and the identity checks."""
    print("\n" + "=" * 72)
    print(
        f"  r = {values['r']:g}     theta = {values['theta_deg']:.4g} deg "
        f"= {values['theta_rad']:.6f} rad"
    )
    print("=" * 72)
    print(f"  Point on circle:  (x, y) = ({values['x']:.6f}, {values['y']:.6f})")
    print("     x = r*cos(theta),  y = r*sin(theta)\n")

    print("  Six trigonometric functions:")
    for name in ("sin", "cos", "tan", "csc", "sec", "cot"):
        print(f"    {name}(theta) = {fmt(values[name])}")
    print()

    for line in identity_lines(identities):
        print("  " + line if line else "")
    print("=" * 72)


# --------------------------------------------------------------------------
# Interactive visualization
# --------------------------------------------------------------------------
def create_interactive_visualization(initial_r: float = 1.0, initial_theta_deg: float = 45.0) -> None:
    """
    Dynamic view: a rotating radius on a circle of radius r together with its
    sin/cos projections, and the sin/cos/tan waveforms unrolling as the
    sliders move.

    Artists are created once and updated with set_data() so redraws stay cheap;
    fig.canvas.draw_idle() schedules the repaint.
    """
    initial_theta_deg = float(np.clip(initial_theta_deg % 360.0, 0.0, 360.0))
    initial_r = float(np.clip(initial_r, 0.1, 3.0))

    fig = plt.figure(figsize=(15, 8.5))
    fig.canvas.manager.set_window_title("Unit Circle Trigonometry")
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1.15, 1.0, 0.95],
        left=0.06, right=0.985, top=0.93, bottom=0.16,
        wspace=0.30, hspace=0.55,
    )

    ax_circle = fig.add_subplot(gs[:, 0])
    ax_sin = fig.add_subplot(gs[0, 1])
    ax_cos = fig.add_subplot(gs[1, 1])
    ax_tan = fig.add_subplot(gs[2, 1])
    ax_info = fig.add_subplot(gs[:, 2])
    ax_info.axis("off")

    # ---- angle arrays (numpy) --------------------------------------------
    wave_deg = np.linspace(0.0, 360.0, 721)
    wave_rad = np.radians(wave_deg)
    sin_wave = np.sin(wave_rad)
    cos_wave = np.cos(wave_rad)
    tan_wave = np.tan(wave_rad)
    tan_wave = np.where(np.abs(tan_wave) > TAN_CLIP, np.nan, tan_wave)  # hide poles
    circle_rad = np.linspace(0.0, 2 * np.pi, 361)

    # ---- circle panel ----------------------------------------------------
    ax_circle.set_aspect("equal", adjustable="box")
    ax_circle.grid(True, linestyle="--", alpha=0.35)
    ax_circle.axhline(0, color="black", linewidth=1.0)
    ax_circle.axvline(0, color="black", linewidth=1.0)
    ax_circle.set_xlabel("x = r*cos(theta)")
    ax_circle.set_ylabel("y = r*sin(theta)")

    (circle_line,) = ax_circle.plot([], [], color="steelblue", linewidth=2.0)
    (arc_line,) = ax_circle.plot([], [], color="darkorange", linewidth=2.0, alpha=0.9)
    (radius_line,) = ax_circle.plot([], [], color="crimson", linewidth=2.5, label="radius r")
    (point_dot,) = ax_circle.plot([], [], "o", color="crimson", markersize=9)
    (sin_proj,) = ax_circle.plot([], [], color="green", linewidth=2.2, linestyle="--",
                                 label="y = r*sin(theta)")
    (cos_proj,) = ax_circle.plot([], [], color="magenta", linewidth=2.2, linestyle="--",
                                 label="x = r*cos(theta)")
    (sin_foot,) = ax_circle.plot([], [], "o", color="green", markersize=6)
    (cos_foot,) = ax_circle.plot([], [], "o", color="magenta", markersize=6)
    point_label = ax_circle.text(0, 0, "", fontsize=9, color="crimson",
                                 ha="left", va="bottom")
    ax_circle.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ---- waveform panels -------------------------------------------------
    wave_axes = (
        (ax_sin, "sin", "green"),
        (ax_cos, "cos", "magenta"),
        (ax_tan, "tan", "darkorange"),
    )
    traces = {}
    for ax, name, color in wave_axes:
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("theta (degrees)", fontsize=8)
        ax.tick_params(labelsize=8)
        # full curve in the background, traced portion on top
        (ghost,) = ax.plot([], [], color="0.75", linewidth=1.2)
        (trace,) = ax.plot([], [], color=color, linewidth=2.2)
        (marker,) = ax.plot([], [], "o", color="crimson", markersize=7)
        (cursor,) = ax.plot([], [], color="crimson", linewidth=1.0, linestyle=":")
        traces[name] = (ghost, trace, marker, cursor)

    for pole in (90, 270):
        ax_tan.axvline(pole, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)
    ax_tan.set_ylim(-TAN_CLIP, TAN_CLIP)
    ax_tan.set_title(f"tan(theta)  (clipped at +/-{TAN_CLIP:g}, independent of r)",
                     fontsize=9, fontweight="bold")

    info_text = ax_info.text(
        0.0, 1.0, "", transform=ax_info.transAxes, family="monospace",
        fontsize=8.2, va="top", ha="left",
    )

    # ---- sliders ---------------------------------------------------------
    ax_theta_slider = fig.add_axes([0.10, 0.075, 0.56, 0.03])
    ax_radius_slider = fig.add_axes([0.10, 0.025, 0.56, 0.03])
    theta_slider = Slider(ax_theta_slider, "theta (deg)", 0.0, 360.0,
                          valinit=initial_theta_deg, valstep=0.5, color="crimson")
    radius_slider = Slider(ax_radius_slider, "radius r", 0.1, 3.0,
                           valinit=initial_r, valstep=0.05, color="steelblue")

    def update(val):
        """Slider callback: recompute everything and refresh the artists."""
        theta_deg = float(theta_slider.val)
        r = float(radius_slider.val)
        theta_rad = math.radians(theta_deg)

        values = compute_trig_values(r, theta_rad)
        identities = check_identities(values)
        x, y = values["x"], values["y"]

        # --- circle -------------------------------------------------------
        circle_line.set_data(r * np.cos(circle_rad), r * np.sin(circle_rad))
        radius_line.set_data([0.0, x], [0.0, y])
        point_dot.set_data([x], [y])
        # vertical leg = the sine projection, horizontal leg = the cosine projection
        sin_proj.set_data([x, x], [0.0, y])
        cos_proj.set_data([0.0, x], [0.0, 0.0])
        sin_foot.set_data([0.0], [y])
        cos_foot.set_data([x], [0.0])
        point_label.set_position((x + 0.05 * r, y + 0.05 * r))
        point_label.set_text(f"({x:.3f}, {y:.3f})")

        arc_r = 0.22 * r
        arc = np.linspace(0.0, theta_rad, max(2, int(abs(theta_deg)) + 2))
        arc_line.set_data(arc_r * np.cos(arc), arc_r * np.sin(arc))

        lim = 1.35 * r
        ax_circle.set_xlim(-lim, lim)
        ax_circle.set_ylim(-lim, lim)
        ax_circle.set_title(
            f"r = {r:.2f}   theta = {theta_deg:.1f} deg = {theta_rad:.4f} rad",
            fontsize=11, fontweight="bold",
        )

        # --- waveforms unrolling up to theta -------------------------------
        k = int(np.searchsorted(wave_deg, theta_deg, side="right"))
        for name, wave, current in (
            ("sin", r * sin_wave, y),          # amplitude scales with r
            ("cos", r * cos_wave, x),
            ("tan", tan_wave, values["tan"]),  # tan is a pure ratio
        ):
            ghost, trace, marker, cursor = traces[name]
            ghost.set_data(wave_deg, wave)
            trace.set_data(wave_deg[:k], wave[:k])
            cursor.set_data([theta_deg, theta_deg],
                            [-1e3, 1e3])  # clipped by the axes limits
            if current is None or abs(current) > TAN_CLIP and name == "tan":
                marker.set_data([], [])
            else:
                marker.set_data([theta_deg], [current])

        amp = 1.2 * r
        ax_sin.set_ylim(-amp, amp)
        ax_cos.set_ylim(-amp, amp)
        ax_sin.set_title(f"r*sin(theta) = {y:.4f}", fontsize=9, fontweight="bold")
        ax_cos.set_title(f"r*cos(theta) = {x:.4f}", fontsize=9, fontweight="bold")

        # --- live numbers and identity checks -------------------------------
        lines = [
            "SIX FUNCTIONS",
            "-" * 40,
        ]
        for name in ("sin", "cos", "tan", "csc", "sec", "cot"):
            lines.append(f"  {name}(t) = {fmt(values[name], width=14)}")
        lines.append("")
        lines.extend(identity_lines(identities))
        info_text.set_text("\n".join(lines))

        fig.canvas.draw_idle()

    theta_slider.on_changed(update)
    radius_slider.on_changed(update)
    update(None)

    fig.suptitle("Trigonometry on the unit circle - drag the sliders",
                 fontsize=13, fontweight="bold")
    plt.show()


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------
def ask_radius() -> float:
    """Prompt until a positive radius is given (blank = 1.0)."""
    while True:
        raw = input("Radius r [default 1]: ").strip()
        if not raw:
            return 1.0
        try:
            r = float(raw)
        except ValueError:
            print("  Please enter a number, e.g. 1 or 2.5")
            continue
        if r <= 0:
            print("  The radius must be positive.")
            continue
        return r


def ask_angle() -> float:
    """Prompt for the unit and the angle; returns the angle in radians."""
    unit = input("Angle in (d)egrees or (r)adians? [default d]: ").strip().lower()
    use_radians = unit.startswith("r")

    prompt = (
        "Angle theta in radians (you may type pi/4) [default pi/4]: "
        if use_radians
        else "Angle theta in degrees [default 45]: "
    )
    while True:
        raw = input(prompt).strip()
        if not raw:
            return math.pi / 4 if use_radians else math.radians(45.0)
        try:
            value = _eval_angle(raw)
        except ValueError as exc:
            print(f"  {exc}")
            continue
        return value if use_radians else math.radians(value)


def _eval_angle(text: str) -> float:
    """Evaluate a simple numeric angle expression, allowing 'pi'."""
    allowed = {"pi": math.pi, "PI": math.pi, "tau": math.tau, "e": math.e}
    try:
        value = eval(text, {"__builtins__": {}}, allowed)  # noqa: S307 - restricted namespace
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read '{text}' as an angle ({exc}).") from exc
    if not isinstance(value, (int, float)):
        raise ValueError(f"Could not read '{text}' as an angle.")
    return float(value)


# --------------------------------------------------------------------------
# Demo / self-check mode
# --------------------------------------------------------------------------
def run_demo() -> int:
    """Print a table over the standard angles and confirm every identity holds."""
    angles = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360]
    header = f"{'deg':>5} | {'sin':>11} {'cos':>11} {'tan':>13} {'csc':>11} {'sec':>13} {'cot':>11}"
    print(header)
    print("-" * len(header))

    failures = 0
    for deg in angles:
        values = compute_trig_values(2.0, math.radians(deg))
        row = " ".join(
            fmt(values[name], width=11 if name in ("sin", "cos", "csc", "cot") else 13, digits=5)
            for name in ("sin", "cos", "tan", "csc", "sec", "cot")
        )
        print(f"{deg:>5} | {row}")
        for key, item in check_identities(values).items():
            if item["match"] is False:
                failures += 1
                print(f"        !! {key} failed: {item['left']} vs {item['right']}")

    print("-" * len(header))
    print("All defined identities hold." if failures == 0 else f"{failures} identity check(s) failed.")
    return 0 if failures == 0 else 1


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> int:
    args = set(sys.argv[1:])
    if "--demo" in args:
        return run_demo()

    show_plot = "--no-plot" not in args

    print("=" * 72)
    print("  UNIT CIRCLE TRIGONOMETRY")
    print("=" * 72)
    print("Enter a radius and an angle; the program locates (x, y) on the")
    print("circle, evaluates all six trig functions and verifies the")
    print("reciprocal, quotient and Pythagorean identities.")
    print("Press Ctrl-C (or type 'quit') to stop.\n")

    while True:
        try:
            r = ask_radius()
            theta_rad = ask_angle()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        values = compute_trig_values(r, theta_rad)
        identities = check_identities(values)
        print_report(values, identities)

        if show_plot:
            try:
                answer = input("\nOpen the interactive plot? (Y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return 0
            if not answer.startswith("n"):
                print("Close the plot window to return to the prompt.")
                create_interactive_visualization(r, values["theta_deg"])

        try:
            again = input("\nAnother angle? (Y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0
        if again.startswith(("n", "q")):
            print("Goodbye!")
            return 0
        print()


if __name__ == "__main__":
    raise SystemExit(main())
