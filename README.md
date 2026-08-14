# Mathmetics-Of-God

Visualise the every concepts of Mathematics as if you are the creator of this Universe.

A collection of Python scripts that visualize fundamental mathematical functions using plots — exploring how simple equations shape the patterns we see in nature.

## What's Inside

| Script | Description |
|---|---|
| `linear_equation_plotter.py` | Plots linear equations (straight-line relationships) |
| `quadratic_equation_plotter.py` | Plots quadratic equations (parabolic curves) |
| `exponential_equation_plotter.py` | Plots exponential equations (nonlinear curve) |
| `unit_circle_trigonometry.py` | Visualizes the unit circle and trigonometric relationships |
| `conic_sections_equation  _plotter.py` | Plots equations of conic sections (circles, ellipses, parabolas, hyperbolas) |
| `conic_sections_visualizer.py` | Slices a double cone with a plane to build a circle, ellipse, parabola or hyperbola — shown as a 3D construction next to the 2D curve |
| `derivative_limit_visualiser.py` | Builds the derivative from its limit definition — drag h → 0 and watch a secant line collapse onto the tangent |
| `derivative_curve_tracer.py` | Animates the derivative *function* — a dot walks along f(x) carrying its tangent, and the panel below draws f'(x) stroke by stroke from the slope it reports; pause and drag the x slider (or arrow-key step) to park on any point and read its slope, angle and condition |
| `derivative_step_by_step.py` | Differentiates a function using an explicit rule engine, prints every intermediate rule and result, and provides an interactive Matplotlib viewer for paging through the working |

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries (`matplotlib`, `numpy`, `sympy`) — install with:

```bash
pip install matplotlib numpy sympy
```

### Running the Scripts

Clone the repo and run any script directly:

```bash
git clone https://github.com/maleo7/Mathmetics-Of-God.git
cd Mathmetics-Of-God
python linear_equation_plotter.py
```

Replace the filename with whichever plotter you'd like to run.

### Step-by-Step Derivative Teacher

Run the interactive prompt and enter functions of the real variable `x`:

```bash
python derivative_step_by_step.py
```

Pass a function directly, or request console-only output:

```bash
python derivative_step_by_step.py --func "(2*x+1)/(x**2-3)"
python derivative_step_by_step.py --func "exp(sin(x**2))" --no-plot
python derivative_step_by_step.py --demo
```

The visualizer uses **Previous**, **Next**, **First**, and **Last** buttons. The
arrow keys move one step, while Home and End jump to the beginning and final
simplification. Closing the window returns to the prompt for another function.

## License

No license specified yet. Feel free to add one (e.g. MIT) if you'd like others to freely use or contribute to this project.
