"""
Name: Joshua Ashkinaze

Date: 2026-09-15

Description: Draws the multiple-mediator model as a path diagram for each control comparison, in
the style of Panel A of the mechanisms figure. Two models share this geometry.

The serial model is the primary specification: closeness routes into transportation with a
directional arrow, an ordering the data are consistent with and that prior work assumes (SI:
Mediation Model Selection). draw() draws that arrow only when the paths dict it is given contains
it, so the same function renders both models. The parallel model is the alternative: AIM may
affect each mediator, each mediator may independently predict attitudes, and closeness and
transportation's residuals are left free to covary rather than fixed at zero. That covariance is
not drawn, since the model asserts no direction between the mediators; it is reported in the
tables instead.

Each path carries its own weight twice over: the line widens in steps with the size of the
coefficient, and anything that misses $p<.05$ is drawn faded. Stars stay on the labels, so the
encoding is redundant with the printed number rather than the only way to read the figure. The
serial arrow is kept distinct from the direct path three ways: it runs vertically between the two
mediators while the direct path runs horizontally, it is solid teal while the direct path is
dashed blue in the treatment's own colour, and their labels sit on opposite sides of the
crossing.

Nothing here refits a model. 1_run_mediation_analyses.R fits every study-by-control cell for both
models and saves the estimates; this reads the cell and model it is asked for, so the diagram and
the tables in 2_analyze_mediation_analyses.ipynb always show the same numbers.

Inputs:
- study_comparison/tables/mediation_model_estimates.csv, written by 1_run_mediation_analyses.R
- study_comparison/tables/mediation_samples.csv, the arm counts per cell

Outputs:
- plots/mediation_paths.pdf, the pooled-studies panels, parallel model
- plots/mediation_paths_grid.pdf, all nine cells as a 3x3, parallel model
- plots/mediation_paths_serial.pdf, plots/mediation_paths_grid_serial.pdf, the same for the
  serial model
"""

import sys
from pathlib import Path

import bisect

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from src.helpers import make_aesthetic

ATTITUDE, CLOSENESS, TRANSPORT = "attitude_index", "closeness_index", "transport_index"
TABLE_DIR = HERE / "tables"

# the model labels 1_run_mediation_analyses.R writes to the "model" column
MODEL_NAMES = {"parallel": "Multiple parallel",
               "serial": "Multiple serial: closeness -> transportation"}
MODEL_TITLES = {"parallel": "Parallel mediation", "serial": "Serial mediation"}
# the serial model's paths dict carries this key; the parallel model's does not, since it has
# no directional path between the mediators, so draw() just checks which it was given
SERIAL_KEY = (TRANSPORT, CLOSENESS)

# transportation takes Wong's vermillion rather than his reddish purple: the purple sits at
# 3.1:1 against white before any fading, and these paths are the ones most often faded
BLUE, TEAL, VERM = "#0072B2", "#009E73", "#D55E00"
DGRAY, NODE_FILL = "#444444", "#DDEEFF"
FS, FSN, FSA = 8.0, 7.0, 6.5
BW, BH = 3.4, 1.7

# line width steps with |b| at these breaks; opacity marks significance
WIDTH_BREAKS = [0.2, 0.4, 0.6, 0.8]
WIDTHS = [0.7, 1.3, 1.9, 2.5, 3.1]
ALPHA_CUTOFF = 0.05
FADED = 0.4

# where the serial model's two middle labels sit, on opposite sides of the crossing
SERIAL_LABEL_XY = (7.8, 4.6)
DIRECT_LABEL_XY = (5.0, 3.9)

TREAT_XY = (1.7, 4.25)
CLOSE_XY = (6.4, 6.40)
TRANS_XY = (6.4, 2.10)
ATTIT_XY = (11.1, 4.25)

####################
# Model
####################

def load_estimates():
    """Every fitted path and defined effect from 1_run_mediation_analyses.R, keyed by study,
    control and model."""
    estimates = pd.read_csv(TABLE_DIR / "mediation_model_estimates.csv")
    counts = pd.read_csv(TABLE_DIR / "mediation_samples.csv")
    return estimates, counts.set_index(["study", "control"]).n


def model_paths(estimates, study, control, model="parallel"):
    """Estimate and p-value for every regression path in one cell of the grid, for one model.
    The parallel model has no (transport, closeness) entry, since it fits no directional path
    there; its residual covariance lives in the estimates table but is not drawn."""
    cell = estimates[estimates.study.eq(study) & estimates.control.eq(control)
                     & estimates.model.eq(MODEL_NAMES[model]) & estimates.op.eq("~")]
    return {(row.lhs, row.rhs): (row.est, row.pvalue) for row in cell.itertuples()}


def emphasis(paths, outcome, predictor):
    """How heavily to draw one path. Width steps with the effect size, at the breaks in
    WIDTH_BREAKS, and anything that misses p < .05 is drawn faded."""
    estimate, p_value = paths[(outcome, predictor)]
    return {"lw": WIDTHS[bisect.bisect_right(WIDTH_BREAKS, abs(estimate))],
            "alpha": 1.0 if p_value < ALPHA_CUTOFF else FADED}


def stars(p_value):
    """Significance marks in the style of the mechanisms figure."""
    if p_value < .001:
        return "***"
    if p_value < .01:
        return "**"
    if p_value < .05:
        return "*"
    return "n.s."


def label(paths, outcome, predictor, compact=False):
    """A coefficient rendered as the mechanisms figure renders them. The compact form drops the
    $b$ so nine panels stay legible on a page small enough to read at fit-to-width."""
    estimate, p_value = paths[(outcome, predictor)]
    return (f"{estimate:+.2f} {stars(p_value)}" if compact
            else f"$b={estimate:+.2f}$ {stars(p_value)}")

####################
# Drawing
####################

def node(axis, centre, lines, colour, fill="white"):
    """A bordered box whose text takes the border colour."""
    x, y = centre
    axis.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                                  boxstyle="round,pad=0.12", lw=0.9, edgecolor=colour,
                                  facecolor=fill, zorder=3, clip_on=False))
    axis.text(x, y, "\n".join(lines), ha="center", va="center", fontsize=FSN,
              color=colour, fontweight="bold", zorder=4, linespacing=1.35, clip_on=False)


def arrow(axis, start, end, colour, text, text_xy, dashed=False, lw=0.9, alpha=1.0):
    """An arrow with an italic coefficient label beside it. The label fades with the line, so a
    path that misses significance reads as one faint unit rather than a faint line shouting a
    number."""
    axis.annotate("", xy=end, xytext=start, clip_on=False,
                  arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, alpha=alpha,
                                  linestyle="--" if dashed else "-"), zorder=2)
    axis.text(*text_xy, text, ha="center", va="center", fontsize=FSA, color=colour,
              style="italic", zorder=5, clip_on=False, alpha=alpha,
              bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none",
                        alpha=alpha))


NODES = {"treat": ["AIM", "Treatment"], CLOSENESS: ["Psychological", "Closeness"],
         TRANSPORT: ["Narrative", "Transport"], ATTITUDE: ["Policy", "Attitude"]}
NODES_COMPACT = {"treat": ["AIM"], CLOSENESS: ["Closeness"], TRANSPORT: ["Transport"],
                 ATTITUDE: ["Attitudes"]}


def draw(axis, paths, letter=None, title=None, subtitle=None, compact=False):
    """One panel of the diagram. Every annotation is optional, so callers that supply their own
    titling, such as the mechanisms figure, can ask for the diagram alone.

    paths determines which model this draws: a dict carrying a (transport, closeness) regression
    draws the serial model's directional arrow; one carrying a residual covariance instead draws
    the parallel model's dashed double-headed connector. model_paths() builds either.
    """
    serial = SERIAL_KEY in paths
    axis.set_xlim(-0.6, 13.4)
    axis.set_ylim(0.55, 7.95)
    axis.set_aspect("equal")
    axis.axis("off")
    if letter:
        axis.text(0, 1.04, letter, transform=axis.transAxes, fontsize=FS, fontweight="bold",
                  va="bottom", ha="left")
    if title:
        axis.text(0.044 if letter else 0, 1.04, title, transform=axis.transAxes, fontsize=FS,
                  va="bottom", ha="left")
    if subtitle:
        axis.text(0.044 if letter else 0, 0.96, subtitle, transform=axis.transAxes, fontsize=6.5,
                  va="bottom", ha="left", color=DGRAY, style="italic")

    names = NODES_COMPACT if compact else NODES
    node(axis, TREAT_XY, names["treat"], BLUE, NODE_FILL)
    node(axis, CLOSE_XY, names[CLOSENESS], TEAL)
    node(axis, TRANS_XY, names[TRANSPORT], VERM)
    node(axis, ATTIT_XY, names[ATTITUDE], DGRAY, "#F7F7F7")

    half_w, half_h = BW / 2, BH / 2
    arrow(axis, (TREAT_XY[0] + half_w, TREAT_XY[1] + 0.6),
          (CLOSE_XY[0] - half_w, CLOSE_XY[1] - half_h),
          TEAL, label(paths, CLOSENESS, "treat", compact), (2.85, 6.15),
          **emphasis(paths, CLOSENESS, "treat"))
    arrow(axis, (TREAT_XY[0] + half_w, TREAT_XY[1] - 0.6),
          (TRANS_XY[0] - half_w, TRANS_XY[1] + half_h),
          VERM, label(paths, TRANSPORT, "treat", compact), (2.85, 2.35),
          **emphasis(paths, TRANSPORT, "treat"))
    arrow(axis, (CLOSE_XY[0] + half_w, CLOSE_XY[1] - half_h),
          (ATTIT_XY[0] - half_w, ATTIT_XY[1] + 0.6),
          TEAL, label(paths, ATTITUDE, CLOSENESS, compact), (9.95, 6.15),
          **emphasis(paths, ATTITUDE, CLOSENESS))
    arrow(axis, (TRANS_XY[0] + half_w, TRANS_XY[1] + half_h),
          (ATTIT_XY[0] - half_w, ATTIT_XY[1] - 0.6),
          VERM, label(paths, ATTITUDE, TRANSPORT, compact), (9.95, 2.35),
          **emphasis(paths, ATTITUDE, TRANSPORT))

    if serial:
        # the serial path drops vertically between the mediators, labelled to its right; the
        # parallel model has no entry here, so nothing is drawn in that column for it
        arrow(axis, (CLOSE_XY[0], CLOSE_XY[1] - half_h), (TRANS_XY[0], TRANS_XY[1] + half_h + 0.12),
              TEAL, label(paths, *SERIAL_KEY, compact), SERIAL_LABEL_XY,
              **emphasis(paths, *SERIAL_KEY))

    # The direct path runs straight across the middle column in both models. In the serial model
    # the closeness-to-transport arrow drops through that column, so the two cross; the labels sit
    # on opposite sides of the crossing and the dashed blue line reads as the one passing through.
    direct_paths = emphasis(paths, ATTITUDE, "treat")
    direct_text = ("" if compact or serial else "direct ") + label(paths, ATTITUDE, "treat", compact)
    arrow(axis, (TREAT_XY[0] + half_w, TREAT_XY[1]), (ATTIT_XY[0] - half_w, ATTIT_XY[1]),
          BLUE, direct_text, DIRECT_LABEL_XY if serial else (CLOSE_XY[0], 3.75), dashed=True,
          **direct_paths)

####################
# Figure
####################

DEFAULT_PANELS = [("A", "Pooled", "Placebo"), ("B", "Pooled", "Vignette")]
STUDIES = ["Pooled", "Study 1", "Study 2"]
CONTROLS = ["Vignette", "Placebo", "Pooled"]


def figure(panels=None, model="parallel", estimates=None, counts=None):
    """Build the diagram and return the figure, so callers can adjust it before saving."""
    panels = panels or DEFAULT_PANELS
    if estimates is None:
        estimates, counts = load_estimates()

    make_aesthetic(hex_palette="wong", smallest_font_size=8)
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams.update({"font.family": "Arial", "font.size": FS})

    fig, axes = plt.subplots(len(panels), 1, figsize=(6.6, 3.05 * len(panels)), squeeze=False)
    for axis, (letter, study, control) in zip(axes.ravel(), panels):
        scope = "both studies pooled" if study == "Pooled" else study.lower()
        controls = "both controls" if control == "Pooled" else control.lower()
        draw(axis, model_paths(estimates, study, control, model), letter,
             f"AIM versus {controls}",
             f"{MODEL_TITLES[model]}, {scope} (n = {counts[study, control]:,})")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.16)
    return fig


def grid_figure(model="parallel", estimates=None, counts=None):
    """All nine cells as a 3x3, studies down the rows and controls across the columns."""
    if estimates is None:
        estimates, counts = load_estimates()

    make_aesthetic(hex_palette="wong", smallest_font_size=8)
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams.update({"font.family": "Arial", "font.size": FS})

    fig, axes = plt.subplots(len(STUDIES), len(CONTROLS), figsize=(10.8, 6.4))
    for row, study in enumerate(STUDIES):
        for col, control in enumerate(CONTROLS):
            axis = axes[row, col]
            draw(axis, model_paths(estimates, study, control, model), compact=True)
            if row == 0:
                axis.text(0.5, 1.02, f"AIM versus {control.lower()}", transform=axis.transAxes,
                          fontsize=FS, fontweight="bold", va="bottom", ha="center")
            axis.text(0.5, -0.02, f"n = {counts[study, control]:,}", transform=axis.transAxes,
                      fontsize=6.5, va="top", ha="center", color=DGRAY, style="italic")
        axes[row, 0].text(-0.04, 0.5, study, transform=axes[row, 0].transAxes, fontsize=FS,
                          fontweight="bold", va="center", ha="center", rotation=90)

    fig.subplots_adjust(left=0.02, right=0.995, top=0.95, bottom=0.03, wspace=0.05, hspace=0.18)
    return fig


####################
# Apply
####################

def main():
    """Draw the default two-panel figure and the 3x3 grid, for both models."""
    estimates, counts = load_estimates()
    outputs = [("mediation_paths", figure, "parallel"),
               ("mediation_paths_grid", grid_figure, "parallel"),
               ("mediation_paths_serial", figure, "serial"),
               ("mediation_paths_grid_serial", grid_figure, "serial")]
    for name, build, model in outputs:
        fig = build(model=model, estimates=estimates, counts=counts)
        fig.savefig(HERE / "plots" / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote plots/{name}.pdf")


if __name__ == "__main__":
    main()
