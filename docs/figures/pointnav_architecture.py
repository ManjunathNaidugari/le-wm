"""Render the proposed thesis architecture; no experimental results are depicted."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(18, 10))
fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.985)
ax.set(xlim=(0, 18), ylim=(0, 10))
ax.axis("off")
INK = "#203047"
BLUE = "#e6f0fb"
ORANGE = "#fff0d9"
PURPLE = "#eee9fa"
GREEN = "#e5f3eb"


def text(x, y, s, size=10, weight="normal", ha="center", color=INK):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va="center",
            color=color, linespacing=1.5)


def box(x, y, w, h, title, body, fill, planned=False, size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                              linewidth=1.35, edgecolor=INK, facecolor=fill,
                              linestyle=(0, (5, 3)) if planned else "solid"))
    text(x+w/2, y+h-.30, title, 11, "bold")
    text(x+w/2, y+(h-.42)/2, body, size)


def arrow(points, color=INK):
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]), color=color, lw=1.3, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=12, linewidth=1.3, color=color))


text(.25, 9.70, "V-JEPA-based predictive navigation for indoor PointNav", 19, "bold", "left")
text(.25, 9.27, "A   Deployment · observe, predict, choose one action, repeat", 12, "bold", "left")

box(.3, 6.85, 2.15, 1.6, "RGB history", "64 causal frames\nfrom Habitat / Gibson", GREEN)
box(3.05, 6.85, 2.25, 1.6, "Frozen encoder", "V-JEPA 2 ViT-L/16\nTemporal / spatial pooling", BLUE, size=9)
box(6.05, 6.85, 2.4, 1.6, "Action proposals", "Small policy π(zₜ, gₜ)\n+ sequence generation\nK candidates, horizon H", ORANGE, True, 9)
box(9.2, 6.65, 3.15, 2.0, "Predictive world model", "Latent predictor Fθ(z, a) → ẑ′\nMotion head Mφ(z, a)\n→ local translation + heading\nRoll out each candidate", ORANGE, True, 9)
box(13.15, 6.85, 2.3, 1.6, "Candidate scoring", "Compose predicted motion\nEstimate goal progress\nSelect best sequence", PURPLE, True, 9)
box(16.05, 6.85, 1.65, 1.6, "Execute", "First action only\nHabitat steps\nObserve again", GREEN, size=9)

for start, end in [(2.45, 3.05), (5.3, 6.05), (8.45, 9.2), (12.35, 13.15), (15.45, 16.05)]:
    arrow([(start, 7.57), (end, 7.57)])
text(5.67, 7.83, "zₜ", 11)
text(8.81, 7.83, "A⁽ⁱ⁾", 11)

# The observed latent initializes the dynamics rollout as well as the policy.
arrow([(5.65, 7.57), (5.65, 6.20), (10.0, 6.20), (10.0, 6.65)])
text(7.8, 6.37, "Initialize every rollout with observed zₜ", 9)

# The goal is a coordinate vector; there is no goal-image embedding.
text(11.0, 9.03, "Relative coordinate goal gₜ · updated from simulator pose", 10, "bold")
arrow([(10.0, 8.88), (7.25, 8.88), (7.25, 8.45)])
arrow([(12.0, 8.88), (14.3, 8.88), (14.3, 8.45)])

arrow([(16.87, 6.85), (16.87, 5.62), (1.37, 5.62), (1.37, 6.85)])
text(9, 5.83, "New RGB observation + updated relative goal · replan after every action", 10)
text(9, 5.17, "Action set: FORWARD (0.25 m) · TURN_LEFT / TURN_RIGHT (15°) · STOP", 10, "bold")
text(9, 4.83, "Future states are predicted by the model; no future simulator queries, map input or expert path at deployment.", 10)

ax.plot([.25, 17.75], [4.47, 4.47], color="#c6ced8", lw=1)
text(.25, 4.12, "B   Training · learn from recorded transitions", 12, "bold", "left")

box(.35, 1.77, 3.2, 1.68, "Recorded trajectories", "Current / next RGB histories\nExecuted actions aₜ\nRecorded poses pₜ, pₜ₊₁", GREEN)
box(4.35, 1.77, 3.05, 1.68, "Frozen shared encoder", "Current latent zₜ\nNext latent target zₜ₊₁\nNo encoder weight updates", BLUE)
box(8.2, 1.77, 3.25, 1.68, "Trainable predictions", "Fθ: next latent ẑₜ₊₁\nMφ: local motion Δp, Δψ\nShort multi-step rollout training", ORANGE, True, 9)
box(12.35, 1.77, 5.25, 1.68, "Supervision and losses", "Latent loss: prediction vs. frozen next-state target\nMotion loss: prediction vs. recorded pose change\nUpdate predictor and motion-head weights", PURPLE, True, 9)
arrow([(3.55, 2.62), (4.35, 2.62)])
text(3.95, 2.92, "RGB", 9)
arrow([(7.4, 2.62), (8.2, 2.62)])
text(7.8, 2.92, "zₜ", 10)
arrow([(11.45, 2.62), (12.35, 2.62)])

arrow([(6.0, 3.45), (6.0, 3.72), (15.0, 3.72), (15.0, 3.45)])
text(10.5, 3.86, "Frozen next-state target (training only)", 9)
arrow([(2.0, 1.77), (2.0, 1.24), (9.4, 1.24), (9.4, 1.77)])
text(5.7, 1.43, "Executed action aₜ conditions both heads", 9)
arrow([(1.0, 1.77), (1.0, .81), (15.0, .81), (15.0, 1.77)])
text(9.0, .98, "Recorded pose changes → local motion targets (training only)", 9)

text(.35, .31, "Blue: frozen weights   |   Orange: learned modules   |   Dashed border: proposed component / extension", 9, ha="left")
text(17.6, .31, "Proposal policy: separate behavior-cloning supervision", 9, ha="right")

for suffix in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"pointnav_architecture.{suffix}", dpi=220, facecolor="white")
plt.close(fig)
