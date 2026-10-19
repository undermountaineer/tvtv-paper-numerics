"""Write a meeting note and images from saved states, without running a solver."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tvtv_numerics.data import load_sequence
from tvtv_numerics.original_1024 import Original1024


def number(value, digits=6):
    return "pending" if value is None else f"{value:.{digits}g}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/meeting_512.json"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    exp = Original1024(root, args.config.resolve())
    report = exp.report()
    state = report["learning"]
    search = state["root"]
    rows = report["rows"]
    ds = exp.settings["data"]
    lines = ["# TV–TV experiment for discussion", "", f"Report generated: {datetime.now(timezone.utc).isoformat()}", "",
             f"Configuration: `{args.config.name}`; protocol: `{exp.token}`.",
             f"Five frames; observation and reconstruction grid: {ds['resolution']} × {ds['resolution']}.",
             f"Spatially uniform, unclipped Gaussian noise: σ={ds['noise_std']}.",
             "The clean 1024 master is block-averaged before noise is added; alpha-map resolution is independent of image resolution.", "",
             "## Current status", "",
             f"- Constant search complete: **{bool(search['complete'])}**.",
             f"- Spatial learning complete: **{bool(state['complete'])}**.",
             f"- Accepted leaves: **{len(state['leaves'])}**."]
    selected = search.get("selected")
    if selected:
        lines += [f"- Design-selected constant alpha: **{selected['alpha']:.9g}** (finite search).",
                  f"- Search boundary limited: **{search.get('boundary_limited', False)}**."]
    if state.get("blocked"):
        lines += [f"- Blocked: {state['blocked']}"]
    if not state["complete"]:
        lines += ["", "No completed spatial-versus-constant conclusion is available yet."]
    for decision in state["decisions"]:
        change = 100 * (decision["children_loss"] / decision["parent_loss"] - 1)
        lines += ["", f"First spatial split accepted: **{decision['accepted']}**. Sum of crop losses changes by **{change:+.4f}%**.",
                  "This is the local DFFI-style acceptance test, not a full-domain reconstruction comparison."]
    lines += ["", "## Global comparisons", "", "| Noise seed | Complete | Training-loss gain | RMSE gain |", "|---|---|---|---|"]
    for pair in report["comparisons"]:
        gain = f"{100 * pair['relative_gain']:.4f}%" if pair["complete"] else "pending"
        rmse_gain = f"{100 * pair['RMSE_relative_gain']:.4f}%" if pair["complete"] else "pending"
        lines += [f"| {pair['seed']} | {pair['complete']} | {gain} | {rmse_gain} |"]
    if state["complete"] and len(state["leaves"]) == 1:
        lines += ["", "**The accepted map is constant.** Its zero gain is expected. Do not spend time validating it as a spatial improvement."]
    if state["complete"]:
        lines += ["", "## Accepted parameter map", "", "| Spatial bounds | Alpha |", "|---|---|"]
        for leaf in state["leaves"]:
            cell = leaf["cell"]
            lines += [f"| y{cell[3]}:{cell[4]}, x{cell[5]}:{cell[6]} | {leaf['alpha']:.9g} |"]
    diagnostic = report.get("rejected_split_global_diagnostic")
    if diagnostic:
        design = report["comparisons"][0]["constant"]
        if diagnostic["eligible"] and design["eligible"]:
            gain = 100 * (1 - diagnostic["upper_loss"] / design["upper_loss"])
            lines += ["", f"Rejected-split proposed map: **{gain:.4f}%** global design-loss gain. This is a diagnostic, not an accepted learned partition or fresh-noise result."]
        else:
            lines += ["", "Rejected-split global diagnostic: pending."]
    lines += ["", "## Candidate progress", "", "Partial losses are excluded from alpha selection and split acceptance.", "",
              "| Seed | Region | Alpha | Mode | Status | Outer steps | Solver seconds | Eligible loss |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        cell = row["cell"]
        region = "whole image" if cell == list(vars(exp.root_cell).values()) else f"level {cell[0]} / y{cell[3]}:{cell[4]} x{cell[5]}:{cell[6]}"
        alpha = "spatial map" if row["alpha"] is None else f"{row['alpha']:.9g}"
        lines += [f"| {row['seed']} | {region} | {alpha} | {row['mode']} | {row['status']} | {row.get('outer_iterations', 0)} | {number(row.get('cumulative_solver_seconds'))} | {number(row.get('upper_loss'))} |"]
    lines += ["", "## Decision for the next session", "",
              "- Time-limited checkpoint: repeat the same run command; completed work is reused.",
              "- Iteration cap, numerical failure, or boundary-limited search: inspect diagnostics/settings before continuing. Repeating does not fix these.",
              "- Accepted nonconstant map with a credible global benefit: freeze it and run the three declared fresh-noise seeds.",
              "- Rejected split or negligible benefit: discuss the image/noise design or the local crop criterion; increasing tree depth cannot repair a rejected first split.",
              "", "A design-seed benefit alone is not evidence of generalization. Positive percentages denote improvement."]
    (exp.out / "MEETING_REPORT.md").write_text("\n".join(lines) + "\n")
    data, _, _ = load_sequence(exp.input_path(ds["design_seed"]))
    completed = [r for r in search["rows"] if r["eligible"]]
    chosen = selected or (min(completed, key=lambda r: r["upper_loss"]) if completed else None)
    constant = None
    if report["comparisons"] and report["comparisons"][0]["constant"]["eligible"]:
        chosen = report["comparisons"][0]["constant"]
    if chosen:
        _, arrays = exp.inspect(chosen, with_arrays=True)
        constant = arrays[0]
        constant_label = f"Constant α={chosen['alpha']:.6g}"
        if not search["complete"]:
            constant_label += "\nBest completed candidate; search incomplete"
        elif chosen["mode"] != "tighter":
            constant_label += "\nLocal tolerance; tighter comparison pending"
    else:
        constant_label = "Constant reconstruction pending"
    spatial, spatial_label = None, "Spatial reconstruction pending"
    if report["comparisons"] and report["comparisons"][0]["spatial"]["eligible"]:
        _, arrays = exp.inspect(report["comparisons"][0]["spatial"], with_arrays=True)
        spatial = arrays[0]
        spatial_label = "Accepted spatial map" if len(state["leaves"]) > 1 else "Accepted map is constant"
    frame = len(data.times) // 2
    panels = [(data.reference, "Clean reference"), (data.noisy, "Observation, σ=0.1"),
              (constant, constant_label), (spatial, spatial_label)]
    for zoom in (False, True):
        fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
        if zoom:
            mask = data.region_masks[frame, 2]
            yy, xx = np.where(mask)
            if len(yy):
                bounds = (slice(max(0, yy.min()-4), min(ds['resolution'], yy.max()+5)),
                          slice(max(0, xx.min()-4), min(ds['resolution'], xx.max()+5)))
            else:
                n = ds["resolution"]
                bounds = (slice(n//4, 3*n//4), slice(n//4, 3*n//4))
        for ax, (array, title) in zip(axes, panels):
            if array is not None:
                plane = array[frame]
                ax.imshow(plane[bounds] if zoom else plane, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{ds['resolution']}×{ds['resolution']}, frame {frame}; " + ("texture detail" if zoom else "full image"))
        fig.savefig(exp.out / ("meeting_texture.png" if zoom else "meeting_comparison.png"), dpi=150)
        plt.close(fig)
    pointer = root / f"MEETING_REPORT_{ds['resolution']}.txt"
    pointer.write_text(str(exp.out / "MEETING_REPORT.md") + "\n")
    if state["complete"]:
        alpha_map = np.empty((ds["resolution"], ds["resolution"]))
        for leaf in state["leaves"]:
            cell = leaf["cell"]
            alpha_map[cell[3]:cell[4], cell[5]:cell[6]] = leaf["alpha"]
        fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
        image = ax.imshow(alpha_map, cmap="viridis", interpolation="nearest")
        ax.set_title("Accepted image-TV alpha map")
        ax.axis("off")
        fig.colorbar(image, ax=ax, label="alpha")
        fig.savefig(exp.out / "meeting_weights.png", dpi=150)
        plt.close(fig)
    print("Meeting note:", exp.out / "MEETING_REPORT.md")
    print("Figures:", exp.out / "meeting_comparison.png", "and", exp.out / "meeting_texture.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
