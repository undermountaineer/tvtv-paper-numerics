"""Figures from saved inputs and completed reconstructions; no computation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_sequence
from .workflow import spatial_children
from .dyadic import assemble_weight_maps


def save_figures(experiment, report):
    design_seed = experiment.settings["data"]["design_seed"]
    data, _, _ = load_sequence(experiment.input_path(design_seed))
    model = experiment.settings["model"]
    cells = spatial_children(data.noisy.shape)
    a, _ = assemble_weight_maps(data.noisy.shape, {cell: (model["frozen_map"][name], model["beta"])
                                                for name, cell in cells.items()})
    t = data.noisy.shape[0]//2
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for ax, field, title in zip(axes, (data.reference[t], data.noisy[t], a[t]), ("Reference", "Noisy design observation", "Frozen alpha map")):
        im = ax.imshow(field, cmap="viridis" if title.endswith("map") else "gray",
                       **({} if title.endswith("map") else dict(vmin=0, vmax=1)))
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes[-1], fraction=.04)
    fig.savefig(experiment.out/"preview.png", dpi=150)
    plt.close(fig)
    complete = [p for p in report["comparisons"] if p["complete"]]
    if not complete:
        return
    fig, axes = plt.subplots(len(complete), 4, figsize=(12, 3*len(complete)), squeeze=False, constrained_layout=True)
    for line, pair in zip(axes, complete):
        data, _, _ = load_sequence(experiment.input_path(pair["seed"]))
        group = [r for r in report["rows"] if r["stage"] == pair["stage"] and r["seed"] == pair["seed"]]
        constant = next(r for r in group if r["alpha"] == pair["constant_alpha"])
        spatial = next(r for r in group if r["alpha"] is None)
        fields = [data.reference[t], data.noisy[t]]
        for row in (constant, spatial):
            job = {k: row[k] for k in ("stage", "seed", "cell", "alpha", "mode")}
            _, arrays = experiment.inspect(job, with_arrays=True)
            fields.append(arrays[0][t])
        titles = ("Reference", f"Noisy: seed {pair['seed']}",
                  f"Constant: loss {pair['constant_loss']:.6f}",
                  f"Spatial: gain {100*pair['relative_gain']:.2f}%")
        for ax, field, title in zip(line, fields, titles):
            ax.imshow(field, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.savefig(experiment.out/"comparison.png", dpi=150)
    plt.close(fig)
