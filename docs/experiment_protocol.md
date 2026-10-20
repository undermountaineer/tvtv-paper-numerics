# Synthetic experiment protocol

## Data and initialization

The authoritative input is the clean five-frame sequence in
`data/reference/synthetic_original_1024/reference_1024.npz`. A declared SHA-256
checksum identifies it. Block-average the clean 1024 master to the selected
working grid, then add independent Gaussian noise without clipping. Both the
observation and reconstruction use the selected working resolution.

The current protocols use sigma=0.1, design seed 20260930 and fresh-noise seeds
202610071, 202610072 and 202610073. Set the initial image to the observation;
estimate a constant initial translation from the noisy frames only. Clean
references are used to evaluate supervised loss. True motion is diagnostic.

Only alpha is learned. Beta=0.0125 and kappa=1 are fixed. Every fresh candidate
starts from the same observation-based initialization for its noise seed.
Any explicit continuation from a previous stopping policy is recorded separately;
final tighter comparisons start fresh with matched initialization and settings.

## Constant parameter selection

The initial candidates are 0.00075, 0.000875 and 0.001. Complete all declared
solves before ranking them by the image-only 3/2 loss. If an endpoint wins,
extend that side by a factor of two within the configured bounds, up to the
declared expansion allowance. For an interior winner, test geometric midpoints
to its neighbours. A boundary-limited search blocks refinement.

This finite search approximates scalar parameter learning. It does not certify
a continuous or global optimum. Reference data from fresh-noise validation is
not used to adjust the candidates or select the design parameter.

## Spatial parameter learning

The current depth-one protocols test four spatial children of the root region.
Each child retains all five frames, the original spacing and its parent's
selected alpha as a tested control. Fit child parameters using the same scalar
search. Accept the split only if

    sum(selected child crop losses) < rho × selected parent loss,

with rho=1. Equal costs reject the split. A rejected split retains its parent.
The implementation supports deeper trees, but the supplied protocols stop at
one split. Partition depth describes the alpha map, not the image resolution.

The child sum is a crop-based selection criterion with independent crop
boundaries. It is not the loss of a global weighted reconstruction. This is
why a separate full-domain comparison is required.

## Global comparison

Assemble the accepted leaves into an alpha map and reconstruct the whole image
sequence. Compare with the selected constant using the same observation,
initial image/motion and tighter solver settings. Report image loss, RMSE,
texture error, motion diagnostics and convergence status.

If the first split is rejected, the workflow also evaluates the proposed map
globally as a design-only diagnostic. It does not override split rejection or
participate as the accepted map in validation. A one-leaf accepted map is
constant; its zero gain is expected.

Before interpreting a small gain, check sensitivity to stopping accuracy and
the finite constant-parameter search. Inspect the reconstruction and the
fine-structure region as well as scalar metrics.

## Frozen fresh-noise evaluation

Freeze the design-selected constant and accepted map, then evaluate both on
the three declared fresh-noise observations. Each pair shares its own
observation-based initialization and tighter settings. Positive relative gain
is defined as one minus spatial loss divided by constant loss.

The comparator is the design-selected constant, not a separately retuned oracle
on each validation seed. Three realizations of one scene assess repeatability
within that setup; they do not establish broad statistical generalization.

## Numerical acceptance and provenance

Only completed, eligible solves enter selection and comparisons. The scaled
block solver verifies residuals at the state returned by the objective
safeguard. Outer convergence additionally requires small image/motion/energy
changes and two consecutive successful outer iterations.

Outputs identify the master, configuration, scientific source and workflow.
An interrupted run can resume complete outer checkpoints. A time budget is
an execution limit, not a substitute for convergence. Preserve source-transition
records when using explicitly continued checkpoints. Details:
[reproducibility](reproducibility.md).
