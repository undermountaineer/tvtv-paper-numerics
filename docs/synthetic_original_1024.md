# Original high-resolution phantom experiment

This branch uses the same original phantom-v2 geometry, texture and translation.
The 1024 master is not the different synthetic scene being developed in parallel.
Increasing the master resolution does not itself imply better spatial-learning gains.

## Inputs and model

- Authoritative clean input: `data/reference/synthetic_original_1024/reference_1024.npz`.
- Average nonoverlapping 8×8 master blocks to obtain the initial 128×128 working grid.
- Add unclipped Gaussian noise on the working grid, sigma=0.1.
- Same design and three fresh noise seeds as the reviewed study.
- Initialize u with the observation and v with a translation estimate from the observation.
- Keep beta=0.0125 and kappa=1; learn only the image-TV alpha.
- Keep the existing lower-level solver and upper-level image loss unchanged.

The coarse clean reference is now the average of the high-resolution master.
This is finer quadrature than the old 4×4 subpixel renderer. Old numerical results
therefore cannot serve as the new experiment's solved candidates.
All training and evaluation use the clean averaged working-grid reference;
the 1024 master is not used directly to score a coarse reconstruction.

## Alpha search

Start with alpha = 0.00075, 0.000875, 0.001, retaining the previously reviewed constant as a control. Include the inherited
parent alpha on each child. Resolve every declared candidate before selection.
If the best value is at an endpoint, extend that end by a factor of two, inside
the declared bounds and at most three times. Once the winner is interior, test
the geometric midpoint to each neighbor, for one round.

An endpoint winner after the expansion allowance is reported as boundary-limited;
it blocks refinement. This requires inspecting the losses and deliberately widening
the protocol, not accepting the endpoint as a learned optimum. The search is a finite
approximation and cannot certify a continuous or global optimum. All alpha candidates
use the same initial u/v; continuation between different alphas is not used.

## Dyadic refinement

Adapt the four-child subdivision and local acceptance test in DFFI, Section 6.3,
Algorithm 2: sum(child learned crop costs) < rho × parent learned crop cost.
Default rho=1, maximum spatial depth=2; time is never split. Every child keeps
the original hx,hy,ht and its parent's alpha as a tested control. Rejected leaves
are retained. Accepted leaves are refined recursively, in deterministic order.
At depth two the finest cells form a 4×4 grid (32×32 pixels per cell at resolution128).
Depth four allows a 16×16 grid, but should only be enabled after reviewing depth-two
results; it may entail many more solves.

DFFI's convex scalar-sensitivity/piggyback update is not transplanted into the
nonconvex joint image-motion solver. Our bounded scalar search replaces that inner
learning step. We retain the project's spatial image-TV weighting and L^(3/2)
training loss; this is an adaptation of the grid logic, not an exact reproduction
of DFFI's weighted-fidelity TV/TGV numerical experiments.

## Global comparison and validation

Assemble the accepted leaves into a full-domain map. Reconstruct both that map and
the learned constant with tighter tolerances, using identical input and initialization.
If the first split was rejected, also reconstruct its proposed map as a clearly
labelled design-only diagnostic. A local rejection can coexist with a global gain;
the diagnostic does not override the learned partition or participate in validation.

Freeze the design constant and accepted map before evaluating three fresh noise
seeds. Do not tune parameters on validation losses. The reported comparison is
against the design-selected finite-search constant, not an oracle best constant
for each fresh seed. If a meaningful gain appears, confirm the constant baseline
with a wider/tighter search before making a manuscript claim.

## Operation and outputs

CLI: `python scripts/run_original_1024.py ACTION`.
Actions: prepare, baseline, refine, compare, validate, report, export.
Config: `configs/synthetic_original_1024.json` (or `--config PATH`).
Notebook: `notebooks/synthetic_original_1024.ipynb`.

Results: `results/synthetic_original_1024/<protocol-id>/`. The protocol includes
the master checksum, all settings, relevant workflow source and scientific solver
source. The cache hashes observations, initial states, maps and numerical settings.
Existing proof-of-concept results and latest pointer are not used by this controller.

Each solve runs in its own process with numerical thread counts set to one,
a hard deadline, complete outer checkpoints and a finite batch budget. Ctrl+C
stops the current run. Repeat an interrupted/time-limited command to resume.
An unfinished candidate never contributes to alpha selection or a split decision.
A maximum-outer-iteration stop needs numerical review; repeated commands alone
will not lift that cap. Planning and reporting never start a reconstruction.

Outputs include protocol.json, learning.json, summary.json, candidates.csv,
prepared input arrays and checkpointed reconstructions. Export bundles these
with the configuration and source for review.

The earlier generic results organizer archives directories other than
`results/proof_of_concept`. Do not apply it blindly after starting this experiment:
its preview would include `results/synthetic_original_1024`.
