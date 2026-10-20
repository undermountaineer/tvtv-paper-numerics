# Spatially weighted TV–TV reconstruction

Numerical experiments for joint image reconstruction and motion estimation with
spatially varying total-variation regularisation. The main experiment compares
a learned piecewise-constant image-TV weight with a tuned constant weight on a
synthetic image sequence.

The reconstruction follows the alternating image/motion structure of
Burger–Dirks–Schönlieb (BDS). Spatial parameter learning adapts the local
selection and dyadic refinement framework of Davoli–Ferreira–Fonseca–Iglesias
(DFFI). The implemented fidelity, motion-TV norm and scalar search are described
in [the numerical model](docs/model.md).

## Installation

From the repository root, with conda or miniforge installed:

```bash
conda env create -f environment.yml
conda activate tvtv-numerics
python -m unittest discover -s tests -v
```

For an existing environment, update its editable installation:

```bash
conda activate tvtv-numerics
python -m pip install --no-build-isolation --no-deps -e .
```

Python 3.12 is required. The numerical dependencies are NumPy, SciPy and
Matplotlib. The environment also includes JupyterLab and an IPython kernel.
The Python commands below work on Linux and Windows.

## Experiments

Both configurations use five frames, spatially uniform unclipped Gaussian noise
with standard deviation 0.1, and one possible spatial split. Only the image-TV
weight alpha is learned; beta=0.0125 and kappa=1 remain fixed.

| Configuration | Observation and reconstruction grid | Spatial parameter partition |
|---|---|---|
| `configs/synthetic_256.json` | 256×256 | One weight, or four 128×128 regions |
| `configs/synthetic_512.json` | 512×512 | One weight, or four 256×256 regions |

The clean input is a versioned 1024×1024 master. Clean pixels are block-averaged
to the selected image grid before noise is added. The parameter partition is
independent of that image grid. Each spatial region retains the full time
interval and the original pixel spacing.

Run the 256 experiment:

```bash
python -u scripts/run_experiment.py --config configs/synthetic_256.json
```

This prepares observations, selects a constant alpha, tests the spatial split,
and compares the accepted map with the selected constant in full-domain solves
at tighter tolerances. The default stage budgets are 45, 45 and 30 minutes.
Budgets bound computation time; they are not runtime predictions. Repeating
the command reuses completed solves and resumes compatible checkpoints.

For a 512 experiment with larger computation budgets:

```bash
python -u scripts/run_experiment.py --config configs/synthetic_512.json --baseline-minutes 180 --refine-minutes 180 --compare-minutes 90
```

Individual stages can be run explicitly:

```bash
python scripts/run_experiment.py prepare --config configs/synthetic_256.json
python scripts/run_experiment.py baseline --config configs/synthetic_256.json --budget-minutes 45
python scripts/run_experiment.py refine --config configs/synthetic_256.json --budget-minutes 45
python scripts/run_experiment.py compare --config configs/synthetic_256.json --budget-minutes 30
```

The workflow stops at the first unresolved candidate. Time-limited work can be
resumed; an iteration cap, numerical failure or boundary-limited search needs
inspection. Incomplete candidates never enter parameter selection or split
acceptance. Run one computation sequence per checkout at a time.

## Reports and validation

Build the report and figures from saved states without running a solver:

```bash
python scripts/report_experiment.py --config configs/synthetic_256.json
```

The command prints the report path under
`results/synthetic_original_1024/<protocol-id>/`. Main outputs are:

| File | Contents |
|---|---|
| `EXPERIMENT_REPORT.md` | Parameter choices, convergence status, split decisions and comparisons |
| `comparison.png`, `texture.png` | Full-image reconstruction and fine-structure detail |
| `weights.png` | Accepted alpha map, when parameter learning is complete |
| `summary.json`, `candidates.csv` | Metrics and candidate diagnostics |
| `protocol.json`, `learning.json` | Configuration/source identity and partition decisions |

Freeze the design-selected constant and accepted map before evaluating the
three declared fresh-noise seeds:

```bash
python scripts/run_experiment.py validate --config configs/synthetic_256.json --budget-minutes 120
python scripts/run_experiment.py report --config configs/synthetic_256.json
```

Validation seeds are not used for tuning. Local split acceptance and global
reconstruction gain are separate results. A rejected split can have a useful
proposed-map diagnostic, but that diagnostic is not an accepted learned map.
Small gains require matched accuracy checks and a competitive constant control.
See [the experiment protocol](docs/experiment_protocol.md).

## Reproducibility

Git tracks source, configurations, documentation and the clean master.
Generated observations, checkpoints, figures and result exports are kept under
the ignored `results/` directory. Existing configuration filenames remain
supported. The neutral 256 configuration has the same settings as the corrected
256 continuation configuration, so it identifies the same saved experiment.

Export the selected experiment, including available numerical arrays:

```bash
python scripts/run_experiment.py export --config configs/synthetic_256.json
```

Keep the configuration and source identity with every reported result. Detailed
instructions are in [reproducibility](docs/reproducibility.md). The numerical
experiment is a finite approximation of a nonconvex joint problem; its stopping
criteria do not establish a global joint or bilevel optimum.

## Repository structure

| Path | Purpose |
|---|---|
| `src/tvtv_numerics/` | Operators, energies, proximal maps, block solver and parameter-learning workflow |
| `scripts/run_experiment.py` | Experiment stages and bounded computation sequence |
| `scripts/report_experiment.py` | Reports and figures from saved states |
| `configs/synthetic_*.json` | Current synthetic protocols |
| `data/reference/synthetic_original_1024/` | Clean master and its validation metadata |
| `notebooks/synthetic_original_1024.ipynb` | Interactive walkthrough with explicit computation switches |
| `tests/` | Numerical and checkpoint regression tests |
| `evidence/` | Historical 128-grid summaries; see [their scope](docs/status.md) |

The earlier frozen-map protocol and notebook remain available as historical
experiments. Their numerical summaries do not describe the current 256 or 512
protocols.
