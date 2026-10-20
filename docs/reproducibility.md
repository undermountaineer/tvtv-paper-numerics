# Reproducing and extending experiments

## Environment and source

Use Python 3.12 and the supplied conda environment. Run commands from the
repository root. An editable installation points the environment to one
checkout; reinstall it when switching checkouts. Verify the numerical tests
after code changes. Record the Git commit with exported results.

The clean master is tracked in Git; no external dataset download is required
for the synthetic experiments. Generated observations, caches and figures are
ignored. To reproduce computed arrays directly, use an experiment export rather
than assuming a source checkout contains previous results.

## Configuration and result identity

Configurations declare the data resolution, noise seeds, alpha search, tree
depth, model parameters and solver settings. The experiment ID includes the
configuration contents and relevant source identities; the configuration's
filename is not part of that identity.

`configs/synthetic_256.json` is identical in contents to the generated
`configs/meeting_256_stopping_fixed.json`, if that continuation config exists.
`configs/synthetic_512.json` is identical to `configs/meeting_512.json`.
These are naming aliases; switching between each pair does not start a different
protocol. Earlier filenames and commands remain available for compatibility.

Use the same configuration for preparing, solving, reporting, validating and
exporting. The latest-output pointer is only a convenience; the protocol ID and
configuration identify the result. Never infer a run's settings from a folder
name or image dimensions alone.

## Running and resuming

```bash
python scripts/run_experiment.py --help
python scripts/run_experiment.py --config configs/synthetic_256.json
```

The default sequence runs preparation, constant selection, spatial selection,
and the design comparison. Validation is an explicit subsequent stage. Each
solver runs in an isolated process with numerical thread counts set to one.
Stage budgets include planning and solve processes; setup and reporting add
overhead. Keep the machine awake. Ctrl+C stops the active computation and
retains complete outer checkpoints.

Repeat the same command after a time-limit or interruption. Completed candidates
are reused. A maximum-iteration stop, numerical failure or boundary-limited
search needs diagnosis before extending the computation. Avoid concurrent
writers to one protocol. The cache locks individual solves against duplicate
computation.

Scientific source changes create new cache identities. Existing results are
kept but are not silently relabelled. Explicit continuation between known
source versions verifies the saved problem and records its transition; imported
completed flags are reset and fresh checks are required. Cumulative times and
outer counts include the retained prefix.

## Reports

```bash
python scripts/report_experiment.py --config configs/synthetic_256.json
```

Reporting starts no solver. It generates `EXPERIMENT_REPORT.md`, comparison and
texture figures, and an accepted-weight figure when available. The pointer
`EXPERIMENT_REPORT_256.txt` gives the report's absolute local path. Pending
reconstructions are explicitly labelled; a best completed constant during an
unfinished search is not presented as the final selected optimum.

The raw `summary.json`, `learning.json`, `candidates.csv` and `protocol.json`
remain the machine-readable records. Protocol output stays under
`results/synthetic_original_1024/` for compatibility with existing caches.
Historical output names remain in earlier result directories.

## Exports

```bash
python scripts/run_experiment.py export --config configs/synthetic_256.json
```

The command prints a ZIP path. The archive contains the configuration, scientific
source, prepared inputs, summaries, generated report, available results and
checkpoints, including any source-transition record. It can be large. Keep it
outside the Git repository's tracked files.

When adding an experiment, copy a configuration and state the scientific
change explicitly. Keep design and validation noise separate, compare constant
and spatial weights at matching accuracy, and distinguish accepted partitions
from proposed-map diagnostics. Historical 128-grid evidence is documented in
`docs/status.md` and must not be labelled as a current higher-resolution result.
