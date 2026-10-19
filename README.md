# TV–TV numerical proof of concept

The active experiment asks whether spatial image-TV weights improve joint image
reconstruction and motion estimation over a tuned constant weight. Start with
`configs/meeting_512.json` and the commands below. This repository begins with
one `main` branch. `SOURCE_PROVENANCE.json` records the original source commit.

## Active experiment

- Five synthetic frames, each observed and reconstructed at **512×512**.
- The existing clean 1024 master is averaged in 2×2 blocks **before** noise is
  added. This is a 512 reconstruction, not a 1024 reconstruction displayed small.
- Spatially uniform, unclipped Gaussian noise, standard deviation 0.1.
- Existing observation-only initialization, model, solver, tolerances and loss.
- Learn image-TV alpha; beta=0.0125 and kappa=1 remain fixed.
- Existing bounded alpha search, including endpoint expansion and midpoint tests.
- At most **one spatial split**: one constant or four 256×256 cells. Every crop
  retains all five frames and the original spatial spacing.
- Strict existing local split criterion, rho=1. Rejection retains the constant.
- Global comparison uses the existing tighter settings and identical input and
  initialization. A proposed rejected split is a separately labelled diagnostic.
- Three declared fresh-noise seeds remain untouched until the design is frozen.

Only image resolution and maximum tree depth differ from the original config;
the step sizes, stopping rules, alpha bounds, scene and noise design do not change.
`configs/meeting_256.json` is a separate, optional deadline fallback. It makes a
256×256 observation and reconstruction; it must be reported as a pilot and is
not interchangeable with the 512 experiment. It is not run automatically.

## Install

Use the environment that already passed the performance checks:

```bash
conda activate tvtv-numerics
python -m pip install --no-build-isolation --no-deps -e .
```

For a colleague's fresh installation:

```bash
conda env create -f environment.yml
conda activate tvtv-numerics
python -m unittest discover -s tests -v
```

## Run today

From this repository's root:

```bash
mkdir -p results
python -u scripts/run_meeting.py 2>&1 | tee results/meeting_512.log
```

Keep the terminal open and the computer awake. The sequence is prepare →
constant search → one-level spatial search → tighter global comparison. Maximum
batch budgets are 180, 180 and 90 minutes for the three compute stages. These
are ceilings, **not runtime predictions**. A stage can finish much sooner.
Preparation and final reporting add overhead. Heavy solver workers use one
numerical thread. Run only one compute sequence at a time.

The sequence stops at the first unresolved stage and saves a meeting report.
Time limits preserve complete outer checkpoints. Repeat the **same command**
to reuse completed work and resume partial solves. An iteration cap, numerical
failure or search-boundary stop needs inspection; repeating cannot resolve it.
Do not change tolerances midway through a comparison or count partial candidates
as converged results. Ctrl+C in this foreground terminal stops the active worker.

For an unattended run that survives closing the terminal, instead use:

```bash
mkdir -p results
nohup python -u scripts/run_meeting.py > results/meeting_512.log 2>&1 < /dev/null &
echo "Runner PID: $!"
```

Do not start this as well as the foreground command. Monitor in another terminal:

```bash
tail -f results/meeting_512.log
```

Ctrl+C in the monitoring terminal only stops `tail`. It does not stop the run.

## Meeting output

Rebuild the note and figures from saved states; this starts **no solver**:

```bash
python scripts/meeting_report.py --config configs/meeting_512.json
cat MEETING_REPORT_512.txt
```

The pointer gives the exact report directory under
`results/synthetic_original_1024/<protocol-id>/`. Open:

- `MEETING_REPORT.md`: constant search, split decision, completed comparisons,
  candidate status and solver seconds. Unfinished work is labelled.
- `meeting_comparison.png`: clean, noisy, constant, accepted-map reconstruction.
- `meeting_texture.png`: the same panels in the fine-structure region.
- `meeting_weights.png`: the accepted alpha map, once learning is complete.
- `summary.json`, `learning.json`, `candidates.csv`, `protocol.json`: full metrics
  and the configuration/source identity.

If the first split is rejected, the accepted map is constant and its zero gain
is expected. A global gain of a rejected proposed map is a diagnostic; it does
not establish successful automatic partition learning. Increasing depth does
not help a rejected first split. Discuss scene/noise design or the local crop
criterion before launching more runs.

If a nonconstant accepted map shows a credible global benefit, freeze it and run
the existing three fresh-noise comparisons:

```bash
python scripts/run_original_1024.py validate --config configs/meeting_512.json --seconds 21600 --budget-minutes 360
python scripts/meeting_report.py --config configs/meeting_512.json
```

Validation is deliberately not in today's automatic sequence. Do not tune on
those seeds. Before a manuscript claim, check that the finite-search constant
remains competitive at the tighter settings; this is not a certified global
parameter optimum. A design-seed improvement alone is not generalization.

## Deadline fallback

Only if the 512 run is too slow for tomorrow, stop it before switching, then:

```bash
python -u scripts/run_meeting.py --config configs/meeting_256.json --baseline-minutes 90 --refine-minutes 90 --compare-minutes 45 2>&1 | tee results/meeting_256.log
```

This uses separate input, protocol and solve identities; it does not overwrite
or resume the 512 experiment. Retune alpha at each resolution; the existing
endpoint expansion handles this within its declared bounds.

## Share

Git contains code, configurations, documentation and the 23 MiB clean master.
Generated observations, checkpoints, results and logs are ignored. The summary
pointer `MEETING_REPORT_*.txt` is also ignored. Do not add the result cache to Git.
To give a colleague complete numerical outputs, use the existing separate export:

```bash
python scripts/run_original_1024.py export --config configs/meeting_512.json
```

Its path is printed. It contains source/configuration and available result and
checkpoint arrays and may be large. Sharing the repository alone shares the
implementation and reproducible clean input, not your computed outputs.

## Background

`docs/model.md` describes the numerical model. `docs/performance_pass.md` records
the runtime-only update; `docs/meeting_protocol.md` states the stop/go decisions.
`docs/synthetic_original_1024.md` describes the original **128/depth-two** protocol;
use the explicit meeting configs for the active experiment.

`configs/proof_of_concept.json`, `scripts/run.py`,
`notebooks/proof_of_concept.ipynb` and `evidence/` retain the earlier frozen-map
study for provenance. Its small gains are not results for this new 512 protocol.
The model follows the BDS alternating-block structure and adapts the DFFI local
learning/refinement logic. The existing scalar search and joint nonconvex model
make this an adaptation, not an exact reproduction of either paper.
