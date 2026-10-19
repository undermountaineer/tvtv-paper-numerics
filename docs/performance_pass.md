# Small TV-TV performance pass - 15 October 2026

The default update reduces solver work without changing the energy, operators,
weights, initialization, step sizes, tolerances, image resolution, noise,
parameter search, or subdivision rule. On the checks below, the default
produces bit-for-bit identical image and motion arrays.

## Measured results

All cases use five frames, uniform Gaussian noise with sigma=0.1, design seed
20260930, alpha=0.000875, beta=0.0125 and kappa=1. The clean input is obtained
from the existing 1024 master at the declared reconstruction resolution.
Each timing is one run in a separate process, with numerical library thread
limits set to one, on the same execution machine. Timings are not predictions
for a different computer.

| Workload | Original | Updated default | Optional checks every 5 steps |
| --- | ---: | ---: | ---: |
| Complete converged 128 x 128 solve | 45.93 s | 23.50 s (1.95x) | 11.22 s (4.09x) |
| Fixed 512 x 512 work: 50 image + 50 motion iterations | 28.17 s | 17.37 s (1.62x) | 10.02 s (2.81x) |

The 512 row is a fixed-work diagnostic, **not a completed reconstruction**.
The 128 original and default runs both use 40 outer updates, 1,392 image
iterations and 1,705 motion iterations. Their image and motion arrays are
bit-for-bit identical, as are the original/default fixed-work 512 arrays.
The 512 fixed-work arrays also match with the optional interval of five.

The timings include the solve and lightweight progress JSON writes. They
exclude input setup, final array export, parameter learning and the production
workflow's full-array checkpoint writes. That checkpoint policy is unchanged;
actual workflow speedup depends partly on the user's disk and operating system.

## Changes

- Express the optical-flow proximal map as a clipped scalar correction, avoiding
  repeated boolean indexing into vector arrays.
- Cache the squared image-gradient norm for each motion block.
- Evaluate the two- and four-component projection norms directly, avoiding
  full stacked square arrays and generic norm reductions.
- Keep references to preceding iterates where the updates already allocate
  new arrays; avoid redundant full-array copies.
- Cache the observation RMS within an image block and avoid calculating
  legacy residual limits that scaled residual mode immediately discards.
- Skip residual calculations before the minimum eligible iteration. For
  `inner_check_interval=1`, check every eligible iterate as before.

The default field is `inner_check_interval=1`. Larger values are an explicit
opt-in. Checks occur at the first eligible iterate, every requested interval
thereafter, and at the final iteration cap. The update equations still execute
at every iteration. Reported residuals are calculated at the last returned
raw iterate, rather than inherited from an earlier check.

No step-size/preconditioner change or compiled dependency is introduced in
this contained pass. Existing objective safeguards are retained.

## Optional interval of five

Less frequent checks can change the stopping iteration and hence the subsequent
alternating trajectory. This is why they are not enabled in the supplied
experiment configurations.

The complete 128 run with interval five converges in 33 outer updates, with
1,240 image and 1,375 motion iterations. Relative to the original:

- primary image loss increases by 0.0658%;
- RMSE increases by approximately 0.0158%;
- maximum image difference is 0.001460 on the grayscale scale;
- maximum componentwise motion difference is about 0.0767 pixels.

This is a small numerical change, but it matters when claimed spatial gains
are themselves small. Before using interval five for a scientific comparison,
confirm the constant/spatial ranking and final gain with dense or tighter
controls. For an explicit opt-in, add `"inner_check_interval": 5` inside each
chosen `solver` configuration block. Keep matched comparison settings.

## What remains expensive at 512

A separate default 512 convergence probe was stopped during testing. At its
last recorded outer boundary, 17 updates and 430.31 solver seconds:

- image RMS change was 8.4906e-5, below its 1.8958e-4 limit;
- motion RMS change was 3.3118e-5, above its 9.7656e-6 limit;
- the most recent blocks used 22 image and 77 motion iterations.

This probe is **not converged** and is not used to claim a completed 512 runtime.
It indicates that motion convergence remains a cost in addition to the array
implementation overhead addressed here. The partial probe contains progress
metadata only; it is not a resumable reconstruction checkpoint.

## Verification

All 41 tests pass. The added tests check proximal optimality across saturated,
unsaturated and zero-gradient cases; cached-gradient agreement; input
immutability; residual scheduling and final-cap reporting; and exact checkpoint
resume with sparse checks. Existing operator, objective, projection, learning
selection and checkpoint checks also pass.

The benchmark JSON files record settings, histories, input/source checksums,
environment, iteration counts, quality metrics and explicit convergence flags.
The verification summary records output differences and measured speedups.

## Applying the download

Extract the update archive first. It contains `apply_update.py`, a checksum
manifest, replacement/new files, an ordinary Git patch, the original two core
files for benchmark reproduction, and benchmark records. The updater checks
all affected files before writing. It refuses unrelated edits, accepts the
supported LF/CRLF line-ending difference, backs up replaced files beside the
download, and can be safely rerun if already applied.

### Linux

Use the checkout corresponding to the supplied branch. Adjust the download
path if the archive was extracted somewhere else.

```bash
conda activate tvtv-numerics
cd ~/projects/tvtv-dyadic-numerics
git switch -c performance/small-pass
python ~/Downloads/tvtv-performance-pass-2026-10-15/apply_update.py --repo . --check
python ~/Downloads/tvtv-performance-pass-2026-10-15/apply_update.py --repo .
python -m pip install --no-build-isolation --no-deps -e .
python -m unittest discover -s tests -v
```

### Windows PowerShell

Adjust the checkout and extracted-download paths as needed.

```powershell
conda activate tvtv-numerics
Set-Location C:\Users\Tobias\tvtv-original-1024
git switch -c performance/small-pass
python "$env:USERPROFILE\Downloads\tvtv-performance-pass-2026-10-15\apply_update.py" --repo . --check
python "$env:USERPROFILE\Downloads\tvtv-performance-pass-2026-10-15\apply_update.py" --repo .
python -m pip install --no-build-isolation --no-deps -e .
python -m unittest discover -s tests -v
```

Installation uses the existing dependencies. Reinstalling the editable project
ensures the active environment points to this checkout rather than another clone.
If the updater reports a conflict, review those files; it has not changed any
files at that point. The included `performance.patch` is also suitable for
`git apply --check` and `git apply` against the original snapshot.

## Reproducing timings

The default quick complete check is:

```bash
python scripts/benchmark_performance.py --resolution 128 --case optimized --seconds 300 --output results/performance_check
```

A bounded 512 work check is:

```bash
python scripts/benchmark_performance.py --resolution 512 --case optimized --fixed-inner 50 --seconds 300 --output results/performance_check
```

For the optional scheduling experiment, change `--case optimized` to
`--case interval5`. For the original version use `--case original` and
`--baseline-dir /path/to/extracted-update/reference_baseline`. The benchmark
loads those original files in a separate module namespace; it does not replace
the updated source. Run cases sequentially without other reconstructions on
the same machine. Save repeated trials in separate output directories.

Omitting `--fixed-inner` requests a normal convergence-controlled solve. A time
budget or iteration cap may leave it unfinished; inspect `converged` and
`stop_reason`. The time budget is observed at outer boundaries. Ctrl+C aborts
the benchmark; its lightweight progress file is not a full resumable checkpoint.
The benchmark is independent of the normal experiment cache.

## Existing results and caches

The production cache and experiment identities include scientific source hashes.
Updating the core therefore creates new identities even with the default
schedule. Existing results are retained in their original directories; they
are not automatically relabelled as outputs of this source version. New runs
need fresh preparation and reconstructions under their new identities.

Old checkpoint/handoff import guards remain in place. Do not copy or rename old
cache entries to bypass them. Use the previous Git branch when reproducing an
old source-specific run. The default update has been numerically compared with
the original here, but this is not a blanket migration of historical caches.

The update itself does not launch parameter learning, alter datasets, change
the approved noise plan, or change the reconstruction/alpha grids in any
experiment configuration.
