# Historical 128-grid evidence — October 7, 2026

These results describe the earlier directly rendered 128×128 frozen-map
experiment. They are historical evidence, not outputs of the current 1024-master
256/512 protocols. Current result records are generated per configuration and
source identity under `results/`.

The historical local alpha choices produced a small improvement after global
joint reconstruction, while the strict crop-cost test rejected the split.

| Observation | Constant image loss | Spatial image loss | Relative loss gain |
|---|---:|---:|---:|
| Design, seed 20260930; tighter control alpha=0.000875 | 0.007707352 | 0.007643402 | 0.830% |
| Fresh seed 202610071; best of 3 constants | 0.007643506 | 0.007580603 | 0.823% |
| Fresh seed 202610072; best of 3 constants | 0.007801311 | 0.007708078 | 1.195% |
| Fresh seed 202610073; best of 3 constants | 0.007442811 | 0.007324475 | 1.590% |

All 12 fresh-noise solves converged. Alpha=0.000875 was the best tested constant
in every fresh observation. The mean loss gain is 1.203%. The RMSE gains are
only about 0.16–0.44%; texture RMSE is essentially unchanged. A large visual or
texture-recovery improvement should not be claimed.

The strict crop comparison is a separate result:

| Quantity | Image loss |
|---|---:|
| Parent global nominal reconstruction | 0.007726162 |
| Four independent crops at parent alpha | 0.007914797 |
| Four independent crops at their selected alphas | 0.007850382 |

Local alpha selection lowers the independent-crop cost, but the selected crop
sum is still above the parent's global cost. The strict split test therefore
rejects the split. The top-cell choices are on the upper shortlist boundary;
the global constant optimum has only been tested in a finite shortlist.

## Evidence and provenance

`evidence/reviewed_summary.json`, `local_selection.csv` and `fresh_noise.csv`
are condensed from the supplied `tvtv-handoff-2026-10-07.zip`. They describe
historical completed runs from branch `feature/minimal-dyadic-proof-of-concept`,
commit `10d6450`. They are not outputs of new cleanup computations. Full arrays
and histories belong to their original archive and scientific source version.
Source-compatibility checks apply to any attempted import. Newly generated
reports belong under `results/`.
