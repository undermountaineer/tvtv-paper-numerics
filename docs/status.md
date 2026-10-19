# Reviewed status — October 7, 2026

The local alpha choices produce a small, repeatable improvement after a global
joint reconstruction. This is enough to keep the current example as a proof
of concept; it is not yet a persuasive automatic refinement result.

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

## Sensible continuation

Use the imported results and single notebook to explain the current example
to the collaborator. The immediate scientific decision is how to improve the
size of the global gain while addressing crop-boundary costs in a refinement
rule. Keep that distinction explicit before introducing deeper partitions,
real data, or a wider beta search. The current branch retains everything
needed to reproduce the one-level selection and global/fresh-noise comparisons.

## Evidence and provenance

`evidence/reviewed_summary.json`, `local_selection.csv` and `fresh_noise.csv`
are condensed from the supplied `tvtv-handoff-2026-10-07.zip`. They describe
historical completed runs from branch `feature/minimal-dyadic-proof-of-concept`,
commit `10d6450`. They are not outputs of new cleanup computations. Full arrays
and histories remain in the original handoff and can be imported by the single
runner. Newly generated reports belong under `results/`.
