# Minimal route to a numerical proof of concept

## Question and scope

One synthetic scene with fine structure, uniform noise, and one-level spatial
learning. Compare a tuned constant alpha with the accepted learned alpha map.
Keep the joint image/motion model, source and numerical accuracy unchanged.

The default 512×512 grid is both the observation grid and reconstruction grid.
The regularization partition is a different grid: one whole-domain weight or
four quadrant weights. The main config changes resolution 128→512 and maximum
spatial depth 2→1. Spatial spacings are set consistently; crop spacings remain
those of the full image. No artificial split is forced to obtain a positive result.

## Today's sequence

1. Tune a full-domain constant on the design observation using the existing
   bounded scalar search. All declared candidates must converge before ranking.
   This also measures actual 512 convergence times on the user's computer.
2. Fit the four spatial children and apply the strict existing crop-cost test.
   Stop at that level even if the split is accepted.
3. At tighter, matched settings, reconstruct the selected constant and accepted
   map on the full image. If the split is rejected, the existing proposed-map
   diagnostic can reveal whether the local criterion and global benefit disagree.
4. Write figures, candidate timings, losses, and the acceptance decision.

The budget is a scheduling bound, not permission to rank unfinished solves.
No numerical result from another protocol/resolution is treated as a new result.

## What to decide with the colleague

| Result | Decision |
|---|---|
| Accepted nonconstant map, meaningful global improvement | Freeze it; run the three declared fresh-noise seeds, then write up this example. |
| Accepted map, improvement comparable to accuracy/search effects | Check the constant parameter search and tighter accuracy before claiming a gain. |
| Split rejected, proposed map helps globally | Discuss the local crop criterion/boundary effects; deeper refinement cannot be reached. |
| Split rejected and proposed map does not help | Change the example deliberately; more tuning on this same setup is low priority. |
| 512 solve remains too expensive | Measure time and convergence status; use a clearly labelled 256 pilot for the meeting, or review accuracy requirements together. |
| Search hits its allowed boundary | Review/widen the declared search; do not call the endpoint optimal. |
| Iteration cap or numerical failure | Inspect final diagnostics; do not launch a larger study with unresolved solves. |

No fixed numerical gain threshold is invented here. Inspect upper loss, RMSE,
fine-structure images, parameter selection accuracy, and eventual fresh-noise
consistency together. If the gain remains tiny, prioritize a simpler convincing
example over increasing tree depth or adding many images.

## Smallest useful manuscript package

One synthetic sequence; one table with constant versus learned-map loss/RMSE;
one overview/detail figure; the learned map; three frozen fresh-noise checks;
and an explicit statement of initialization, discretization and stopping rules.
The meeting figures/report are working outputs, not final publication figures.
An ordinary image with spatially varying noise can be an optional second
experiment after the main example succeeds. It is not required to start today.
