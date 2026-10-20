# Correct the interaction between residual stopping and objective damping

The October 19 256-grid run completed alpha=0.00075 in 52 outer iterations,
but alpha=0.000875 reached the cap of 128. In its last image block, recorded
primal and dual residuals passed their limits. A separate condition requiring
the *raw* PDHG image candidate to avoid even a roughly 1e-8 objective increase
prevented that block from being marked converged. It consumed 250 inner steps,
then accepted a 1/16 damped step. Outer changes and energy change passed, but
the false inner flag reset the consecutive-success counter. No spatial split
or global constant-versus-spatial comparison was reached.

## Corrected policy

The PDHG updates, energy, stencil operators, proximal maps, step sizes, weights,
data, initialization, nominal/tighter accuracy settings and objective safeguard
are unchanged. The change is in the stopping decisions for `residual_mode=scaled`.

1. Test the existing incremental primal/dual residuals after the minimum inner
   iteration count.
2. Apply the existing objective safeguard to a residual-qualified candidate.
3. Check proximal fixed-point residuals at the **returned safeguarded primal
   state and current feasible dual state**. If they fail, or damping changes
   the candidate too much, continue the inner PDHG loop from its raw state.
4. Stop that block only when the safeguarded state passes. Recheck the final
   returned state, including when the iteration cap was reached. A cap below
   the minimum iteration count cannot be labelled converged.

The same procedure applies to image and motion blocks. The stricter raw
objective gate is removed; the objective safeguard itself retains its original
tolerances and backtracking. A zero or tiny damped step alone does not prove
convergence. Legacy residual mode retains its previous stopping behavior.

The new reported scaled residuals are the normalized fixed-point defects

    u = prox_(tau G)(u - tau K* p)
    p = prox_(Sigma F*)(p + Sigma K u)

at the returned state. Image primal differences are normalized by observation
RMS; motion differences by pixel spacing. Dual differences are normalized by
the unchanged constraint-ball radii. These defects vanish at a saddle point of
each fixed-block convex problem. Passing finite thresholds is an approximate
block certificate, not a proof of a global joint minimizer. The outer loop
still requires both blocks, image/motion/energy changes, and two consecutive
successful outer iterations. Histories mark the new checks with
`u_returned_state_checked` and `v_returned_state_checked`.

The final stopping iterate can change. This is not a bitwise-equivalent runtime
optimization. Controlled fixed-work comparisons retain bitwise-identical image,
motion and dual updates. Check tightness before interpreting small learning
gains; all final comparisons must use matched corrected settings.

## Explicit continuation

`scripts/resume_stopping_fix.py --config configs/meeting_256.json` creates
`configs/meeting_256_stopping_fixed.json`. Its outer cap is at least 256 so a
checkpoint already at outer=128 has room for fresh checks. This execution limit
does not change the residual/change tolerances. Original outputs stay in place.

Continuation accepts only the exact reviewed old and new scientific source
identities, matching master/workflow/settings, and exact saved observations and
initial states. It verifies each checkpoint's problem identity, shapes, finite
values, dual feasibility and recorded energy. It copies the local primal/dual
state and its history, resets the consecutive-success counter, and imports no
completed-result flag. Two new successful outer iterations are required.

Even previously completed local candidates are rechecked. Existing progress
under the corrected source is never overwritten by a repeated import. The
transition and old history prefix are recorded in
`STOPPING_FIX_CONTINUATION.json` and exposed in the meeting report. Old history
records retain their original meaning; new fixed-point metrics begin after
the recorded source transition.

Only local training states are continued. The final tighter constant/spatial
comparisons run afresh from the common observation-based initialization under
the corrected policy. The scalar parameter search and strict local partition
acceptance rule are unchanged. The script does not bypass arbitrary source
changes or reuse old result eligibility.

See the update bundle's verification records for regression tests and actual
256-grid probes. Server timings are not predictions for another computer.
