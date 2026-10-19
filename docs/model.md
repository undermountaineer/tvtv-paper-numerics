# Numerical model and experiment contract

For an observation sequence `f`, the implementation alternates image and
velocity blocks for the penalized energy

\[
E(u,v)=h_xh_yh_t\left[
\frac23\sum |u-f|^{3/2}
+\sum \alpha\,|\nabla^+u|
+\sum \beta\,|\nabla^+v|_F
+\kappa\sum|D_tu+\nabla^c u\cdot v|
\right].
\]

The image TV is isotropic; velocity TV uses the joint Frobenius norm.
Forward spatial differences define TV and central spatial differences define
transport. Motion uses `u[t]` in the transport term. Grid spacings are explicit;
spatial boundaries use zero differences/Neumann conventions. The solver takes
the minimum of weights touched by each forward gradient stencil at a spatial
interface. It never averages discontinuous weights across that interface.

`u` has shape `(nt, ny, nx)`, and `v` has shape `(nt-1, ny, nx, 2)`, with
components `(vx, vy)`. Computations use float64. Both blocks are estimated in
every joint solve. `u0=f`; a pooled small-translation fit on smoothed
observations provides `v0`. Reference images and motion never enter that fit.
Reference motion is used only for evaluation.

The supervised image loss is

\[
J(u)=h_xh_yh_t\sum_t w_t\sum_{y,x}|u-u_{\rm ref}|^{3/2},
\qquad w=(1/2,1,\ldots,1,1/2).
\]

This is the image-only upper loss (`gamma_u=1`, `gamma_v=0`), without a factor
`2/3`. RMSE, PSNR and motion error are additional diagnostics. Crop losses
retain the original physical spacings and full time interval.

## Current proof of concept

Five 128×128 frames show a translating scene with smooth shading, sharp
interfaces and resolved texture. Independent unclipped Gaussian noise has
standard deviation 0.1. The scene is rendered by 4×4 midpoint sampling per
pixel. The continuous velocity is `(0.003, 0.0015)` in domain units per time.

Only alpha is varied in this experiment; beta=0.0125 and kappa=1 are fixed.
The tested alpha shortlist is `0.00075, 0.000875, 0.001`. Four spatial crops
each contain all five frames. The reviewed local choices are 0.001 on the top
half and 0.000875 on the bottom half. The assembled map is always reconstructed
in a global joint solve; a crop mosaic is not the global solution.

The local velocity block has the reviewed 500-iteration cap. Global nominal
blocks have caps of 250; tighter blocks have caps of 400 and halved stopping
tolerances. All current exact settings are in the one configuration file.
Constant and spatial results are compared at matching tighter settings.

The map is frozen before the three validation observations are generated.
Each observation has its own observation-derived initial motion shared by all
four competitors. The best of three constants on each validation observation
is an oracle shortlist comparator, not a globally optimal parameter choice.

## Interpretation and relation to the papers

Burger–Dirks–Schönlieb's joint reconstruction/motion model supplies the basic
TV–TV coupling and alternating-solver context. The active manuscript version
uses the 3/2 fidelity and vectorial velocity TV specified above. The separate
squared-L2 BDS solver is unnecessary for this comparison and has been removed.

Davoli–Ferreira–Fonseca–Iglesias motivates local parameter selection and dyadic
partitions. Its strict cost logic must be distinguished from assembling a
proposed map and evaluating its global reconstruction. Here the strict local
split is rejected. A global gain therefore supports the usefulness of spatial
weights, while successful automatic dyadic refinement remains unresolved.

The numerical problem uses a finite transport penalty, alternating block
minimization and finite parameter shortlists. It does not enforce the exact
optical-flow constraint or the manuscript's theoretical velocity/divergence
bounds. Convergence means the implemented residual, change and energy criteria
were met. It does not establish a global joint minimizer or an exact optimistic
bilevel solution. This example is evidence for the numerical approximation of
the model; the three noise samples do not establish statistical significance
or generalization to other scenes.
