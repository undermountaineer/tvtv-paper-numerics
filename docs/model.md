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

## Synthetic protocols

The clean input is the versioned five-frame 1024 master. Each protocol declares
the observation/reconstruction grid; clean block averaging precedes independent
Gaussian noise. The current protocols use 256 or 512 pixels per spatial axis,
noise standard deviation 0.1, and the physical translation `(0.003, 0.0015)`.
The same working-grid reference is used to score each reconstruction.

Alpha is selected by bounded scalar search. Beta=0.0125 and kappa=1 are fixed.
Each local region retains all time frames and the physical pixel spacing.
The current depth-one partition is constant or has four spatial children.
The accepted map is assembled into a single global reconstruction; independent
crop reconstructions are used for parameter selection and the split test only.

See [experiment protocol](experiment_protocol.md) for the constant search,
local acceptance test, full-domain comparison and frozen fresh-noise evaluation.
Exact solver caps and stopping thresholds are in the configuration files.

## Solver and interpretation

The implementation alternates convex image and motion subproblems using PDHG.
The scaled solver checks the residuals of the safeguarded returned primal/dual
state. The outer loop also checks image and motion changes, energy change and
two consecutive successful updates. An unfinished solve cannot enter parameter
selection. Source and problem identities guard checkpoint reuse.

BDS supplies the alternating joint reconstruction/motion structure. The 3/2
fidelity and joint Frobenius motion-TV norm follow the model implemented here.
DFFI motivates local scalar learning and spatial dyadic refinement. Bounded
scalar search replaces its convex sensitivity/piggyback parameter update in
this nonconvex joint problem. These are documented adaptations, not an exact
reproduction of either paper's numerical experiments.

The numerical energy uses a finite transport penalty. It does not enforce an
exact optical-flow constraint or the theoretical velocity/divergence bounds.
Finite residual convergence does not prove a global joint minimizer or an exact
optimistic bilevel solution. Local split acceptance and global reconstruction
gain must be reported separately. Fresh-noise checks on one scene assess that
setup; they do not establish generalization to other scenes.

## Reference works

- Burger, Dirks and Schönlieb: *A variational model for joint motion estimation
  and image reconstruction* (2018).
- Davoli, Ferreira, Fonseca and Iglesias: *Spatially varying regularization
  parameters for TV* (2025), local learning and dyadic refinement framework.
