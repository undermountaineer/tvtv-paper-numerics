"""Alternating PDHG solver for the weighted manuscript model."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from dataclasses import asdict

import numpy as np
from numpy.typing import NDArray

from .energy import transport_residual
from .operators import (
    dt_forward,
    dx_central,
    dy_central,
    estimate_operator_norm_squared,
    spatial_gradient,
    spatial_gradient_adjoint,
    transport_adjoint,
    transport_operator,
    vector_field_gradient,
    vector_field_gradient_adjoint,
)
from .proximal import (
    project_frobenius_tv_ball,
    project_scalar_tv_ball,
    prox_l32_fidelity,
    prox_optical_flow_affine,
)
from .weighted_energy import (
    manuscript_energy,
    weighted_u_block_objective,
    weighted_v_block_objective,
)
from .weights import (
    forward_stencil_lower_envelope,
    validate_weight_map,
)


Array = NDArray[np.floating]


@dataclass(frozen=True)
class WeightedTVTVConfig:
    kappa: float = 1.0

    hx: float = 1.0
    hy: float = 1.0
    ht: float = 1.0

    max_outer_iterations: int = 20
    max_inner_u_iterations: int = 400
    max_inner_v_iterations: int = 400
    min_inner_iterations: int = 10
    # One preserves the existing stopping schedule. Larger intervals are
    # explicit opt-ins; the first eligible and final iterates are always checked.
    inner_check_interval: int = 1

    # The legacy tolerance names are retained as relative tolerances.  The
    # absolute terms make the tests independent of the number of pixels.
    inner_tolerance: float = 1.0e-5
    inner_absolute_tolerance: float = 1.0e-7
    outer_tolerance: float = 1.0e-4
    outer_absolute_tolerance: float = 1.0e-7
    outer_velocity_tolerance_pixels: float | None = None
    outer_energy_tolerance: float = 1.0e-5
    outer_energy_absolute_tolerance: float = 1.0e-10
    required_outer_successes: int = 2

    theta: float = 1.0
    step_safety: float = 0.90
    norm_iterations: int = 40
    update_order: str = "v_then_u"

    objective_increase_tolerance: float = 1.0e-8
    objective_increase_absolute_tolerance: float = 1.0e-10
    max_objective_backtracks: int = 12

    # Explicit opt-in keeps older experiments on their original policy.
    step_policy: str = "legacy"
    residual_mode: str = "legacy"
    balanced_u_step: float = 0.1

    verbose: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.kappa,
            self.balanced_u_step,
            self.hx,
            self.hy,
            self.ht,
            self.inner_tolerance,
            self.inner_absolute_tolerance,
            self.outer_tolerance,
            self.outer_absolute_tolerance,
            self.outer_energy_tolerance,
            self.outer_energy_absolute_tolerance,
            self.step_safety,
            self.objective_increase_tolerance,
            self.objective_increase_absolute_tolerance,
        )

        if min(positive) <= 0.0:
            raise ValueError(
                "parameters, spacings, tolerances, and "
                "safety must be positive"
            )

        if not 0.0 < self.step_safety < 1.0:
            raise ValueError(
                "step_safety must lie strictly between "
                "zero and one"
            )

        if not 0.0 <= self.theta <= 1.0:
            raise ValueError(
                "theta must lie between zero and one"
            )

        counts = (
            self.max_outer_iterations,
            self.max_inner_u_iterations,
            self.max_inner_v_iterations,
            self.min_inner_iterations,
            self.norm_iterations,
            self.required_outer_successes,
            self.max_objective_backtracks,
        )

        if min(counts) < 1:
            raise ValueError(
                "iteration counts must be positive"
            )

        if type(self.inner_check_interval) is not int or self.inner_check_interval < 1:
            raise ValueError("inner_check_interval must be a positive integer")

        if self.outer_velocity_tolerance_pixels is not None and self.outer_velocity_tolerance_pixels <= 0:
            raise ValueError("outer_velocity_tolerance_pixels must be positive")

        if self.step_policy not in {"legacy", "balanced"}:
            raise ValueError("step_policy must be legacy or balanced")
        if self.residual_mode not in {"legacy", "scaled"}:
            raise ValueError("residual_mode must be legacy or scaled")

        if self.update_order not in {"v_then_u", "u_then_v"}:
            raise ValueError(
                "update_order must be 'v_then_u' or 'u_then_v'"
            )


@dataclass
class WeightedBlockResult:
    value: Array
    iterations: int
    residual: float
    converged: bool
    raw_primal_residual: float = np.inf
    raw_dual_residual: float = np.inf
    primal_step_size: float = np.nan
    dual_tv_step_size: float = np.nan
    dual_transport_step_size: float = 0.0
    step_stability_bound: float = np.nan
    primal_residual: float = np.inf
    dual_residual: float = np.inf
    primal_residual_limit: float = np.inf
    dual_residual_limit: float = np.inf
    iterate_change: float = np.inf
    objective_before: float = np.nan
    objective_after: float = np.nan
    accepted_step_scale: float = 1.0
    step_size: float = np.nan
    norm_squared: float = np.nan
    safeguarded: bool = False
    safeguard_adjustment: float = 0.0
    dual: Array | tuple[Array, Array] | None = None
    residual_evaluations: int = 0


@dataclass
class WeightedTVTVResult:
    u: Array
    v: Array
    history: list[dict[str, float]] = field(
        default_factory=list
    )
    outer_iterations: int = 0
    converged: bool = False
    stop_reason: str = "maximum_outer_iterations"
    update_order: str = "v_then_u"


def _rms(value: Array) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(array * array)))


def _combined_rms(values: tuple[Array, ...]) -> float:
    total_size = sum(np.asarray(value).size for value in values)
    if total_size == 0:
        return 0.0
    total = sum(
        float(np.sum(value * value))
        for value in values
    )
    return float(np.sqrt(total / total_size))


def _relative_rms_change(new: Array, old: Array) -> float:
    scale = max(_rms(new), _rms(old), np.finfo(float).eps)
    return float(_rms(new - old) / scale)


def _residual_limit(
    absolute_tolerance: float,
    relative_tolerance: float,
    *scales: float,
) -> float:
    return float(
        absolute_tolerance
        + relative_tolerance * max(0.0, *scales)
    )


@lru_cache(maxsize=128)
def spatial_norm_squared(ny: int, nx: int, hx: float, hy: float) -> float:
    """Exact norm squared of the forward gradient with its Neumann boundary.

    Velocity components and temporal copies do not change this norm. This
    fixed quantity is shared by all outer iterations, parameters and cells.
    """
    lx = 0.0 if nx == 1 else 4*np.cos(np.pi/(2*nx))**2/hx**2
    ly = 0.0 if ny == 1 else 4*np.cos(np.pi/(2*ny))**2/hy**2
    return float(lx + ly)


@lru_cache(maxsize=128)
def _legacy_velocity_norm(shape, hx, hy, iterations):
    return estimate_operator_norm_squared(
        lambda x: vector_field_gradient(x, hx, hy),
        lambda y: vector_field_gradient_adjoint(y, hx, hy),
        shape, iterations=iterations)


def balanced_u_steps(v, shape, config):
    """Scalar primal step and two scalar dual steps with a safe block bound.

    ||T(v)|| <= 2/ht + max|vx|/hx + max|vy|/hy. Allocate half the
    stability budget to each dual block. Each TV component has the same
    dual step, so the existing Euclidean projection remains its exact prox.
    """
    spatial = spatial_norm_squared(shape[1], shape[2], config.hx, config.hy)
    transport = (2/config.ht + np.max(np.abs(v[...,0]))/config.hx
                 + np.max(np.abs(v[...,1]))/config.hy)**2
    tau = config.balanced_u_step
    budget = config.step_safety**2
    sigma_tv = budget/(2*tau*max(spatial,1e-12))
    sigma_transport = budget/(2*tau*max(transport,1e-12))
    return tau, sigma_tv, sigma_transport, float(spatial+transport)


def _scaled_limit(config):
    return config.inner_absolute_tolerance + config.inner_tolerance


def _inner_residual_due(iteration, maximum, config):
    return iteration == maximum or (
        iteration >= config.min_inner_iterations
        and (iteration - config.min_inner_iterations) % config.inner_check_interval == 0
    )


def _objective_safeguard(
    initial: Array,
    candidate: Array,
    objective,
    config: WeightedTVTVConfig,
) -> tuple[Array, float, float, float, bool]:
    """Backtrack an inexact block step until its objective is nonincreasing."""
    objective_before = float(objective(initial))
    objective_after = float(objective(candidate))
    tolerance = (
        config.objective_increase_absolute_tolerance
        + config.objective_increase_tolerance
        * max(1.0, abs(objective_before))
    )

    if objective_after <= objective_before + tolerance:
        return candidate, objective_before, objective_after, 1.0, False

    direction = candidate - initial
    scale = 1.0
    for _ in range(config.max_objective_backtracks):
        scale *= 0.5
        trial = initial + scale * direction
        trial_objective = float(objective(trial))
        if trial_objective <= objective_before + tolerance:
            return trial, objective_before, trial_objective, scale, True

    return initial.copy(), objective_before, objective_before, 0.0, True


def solve_weighted_u_block(
    f: Array,
    v: Array,
    alpha_weight: Array,
    initial_u: Array,
    config: WeightedTVTVConfig,
    initial_dual: tuple[Array, Array] | None = None,
) -> WeightedBlockResult:
    """Solve the weighted image subproblem for fixed velocity."""
    f = np.asarray(f, dtype=float)
    v = np.asarray(v, dtype=float)
    u = np.asarray(
        initial_u,
        dtype=float,
    ).copy()

    if u.shape != f.shape or u.ndim != 3:
        raise ValueError(
            "initial_u and f must have shape (nt, ny, nx)"
        )

    expected_v_shape = (
        f.shape[0] - 1,
        *f.shape[1:],
        2,
    )

    if v.shape != expected_v_shape:
        raise ValueError(
            "v must have shape (nt-1, ny, nx, 2)"
        )

    raw_alpha = validate_weight_map(
        alpha_weight,
        f.shape,
        "alpha_weight",
    )

    alpha = forward_stencil_lower_envelope(raw_alpha)

    def forward(
        x: Array,
    ) -> tuple[Array, Array]:
        return (
            spatial_gradient(
                x,
                config.hx,
                config.hy,
            ),
            transport_operator(
                x,
                v,
                config.ht,
                config.hx,
                config.hy,
            ),
        )

    def adjoint(
        y: tuple[Array, Array],
    ) -> Array:
        return (
            spatial_gradient_adjoint(
                y[0],
                config.hx,
                config.hy,
            )
            + transport_adjoint(
                y[1],
                v,
                nt=f.shape[0],
                ht=config.ht,
                hx=config.hx,
                hy=config.hy,
            )
        )

    if config.step_policy == "balanced":
        tau, sigma_tv, sigma_transport, norm_squared = balanced_u_steps(v,u.shape,config)
        step = tau
        stability = config.step_safety**2
    else:
        norm_squared = max(1e-12, 1.05*estimate_operator_norm_squared(
            forward, adjoint, u.shape, iterations=config.norm_iterations))
        step = config.step_safety/np.sqrt(norm_squared)
        tau = sigma_tv = sigma_transport = step
        stability = tau*sigma_tv*norm_squared

    dual_tv_shape = (*u.shape, 2)
    dual_transport_shape = v.shape[:-1]
    if initial_dual is None:
        dual_tv = np.zeros(dual_tv_shape, dtype=float)
        dual_transport = np.zeros(dual_transport_shape, dtype=float)
    else:
        dual_tv = np.asarray(initial_dual[0], dtype=float).copy()
        dual_transport = np.asarray(initial_dual[1], dtype=float).copy()
        if (
            dual_tv.shape != dual_tv_shape
            or dual_transport.shape != dual_transport_shape
        ):
            raise ValueError("initial_dual has incompatible shapes")

    u_start = u.copy()
    u_bar = u.copy()
    primal_residual = np.inf
    dual_residual = np.inf
    primal_limit = 0.0
    dual_limit = 0.0
    iterate_change = np.inf
    inner_converged = False

    image_scale = max(_rms(f), 0.1)
    residual_evaluations = 0

    objective = lambda candidate: weighted_u_block_objective(
        candidate,
        f,
        v,
        raw_alpha,  # Objective evaluation applies the stencil envelope itself.
        config.kappa,
        config.ht,
        config.hx,
        config.hy,
    )
    initial_block_objective = objective(u_start) if config.residual_mode == "scaled" else 0.0
    early_stop_allowance = (config.objective_increase_absolute_tolerance
        + config.objective_increase_tolerance*max(1.,abs(initial_block_objective)))

    for iteration in range(
        1,
        config.max_inner_u_iterations + 1,
    ):
        # Updates below allocate new primal/dual arrays, so these references
        # remain the preceding iterate without an additional full-array copy.
        old_u = u
        old_dual_tv = dual_tv
        old_dual_transport = dual_transport

        dual_tv = project_scalar_tv_ball(
            dual_tv
            + sigma_tv
            * spatial_gradient(
                u_bar,
                config.hx,
                config.hy,
            ),
            alpha,
        )

        dual_transport = np.clip(
            dual_transport
            + sigma_transport
            * transport_operator(
                u_bar,
                v,
                config.ht,
                config.hx,
                config.hy,
            ),
            -config.kappa,
            config.kappa,
        )

        gradient = spatial_gradient_adjoint(
            dual_tv,
            config.hx,
            config.hy,
        )

        gradient += transport_adjoint(
            dual_transport,
            v,
            nt=f.shape[0],
            ht=config.ht,
            hx=config.hx,
            hy=config.hy,
        )

        u = prox_l32_fidelity(
            u - tau * gradient,
            f,
            tau,
        )

        u_bar = (
            u
            + config.theta * (u - old_u)
        )

        if not _inner_residual_due(iteration, config.max_inner_u_iterations, config):
            continue
        residual_evaluations += 1

        delta_u = old_u - u
        delta_duals = (
            old_dual_tv - dual_tv,
            old_dual_transport - dual_transport,
        )
        adjoint_delta_duals = adjoint(delta_duals)
        forward_delta_u = forward(delta_u)

        primal_residual_array = (
            delta_u / tau - adjoint_delta_duals
        )
        dual_residual_arrays = (
            delta_duals[0] / sigma_tv - forward_delta_u[0],
            delta_duals[1] / sigma_transport - forward_delta_u[1],
        )

        primal_residual = _rms(primal_residual_array)
        dual_residual = _combined_rms(dual_residual_arrays)
        raw_primal_residual, raw_dual_residual = primal_residual, dual_residual
        if config.residual_mode == "scaled":
            primal_residual = tau * raw_primal_residual / image_scale
            dual_residual = max(
                _rms(sigma_tv*dual_residual_arrays[0]/alpha[...,None]),
                _rms(sigma_transport*dual_residual_arrays[1]/config.kappa))
            primal_limit = dual_limit = _scaled_limit(config)
        else:
            primal_limit = _residual_limit(
                config.inner_absolute_tolerance, config.inner_tolerance,
                _rms(delta_u / tau), _rms(adjoint_delta_duals),
            )
            dual_limit = _residual_limit(
                config.inner_absolute_tolerance, config.inner_tolerance,
                _combined_rms((delta_duals[0] / sigma_tv, delta_duals[1] / sigma_transport)),
                _combined_rms(forward_delta_u),
            )
        iterate_change = _relative_rms_change(u, old_u)

        if (
            iteration
            >= config.min_inner_iterations
            and primal_residual <= primal_limit
            and dual_residual <= dual_limit
        ):
            # A residual-qualified iterate that still increases the block
            # objective needs more inner work, rather than an early exit that
            # is subsequently rejected by the safeguard.
            if config.residual_mode == "scaled" and objective(u) > initial_block_objective + early_stop_allowance:
                continue
            inner_converged = True
            break

    raw_candidate = u.copy()
    (
        u,
        objective_before,
        objective_after,
        accepted_scale,
        safeguarded,
    ) = _objective_safeguard(u_start, u, objective, config)
    safeguard_adjustment = _rms(u - raw_candidate)
    safeguard_limit = _residual_limit(
        config.inner_absolute_tolerance,
        config.inner_tolerance,
        _rms(u),
        _rms(raw_candidate),
    )
    if config.residual_mode == "scaled":
        safeguard_limit = _scaled_limit(config) * image_scale
    returned_converged = (
        inner_converged and safeguard_adjustment <= safeguard_limit
    )

    return WeightedBlockResult(
        value=u,
        iterations=iteration,
        residual=max(primal_residual, dual_residual),
        converged=returned_converged,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        primal_residual_limit=primal_limit,
        dual_residual_limit=dual_limit,
        iterate_change=iterate_change,
        objective_before=objective_before,
        objective_after=objective_after,
        accepted_step_scale=accepted_scale,
        step_size=step,
        raw_primal_residual=raw_primal_residual,
        raw_dual_residual=raw_dual_residual,
        primal_step_size=tau, dual_tv_step_size=sigma_tv,
        dual_transport_step_size=sigma_transport, step_stability_bound=stability,
        norm_squared=norm_squared,
        safeguarded=safeguarded,
        safeguard_adjustment=safeguard_adjustment,
        # Both duals remain feasible after projection and are useful warm
        # starts even when the primal candidate was relaxed.
        dual=(dual_tv, dual_transport),
        residual_evaluations=residual_evaluations,
    )


def solve_weighted_v_block(
    u: Array,
    beta_weight: Array,
    initial_v: Array,
    config: WeightedTVTVConfig,
    initial_dual: Array | None = None,
) -> WeightedBlockResult:
    """Solve the weighted velocity subproblem for fixed image."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(
        initial_v,
        dtype=float,
    ).copy()

    expected_shape = (
        u.shape[0] - 1,
        *u.shape[1:],
        2,
    )

    if (
        u.ndim != 3
        or v.shape != expected_shape
    ):
        raise ValueError(
            "u must be (nt, ny, nx) and "
            "v must be (nt-1, ny, nx, 2)"
        )

    raw_beta = validate_weight_map(
        beta_weight,
        v.shape[:-1],
        "beta_weight",
    )

    beta = forward_stencil_lower_envelope(raw_beta)

    def forward(
        x: Array,
    ) -> Array:
        return vector_field_gradient(
            x,
            config.hx,
            config.hy,
        )

    def adjoint(
        y: Array,
    ) -> Array:
        return vector_field_gradient_adjoint(
            y,
            config.hx,
            config.hy,
        )

    # Dividing the v-subproblem by kappa does not change its minimizer.
    tv_radius = beta / config.kappa
    if config.step_policy == "balanced":
        norm_squared = max(1e-12, spatial_norm_squared(
            u.shape[1],u.shape[2],config.hx,config.hy))
        step = config.step_safety/np.sqrt(norm_squared)
        # Balance physical displacement (one pixel) against the dual TV radius.
        balance = min(config.hx,config.hy)/float(np.median(tv_radius))
        tau, sigma = step*balance, step/balance
    else:
        norm_squared = max(1e-12,1.05*_legacy_velocity_norm(
            v.shape,config.hx,config.hy,config.norm_iterations))
        step = config.step_safety/np.sqrt(norm_squared)
        tau = sigma = step

    dual_shape = (*v.shape, 2)
    if initial_dual is None:
        dual_tv = np.zeros(dual_shape, dtype=float)
    else:
        dual_tv = np.asarray(initial_dual, dtype=float).copy()
        if dual_tv.shape != dual_shape:
            raise ValueError("initial_dual has incompatible shape")

    v_start = v.copy()
    v_bar = v.copy()

    u_t = dt_forward(
        u,
        config.ht,
    )

    grad_u = np.stack(
        (
            dx_central(
                u[:-1],
                config.hx,
            ),
            dy_central(
                u[:-1],
                config.hy,
            ),
        ),
        axis=-1,
    )
    grad_norm_squared = (
        grad_u[..., 0] * grad_u[..., 0] + grad_u[..., 1] * grad_u[..., 1]
    )
    residual_evaluations = 0

    primal_residual = np.inf
    dual_residual = np.inf
    primal_limit = 0.0
    dual_limit = 0.0
    iterate_change = np.inf
    inner_converged = False

    objective = lambda candidate: weighted_v_block_objective(
        candidate,
        u,
        raw_beta,  # Objective evaluation applies the stencil envelope itself.
        config.kappa,
        config.ht,
        config.hx,
        config.hy,
    )
    initial_block_objective = objective(v_start) if config.residual_mode == "scaled" else 0.0
    early_stop_allowance = (config.objective_increase_absolute_tolerance
        + config.objective_increase_tolerance*max(1.,abs(initial_block_objective)))

    for iteration in range(
        1,
        config.max_inner_v_iterations + 1,
    ):
        old_v = v
        old_dual_tv = dual_tv

        dual_tv = project_frobenius_tv_ball(
            dual_tv
            + sigma
            * vector_field_gradient(
                v_bar,
                config.hx,
                config.hy,
            ),
            tv_radius,
        )

        proximal_argument = (
            v
            - tau
            * vector_field_gradient_adjoint(
                dual_tv,
                config.hx,
                config.hy,
            )
        )

        v = prox_optical_flow_affine(
            proximal_argument,
            u_t,
            grad_u,
            tau,
            gradient_norm_squared=grad_norm_squared,
        )

        v_bar = (
            v
            + config.theta * (v - old_v)
        )

        if not _inner_residual_due(iteration, config.max_inner_v_iterations, config):
            continue
        residual_evaluations += 1

        delta_v = old_v - v
        delta_dual = old_dual_tv - dual_tv
        adjoint_delta_dual = adjoint(delta_dual)
        forward_delta_v = forward(delta_v)

        primal_residual_array = (
            delta_v / tau - adjoint_delta_dual
        )
        dual_residual_array = (
            delta_dual / sigma - forward_delta_v
        )
        primal_residual = _rms(primal_residual_array)
        dual_residual = _rms(dual_residual_array)
        raw_primal_residual, raw_dual_residual = primal_residual, dual_residual
        if config.residual_mode == "scaled":
            primal_residual = _rms(tau*primal_residual_array/np.array([config.hx,config.hy]))
            dual_residual = _rms(sigma*dual_residual_array/tv_radius[...,None,None])
            primal_limit = dual_limit = _scaled_limit(config)
        else:
            primal_limit = _residual_limit(
                config.inner_absolute_tolerance, config.inner_tolerance,
                _rms(delta_v / tau), _rms(adjoint_delta_dual),
            )
            dual_limit = _residual_limit(
                config.inner_absolute_tolerance, config.inner_tolerance,
                _rms(delta_dual / sigma), _rms(forward_delta_v),
            )
        iterate_change = _relative_rms_change(v, old_v)

        if (
            iteration
            >= config.min_inner_iterations
            and primal_residual <= primal_limit
            and dual_residual <= dual_limit
        ):
            # A residual-qualified iterate that still increases the block
            # objective needs more inner work, rather than an early exit that
            # is subsequently rejected by the safeguard.
            if config.residual_mode == "scaled" and objective(v) > initial_block_objective + early_stop_allowance:
                continue
            inner_converged = True
            break

    raw_candidate = v.copy()
    (
        v,
        objective_before,
        objective_after,
        accepted_scale,
        safeguarded,
    ) = _objective_safeguard(v_start, v, objective, config)
    safeguard_adjustment = _rms(v - raw_candidate)
    safeguard_limit = _residual_limit(
        config.inner_absolute_tolerance,
        config.inner_tolerance,
        _rms(v),
        _rms(raw_candidate),
    )
    if config.residual_mode == "scaled":
        safeguard_limit = _scaled_limit(config)*min(config.hx,config.hy)
    returned_converged = (
        inner_converged and safeguard_adjustment <= safeguard_limit
    )

    return WeightedBlockResult(
        value=v,
        iterations=iteration,
        residual=max(primal_residual, dual_residual),
        converged=returned_converged,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        primal_residual_limit=primal_limit,
        dual_residual_limit=dual_limit,
        iterate_change=iterate_change,
        objective_before=objective_before,
        objective_after=objective_after,
        accepted_step_scale=accepted_scale,
        step_size=step,
        raw_primal_residual=raw_primal_residual,
        raw_dual_residual=raw_dual_residual,
        primal_step_size=tau, dual_tv_step_size=sigma,
        step_stability_bound=tau*sigma*norm_squared,
        norm_squared=norm_squared,
        safeguarded=safeguarded,
        safeguard_adjustment=safeguard_adjustment,
        # Feasibility is preserved by the projection, so keep the dual state
        # when the primal safeguard is active instead of restarting PDHG.
        dual=dual_tv,
        residual_evaluations=residual_evaluations,
    )


CHECKPOINT_PROTOCOL = 1


def _checkpoint_problem_id(f, alpha, beta, u0, v0, config):
    settings = asdict(config)
    # Outer cap and logging are execution limits, not update rules.
    settings.pop("max_outer_iterations")
    settings.pop("verbose")
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for value in (f, alpha, beta, u0, v0):
        array = np.ascontiguousarray(value, dtype=float)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def solve_weighted_tvtv(
    f: Array,
    alpha_weight: Array,
    beta_weight: Array,
    config: WeightedTVTVConfig,
    initial_u: Array | None = None,
    initial_v: Array | None = None,
    *,
    resume_state: dict | None = None,
    checkpoint_callback=None,
    should_stop=None,
) -> WeightedTVTVResult:
    """Run alternating minimization for fixed weight maps.

    Optional checkpoints are taken at completed outer iterations. They retain
    primal/dual arrays and the stopping history. A stop request is observed at
    an outer boundary; an unfinished inner solve is not checkpointed.
    """
    f = np.asarray(f, dtype=float)

    if f.ndim != 3 or f.shape[0] < 2:
        raise ValueError(
            "f must have shape (nt, ny, nx) with nt>=2"
        )

    alpha = validate_weight_map(
        alpha_weight,
        f.shape,
        "alpha_weight",
    )

    v_shape = (
        f.shape[0] - 1,
        *f.shape[1:],
        2,
    )

    beta = validate_weight_map(
        beta_weight,
        v_shape[:-1],
        "beta_weight",
    )

    if initial_u is None:
        u = f.copy()
    else:
        u = np.asarray(
            initial_u,
            dtype=float,
        ).copy()

    if initial_v is None:
        v = np.zeros(
            v_shape,
            dtype=float,
        )
    else:
        v = np.asarray(
            initial_v,
            dtype=float,
        ).copy()

    if (
        u.shape != f.shape
        or v.shape != v_shape
    ):
        raise ValueError(
            "initial values have incompatible shapes"
        )

    initial_energy = manuscript_energy(
        u,
        v,
        f,
        alpha,
        beta,
        config.kappa,
        config.ht,
        config.hx,
        config.hy,
    )

    history: list[dict[str, float]] = [
        {
            "iteration": 0.0,
            **initial_energy,
            "transport_residual": transport_residual(
                u, v, config.ht, config.hx, config.hy
            ),
            "relative_u_change": 0.0,
            "relative_v_change": 0.0,
            "absolute_u_change": 0.0,
            "absolute_v_change": 0.0,
            "u_change_limit": 0.0,
            "v_change_limit": 0.0,
            "energy_relative_change": 0.0,
            "energy_change_limit": 0.0,
            "outer_consecutive_successes": 0.0,
            "u_inner_iterations": 0.0,
            "v_inner_iterations": 0.0,
            "u_residual_evaluations": 0.0,
            "v_residual_evaluations": 0.0,
            "u_inner_residual": 0.0,
            "v_inner_residual": 0.0,
            "u_primal_residual": 0.0,
            "u_dual_residual": 0.0,
            "v_primal_residual": 0.0,
            "v_dual_residual": 0.0,
            "u_primal_residual_ratio": 0.0,
            "u_dual_residual_ratio": 0.0,
            "v_primal_residual_ratio": 0.0,
            "v_dual_residual_ratio": 0.0,
            "u_raw_primal_residual": 0.0,
            "u_raw_dual_residual": 0.0,
            "v_raw_primal_residual": 0.0,
            "v_raw_dual_residual": 0.0,
            "u_primal_step": 0.0,
            "u_dual_tv_step": 0.0,
            "u_dual_transport_step": 0.0,
            "v_primal_step": 0.0,
            "v_dual_tv_step": 0.0,
            "u_iterate_change": 0.0,
            "v_iterate_change": 0.0,
            "u_block_objective_change": 0.0,
            "v_block_objective_change": 0.0,
            "u_accepted_step_scale": 1.0,
            "v_accepted_step_scale": 1.0,
            "u_objective_safeguarded": 0.0,
            "v_objective_safeguarded": 0.0,
            "u_safeguard_adjustment": 0.0,
            "v_safeguard_adjustment": 0.0,
        }
    ]

    previous_total = float(initial_energy["total"])
    consecutive_successes = 0
    u_dual: tuple[Array, Array] | None = None
    v_dual: Array | None = None

    completed_outer = 0
    problem_id = None
    if resume_state is not None or checkpoint_callback is not None:
        problem_id = _checkpoint_problem_id(f, alpha, beta, u, v, config)
    if resume_state is not None:
        if resume_state.get("schema") != CHECKPOINT_PROTOCOL or resume_state.get("problem_id") != problem_id:
            raise ValueError("Checkpoint does not match the observations, weights, initial state or solver settings.")
        completed_outer = int(resume_state["outer_iterations"])
        u = np.asarray(resume_state["u"], dtype=float).copy()
        v = np.asarray(resume_state["v"], dtype=float).copy()
        if u.shape != f.shape or v.shape != v_shape:
            raise ValueError("Checkpoint primal shapes are incompatible.")
        saved_u_dual = resume_state["u_dual"]
        saved_v_dual = resume_state["v_dual"]
        u_dual = None if saved_u_dual is None else tuple(np.asarray(x, dtype=float).copy() for x in saved_u_dual)
        v_dual = None if saved_v_dual is None else np.asarray(saved_v_dual, dtype=float).copy()
        history = [dict(record) for record in resume_state["history"]]
        previous_total = float(resume_state["previous_total"])
        consecutive_successes = int(resume_state["consecutive_successes"])
        if completed_outer < 0 or len(history) != completed_outer + 1 or int(history[-1]["iteration"]) != completed_outer:
            raise ValueError("Checkpoint iteration history is inconsistent.")
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError("Checkpoint contains non-finite primal values.")
        if completed_outer and (u_dual is None or v_dual is None):
            raise ValueError("A completed outer checkpoint must include both dual states.")
        if consecutive_successes >= config.required_outer_successes:
            return WeightedTVTVResult(u=u, v=v, history=history,
                outer_iterations=completed_outer, converged=True,
                stop_reason="converged", update_order=config.update_order)

    def checkpoint():
        if checkpoint_callback is not None:
            checkpoint_callback(dict(schema=CHECKPOINT_PROTOCOL,
                problem_id=problem_id, u=u, v=v, u_dual=u_dual, v_dual=v_dual,
                history=history, outer_iterations=completed_outer,
                previous_total=previous_total,
                consecutive_successes=consecutive_successes))

    if resume_state is None:
        checkpoint()
    for outer_iteration in range(
        completed_outer + 1,
        config.max_outer_iterations + 1,
    ):
        if should_stop is not None and should_stop():
            return WeightedTVTVResult(u=u, v=v, history=history,
                outer_iterations=completed_outer, converged=False,
                stop_reason="checkpoint_stop", update_order=config.update_order)
        old_u = u.copy()
        old_v = v.copy()

        if config.update_order == "v_then_u":
            v_result = solve_weighted_v_block(
                u, beta, v, config, initial_dual=v_dual
            )
            v = v_result.value
            v_dual = (
                v_result.dual
                if isinstance(v_result.dual, np.ndarray)
                else None
            )
            u_result = solve_weighted_u_block(
                f, v, alpha, u, config, initial_dual=u_dual
            )
            u = u_result.value
            u_dual = (
                u_result.dual
                if isinstance(u_result.dual, tuple)
                else None
            )
        else:
            u_result = solve_weighted_u_block(
                f, v, alpha, u, config, initial_dual=u_dual
            )
            u = u_result.value
            u_dual = (
                u_result.dual
                if isinstance(u_result.dual, tuple)
                else None
            )
            v_result = solve_weighted_v_block(
                u, beta, v, config, initial_dual=v_dual
            )
            v = v_result.value
            v_dual = (
                v_result.dual
                if isinstance(v_result.dual, np.ndarray)
                else None
            )

        absolute_u_change = _rms(u - old_u)
        absolute_v_change = _rms(v - old_v)
        u_change_limit = _residual_limit(
            config.outer_absolute_tolerance,
            config.outer_tolerance,
            _rms(u),
            _rms(old_u),
        )
        v_change_limit = _residual_limit(
            config.outer_absolute_tolerance,
            config.outer_tolerance,
            _rms(v),
            _rms(old_v),
        )
        if config.residual_mode == "scaled":
            u_change_limit = (config.outer_absolute_tolerance + config.outer_tolerance)*max(_rms(f),0.1)
            pixels = np.array([config.hx,config.hy])
            v_change_limit = min(config.hx,config.hy)*(config.outer_absolute_tolerance
                + config.outer_tolerance*max(1.,_rms(v/pixels),_rms(old_v/pixels)))
            if config.outer_velocity_tolerance_pixels is not None:
                v_change_limit = config.outer_velocity_tolerance_pixels*min(config.hx,config.hy)
        u_change = _relative_rms_change(u, old_u)
        v_change = _relative_rms_change(v, old_v)

        energy = manuscript_energy(
            u,
            v,
            f,
            alpha,
            beta,
            config.kappa,
            config.ht,
            config.hx,
            config.hy,
        )
        total = float(energy["total"])
        absolute_energy_change = abs(total - previous_total)
        energy_scale = max(abs(total), abs(previous_total), np.finfo(float).eps)
        energy_relative_change = absolute_energy_change / energy_scale
        energy_change_limit = (
            config.outer_energy_absolute_tolerance
            + config.outer_energy_tolerance * energy_scale
        )

        outer_success = (
            absolute_u_change <= u_change_limit
            and absolute_v_change <= v_change_limit
            and absolute_energy_change <= energy_change_limit
            and u_result.converged
            and v_result.converged
        )
        consecutive_successes = (
            consecutive_successes + 1 if outer_success else 0
        )

        record = {
            "iteration": float(outer_iteration),
            **energy,
            "transport_residual": transport_residual(
                u, v, config.ht, config.hx, config.hy
            ),
            "relative_u_change": u_change,
            "relative_v_change": v_change,
            "absolute_u_change": absolute_u_change,
            "absolute_v_change": absolute_v_change,
            "u_change_limit": u_change_limit,
            "v_change_limit": v_change_limit,
            "energy_relative_change": energy_relative_change,
            "energy_change_limit": energy_change_limit,
            "outer_consecutive_successes": float(consecutive_successes),
            "u_inner_iterations": float(u_result.iterations),
            "v_inner_iterations": float(v_result.iterations),
            "u_residual_evaluations": float(u_result.residual_evaluations),
            "v_residual_evaluations": float(v_result.residual_evaluations),
            "u_inner_residual": u_result.residual,
            "v_inner_residual": v_result.residual,
            "u_primal_residual": u_result.primal_residual,
            "u_dual_residual": u_result.dual_residual,
            "v_primal_residual": v_result.primal_residual,
            "v_dual_residual": v_result.dual_residual,
            "u_primal_residual_ratio": (
                u_result.primal_residual / u_result.primal_residual_limit
            ),
            "u_dual_residual_ratio": (
                u_result.dual_residual / u_result.dual_residual_limit
            ),
            "v_primal_residual_ratio": (
                v_result.primal_residual / v_result.primal_residual_limit
            ),
            "v_dual_residual_ratio": (
                v_result.dual_residual / v_result.dual_residual_limit
            ),
            "u_raw_primal_residual": u_result.raw_primal_residual,
            "u_raw_dual_residual": u_result.raw_dual_residual,
            "v_raw_primal_residual": v_result.raw_primal_residual,
            "v_raw_dual_residual": v_result.raw_dual_residual,
            "u_primal_step": u_result.primal_step_size,
            "u_dual_tv_step": u_result.dual_tv_step_size,
            "u_dual_transport_step": u_result.dual_transport_step_size,
            "v_primal_step": v_result.primal_step_size,
            "v_dual_tv_step": v_result.dual_tv_step_size,
            "u_iterate_change": u_result.iterate_change,
            "v_iterate_change": v_result.iterate_change,
            "u_block_objective_change": (
                u_result.objective_after - u_result.objective_before
            ),
            "v_block_objective_change": (
                v_result.objective_after - v_result.objective_before
            ),
            "u_accepted_step_scale": u_result.accepted_step_scale,
            "v_accepted_step_scale": v_result.accepted_step_scale,
            "u_objective_safeguarded": float(u_result.safeguarded),
            "v_objective_safeguarded": float(v_result.safeguarded),
            "u_safeguard_adjustment": u_result.safeguard_adjustment,
            "v_safeguard_adjustment": v_result.safeguard_adjustment,
        }
        history.append(record)

        if config.verbose:
            print(
                f"outer={outer_iteration:3d} "
                f"energy={total:.6e} "
                f"transport={record['transport_residual']:.3e} "
                f"du={absolute_u_change:.3e}/{u_change_limit:.3e} "
                f"dv={absolute_v_change:.3e}/{v_change_limit:.3e} "
                f"primal-dual="
                f"{max(record['u_primal_residual_ratio'], record['u_dual_residual_ratio'], record['v_primal_residual_ratio'], record['v_dual_residual_ratio']):.2e}"
            )

        previous_total = total
        completed_outer = outer_iteration
        checkpoint()

        if consecutive_successes >= config.required_outer_successes:
            return WeightedTVTVResult(
                u=u,
                v=v,
                history=history,
                outer_iterations=outer_iteration,
                converged=True,
                stop_reason="converged",
                update_order=config.update_order,
            )

    return WeightedTVTVResult(
        u=u,
        v=v,
        history=history,
        outer_iterations=completed_outer,
        converged=False,
        stop_reason="maximum_outer_iterations",
        update_order=config.update_order,
    )
