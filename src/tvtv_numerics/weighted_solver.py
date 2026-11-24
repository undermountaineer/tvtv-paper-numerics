"""Alternating PDHG solver for the weighted manuscript model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from .operators import (
    dt_forward, dx_central, dy_central, spatial_gradient, spatial_gradient_adjoint,
    transport_adjoint, transport_operator, vector_field_gradient, vector_field_gradient_adjoint,
)
from .proximal import (
    project_frobenius_tv_ball, project_scalar_tv_ball, prox_l32_fidelity, prox_optical_flow_affine,
)
from .weighted_energy import (
    manuscript_energy, transport_residual, weighted_u_block_objective, weighted_v_block_objective,
)
from .weights import forward_stencil_lower_envelope, validate_weight_map


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
    # Residuals reuse K x and K* p, so checking every eligible iterate is cheap.
    # Larger intervals can change the stopping iterate.
    inner_check_interval: int = 1

    # Scaled inner limit: inner_absolute_tolerance + inner_tolerance.
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
    balanced_u_step: float = 0.1
    update_order: str = "v_then_u"

    objective_increase_tolerance: float = 1.0e-8
    objective_increase_absolute_tolerance: float = 1.0e-10
    max_objective_backtracks: int = 12

    verbose: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.kappa, self.balanced_u_step, self.hx, self.hy, self.ht,
            self.inner_tolerance, self.inner_absolute_tolerance,
            self.outer_tolerance, self.outer_absolute_tolerance,
            self.outer_energy_tolerance, self.outer_energy_absolute_tolerance,
            self.objective_increase_tolerance, self.objective_increase_absolute_tolerance,
        )
        if min(positive) <= 0.0:
            raise ValueError("parameters, spacings and tolerances must be positive")
        if not 0.0 < self.step_safety < 1.0:
            raise ValueError("step_safety must lie strictly between zero and one")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must lie between zero and one")
        counts = (
            self.max_outer_iterations, self.max_inner_u_iterations, self.max_inner_v_iterations,
            self.min_inner_iterations, self.required_outer_successes, self.max_objective_backtracks,
        )
        if min(counts) < 1:
            raise ValueError("iteration counts must be positive")
        if type(self.inner_check_interval) is not int or self.inner_check_interval < 1:
            raise ValueError("inner_check_interval must be a positive integer")
        if self.outer_velocity_tolerance_pixels is not None and self.outer_velocity_tolerance_pixels <= 0:
            raise ValueError("outer_velocity_tolerance_pixels must be positive")
        if self.update_order not in {"v_then_u", "u_then_v"}:
            raise ValueError("update_order must be 'v_then_u' or 'u_then_v'")


@dataclass
class WeightedBlockResult:
    value: Array
    dual: Array | tuple[Array, ...]
    converged: bool
    iterations: int
    residual_evaluations: int
    # Scaled and raw fixed-point defects of the returned (safeguarded) state.
    residual: float
    primal_residual: float
    dual_residual: float
    residual_limit: float
    raw_primal_residual: float
    raw_dual_residual: float
    iterate_change: float
    objective_before: float
    objective_after: float
    accepted_step_scale: float
    safeguarded: bool
    safeguard_adjustment: float
    primal_step_size: float
    dual_step_sizes: tuple[float, ...]


@dataclass
class WeightedTVTVResult:
    u: Array
    v: Array
    history: list[dict[str, float]] = field(default_factory=list)
    outer_iterations: int = 0
    converged: bool = False
    stop_reason: str = "maximum_outer_iterations"
    update_order: str = "v_then_u"


def _rms(value: Array) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.sqrt(np.vdot(array, array) / array.size)) if array.size else 0.0


def _combined_rms(values) -> float:
    arrays = [np.asarray(value, dtype=float) for value in values]
    size = sum(array.size for array in arrays)
    return float(np.sqrt(sum(np.vdot(a, a) for a in arrays) / size)) if size else 0.0


def _relative_rms_change(new: Array, old: Array) -> float:
    scale = max(_rms(new), _rms(old), np.finfo(float).eps)
    return float(_rms(new - old) / scale)


@lru_cache(maxsize=128)
def spatial_norm_squared(ny: int, nx: int, hx: float, hy: float) -> float:
    """Exact norm squared of the forward gradient with its Neumann boundary.

    Velocity components and temporal copies do not change this norm. This
    fixed quantity is shared by all outer iterations, parameters and cells.
    """
    lx = 0.0 if nx == 1 else 4 * np.cos(np.pi / (2 * nx)) ** 2 / hx**2
    ly = 0.0 if ny == 1 else 4 * np.cos(np.pi / (2 * ny)) ** 2 / hy**2
    return float(lx + ly)


def balanced_u_steps(v, shape, config):
    """Scalar primal step and two scalar dual steps with a safe block bound.

    ||T(v)|| <= 2/ht + max|vx|/hx + max|vy|/hy. Allocate half the
    stability budget to each dual block. Each TV component has the same
    dual step, so the existing Euclidean projection remains its exact prox.
    """
    spatial = spatial_norm_squared(shape[1], shape[2], config.hx, config.hy)
    transport = (2 / config.ht + np.max(np.abs(v[..., 0])) / config.hx
                 + np.max(np.abs(v[..., 1])) / config.hy) ** 2
    tau = config.balanced_u_step
    budget = config.step_safety**2
    sigma_tv = budget / (2 * tau * max(spatial, 1e-12))
    sigma_transport = budget / (2 * tau * max(transport, 1e-12))
    return tau, sigma_tv, sigma_transport, float(spatial + transport)


def _inner_residual_due(iteration, maximum, config):
    return iteration == maximum or (
        iteration >= config.min_inner_iterations
        and (iteration - config.min_inner_iterations) % config.inner_check_interval == 0
    )


def _objective_safeguard(initial: Array, candidate: Array, objective, config: WeightedTVTVConfig):
    """Backtrack an inexact block step until its objective is nonincreasing."""
    objective_before = float(objective(initial))
    objective_after = float(objective(candidate))
    tolerance = (config.objective_increase_absolute_tolerance
                 + config.objective_increase_tolerance * max(1.0, abs(objective_before)))
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


def _initial_duals(initial, shapes):
    if initial is None:
        return tuple(np.zeros(shape) for shape in shapes)
    duals = tuple(np.asarray(dual, dtype=float) for dual in initial)
    if tuple(dual.shape for dual in duals) != tuple(shapes):
        raise ValueError("initial_dual has incompatible shapes")
    return duals


def _pdhg_block(x, duals, *, forward, adjoint, prox_primal, prox_duals, tau, sigmas,
                primal_scale, dual_scales, objective, max_iterations, config) -> WeightedBlockResult:
    """PDHG for min_x G(x) + sum_i F_i(K_i x), with one dual step per block.

    K x and K* p are carried between iterations, so the incremental residuals
    need no extra operator applications: K xbar_k = (1+theta) K x_k - theta K x_{k-1}.
    A residual-qualified iterate is safeguarded against objective increase.
    The block stops only if the proximal fixed-point defects of that
    safeguarded state pass; otherwise PDHG continues from the raw iterate.
    Defects are scaled by primal_scale and by the dual ball radii.
    """
    x_start = x
    kx = kx_bar = forward(x)
    kty = adjoint(duals)
    limit = config.inner_absolute_tolerance + config.inner_tolerance
    adjustment_limit = limit * float(np.min(primal_scale))
    residual_evaluations = 0
    iterate_change = np.inf
    accepted = None

    def fixed_point_defects(value):
        """Scaled and raw defects of x = prox(x - tau K*p), p = prox(p + sigma K x)."""
        probe = prox_primal(value - tau * kty, tau)
        kv = kx if value is x else forward(value)
        probe_duals = prox_duals(tuple(p + s * k for p, s, k in zip(duals, sigmas, kv)))
        gaps = [p - q for p, q in zip(duals, probe_duals)]
        return (_rms((value - probe) / primal_scale),
                max(_rms(g / d) for g, d in zip(gaps, dual_scales)),
                _rms((value - probe) / tau),
                _combined_rms([g / s for g, s in zip(gaps, sigmas)]))

    for iteration in range(1, max_iterations + 1):
        x_old, duals_old, kx_old, kty_old = x, duals, kx, kty
        duals = prox_duals(tuple(p + s * k for p, s, k in zip(duals, sigmas, kx_bar)))
        kty = adjoint(duals)
        x = prox_primal(x - tau * kty, tau)
        kx = forward(x)
        kx_bar = tuple(k + config.theta * (k - k_old) for k, k_old in zip(kx, kx_old))

        if not _inner_residual_due(iteration, max_iterations, config):
            continue
        residual_evaluations += 1
        primal_gap = (x_old - x) / tau - (kty_old - kty)
        dual_gaps = [(p_old - p) / s - (k_old - k)
                     for p_old, p, s, k_old, k in zip(duals_old, duals, sigmas, kx_old, kx)]
        primal_residual = _rms(tau * primal_gap / primal_scale)
        dual_residual = max(_rms(s * g / d) for s, g, d in zip(sigmas, dual_gaps, dual_scales))
        iterate_change = _relative_rms_change(x, x_old)
        if iteration < config.min_inner_iterations or primal_residual > limit or dual_residual > limit:
            continue
        guarded = _objective_safeguard(x_start, x, objective, config)
        defects = fixed_point_defects(guarded[0])
        if defects[0] <= limit and defects[1] <= limit and _rms(guarded[0] - x) <= adjustment_limit:
            accepted = guarded, defects
            break

    if accepted is None:
        guarded = _objective_safeguard(x_start, x, objective, config)
        defects = fixed_point_defects(guarded[0])
    else:
        guarded, defects = accepted
    value, objective_before, objective_after, scale, safeguarded = guarded
    primal_residual, dual_residual, raw_primal_residual, raw_dual_residual = defects
    return WeightedBlockResult(
        value=value,
        # Duals stay feasible after projection; keep them as warm starts even
        # when the primal candidate was damped.
        dual=duals,
        converged=accepted is not None,
        iterations=iteration,
        residual_evaluations=residual_evaluations,
        residual=max(primal_residual, dual_residual),
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        residual_limit=limit,
        raw_primal_residual=raw_primal_residual,
        raw_dual_residual=raw_dual_residual,
        iterate_change=iterate_change,
        objective_before=objective_before,
        objective_after=objective_after,
        accepted_step_scale=scale,
        safeguarded=safeguarded,
        safeguard_adjustment=_rms(value - x),
        primal_step_size=tau,
        dual_step_sizes=tuple(sigmas),
    )


def solve_weighted_u_block(f, v, alpha_weight, initial_u, config, initial_dual=None) -> WeightedBlockResult:
    """Solve the weighted image subproblem for fixed velocity.

    Duals: (spatial TV, transport). Primal defects are scaled by the
    observation RMS (at least 0.1), dual defects by alpha and kappa.
    """
    f = np.asarray(f, dtype=float)
    v = np.asarray(v, dtype=float)
    u = np.asarray(initial_u, dtype=float)
    if u.shape != f.shape or u.ndim != 3:
        raise ValueError("initial_u and f must have shape (nt, ny, nx)")
    if v.shape != (f.shape[0] - 1, *f.shape[1:], 2):
        raise ValueError("v must have shape (nt-1, ny, nx, 2)")

    raw_alpha = validate_weight_map(alpha_weight, f.shape, "alpha_weight")
    alpha = forward_stencil_lower_envelope(raw_alpha)
    hx, hy, ht, kappa = config.hx, config.hy, config.ht, config.kappa
    tau, sigma_tv, sigma_transport, _ = balanced_u_steps(v, u.shape, config)

    return _pdhg_block(
        u, _initial_duals(initial_dual, ((*u.shape, 2), v.shape[:-1])),
        forward=lambda x: (spatial_gradient(x, hx, hy), transport_operator(x, v, ht, hx, hy)),
        adjoint=lambda p: (spatial_gradient_adjoint(p[0], hx, hy)
                           + transport_adjoint(p[1], v, f.shape[0], ht, hx, hy)),
        prox_primal=lambda z, t: prox_l32_fidelity(z, f, t),
        prox_duals=lambda p: (project_scalar_tv_ball(p[0], alpha), np.clip(p[1], -kappa, kappa)),
        tau=tau,
        sigmas=(sigma_tv, sigma_transport),
        primal_scale=max(_rms(f), 0.1),
        dual_scales=(alpha[..., None], kappa),
        # Objective evaluation applies the stencil envelope itself.
        objective=lambda x: weighted_u_block_objective(x, f, v, raw_alpha, kappa, ht, hx, hy),
        max_iterations=config.max_inner_u_iterations,
        config=config,
    )


def solve_weighted_v_block(u, beta_weight, initial_v, config, initial_dual=None) -> WeightedBlockResult:
    """Solve the weighted velocity subproblem for fixed image.

    Primal defects are scaled by the pixel spacing, dual defects by beta/kappa.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(initial_v, dtype=float)
    if u.ndim != 3 or v.shape != (u.shape[0] - 1, *u.shape[1:], 2):
        raise ValueError("u must be (nt, ny, nx) and v must be (nt-1, ny, nx, 2)")

    raw_beta = validate_weight_map(beta_weight, v.shape[:-1], "beta_weight")
    hx, hy, ht, kappa = config.hx, config.hy, config.ht, config.kappa
    # Dividing the v-subproblem by kappa does not change its minimizer.
    tv_radius = forward_stencil_lower_envelope(raw_beta) / kappa
    step = config.step_safety / np.sqrt(max(1e-12, spatial_norm_squared(u.shape[1], u.shape[2], hx, hy)))
    # Balance physical displacement (one pixel) against the dual TV radius.
    balance = min(hx, hy) / float(np.median(tv_radius))

    u_t = dt_forward(u, ht)
    grad_u = np.stack((dx_central(u[:-1], hx), dy_central(u[:-1], hy)), axis=-1)
    grad_norm_squared = grad_u[..., 0] * grad_u[..., 0] + grad_u[..., 1] * grad_u[..., 1]

    result = _pdhg_block(
        v, _initial_duals(None if initial_dual is None else (initial_dual,), ((*v.shape, 2),)),
        forward=lambda x: (vector_field_gradient(x, hx, hy),),
        adjoint=lambda p: vector_field_gradient_adjoint(p[0], hx, hy),
        prox_primal=lambda z, t: prox_optical_flow_affine(
            z, u_t, grad_u, t, gradient_norm_squared=grad_norm_squared),
        prox_duals=lambda p: (project_frobenius_tv_ball(p[0], tv_radius),),
        tau=step * balance,
        sigmas=(step / balance,),
        primal_scale=np.array([hx, hy]),
        dual_scales=(tv_radius[..., None, None],),
        objective=lambda x: weighted_v_block_objective(x, u, raw_beta, kappa, ht, hx, hy),
        max_iterations=config.max_inner_v_iterations,
        config=config,
    )
    return replace(result, dual=result.dual[0])


def _block_record(prefix: str, result: WeightedBlockResult) -> dict[str, float]:
    values = dict(
        inner_iterations=result.iterations,
        residual_evaluations=result.residual_evaluations,
        inner_residual=result.residual,
        primal_residual=result.primal_residual,
        dual_residual=result.dual_residual,
        primal_residual_ratio=result.primal_residual / result.residual_limit,
        dual_residual_ratio=result.dual_residual / result.residual_limit,
        raw_primal_residual=result.raw_primal_residual,
        raw_dual_residual=result.raw_dual_residual,
        primal_step=result.primal_step_size,
        iterate_change=result.iterate_change,
        block_objective_change=result.objective_after - result.objective_before,
        accepted_step_scale=result.accepted_step_scale,
        objective_safeguarded=result.safeguarded,
        safeguard_adjustment=result.safeguard_adjustment,
        **dict(zip(("dual_tv_step", "dual_transport_step"), result.dual_step_sizes)),
    )
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


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


def solve_weighted_tvtv(f, alpha_weight, beta_weight, config, initial_u=None, initial_v=None, *,
                        resume_state=None, checkpoint_callback=None, should_stop=None) -> WeightedTVTVResult:
    """Run alternating minimization for fixed weight maps.

    Optional checkpoints are taken at completed outer iterations. They retain
    primal/dual arrays and the stopping history. A stop request is observed at
    an outer boundary; an unfinished inner solve is not checkpointed.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 3 or f.shape[0] < 2:
        raise ValueError("f must have shape (nt, ny, nx) with nt>=2")
    v_shape = (f.shape[0] - 1, *f.shape[1:], 2)
    alpha = validate_weight_map(alpha_weight, f.shape, "alpha_weight")
    beta = validate_weight_map(beta_weight, v_shape[:-1], "beta_weight")
    u = f.copy() if initial_u is None else np.asarray(initial_u, dtype=float).copy()
    v = np.zeros(v_shape) if initial_v is None else np.asarray(initial_v, dtype=float).copy()
    if u.shape != f.shape or v.shape != v_shape:
        raise ValueError("initial values have incompatible shapes")

    spacings = (config.ht, config.hx, config.hy)
    initial_energy = manuscript_energy(u, v, f, alpha, beta, config.kappa, *spacings)
    history = [{"iteration": 0.0, **initial_energy, "transport_residual": transport_residual(u, v, *spacings)}]
    previous_total = float(initial_energy["total"])
    consecutive_successes = 0
    u_dual = v_dual = None
    completed_outer = 0

    problem_id = None
    if resume_state is not None or checkpoint_callback is not None:
        problem_id = _checkpoint_problem_id(f, alpha, beta, u, v, config)

    def result(converged, stop_reason):
        return WeightedTVTVResult(u=u, v=v, history=history, outer_iterations=completed_outer,
                                  converged=converged, stop_reason=stop_reason, update_order=config.update_order)

    def checkpoint():
        if checkpoint_callback is not None:
            checkpoint_callback(dict(schema=CHECKPOINT_PROTOCOL, problem_id=problem_id, u=u, v=v,
                                     u_dual=u_dual, v_dual=v_dual, history=history,
                                     outer_iterations=completed_outer, previous_total=previous_total,
                                     consecutive_successes=consecutive_successes))

    if resume_state is not None:
        if resume_state.get("schema") != CHECKPOINT_PROTOCOL or resume_state.get("problem_id") != problem_id:
            raise ValueError("Checkpoint does not match the observations, weights, initial state or solver settings.")
        completed_outer = int(resume_state["outer_iterations"])
        u = np.asarray(resume_state["u"], dtype=float).copy()
        v = np.asarray(resume_state["v"], dtype=float).copy()
        if u.shape != f.shape or v.shape != v_shape:
            raise ValueError("Checkpoint primal shapes are incompatible.")
        saved_u_dual, saved_v_dual = resume_state["u_dual"], resume_state["v_dual"]
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
            return result(True, "converged")
    else:
        checkpoint()

    image_scale = max(_rms(f), 0.1)
    pixel = min(config.hx, config.hy)
    pixels = np.array([config.hx, config.hy])
    for outer_iteration in range(completed_outer + 1, config.max_outer_iterations + 1):
        if should_stop is not None and should_stop():
            return result(False, "checkpoint_stop")
        # Block solves return new arrays, so these references keep the old iterate.
        old_u, old_v = u, v
        if config.update_order == "u_then_v":
            u_result = solve_weighted_u_block(f, v, alpha, u, config, u_dual)
            v_result = solve_weighted_v_block(u_result.value, beta, v, config, v_dual)
        else:
            v_result = solve_weighted_v_block(u, beta, v, config, v_dual)
            u_result = solve_weighted_u_block(f, v_result.value, alpha, u, config, u_dual)
        u, v = u_result.value, v_result.value
        u_dual, v_dual = u_result.dual, v_result.dual

        absolute_u_change = _rms(u - old_u)
        absolute_v_change = _rms(v - old_v)
        u_change_limit = (config.outer_absolute_tolerance + config.outer_tolerance) * image_scale
        if config.outer_velocity_tolerance_pixels is None:
            v_change_limit = pixel * (config.outer_absolute_tolerance + config.outer_tolerance
                                      * max(1.0, _rms(v / pixels), _rms(old_v / pixels)))
        else:
            v_change_limit = config.outer_velocity_tolerance_pixels * pixel

        energy = manuscript_energy(u, v, f, alpha, beta, config.kappa, *spacings)
        total = float(energy["total"])
        absolute_energy_change = abs(total - previous_total)
        energy_scale = max(abs(total), abs(previous_total), np.finfo(float).eps)
        energy_change_limit = config.outer_energy_absolute_tolerance + config.outer_energy_tolerance * energy_scale

        outer_success = (absolute_u_change <= u_change_limit and absolute_v_change <= v_change_limit
                         and absolute_energy_change <= energy_change_limit
                         and u_result.converged and v_result.converged)
        consecutive_successes = consecutive_successes + 1 if outer_success else 0

        record = {
            "iteration": float(outer_iteration),
            **energy,
            "transport_residual": transport_residual(u, v, *spacings),
            "relative_u_change": _relative_rms_change(u, old_u),
            "relative_v_change": _relative_rms_change(v, old_v),
            "absolute_u_change": absolute_u_change,
            "absolute_v_change": absolute_v_change,
            "u_change_limit": u_change_limit,
            "v_change_limit": v_change_limit,
            "energy_relative_change": absolute_energy_change / energy_scale,
            "energy_change_limit": energy_change_limit,
            "outer_consecutive_successes": float(consecutive_successes),
            **_block_record("u", u_result),
            **_block_record("v", v_result),
        }
        history.append(record)

        if config.verbose:
            ratio = max(record[f"{block}_{kind}_residual_ratio"] for block in "uv" for kind in ("primal", "dual"))
            print(f"outer={outer_iteration:3d} energy={total:.6e} transport={record['transport_residual']:.3e} "
                  f"du={absolute_u_change:.3e}/{u_change_limit:.3e} "
                  f"dv={absolute_v_change:.3e}/{v_change_limit:.3e} primal-dual={ratio:.2e}")

        previous_total = total
        completed_outer = outer_iteration
        checkpoint()
        if consecutive_successes >= config.required_outer_successes:
            return result(True, "converged")

    return result(False, "maximum_outer_iterations")
