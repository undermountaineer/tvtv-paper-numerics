"""Initial-state-aware cache with complete outer-iteration checkpoints.

Use in an isolated process with numerical thread limits set at startup.
Existing minimal_experiment.CachedSolver remains unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from .weighted_solver import WeightedTVTVConfig, WeightedTVTVResult, solve_weighted_tvtv

CACHE_PROTOCOL = 1
SOURCE_FILES = ('weighted_solver.py', 'weighted_energy.py', 'weights.py',
                'operators.py', 'proximal.py', 'energy.py', 'checkpoint_cache.py')


def settings_identity(config):
    value = asdict(config)
    value.pop('max_outer_iterations')
    value.pop('verbose')
    return value


def _array_hash(value):
    array = np.ascontiguousarray(value, dtype=float)
    return dict(shape=list(array.shape), dtype='float64',
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def scientific_code_id():
    directory = Path(__file__).parent
    return hashlib.sha256(b''.join((directory / name).read_bytes()
                                  for name in SOURCE_FILES)).hexdigest()


def solve_identity(noisy, alpha, beta, config, initial_u=None, initial_v=None):
    f = np.asarray(noisy, dtype=float)
    u0 = f if initial_u is None else np.asarray(initial_u, dtype=float)
    v0 = np.zeros((f.shape[0]-1, *f.shape[1:], 2)) if initial_v is None else np.asarray(initial_v, dtype=float)
    if f.ndim != 3 or f.shape[0] < 2 or u0.shape != f.shape or v0.shape != (f.shape[0]-1, *f.shape[1:], 2):
        raise ValueError('Observation or initial-state shapes are incompatible.')
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    if a.shape != f.shape or b.shape != v0.shape[:-1]:
        raise ValueError('Supply explicit alpha/beta maps with the correct shapes.')
    if not all(np.isfinite(x).all() for x in (f, a, b, u0, v0)) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError('Inputs must be finite and weights positive.')
    identity = dict(protocol=CACHE_PROTOCOL, solver=settings_identity(config),
                    source=scientific_code_id(),
                    arrays={name: _array_hash(x) for name, x in
                            zip(('noisy','alpha','beta','initial_u','initial_v'), (f,a,b,u0,v0))})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return key, identity


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _save_npz(path, arrays, metadata):
    # Uncompressed checkpoints avoid compression cost each outer iteration.
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays, metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _entry_lock(path):
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('This cache entry is already in use by another process.') from exc
            try:
                yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('This cache entry is already in use by another process.') from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata_json']))
        arrays = {name: saved[name].copy() for name in saved.files if name != 'metadata_json'}
    return arrays, metadata


def _save_result(entry, identity, result, seconds, provenance=None):
    if not np.isfinite(result.u).all() or not np.isfinite(result.v).all():
        raise ValueError('Non-finite solver result cannot be cached.')
    fields = {name:getattr(result,name) for name in
              ('history','outer_iterations','converged','stop_reason','update_order')}
    metadata = dict(identity=identity, result=fields,
                    cumulative_solver_seconds=float(seconds), provenance=provenance)
    _save_npz(entry/'result.npz', dict(u=result.u, v=result.v), metadata)


class ResumableSolverCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last_key = None
        self.last_action = None
        self.last_metadata = None

    def run(self, noisy, alpha, beta, config, initial_u=None, initial_v=None,
            *, seconds=None, outer_iteration_budget=None, resume=True):
        if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
            raise ValueError('seconds must be finite and positive.')
        if outer_iteration_budget is not None and (outer_iteration_budget < 1 or int(outer_iteration_budget) != outer_iteration_budget):
            raise ValueError('outer_iteration_budget must be a positive integer.')
        key, identity = solve_identity(noisy, alpha, beta, config, initial_u, initial_v)
        self.last_key = key
        entry = self.directory/key
        entry.mkdir(parents=True, exist_ok=True)
        with _entry_lock(entry/'entry.lock'):
            result_path = entry/'result.npz'
            if result_path.is_file():
                arrays, metadata = _read_npz(result_path)
                if metadata['identity'] != identity:
                    raise ValueError('Saved result identity is inconsistent.')
                result = WeightedTVTVResult(u=arrays['u'], v=arrays['v'], **metadata['result'])
                if result.converged:
                    self.last_action = 'reused_completed'
                    self.last_metadata = metadata
                    print(f'REUSED completed solve: {key[:12]}; outer={result.outer_iterations}', flush=True)
                    return result
            state = None
            cumulative = 0.0
            checkpoint_path = entry/'checkpoint.npz'
            if checkpoint_path.is_file():
                if not resume:
                    raise RuntimeError('Checkpoint exists. Resume it or use a distinct cache directory; no implicit restart.')
                arrays, metadata = _read_npz(checkpoint_path)
                if metadata['identity'] != identity:
                    raise ValueError('Checkpoint identity is inconsistent.')
                state = metadata['state']
                state.update(u=arrays['u'], v=arrays['v'],
                             u_dual=(arrays['u_dual_tv'], arrays['u_dual_transport']) if metadata['has_duals'] else None,
                             v_dual=arrays.get('v_dual_tv'))
                cumulative = float(metadata['cumulative_solver_seconds'])
            completed = state['outer_iterations'] if state is not None else 0
            self.last_action = 'resumed' if state is not None else 'computed'
            print(f'{self.last_action.upper()} solve: {key[:12]}; from outer={completed}', flush=True)
            started = time.perf_counter()
            latest_outer = completed
            def save_checkpoint(current):
                nonlocal latest_outer
                latest_outer = current['outer_iterations']
                arrays = dict(u=current['u'], v=current['v'])
                if current['u_dual'] is not None:
                    arrays.update(u_dual_tv=current['u_dual'][0],
                                  u_dual_transport=current['u_dual'][1], v_dual_tv=current['v_dual'])
                if not all(np.isfinite(x).all() for x in arrays.values()):
                    raise ValueError('Non-finite checkpoint rejected.')
                metadata = dict(identity=identity,
                    state={k:v for k,v in current.items() if k not in ('u','v','u_dual','v_dual')},
                    has_duals=current['u_dual'] is not None,
                    cumulative_solver_seconds=cumulative+time.perf_counter()-started)
                _save_npz(checkpoint_path, arrays, metadata)
                _write_json(entry/'progress.json', dict(key=key, outer_iterations=latest_outer,
                    converged=current['consecutive_successes'] >= config.required_outer_successes,
                    cumulative_solver_seconds=metadata['cumulative_solver_seconds']))
            def should_stop():
                return ((seconds is not None and time.perf_counter()-started >= seconds) or
                        (outer_iteration_budget is not None and latest_outer-completed >= outer_iteration_budget))
            result = solve_weighted_tvtv(noisy, alpha, beta, config, initial_u, initial_v,
                resume_state=state, checkpoint_callback=save_checkpoint, should_stop=should_stop)
            _save_result(entry, identity, result, cumulative+time.perf_counter()-started)
            self.last_metadata = _read_npz(result_path)[1]
            print(f'SAVED outer={result.outer_iterations}; converged={result.converged}; {result.stop_reason}', flush=True)
            return result

    __call__ = run

    def adopt_completed(self, noisy, alpha, beta, config, initial_u, initial_v,
                        result, seconds, provenance):
        if not result.converged or result.stop_reason != 'converged':
            raise ValueError('Only validated converged results can be adopted.')
        key, identity = solve_identity(noisy, alpha, beta, config, initial_u, initial_v)
        entry = self.directory/key
        entry.mkdir(parents=True, exist_ok=True)
        with _entry_lock(entry/'entry.lock'):
            if (entry/'result.npz').is_file():
                arrays, metadata = _read_npz(entry/'result.npz')
                if metadata['identity'] != identity:
                    raise ValueError('Existing result identity is inconsistent.')
                if metadata['result']['converged']:
                    self.last_key = key; self.last_action = 'reused_completed'; self.last_metadata = metadata
                    return key
            _save_result(entry, identity, result, seconds, provenance)
            self.last_key = key; self.last_action = 'adopted_completed'
            self.last_metadata = _read_npz(entry/'result.npz')[1]
        return key
