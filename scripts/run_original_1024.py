"""Separate, resumable controller for the original 1024 master experiment."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def replace_with_retry(path, target, original_replace):
    """Retry transient Windows access/sharing errors; retain atomic replacement."""
    for attempt in range(20):
        try:
            return original_replace(path, target)
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 19:
                raise
            if attempt == 0:
                print("Windows blocked file replacement; retrying:", target, flush=True)
            time.sleep(min(.1 * (attempt + 1), .5))


def install_windows_write_retries():
    # Worker-only hook: scientific source and existing cache identities stay intact.
    if os.name != "nt":
        return
    original_replace = Path.replace
    def retry(path, target):
        return replace_with_retry(path, target, original_replace)
    Path.replace = retry


def worker(request):
    install_windows_write_retries()
    from tvtv_numerics.original_1024 import Original1024
    from tvtv_numerics.workflow import write_json
    experiment = Original1024(request["root"], request["config"])
    action = request["action"]
    if action == "plan":
        result = experiment.plan(request["stage"])
    elif action == "solve":
        result = experiment.solve(request["job"], request["seconds"])
    else:
        result = getattr(experiment, action)()
    write_json(request["response"], result)


def stop(process):
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def invoke(args, action, limit=120, **details):
    out = args.root/"results/synthetic_original_1024/launcher"
    out.mkdir(parents=True, exist_ok=True)
    token = str(time.time_ns())
    request, response, log = (out/f"{token}.{name}" for name in ("request.json", "response.json", "log.txt"))
    request.write_text(json.dumps(dict(root=str(args.root), config=str(args.config) if args.config else None,
                                      action=action, response=str(response), **details)))
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONPATH"] = str(args.root/"src") + (os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MPLBACKEND"] = "Agg"
    started = time.monotonic()
    with log.open("w") as stream:
        p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(request)],
                             cwd=args.root, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            while p.poll() is None:
                left = limit-(time.monotonic()-started)
                if left <= 0:
                    stop(p)
                    raise RuntimeError(f"Time limit reached; checkpoints retained. Log: {log}")
                try:
                    p.wait(timeout=min(25, left))
                except subprocess.TimeoutExpired:
                    print(f"  {time.monotonic()-started:.0f}s elapsed; log: {log}", flush=True)
        except KeyboardInterrupt:
            stop(p)
            raise
    if p.returncode or not response.exists():
        raise RuntimeError(log.read_text(errors="replace")[-4000:])
    return json.loads(response.read_text())


def show(result):
    if "archive" in result:
        print("Archive:", result["archive"])
        return
    print("Results:", result["output_directory"])
    state = result.get("learning", {})
    root = state.get("root", {})
    if root.get("complete"):
        print("Design constant alpha:", root["selected"]["alpha"], "; boundary limited:", root["boundary_limited"])
    print("Refinement complete:", state.get("complete"), "; leaves:", len(state.get("leaves", [])))
    for decision in state.get("decisions", []):
        print("Cell:", decision["cell"], "; split accepted:", decision["accepted"])
    if state.get("blocked"):
        print("Blocked:", state["blocked"])
    for pair in result.get("comparisons", []):
        if pair["complete"]:
            print(f"Seed {pair['seed']}: loss gain {100*pair['relative_gain']:.3f}%; RMSE gain {100*pair['RMSE_relative_gain']:.3f}%")
    diagnostic = result.get("rejected_split_global_diagnostic")
    if diagnostic and diagnostic["eligible"]:
        print("Rejected first split, global diagnostic loss:", diagnostic["upper_loss"])


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(json.loads(Path(sys.argv[2]).read_text()))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "baseline", "refine", "compare", "validate", "report", "export"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seconds", type=float, default=300, help="Hard limit per solve, including startup.")
    parser.add_argument("--budget-minutes", type=float, default=30, help="Total budget for this command, including planning.")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.config:
        args.config = args.config.resolve()
    if not math.isfinite(args.seconds) or args.seconds < 30 or not math.isfinite(args.budget_minutes) or args.budget_minutes < 1:
        parser.error("Use seconds >=30 and a finite budget >=1 minute.")
    try:
        if args.action in ("prepare", "report", "export"):
            show(invoke(args, args.action))
            return 0
        deadline = time.monotonic()+args.budget_minutes*60
        while True:
            left = deadline-time.monotonic()
            if left < 30:
                print("Batch budget reached. Repeat the same command to resume.")
                show(invoke(args, "report"))
                return 1
            plan = invoke(args, "plan", limit=min(120, left), stage=args.action)
            if plan["blocked"]:
                print(plan["blocked"])
                show(invoke(args, "report"))
                return 1
            if plan["complete"]:
                show(invoke(args, "report"))
                return 0
            # Replan after every solve: expansions and descendants are adaptive.
            row = plan["jobs"][0]
            job = {key: row[key] for key in ("stage", "seed", "cell", "alpha", "mode", "leaves")}
            limit = min(args.seconds, deadline-time.monotonic())
            if limit < 30:
                continue
            print("Solving:", job, flush=True)
            result = invoke(args, "solve", limit=limit, job=job, seconds=max(1, limit-15))
            if not result["eligible"]:
                print("Unresolved candidate:", result["status"], "; selection stops here.")
                if result["status"] == "maximum_outer_iterations":
                    print("The iteration cap requires numerical review; repeating alone cannot lift it.")
                else:
                    print("Repeat the same command to resume a time-limited checkpoint.")
                show(invoke(args, "report"))
                return 1
    except (RuntimeError, KeyboardInterrupt) as exc:
        print("Stopped; checkpoints retained.", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
