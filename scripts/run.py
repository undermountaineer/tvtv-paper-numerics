"""Single, bounded entry point. The controller imports only the standard library."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ACTIONS = ("prepare", "import-handoff", "learn", "compare", "validate", "report", "export", "smoke")


def worker(request):
    # These limits must be set before NumPy loads, including on Windows.
    from tvtv_numerics.workflow import Experiment, smoke, write_json
    if request["action"] == "smoke":
        result = smoke(request["root"])
    else:
        experiment = Experiment(request["root"], request["config"])
        action = request["action"]
        if action == "import-handoff":
            from tvtv_numerics.handoff import import_handoff
            result = import_handoff(experiment, request["handoff"])
        elif action == "plan":
            result = experiment.plan(request["stage"])
        elif action == "solve":
            result = experiment.solve(request["job"], request["soft_seconds"])
        elif action == "prepare":
            result = experiment.prepare()
        elif action == "report":
            result = experiment.report()
        elif action == "export":
            result = experiment.export()
        else:
            raise ValueError("Unknown worker action.")
    write_json(request["response"], result)


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def invoke(args, action, limit=120, **details):
    out = args.root/"results/proof_of_concept/launcher"
    out.mkdir(parents=True, exist_ok=True)
    token = str(time.time_ns())
    request_path = out/f"{token}.request.json"
    response = out/f"{token}.response.json"
    log = out/f"{token}.log.txt"
    request = dict(root=str(args.root), config=str(args.config) if args.config else None,
                   action=action, response=str(response), **details)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = str(args.root/"src") + (os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = time.monotonic()
    status = "completed"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(request_path)],
                                   cwd=args.root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        try:
            while process.poll() is None:
                left = limit-(time.monotonic()-started)
                if left <= 0:
                    stop_process(process)
                    status = "hard_time_limit"
                    break
                try:
                    process.wait(timeout=min(25, left))
                except subprocess.TimeoutExpired:
                    print(f"  {time.monotonic()-started:.0f}s elapsed; log: {log}", flush=True)
        except KeyboardInterrupt:
            stop_process(process)
            status = "interrupted"
    if status == "completed" and (process.returncode != 0 or not response.is_file()):
        status = "worker_error"
    if status != "completed":
        print(f"{status}; completed outer checkpoints retained. Log: {log}", flush=True)
        if status == "worker_error":
            print(log.read_text(encoding="utf-8", errors="replace")[-4000:], flush=True)
        raise RuntimeError(status)
    return json.loads(response.read_text(encoding="utf-8"))


def show(result):
    fields = ("output_directory", "archive", "imported_completed", "new_solves", "completed_solves", "planned_solves",
              "validation_complete", "spatial_wins", "mean_relative_gain", "smoke_passed")
    print(json.dumps({k: result[k] for k in fields if k in result}, indent=2), flush=True)
    if "learning" in result:
        learning = result["learning"]
        print("Local selection:", learning["selected_alphas"], "; strict split accepted:", learning["strict_split_accepted"])
    for pair in result.get("comparisons", []):
        if pair["complete"]:
            print(f"{pair['stage']}: seed={pair['seed']}; loss gain={100*pair['relative_gain']:.3f}%")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        worker(request)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("handoff", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seconds", type=float, default=600, help="Hard limit per solve process, including startup.")
    parser.add_argument("--budget-minutes", type=float, default=60, help="Total allowance for solve processes.")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.config:
        args.config = args.config.resolve()
    if not math.isfinite(args.seconds) or args.seconds < 30 or not math.isfinite(args.budget_minutes) or args.budget_minutes <= 0:
        parser.error("Use a per-solve limit >=30 seconds and a positive finite batch budget.")
    if args.action == "import-handoff" and args.handoff is None:
        parser.error("Pass the path to tvtv-handoff-2026-10-07.zip.")
    if args.handoff is not None and args.action != "import-handoff":
        parser.error("A handoff path is used only with import-handoff.")
    try:
        if args.action not in ("learn", "compare", "validate"):
            details = dict(handoff=str(args.handoff.resolve())) if args.handoff else {}
            show(invoke(args, args.action, **details))
            return 0
        plan = invoke(args, "plan", stage=args.action)
        started = time.monotonic()
        stopped = False
        try:
            for entry in plan["jobs"]:
                if entry["eligible"]:
                    print("Reusing completed:", entry["job"], flush=True)
                    continue
                left = args.budget_minutes*60-(time.monotonic()-started)
                if left < 30:
                    print("Batch budget exhausted; run the same command to resume.")
                    stopped = True
                    break
                limit = min(args.seconds, left)
                print("Solving:", entry["job"], flush=True)
                result = invoke(args, "solve", limit=limit, job=entry["job"], soft_seconds=max(1, limit-15))
                if not result["eligible"]:
                    print("Unresolved candidate:", result["status"], "; the batch stops here.")
                    stopped = True
                    break
        except RuntimeError:
            stopped = True
        show(invoke(args, "report"))
        return 1 if stopped else 0
    except RuntimeError as exc:
        print(f"Stopped: {exc}. Run report to inspect retained results.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
