"""Run synthetic TV–TV experiments with bounded, resumable computation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", default="run", choices=(
        "run", "prepare", "baseline", "refine", "compare", "validate", "report", "export"))
    p.add_argument("--config", type=Path, default=Path("configs/synthetic_256.json"))
    p.add_argument("--baseline-minutes", type=int, default=45)
    p.add_argument("--refine-minutes", type=int, default=45)
    p.add_argument("--compare-minutes", type=int, default=30)
    p.add_argument("--budget-minutes", type=int, default=120, help="Budget for a single selected compute stage.")
    p.add_argument("--seconds", type=int, default=3600, help="Per-solve limit for a single selected compute stage.")
    args = p.parse_args()
    if min(args.baseline_minutes, args.refine_minutes, args.compare_minutes, args.budget_minutes) < 1 or args.seconds < 30:
        p.error("Use positive minute budgets and seconds >=30.")
    root = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    report_command = [sys.executable, str(root / "scripts/report_experiment.py"), "--config", str(config)]
    if args.action == "report":
        return subprocess.call(report_command, cwd=root)
    if args.action == "export":
        code = subprocess.call(report_command, cwd=root)
        if code:
            return code
    stages = ([("prepare", None), ("baseline", args.baseline_minutes),
               ("refine", args.refine_minutes), ("compare", args.compare_minutes)]
              if args.action == "run" else [(args.action, args.budget_minutes)])
    code = 0
    try:
        for stage, minutes in stages:
            print(f"\n{datetime.now(timezone.utc).isoformat()} | {stage}", flush=True)
            command = [sys.executable, "-u", str(root / "scripts/run_original_1024.py"), stage,
                       "--root", str(root), "--config", str(config)]
            if minutes is not None and stage not in ("prepare", "export"):
                seconds = minutes*60 if args.action == "run" else args.seconds
                command += ["--seconds", str(seconds), "--budget-minutes", str(minutes)]
            code = subprocess.call(command, cwd=root)
            if code:
                print("Stopped at", stage, "-- inspect the report. Repeat this command to resume time-limited work.", flush=True)
                break
        else:
            print("Requested computation complete.", flush=True)
    except KeyboardInterrupt:
        print("Interrupted. Complete outer checkpoints remain available.", flush=True)
        code = 130
    finally:
        result = subprocess.call(report_command, cwd=root)
        if result:
            print("Report could not be built; inspect the worker log above.", flush=True)
    return code or result


if __name__ == "__main__":
    raise SystemExit(main())
