"""Run the bounded 512 experiment in order; stop at the first unresolved stage."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/meeting_512.json"))
    p.add_argument("--baseline-minutes", type=int, default=180)
    p.add_argument("--refine-minutes", type=int, default=180)
    p.add_argument("--compare-minutes", type=int, default=90)
    args = p.parse_args()
    if min(args.baseline_minutes, args.refine_minutes, args.compare_minutes) < 1:
        p.error("Stage budgets must be positive minutes.")
    root = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    code = 0
    try:
        stages = [("prepare", None), ("baseline", args.baseline_minutes),
                  ("refine", args.refine_minutes), ("compare", args.compare_minutes)]
        for stage, minutes in stages:
            print(f"\n{datetime.now(timezone.utc).isoformat()} | {stage}", flush=True)
            command = [sys.executable, "-u", str(root / "scripts/run_original_1024.py"), stage,
                       "--root", str(root), "--config", str(config)]
            if minutes is not None:
                command += ["--seconds", str(minutes * 60), "--budget-minutes", str(minutes)]
            code = subprocess.call(command, cwd=root)
            if code:
                print("Stopped at", stage, "-- see the report. Repeat this command to resume a time-limited run.", flush=True)
                break
        else:
            print("Design experiment complete. Inspect the report before fresh-noise validation.", flush=True)
    except KeyboardInterrupt:
        print("Interrupted. Complete outer checkpoints remain available.", flush=True)
        code = 130
    finally:
        result = subprocess.call([sys.executable, str(root / "scripts/meeting_report.py"),
                                  "--config", str(config)], cwd=root)
        if result:
            print("Report could not be built; inspect the worker log above.", flush=True)
    return code or result


if __name__ == "__main__":
    raise SystemExit(main())
