"""Advance a detached pilot to formal annotation only after engineering acceptance."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_retrieval_comparison import write_json
from scripts.experiment_manifest import sha256_file
from scripts.run_bench_reannotation import STAGES, MODEL


def accept(out):
    manifest = json.loads((out / "manifest.json").read_text())
    records = []
    for stage in STAGES:
        for sid in manifest["pilot_ids"]:
            path = out / ("arm_" + stage) / sid / "result.json"
            data = json.loads(path.read_text())
            raw = out / data["response_record"]
            if sha256_file(raw) != data["response_record_sha256"]:
                raise ValueError("response evidence hash mismatch")
            response = json.loads(raw.read_text())["response"]
            if response.get("model") != MODEL or response.get("status") != "completed" or response.get("reasoning", {}).get("effort") != "high":
                raise ValueError("pilot model/effort/completion mismatch")
            # Interim unlocated claims remain preserved for later correction.
            # Only final confirmed claims must have validated quotes to expand.
            for e in data["annotation"]["errors"]:
                if stage == "final" and e["status"] == "confirmed" and e["quote"] and not e["evidence_validation"]["exact_match"]:
                    raise ValueError("confirmed pilot error has no exact original quote; operator review needed")
            records.append({"path": str(path.relative_to(out)), "sha256": sha256_file(path)})
    return {"state": "engineering_accepted", "pilot_tasks": 20, "records": records,
            "manifest_sha256": sha256_file(out / "manifest.json"), "accepted_unix": time.time(),
            "physical_gold_validation": "pending human review; engineering acceptance is not label correctness",
            "omission_policy": "Empty-quote omission claims allowed by frozen schema remain unlocated; excluded from location-based gold until reviewed; never fabricated or dropped",
            "reuse_policy": "same frozen inputs/prompts; 20 pilot results reused in formal 396 tasks"}


def main(args):
    out = args.output.resolve()
    with (out / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out / "supervisor.pid").write_text(str(os.getpid()))
        while True:
            state = json.loads((out / "progress.json").read_text())
            if state.get("state") == "pilot_complete_pending_acceptance":
                break
            if state.get("state") == "stopped":
                write_json(out / "supervisor_status.json", {"state": "pilot_stopped", "progress": state})
                return
            try:
                os.kill(int((out / "runner.pid").read_text()), 0)
            except (ProcessLookupError, FileNotFoundError):
                write_json(out / "supervisor_status.json", {"state": "pilot_process_missing", "checked_unix": time.time()})
                return
            time.sleep(20)
        try:
            acceptance = accept(out)
            # Wait for the pilot to release its exclusive runner lock before dispatch.
            with (out / "runner.lock").open("w") as runner_lock:
                fcntl.flock(runner_lock, fcntl.LOCK_EX)
                write_json(out / "pilot_acceptance.json", acceptance)
            command = [sys.executable, str(ROOT / "scripts/run_bench_reannotation.py"),
                       "--source", str(args.source.resolve()), "--output", str(out),
                       "--env", str(args.env.resolve()), "--mode", "formal"]
            if args.recover_technical:
                command.append("--recover-technical")
            write_json(out / "supervisor_status.json", {"state": "starting_formal", "time": time.time()})
            with (out / "runner.log").open("ab") as log:
                result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            write_json(out / "supervisor_status.json", {"state": "formal_exited", "exit_code": result.returncode, "time": time.time()})
        except Exception as exc:
            write_json(out / "supervisor_status.json", {"state": "acceptance_blocked", "error": str(exc), "time": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "env"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--recover-technical", action="store_true")
    main(parser.parse_args())
