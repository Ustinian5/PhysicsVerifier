#!/usr/bin/env python3
"""Parse OpenRLHF GRPO training metrics and save learning-curve plots."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(
    r"Global step\s+(\d+):\s+(\{.*?\})(?:\s|$)",
    re.DOTALL,
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_JOB_RE = re.compile(r"raysubmit_[A-Za-z0-9]+")

ROOT = Path(__file__).resolve().parents[2]

METRIC_KEYS = [
    "reward",
    "policy_loss",
    "return",
    "kl",
    "actor_lr",
    "response_length",
    "total_length",
    "dynamic_filtering_pass_rate",
    "dynamic_filtering_legacy_accept_rate",
    "dynamic_filtering_variance_accept_rate",
    "dynamic_filtering_effective_group_rate",
    "dynamic_filtering_zero_variance_rejects",
    "dynamic_filtering_candidate_groups",
    "dynamic_filtering_accepted_groups",
    "dynamic_filtering_budget_exhausted",
    "dynamic_filtering_gens_per_effective_group",
]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_global_steps_text(text: str) -> dict[int, dict[str, float]]:
    text = strip_ansi(text)
    rows: dict[int, dict[str, float]] = {}
    for match in STEP_RE.finditer(text):
        step = int(match.group(1))
        try:
            metrics = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError):
            continue
        if not isinstance(metrics, dict):
            continue
        parsed: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                continue
        rows[step] = parsed
    return rows


def parse_global_steps(log_path: Path) -> dict[int, dict[str, float]]:
    if not log_path.is_file():
        return {}
    return parse_global_steps_text(log_path.read_text(encoding="utf-8", errors="replace"))


def merge_rows(*parts: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for part in parts:
        merged.update(part)
    return merged


def discover_ray_job_id(save_path: Path) -> str | None:
    job_file = save_path / "ray_job_id.txt"
    if job_file.is_file():
        job_id = job_file.read_text(encoding="utf-8").strip()
        if job_id:
            return job_id

    try:
        proc = subprocess.run(
            ["ray", "job", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    matches = RAY_JOB_RE.findall(proc.stdout + proc.stderr)
    for job_id in reversed(matches):
        if "RUNNING" in proc.stdout and job_id in proc.stdout:
            return job_id
    return matches[-1] if matches else None


def sync_ray_job_log(save_path: Path, ray_log: Path, timeout_sec: int = 120) -> bool:
    job_id = discover_ray_job_id(save_path)
    if not job_id:
        return False

    try:
        proc = subprocess.run(
            ["ray", "job", "logs", job_id],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    if not proc.stdout and proc.returncode != 0:
        return False

    ray_log.parent.mkdir(parents=True, exist_ok=True)
    ray_log.write_text(proc.stdout, encoding="utf-8")
    job_file = save_path / "ray_job_id.txt"
    job_file.write_text(job_id + "\n", encoding="utf-8")
    return True


def load_reward_metrics(metrics_log: Path) -> dict[str, float]:
    if not metrics_log.is_file():
        return {}
    accs: list[float] = []
    latencies: list[float] = []
    verifier_hits = 0
    verifier_failed = 0
    verifier_skipped = 0
    total = 0
    with metrics_log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if row.get("acc"):
                accs.append(1.0)
            else:
                accs.append(0.0)
            latencies.append(float(row.get("latency_ms", 0.0) or 0.0))
            mode = row.get("verifier_mode")
            if mode == "full":
                verifier_hits += 1
            elif mode == "failed":
                verifier_failed += 1
            else:
                verifier_skipped += 1
    if total == 0:
        return {}
    lat_sorted = sorted(latencies)
    p95_idx = min(int(len(lat_sorted) * 0.95), len(lat_sorted) - 1)
    return {
        "reward_acc_mean": sum(accs) / total,
        "reward_latency_ms_mean": sum(latencies) / total,
        "reward_latency_ms_p95": lat_sorted[p95_idx],
        "reward_verifier_trigger_rate": verifier_hits / total * 100.0,
        "reward_verifier_fail_rate": verifier_failed / total * 100.0,
        "reward_verifier_skip_rate": verifier_skipped / total * 100.0,
        "reward_samples": float(total),
    }


def plot_reward_metrics(metrics_log: Path, out_path: Path) -> None:
    if not metrics_log.is_file():
        return
    accs: list[float] = []
    latencies: list[float] = []
    scores: list[float] = []
    with metrics_log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            accs.append(1.0 if row.get("acc") else 0.0)
            latencies.append(float(row.get("latency_ms", 0.0) or 0.0))
            scores.append(float(row.get("score", 0.0) or 0.0))
    if not accs:
        return
    idx = list(range(1, len(accs) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Reward server metrics", fontsize=12)
    axes[0, 0].plot(idx, scores, marker=".", linewidth=1)
    axes[0, 0].set_title("Reward score")
    axes[0, 1].plot(idx, accs, marker=".", color="tab:green")
    axes[0, 1].set_title("Answer accuracy (0/1)")
    axes[1, 0].plot(idx, latencies, marker=".", color="tab:orange")
    axes[1, 0].set_title("Latency (ms)")
    axes[1, 1].hist(scores, bins=20, color="tab:purple", alpha=0.8)
    axes[1, 1].set_title("Reward distribution")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: dict[int, dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["global_step", *METRIC_KEYS]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(rows):
            out = {"global_step": step}
            for key in METRIC_KEYS:
                out[key] = rows[step].get(key, "")
            writer.writerow(out)


def plot_curves(rows: dict[int, dict[str, float]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No Global step metrics yet", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    steps = sorted(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=12)

    def series(key: str) -> list[float]:
        return [rows[s].get(key, float("nan")) for s in steps]

    axes[0, 0].plot(steps, series("reward"), marker="o", linewidth=1.5)
    axes[0, 0].set_title("Reward")
    axes[0, 0].set_xlabel("Global step")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, series("dynamic_filtering_pass_rate"), marker="o", color="tab:orange")
    axes[0, 1].set_title("Dynamic filtering pass rate (%)")
    axes[0, 1].set_xlabel("Global step")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, series("response_length"), marker="o", color="tab:green", label="response")
    axes[1, 0].plot(steps, series("total_length"), marker="s", color="tab:purple", label="total")
    axes[1, 0].set_title("Sequence length")
    axes[1, 0].set_xlabel("Global step")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, series("policy_loss"), marker="o", color="tab:red", label="policy_loss")
    ax_lr = axes[1, 1].twinx()
    ax_lr.plot(steps, series("actor_lr"), marker="x", color="tab:blue", label="actor_lr")
    axes[1, 1].set_title("Policy loss / actor LR")
    axes[1, 1].set_xlabel("Global step")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    default_save = Path("/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf")
    parser = argparse.ArgumentParser(description="Plot OpenRLHF GRPO training curves")
    parser.add_argument(
        "--save-path",
        default=str(default_save),
        help="Checkpoint root (for ray job log sync and defaults)",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=None,
        help="Log file with Global step lines (repeatable)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for PNG/CSV outputs",
    )
    parser.add_argument(
        "--no-sync-ray",
        action="store_true",
        help="Skip syncing Ray job logs before plotting",
    )
    args = parser.parse_args()

    save_path = Path(args.save_path)
    out_dir = Path(args.out_dir or save_path / "plots")
    ray_log = save_path / "ray_train.log"

    log_paths: list[Path] = []
    if args.log:
        log_paths.extend(Path(p) for p in args.log)
    else:
        log_paths.append(save_path / "train_launch.log")
        log_paths.append(save_path / "direct_train.log")
    if ray_log not in log_paths:
        log_paths.append(ray_log)

    synced = False
    if not args.no_sync_ray:
        synced = sync_ray_job_log(save_path, ray_log)

    parts = [parse_global_steps(path) for path in log_paths if path.is_file()]
    rows = merge_rows(*parts)
    reward_logs = [
        save_path / "plots/physics_reward_metrics.jsonl",
        ROOT / "logs/physics_reward_metrics.jsonl",
    ]
    reward_metrics: dict[str, float] = {}
    reward_log_used = None
    for candidate in reward_logs:
        reward_metrics = load_reward_metrics(candidate)
        if reward_metrics:
            reward_log_used = candidate
            break

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    png_path = out_dir / "training_curves.png"
    csv_path = out_dir / "training_metrics.csv"
    snapshot_path = out_dir / f"training_curves_{ts}.png"

    title = f"Qwen3-30B Physics GRPO ({len(rows)} steps)"
    plot_curves(rows, png_path, title)
    plot_curves(rows, snapshot_path, title)
    write_csv(rows, csv_path)
    if reward_metrics:
        (out_dir / "reward_metrics_summary.json").write_text(
            json.dumps(reward_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    reward_png = out_dir / "reward_metrics.png"
    plot_reward_metrics(reward_log_used or ROOT / "logs/physics_reward_metrics.jsonl", reward_png)

    sources = [str(p) for p in log_paths if p.is_file()]
    print(
        f"[plot] steps={len(rows)} synced_ray={synced} reward_metrics={bool(reward_metrics)} sources={sources} "
        f"png={png_path} csv={csv_path} snapshot={snapshot_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
