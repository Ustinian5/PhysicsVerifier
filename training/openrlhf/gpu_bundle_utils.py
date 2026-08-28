#!/usr/bin/env python3
"""GPU idle probing, bundle selection, and best-effort CUDA reservation holders."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def query_gpus() -> list[dict[str, Any]]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "mem_total_mib": int(parts[1]),
                "mem_used_mib": int(parts[2]),
                "mem_free_mib": int(parts[3]),
                "util_pct": float(parts[4]),
            }
        )
    return gpus


def compute_pids_on_gpu(gpu_index: int) -> list[int]:
    """Return PIDs with compute apps on the given GPU (best effort)."""
    try:
        out = _run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ]
        )
    except subprocess.CalledProcessError:
        return []
    pids: list[int] = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line or line.lower() == "n/a":
            continue
        try:
            pids.append(int(line.split()[0]))
        except ValueError:
            continue
    return pids


def is_gpu_idle(
    g: dict[str, Any],
    *,
    free_mib: int,
    util_max: float,
    allow_pids: set[int] | None = None,
) -> bool:
    allow_pids = allow_pids or set()
    pids = [p for p in compute_pids_on_gpu(g["index"]) if p not in allow_pids]
    return (
        not pids
        and g["util_pct"] <= util_max
        and g["mem_free_mib"] >= free_mib
    )


def select_train_judge_bundle(
    *,
    n_train: int = 6,
    n_judge: int = 2,
    free_mib: int = 75000,
    util_max: float = 5.0,
    allow_pids: set[int] | None = None,
    prefer_judge: list[int] | None = None,
) -> dict[str, Any]:
    """Pick n_train strictly idle GPUs plus n_judge dedicated judge GPUs.

    Judge GPUs may already host processes in allow_pids (e.g. a live vLLM).
    prefer_judge is honored first so a loaded 30B replica is not migrated.
    """
    allow_pids = allow_pids or set()
    prefer_judge = list(prefer_judge or [])
    gpus = query_gpus()
    ours: list[dict[str, Any]] = []
    strict_idle: list[dict[str, Any]] = []
    for g in gpus:
        gpids = set(compute_pids_on_gpu(g["index"]))
        foreign = gpids - allow_pids
        if gpids and not foreign:
            ours.append(g)
        elif is_gpu_idle(g, free_mib=free_mib, util_max=util_max, allow_pids=None):
            strict_idle.append(g)

    pool: dict[int, dict[str, Any]] = {g["index"]: g for g in ours + strict_idle}
    ours_idx = {g["index"] for g in ours}
    judges: list[int] = []
    for j in prefer_judge:
        if j in pool and j not in judges:
            judges.append(j)
        if len(judges) >= n_judge:
            break
    if len(judges) < n_judge:
        rest = [i for i in pool if i not in judges]
        rest.sort(key=lambda i: (0 if i in ours_idx else 1, -i))
        for i in rest:
            judges.append(i)
            if len(judges) >= n_judge:
                break
    if len(judges) < n_judge:
        return {
            "ok": False,
            "idle_indices": [g["index"] for g in strict_idle],
            "ours_indices": sorted(ours_idx),
            "reason": f"need {n_judge} judge GPUs, found {len(judges)}",
        }

    judge_set = set(judges)
    train_pool = [g for g in strict_idle if g["index"] not in judge_set]
    if len(train_pool) < n_train:
        return {
            "ok": False,
            "idle_indices": [g["index"] for g in train_pool],
            "judge_gpus": judges,
            "reason": f"need {n_train} idle train GPUs disjoint from judges {judges}, found {len(train_pool)}",
        }

    idxs = sorted(g["index"] for g in train_pool)
    best_train: list[int] | None = None
    for start in range(min(idxs), max(idxs) + 1):
        cand = list(range(start, start + n_train))
        if all(i in idxs for i in cand):
            best_train = cand
            break
    if best_train is None:
        by_free = sorted(train_pool, key=lambda g: (-g["mem_free_mib"], g["index"]))
        best_train = sorted(g["index"] for g in by_free[:n_train])

    by_idx = {g["index"]: g for g in gpus}
    all_gpus = best_train + judges
    return {
        "ok": True,
        "gpus": all_gpus,
        "train_gpus": best_train,
        "judge_gpus": judges,
        "judge_gpu": judges[0],
        "detail": [by_idx[i] for i in all_gpus if i in by_idx],
        "reason": "selected",
    }


def select_four_gpu_bundle(
    *,
    free_mib: int = 75000,
    util_max: float = 5.0,
    allow_pids: set[int] | None = None,
) -> dict[str, Any] | None:
    """Pick any 4 idle GPUs; prefer consecutive; judge = largest free among them."""
    gpus = query_gpus()
    idle = [g for g in gpus if is_gpu_idle(g, free_mib=free_mib, util_max=util_max, allow_pids=allow_pids)]
    if len(idle) < 4:
        return {
            "ok": False,
            "idle_indices": [g["index"] for g in idle],
            "idle_detail": idle,
            "reason": f"need 4 idle GPUs, found {len(idle)}",
        }

    idxs = sorted(g["index"] for g in idle)
    # Prefer consecutive runs of length 4.
    best: list[int] | None = None
    for start in range(min(idxs), max(idxs) + 1):
        cand = list(range(start, start + 4))
        if all(i in idxs for i in cand):
            best = cand
            break
    if best is None:
        # Fall back to four largest-free idle GPUs (sorted ascending for readability).
        by_free = sorted(idle, key=lambda g: (-g["mem_free_mib"], g["index"]))
        best = sorted(g["index"] for g in by_free[:4])

    by_idx = {g["index"]: g for g in idle}
    bundle = [by_idx[i] for i in best]
    # Prefer the highest-index GPU as judge so train CVD is often 0,1,2.
    judge = max(bundle, key=lambda g: (g["mem_free_mib"], g["index"]))
    train = sorted((g["index"] for g in bundle if g["index"] != judge["index"]))
    return {
        "ok": True,
        "gpus": best,
        "judge_gpu": judge["index"],
        "train_gpus": train,
        "detail": bundle,
        "reason": "selected",
    }


def cmd_probe(args: argparse.Namespace) -> int:
    allow = {int(x) for x in args.allow_pids.split(",") if x.strip()} if args.allow_pids else set()
    if args.train_only:
        gpus = query_gpus()
        idle = [
            g
            for g in gpus
            if is_gpu_idle(g, free_mib=args.free_mib, util_max=args.util_max, allow_pids=allow)
        ]
        n_train = int(args.n_train)
        if len(idle) < n_train:
            result = {
                "ok": False,
                "idle_indices": [g["index"] for g in idle],
                "train_gpus": [],
                "judge_gpus": [],
                "gpus": gpus,
                "reason": (
                    f"need {n_train} idle train-only GPUs with no foreign compute PIDs, "
                    f"free>={args.free_mib}MiB, util<={args.util_max}%; found {len(idle)}"
                ),
            }
        else:
            idxs = sorted(g["index"] for g in idle)
            best: list[int] | None = None
            for start in range(min(idxs), max(idxs) + 1):
                cand = list(range(start, start + n_train))
                if all(i in idxs for i in cand):
                    best = cand
                    break
            if best is None:
                by_free = sorted(idle, key=lambda g: (-g["mem_free_mib"], g["index"]))
                best = sorted(g["index"] for g in by_free[:n_train])
            result = {
                "ok": True,
                "gpus": best,
                "train_gpus": best,
                "judge_gpus": [],
                "idle_indices": idxs,
                "detail": [g for g in idle if g["index"] in set(best)],
                "reason": "selected_train_only",
            }
    elif args.bundle8:
        prefer = [int(x) for x in args.prefer_judge.split(",") if x.strip()] if args.prefer_judge else [4, 7]
        result = select_train_judge_bundle(
            n_train=args.n_train,
            n_judge=args.n_judge,
            free_mib=args.free_mib,
            util_max=args.util_max,
            allow_pids=allow,
            prefer_judge=prefer,
        )
    elif args.bundle:
        result = select_four_gpu_bundle(
            free_mib=args.free_mib, util_max=args.util_max, allow_pids=allow
        )
    else:
        gpus = query_gpus()
        idle = [
            g
            for g in gpus
            if is_gpu_idle(g, free_mib=args.free_mib, util_max=args.util_max, allow_pids=allow)
        ]
        result = {
            "ok": len(idle) >= args.min_idle,
            "idle_indices": [g["index"] for g in idle],
            "gpus": gpus,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result.get("ok") else 1


def cmd_reserve(args: argparse.Namespace) -> int:
    """Allocate a small CUDA tensor on each GPU and hold until signal/timeout."""
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        print("[error] no gpus", file=sys.stderr)
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    import torch

    pid_path = Path(args.pid_file) if args.pid_file else None
    if pid_path:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()), encoding="utf-8")

    stop = {"flag": False}

    def _stop(*_a):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    holders = []
    mib = int(args.mib)
    elems = max(1, (mib * 1024 * 1024) // 4)  # float32
    for local_i, physical in enumerate(gpus):
        with torch.cuda.device(local_i):
            t = torch.empty(elems, dtype=torch.float32, device=f"cuda:{local_i}")
            t.fill_(1.0)
            torch.cuda.synchronize()
            holders.append(t)
            print(f"[reserve] GPU{physical} held ~{mib}MiB local={local_i}", flush=True)

    status = {
        "phase": "holding",
        "pid": os.getpid(),
        "gpus": gpus,
        "mib_each": mib,
        "started_at": time.time(),
    }
    if args.status_file:
        Path(args.status_file).write_text(json.dumps(status, indent=2), encoding="utf-8")

    deadline = time.time() + float(args.timeout_secs) if args.timeout_secs > 0 else None
    while not stop["flag"]:
        if deadline is not None and time.time() >= deadline:
            break
        time.sleep(1.0)

    # Drop references so CUDA memory frees promptly.
    holders.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if pid_path and pid_path.exists():
        try:
            pid_path.unlink()
        except OSError:
            pass
    print("[reserve] released", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("probe", help="Probe idle GPUs / select GPU bundle")
    pp.add_argument("--free-mib", type=int, default=75000)
    pp.add_argument("--util-max", type=float, default=5.0)
    pp.add_argument("--min-idle", type=int, default=4)
    pp.add_argument("--bundle", action="store_true", help="select 3 train + 1 judge")
    pp.add_argument("--bundle8", action="store_true", help="select N train + M judge (default 6+2)")
    pp.add_argument("--train-only", action="store_true", help="select N strictly idle train GPUs, no judges")
    pp.add_argument("--n-train", type=int, default=6)
    pp.add_argument("--n-judge", type=int, default=2)
    pp.add_argument("--prefer-judge", default="4,7")
    pp.add_argument("--allow-pids", default="")
    pp.add_argument("--pretty", action="store_true")
    pp.set_defaults(func=cmd_probe)

    pr = sub.add_parser("reserve", help="Hold small CUDA allocations on GPUs")
    pr.add_argument("--gpus", required=True, help="comma-separated physical GPU indices")
    pr.add_argument("--mib", type=int, default=512)
    pr.add_argument("--pid-file", default="")
    pr.add_argument("--status-file", default="")
    pr.add_argument("--timeout-secs", type=int, default=0)
    pr.set_defaults(func=cmd_reserve)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
