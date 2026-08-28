#!/usr/bin/env python3
"""Compare process-paragraph scores from two judge endpoints on the same rollouts.

Used to measure 8B self-judge vs 30B agreement before full GRPO.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

import urllib.request


def _load_pairs(path: Path, limit: int) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts = row.get("prompt") or []
            comps = row.get("completion") or []
            if isinstance(prompts, str):
                prompts = [prompts]
            if isinstance(comps, str):
                comps = [comps]
            for p, c in zip(prompts, comps):
                pairs.append({"prompt": str(p), "completion": str(c)})
                if limit and len(pairs) >= limit:
                    return pairs
    return pairs


def _post_rewards(url: str, queries: List[str], prompts: List[str], timeout: float) -> List[float]:
    payload = json.dumps({"query": queries, "prompts": prompts, "labels": [""] * len(queries)}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return [float(x) for x in data["rewards"]]


def spearman(xs: List[float], ys: List[float]) -> float:
    def ranks(vals: List[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = sum((a - mx) ** 2 for a in rx) ** 0.5
    deny = sum((b - my) ** 2 for b in ry) ** 0.5
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--url-a", default="http://127.0.0.1:8770/get_reward")
    p.add_argument("--url-b", default="")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--output", type=Path, default=Path("/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift/self_judge_smoke.json"))
    args = p.parse_args()

    pairs = _load_pairs(args.rollouts, args.limit)
    scores_a: List[float] = []
    scores_b: List[float] = []
    for i in range(0, len(pairs), args.batch):
        chunk = pairs[i : i + args.batch]
        qs = [c["completion"] for c in chunk]
        ps = [c["prompt"] for c in chunk]
        scores_a.extend(_post_rewards(args.url_a, qs, ps, args.timeout))
        if args.url_b:
            scores_b.extend(_post_rewards(args.url_b, qs, ps, args.timeout))

    report: Dict[str, Any] = {
        "n": len(scores_a),
        "mean_a": statistics.mean(scores_a) if scores_a else None,
        "std_a": statistics.pstdev(scores_a) if len(scores_a) > 1 else 0.0,
        "scores_a": scores_a,
    }
    if scores_b:
        agree = sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) <= 0.1) / max(len(scores_a), 1)
        report.update(
            {
                "mean_b": statistics.mean(scores_b),
                "std_b": statistics.pstdev(scores_b) if len(scores_b) > 1 else 0.0,
                "spearman": spearman(scores_a, scores_b),
                "agree_within_0.1": agree,
                "scores_b": scores_b,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("scores")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
