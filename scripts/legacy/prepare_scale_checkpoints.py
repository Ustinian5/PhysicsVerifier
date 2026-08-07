from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_eval_samples(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    out: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


def _checkpoint_values(step: int, max_size: int) -> List[int]:
    vals: List[int] = []
    cur = step
    while cur <= max_size:
        vals.append(cur)
        cur += step
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare rule-expansion checkpoint datasets at fixed intervals.")
    parser.add_argument("--input", type=str, default="data/evaluation_sample_1000_expansion.json")
    parser.add_argument("--output-dir", type=str, default="data/checkpoints")
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument("--manifest", type=str, default="results/scale_curve/checkpoint_manifest.json")
    args = parser.parse_args()

    if args.step <= 0:
        raise SystemExit("--step must be > 0")
    if args.max_size <= 0:
        raise SystemExit("--max-size must be > 0")

    src_path = Path(args.input)
    out_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    samples = _load_eval_samples(src_path)
    if len(samples) < args.max_size:
        raise SystemExit(f"Input samples={len(samples)} < requested max-size={args.max_size}")

    checkpoints = _checkpoint_values(args.step, args.max_size)

    plan: List[Dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in checkpoints:
        subset = samples[:n]
        ckpt_name = f"ckpt_{n:04d}"
        out_file = out_dir / f"evaluation_sample_{n}.json"
        out_file.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")

        plan.append(
            {
                "checkpoint_name": ckpt_name,
                "expansion_sample_count": n,
                "expansion_input": str(out_file),
                "semantic_output": f"results/checkpoints/{ckpt_name}/semantic_experience.json",
                "distilled_output": f"results/checkpoints/{ckpt_name}/semantic_experience_distilled.json",
                "translation_report": f"results/checkpoints/{ckpt_name}/experience_symbolic_translation_report.json",
                "translation_manifest": f"results/checkpoints/{ckpt_name}/experience_symbolic_program_manifest.json",
                "unified_catalog": f"catalogs/checkpoints/unified_rule_library_{ckpt_name}.json",
                "eval100_input": "data/evaluation_rubric_100.json",
                "eval100_result": f"results/checkpoints/{ckpt_name}/top_down_results_eval100.json",
                "eval100_audit": f"results/checkpoints/{ckpt_name}/symbolic_audit_eval100.json",
                "strict_metrics": f"results/scale_curve/{ckpt_name}/strict_metrics.json",
            }
        )

    manifest_payload = {
        "summary": {
            "source_input": str(src_path),
            "total_source_samples": len(samples),
            "step": args.step,
            "max_size": args.max_size,
            "checkpoint_count": len(plan),
        },
        "checkpoints": plan,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest_payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
