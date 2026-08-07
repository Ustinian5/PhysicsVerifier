from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("checkpoints"), list):
        raise ValueError("manifest must contain 'checkpoints' list")
    return data


def _cmds_for_ckpt(ckpt: Dict[str, Any], model: str) -> List[str]:
    name = str(ckpt.get("checkpoint_name") or "ckpt")
    n = int(ckpt.get("expansion_sample_count") or 0)

    semantic_output = str(ckpt.get("semantic_output"))
    distilled_output = str(ckpt.get("distilled_output"))
    translation_report = str(ckpt.get("translation_report"))
    translation_manifest = str(ckpt.get("translation_manifest"))
    unified_catalog = str(ckpt.get("unified_catalog"))
    eval100_input = str(ckpt.get("eval100_input"))
    eval100_result = str(ckpt.get("eval100_result"))
    eval100_audit = str(ckpt.get("eval100_audit"))
    strict_metrics = str(ckpt.get("strict_metrics"))
    expansion_input = str(ckpt.get("expansion_input"))

    return [
        f"mkdir -p results/checkpoints/{name} results/scale_curve/{name} catalogs/checkpoints",
        (
            "python scripts/generate_experience_rules.py "
            f"--input {expansion_input} "
            "--rules-catalog catalogs/rules_catalog_top_down.json "
            f"--model {model} "
            f"--output {semantic_output} "
            f"--distilled-output {distilled_output}"
        ),
        (
            "python scripts/generate_symbolic_checks.py "
            f"--input {distilled_output} "
            "--model gpt-4.1-mini "
            "--output-module symbolic/generated_experience_checks.py "
            f"--output-manifest {translation_manifest} "
            f"--report {translation_report} --repair"
        ),
        (
            "python scripts/manage_rule_library.py build "
            f"--experience {distilled_output} "
            f"--output {unified_catalog}"
        ),
        (
            "python scripts/run_verifier.py "
            f"--input {eval100_input} "
            f"--output {eval100_result} "
            f"--symbolic-output {eval100_audit} "
            f"--model {model} "
            f"--unified-catalog {unified_catalog} "
            f"--experience-code-manifest {translation_manifest} "
            "--experience-code-module symbolic.generated_experience_checks "
            "--no-agentic"
        ),
        (
            "python scripts/compute_strict_eval_metrics.py "
            f"--predictions {eval100_result} "
            f"--audit {eval100_audit} "
            "--rubric-meta data/rubric_eval_100_meta.json "
            f"--output {strict_metrics} "
            f"--checkpoint-size {n}"
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate runbook commands for scale-checkpoint experiments.")
    parser.add_argument("--manifest", type=str, default="results/scale_curve/checkpoint_manifest.json")
    parser.add_argument("--output-md", type=str, default="results/scale_curve/runbook.md")
    parser.add_argument("--output-sh", type=str, default="scripts/legacy/run_scale_checkpoints.sh")
    parser.add_argument("--model", type=str, default="qwen3-30b-a3b")
    args = parser.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    checkpoints = manifest.get("checkpoints") or []

    lines_md: List[str] = []
    lines_md.append("# Scale Checkpoint Runbook")
    lines_md.append("")
    lines_md.append("下列命令按检查点从小到大执行。此文档只生成步骤，不自动执行。")
    lines_md.append("")

    lines_sh: List[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]

    for ckpt in checkpoints:
        if not isinstance(ckpt, dict):
            continue
        name = str(ckpt.get("checkpoint_name") or "ckpt")
        n = int(ckpt.get("expansion_sample_count") or 0)
        cmds = _cmds_for_ckpt(ckpt, model=args.model)

        lines_md.append(f"## {name} ({n} samples)")
        lines_md.append("")
        lines_md.append("```bash")
        for c in cmds:
            lines_md.append(c)
        lines_md.append("```")
        lines_md.append("")

        lines_sh.append(f"echo '[run] {name} ({n})'")
        lines_sh.extend(cmds)
        lines_sh.append("")

    lines_md.append("## Aggregate Curve")
    lines_md.append("")
    lines_md.append("```bash")
    agg_cmd = (
        "python scripts/legacy/aggregate_scale_curve.py "
        "--metrics-glob 'results/scale_curve/ckpt_*/strict_metrics.json' "
        "--output-csv results/scale_curve/curve_metrics.csv "
        "--output-json results/scale_curve/curve_metrics.json"
    )
    plot_cmd = (
        "python scripts/legacy/plot_scale_curve.py "
        "--input-csv results/scale_curve/curve_metrics.csv "
        "--output results/scale_curve/scale_curve.png"
    )
    lines_md.append(agg_cmd)
    lines_md.append(plot_cmd)
    lines_md.append("```")

    lines_sh.append("echo '[run] aggregate scale curve'")
    lines_sh.append(agg_cmd)
    lines_sh.append("echo '[run] plot scale curve'")
    lines_sh.append(plot_cmd)

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines_md), encoding="utf-8")

    out_sh = Path(args.output_sh)
    out_sh.parent.mkdir(parents=True, exist_ok=True)
    out_sh.write_text("\n".join(lines_sh) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "manifest": args.manifest,
                "runbook_md": args.output_md,
                "runbook_sh": args.output_sh,
                "checkpoint_count": len([c for c in checkpoints if isinstance(c, dict)]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
