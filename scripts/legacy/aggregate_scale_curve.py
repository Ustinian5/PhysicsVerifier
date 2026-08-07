from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_checkpoint_size(path: str, payload: Dict[str, Any]) -> int:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    val = summary.get("checkpoint_size")
    if isinstance(val, int) and val > 0:
        return val

    m = re.search(r"ckpt_(\d+)", path)
    if m:
        return int(m.group(1))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate strict metrics from multiple checkpoint files.")
    parser.add_argument("--metrics-glob", type=str, default="results/scale_curve/ckpt_*/strict_metrics.json")
    parser.add_argument("--output-csv", type=str, default="results/scale_curve/curve_metrics.csv")
    parser.add_argument("--output-json", type=str, default="results/scale_curve/curve_metrics.json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.metrics_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.metrics_glob}")

    rows: List[Dict[str, Any]] = []
    for p in paths:
        payload = _load_json(Path(p))
        if not isinstance(payload, dict):
            continue
        ckpt = _extract_checkpoint_size(p, payload)
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        checks = payload.get("code_check_stats") if isinstance(payload.get("code_check_stats"), dict) else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}

        rows.append(
            {
                "checkpoint_size": ckpt,
                "precision": float(metrics.get("precision") or 0.0),
                "recall": float(metrics.get("recall") or 0.0),
                "f1": float(metrics.get("f1") or 0.0),
                "accuracy": float(metrics.get("accuracy") or 0.0),
                "inconclusive_ratio": float(checks.get("experience_code_inconclusive_ratio") or 0.0),
                "sample_all_inconclusive_ratio": float(checks.get("sample_all_inconclusive_ratio") or 0.0),
                "missing_binding_ratio": float(checks.get("suppressed_missing_binding_ratio") or 0.0),
                "total_evaluable_samples": int(summary.get("total_evaluable_samples") or 0),
                "gt_incorrect_count": int(summary.get("gt_incorrect_count") or 0),
                "source_file": p,
            }
        )

    rows.sort(key=lambda x: x["checkpoint_size"])

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint_size",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "inconclusive_ratio",
                "sample_all_inconclusive_ratio",
                "missing_binding_ratio",
                "total_evaluable_samples",
                "gt_incorrect_count",
                "source_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_json = {
        "summary": {
            "metrics_glob": args.metrics_glob,
            "count": len(rows),
        },
        "points": rows,
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out_json["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
