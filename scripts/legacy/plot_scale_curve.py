from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _load_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ckpt_raw = row.get("checkpoint_size")
            if not ckpt_raw:
                continue
            try:
                checkpoint_size = int(float(ckpt_raw))
            except ValueError:
                continue

            def _to_float(key: str) -> float:
                val = row.get(key)
                if val is None or val == "":
                    return 0.0
                try:
                    return float(val)
                except ValueError:
                    return 0.0

            rows.append(
                {
                    "checkpoint_size": float(checkpoint_size),
                    "precision": _to_float("precision"),
                    "recall": _to_float("recall"),
                    "f1": _to_float("f1"),
                    "accuracy": _to_float("accuracy"),
                    "inconclusive_ratio": _to_float("inconclusive_ratio"),
                    "sample_all_inconclusive_ratio": _to_float("sample_all_inconclusive_ratio"),
                    "missing_binding_ratio": _to_float("missing_binding_ratio"),
                }
            )

    rows.sort(key=lambda x: x["checkpoint_size"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scale-curve metrics from aggregated CSV output.")
    parser.add_argument("--input-csv", type=str, default="results/scale_curve/curve_metrics.csv")
    parser.add_argument("--output", type=str, default="results/scale_curve/scale_curve.png")
    parser.add_argument("--title", type=str, default="Scale Curve: Eval100 Metrics vs Expansion Size")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    out_path = Path(args.output)
    rows = _load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No points found in: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: ./.venv/bin/python -m pip install matplotlib"
        ) from exc

    x = [int(p["checkpoint_size"]) for p in rows]
    precision = [p["precision"] for p in rows]
    recall = [p["recall"] for p in rows]
    f1 = [p["f1"] for p in rows]
    accuracy = [p["accuracy"] for p in rows]
    inconclusive = [p["inconclusive_ratio"] for p in rows]
    sample_inconclusive = [p["sample_all_inconclusive_ratio"] for p in rows]
    missing_binding = [p["missing_binding_ratio"] for p in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(x, precision, marker="o", linewidth=2, label="Precision")
    axes[0].plot(x, recall, marker="o", linewidth=2, label="Recall")
    axes[0].plot(x, f1, marker="o", linewidth=2, label="F1")
    axes[0].plot(x, accuracy, marker="o", linewidth=2, label="Accuracy")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best")

    axes[1].plot(x, inconclusive, marker="o", linewidth=2, label="Inconclusive Ratio")
    axes[1].plot(x, sample_inconclusive, marker="o", linewidth=2, label="All-Inconclusive Ratio")
    axes[1].plot(x, missing_binding, marker="o", linewidth=2, label="Missing-Binding Ratio")
    axes[1].set_xlabel("Expansion Sample Count")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="best")

    fig.suptitle(args.title)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(
        {
            "input_csv": str(csv_path),
            "output": str(out_path),
            "points": len(rows),
        }
    )


if __name__ == "__main__":
    main()
