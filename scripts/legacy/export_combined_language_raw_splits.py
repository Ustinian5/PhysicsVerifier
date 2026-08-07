#!/usr/bin/env python3
"""Export raw combined-language samples according to a normalized split manifest.

`export_combined_language_eval_slices.py` writes verifier-compatible normalized JSON
and a manifest containing global source indices. This script replays the source stream
and writes the corresponding raw `prompt/response/label/reward/metadata` samples for
data provenance and later rule-expansion workflows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.combined_language_io import iter_rollout_batches  # noqa: E402
from scripts.combined_language_samples import sample_to_eval_row  # noqa: E402


SPLIT_INDEX_KEYS = {
    "smoke": "smoke_indices",
    "rule_expansion": "rule_expansion_indices",
    "main_test": "main_test_indices",
    "val_ablation": "val_indices",
}


def _load_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"manifest must be a JSON object: {path}")
    return data


def _want_indices(manifest: Dict[str, Any]) -> Dict[str, Set[int]]:
    raw = manifest.get("index_sets")
    if not isinstance(raw, dict):
        raise SystemExit("manifest missing index_sets")
    out: Dict[str, Set[int]] = {}
    for split, key in SPLIT_INDEX_KEYS.items():
        vals = raw.get(key) or []
        if not isinstance(vals, list):
            raise SystemExit(f"manifest index_sets.{key} must be a list")
        out[split] = {int(x) for x in vals}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw source samples for combined-language splits.")
    parser.add_argument("--input", type=str, default=str(REPO_ROOT / "data" / "combined_language_only.json"))
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--max-rollouts", type=int, default=0)
    args = parser.parse_args()

    src = Path(args.input)
    manifest_path = Path(args.manifest)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_path)
    wants_by_split = _want_indices(manifest)
    split_by_index: Dict[int, str] = {}
    for split, indices in wants_by_split.items():
        for idx in indices:
            if idx in split_by_index:
                raise SystemExit(f"global index appears in multiple splits: {idx}")
            split_by_index[idx] = split

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {split: [] for split in wants_by_split}
    need = set(split_by_index)
    global_idx = 0
    seen_rollouts = 0
    max_rollouts = int(args.max_rollouts or 0)

    for batch in iter_rollout_batches(src):
        rid = batch.get("rollout_id")
        samples = batch.get("samples") if isinstance(batch.get("samples"), list) else []
        for local_idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            if global_idx in need:
                split = split_by_index[global_idx]
                eval_row = sample_to_eval_row(rid, sample, seq_index=local_idx)
                rows_by_split[split].append(
                    {
                        "id": eval_row["id"],
                        "global_index": global_idx,
                        "rollout_id": rid,
                        "sample_index": sample.get("index", local_idx),
                        "question": eval_row["question"],
                        "raw_sample": sample,
                    }
                )
                need.remove(global_idx)
                if not need:
                    break
            global_idx += 1
        seen_rollouts += 1
        if not need:
            break
        if max_rollouts > 0 and seen_rollouts >= max_rollouts:
            break

    outputs: Dict[str, str] = {}
    for split, rows in rows_by_split.items():
        path = outdir / f"raw_{split}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs[split] = str(path)

    summary = {
        "source": str(src),
        "manifest": str(manifest_path),
        "outdir": str(outdir),
        "max_rollouts": max_rollouts or None,
        "uncollected_global_indices": sorted(need),
        "outputs": outputs,
        "counts": {split: len(rows) for split, rows in rows_by_split.items()},
    }
    (outdir / "raw_split_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if need:
        raise SystemExit(f"failed to collect {len(need)} requested raw samples")


if __name__ == "__main__":
    main()
