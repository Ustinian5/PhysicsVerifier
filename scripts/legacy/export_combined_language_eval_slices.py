#!/usr/bin/env python3
"""Export fixed-seed evaluation slices from `combined_language_only.json` (streaming).

Uses a **two-pass** scheme (count → deterministic index sample → collect) so memory stays
O(|outputs|), not O(all samples), when `--max-rollouts 0` scans the full file.

Outputs verifier-compatible JSON lists: `id`, `question`, `prediction`, `answer`.
The generated splits are mutually exclusive and intended for:
  - smoke: tiny sanity checks
  - rule_expansion: rule-library growth / experience mining
  - val_ablation: tuning and ablations
  - main_test: final held-out evaluation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.combined_language_io import iter_rollout_batches  # noqa: E402
from scripts.combined_language_samples import sample_to_eval_row  # noqa: E402


def _scan_sample_indices(path: Path, *, max_rollouts: int) -> Tuple[int, List[int], List[int], List[int]]:
    seen_rollouts = 0
    n = 0
    correct: List[int] = []
    wrong: List[int] = []
    unknown: List[int] = []
    for batch in iter_rollout_batches(path):
        rid = batch.get("rollout_id")
        for s in batch.get("samples") or []:
            if isinstance(s, dict):
                row = sample_to_eval_row(rid, s, seq_index=n)
                acc = row.get("source_reward_acc")
                if acc is True:
                    correct.append(n)
                elif acc is False:
                    wrong.append(n)
                else:
                    unknown.append(n)
                n += 1
        seen_rollouts += 1
        if max_rollouts > 0 and seen_rollouts >= max_rollouts:
            break
    return n, correct, wrong, unknown


def _collect_at_indices(path: Path, want: Set[int], *, max_rollouts: int) -> Dict[int, Dict[str, Any]]:
    seen_rollouts = 0
    idx = 0
    out: Dict[int, Dict[str, Any]] = {}
    need = set(want)
    for batch in iter_rollout_batches(path):
        rid = batch.get("rollout_id")
        for i, s in enumerate(batch.get("samples") or []):
            if not isinstance(s, dict):
                continue
            if idx in need:
                out[idx] = sample_to_eval_row(rid, s, seq_index=i)
                need.remove(idx)
                if not need:
                    return out
            idx += 1
        seen_rollouts += 1
        if max_rollouts > 0 and seen_rollouts >= max_rollouts:
            break
    return out


def _strip_meta(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for r in rows:
        c = dict(r)
        c.pop("meta", None)
        cleaned.append(c)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Export eval JSON slices from combined_language_only.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "combined_language_only.json"),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(REPO_ROOT / "data" / "derived"),
    )
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=0,
        help="Stop after N rollout batches (0 = full file; two passes).",
    )
    parser.add_argument("--expansion-n", type=int, default=600, help="Rule expansion / experience mining set size.")
    parser.add_argument("--test-n", type=int, default=100, help="Main locked test set size.")
    parser.add_argument(
        "--test-right-n",
        type=int,
        default=-1,
        help="If >=0, force this many reward_acc=True samples into main_test.",
    )
    parser.add_argument(
        "--test-wrong-n",
        type=int,
        default=-1,
        help="If >=0, force this many reward_acc=False samples into main_test.",
    )
    parser.add_argument("--val-n", type=int, default=80, help="Validation / ablation tuning set size.")
    parser.add_argument("--smoke-n", type=int, default=20, help="Tiny sanity-check list.")
    args = parser.parse_args()

    src = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    max_r = int(args.max_rollouts)
    seed = int(args.seed)
    smoke_k = max(0, int(args.smoke_n))
    expansion_k = max(0, int(args.expansion_n))
    test_k = max(0, int(args.test_n))
    val_k = max(0, int(args.val_n))

    n_total, correct_indices, wrong_indices, unknown_indices = _scan_sample_indices(src, max_rollouts=max_r)
    if n_total <= 0:
        raise SystemExit("No samples found; check --input / --max-rollouts.")

    rng = random.Random(seed)
    all_indices = list(range(n_total))
    rng.shuffle(all_indices)
    rng.shuffle(correct_indices)
    rng.shuffle(wrong_indices)
    rng.shuffle(unknown_indices)

    take_smoke = min(smoke_k, n_total)
    smoke_idx = set(all_indices[:take_smoke])

    remaining = [idx for idx in all_indices if idx not in smoke_idx]
    forced_right = int(args.test_right_n)
    forced_wrong = int(args.test_wrong_n)
    if forced_right >= 0 or forced_wrong >= 0:
        right_take = max(0, forced_right)
        wrong_take = max(0, forced_wrong)
        if right_take > len([x for x in correct_indices if x not in smoke_idx]):
            raise SystemExit(
                f"Not enough reward_acc=True samples for --test-right-n={right_take}; "
                f"available after smoke={len([x for x in correct_indices if x not in smoke_idx])}"
            )
        if wrong_take > len([x for x in wrong_indices if x not in smoke_idx]):
            raise SystemExit(
                f"Not enough reward_acc=False samples for --test-wrong-n={wrong_take}; "
                f"available after smoke={len([x for x in wrong_indices if x not in smoke_idx])}"
            )
        right_selected = [x for x in correct_indices if x not in smoke_idx][:right_take]
        wrong_selected = [x for x in wrong_indices if x not in smoke_idx][:wrong_take]
        test_idx = set(right_selected + wrong_selected)
        if test_k > len(test_idx):
            fill = [x for x in remaining if x not in test_idx]
            test_idx.update(fill[: test_k - len(test_idx)])
    else:
        take_test = min(test_k, max(0, n_total - take_smoke))
        test_idx = set(remaining[:take_test])

    used = smoke_idx | test_idx
    rest = [idx for idx in remaining if idx not in used]
    take_expansion = min(expansion_k, len(rest))
    expansion_idx = set(rest[:take_expansion])
    used |= expansion_idx
    rest = [idx for idx in rest if idx not in expansion_idx]
    take_val = min(val_k, len(rest))
    val_idx = set(rest[:take_val])

    want = smoke_idx | expansion_idx | test_idx | val_idx
    packed = _collect_at_indices(src, want, max_rollouts=max_r)

    smoke_rows = _strip_meta([packed[i] for i in sorted(smoke_idx) if i in packed])
    expansion_rows = _strip_meta([packed[i] for i in sorted(expansion_idx) if i in packed])
    test_rows = _strip_meta([packed[i] for i in sorted(test_idx) if i in packed])
    val_rows = _strip_meta([packed[i] for i in sorted(val_idx) if i in packed])

    smoke_path = outdir / "combined_language_smoke.json"
    expansion_path = outdir / "combined_language_rule_expansion.json"
    test_path = outdir / "combined_language_main_test.json"
    val_path = outdir / "combined_language_val_ablation.json"
    manifest_path = outdir / "combined_language_export_manifest.json"

    smoke_path.write_text(json.dumps(smoke_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    expansion_path.write_text(json.dumps(expansion_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest: Dict[str, Any] = {
        "source": str(src),
        "seed": seed,
        "max_rollouts": max_r or None,
        "population_samples_seen": n_total,
        "outputs": {
            "smoke": str(smoke_path),
            "rule_expansion": str(expansion_path),
            "main_test": str(test_path),
            "val_ablation": str(val_path),
            "counts": {
                "smoke": len(smoke_rows),
                "rule_expansion": len(expansion_rows),
                "main_test": len(test_rows),
                "val_ablation": len(val_rows),
            },
        },
        "source_reward_acc_counts": {
            "true": len(correct_indices),
            "false": len(wrong_indices),
            "unknown": len(unknown_indices),
        },
        "main_test_reward_acc_targets": {
            "right": forced_right if forced_right >= 0 else None,
            "wrong": forced_wrong if forced_wrong >= 0 else None,
        },
        "index_sets": {
            "smoke_indices": sorted(smoke_idx),
            "rule_expansion_indices": sorted(expansion_idx),
            "main_test_indices": sorted(test_idx),
            "val_indices": sorted(val_idx),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
