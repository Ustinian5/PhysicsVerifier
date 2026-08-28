#!/usr/bin/env python3
"""Three-layer leakage audit between train prompts and eval sets.

Hard-fail on exact sample/question ID overlap or normalized question-text hash
overlap with official HiPhO or held-out eval. High n-gram overlap is reported
for human review but is not a hard fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.hipho.hipho_contract import normalize_question_text

DEFAULT_MANIFEST = ROOT / "data/rl/train_manifest.json"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_id(row: Dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(
        row.get("id")
        or row.get("sample_id")
        or row.get("problem_id")
        or row.get("question_id")
        or meta.get("sample_id")
        or meta.get("id")
        or ""
    )


def row_question(row: Dict[str, Any]) -> str:
    if row.get("question"):
        return str(row.get("question"))
    if row.get("problem"):
        return str(row.get("problem"))
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if meta.get("question"):
        return str(meta["question"])
    for msg in row.get("messages") or row.get("input") or []:
        if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
            return str(msg["content"])
    return ""


def question_hash(text: str) -> str:
    return hashlib.sha256(normalize_question_text(text).encode("utf-8")).hexdigest()


def char_ngrams(text: str, n: int = 12) -> Set[str]:
    norm = normalize_question_text(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def index_rows(rows: Sequence[Dict[str, Any]], *, origin: str) -> List[Dict[str, Any]]:
    indexed: List[Dict[str, Any]] = []
    for row in rows:
        q = row_question(row)
        indexed.append(
            {
                "origin": origin,
                "id": row_id(row),
                "question": q,
                "hash": question_hash(q) if q else "",
                "ngrams": char_ngrams(q),
            }
        )
    return indexed


def audit(
    train_rows: Sequence[Dict[str, Any]],
    eval_sets: Dict[str, Sequence[Dict[str, Any]]],
    *,
    fuzzy_threshold: float = 0.85,
) -> Dict[str, Any]:
    train = index_rows(train_rows, origin="train")
    train_ids = {r["id"] for r in train if r["id"]}
    train_hashes = {r["hash"] for r in train if r["hash"]}
    exact_id: Dict[str, List[str]] = {}
    exact_hash: Dict[str, List[str]] = {}
    fuzzy: Dict[str, List[Dict[str, Any]]] = {}
    hard_fail = False
    for name, rows in eval_sets.items():
        ev = index_rows(rows, origin=name)
        id_hits = sorted(train_ids & {r["id"] for r in ev if r["id"]})
        hash_hits = sorted(train_hashes & {r["hash"] for r in ev if r["hash"]})
        exact_id[name] = id_hits
        exact_hash[name] = hash_hits
        if id_hits or hash_hits:
            hard_fail = True
        reviews: List[Dict[str, Any]] = []
        eval_by_hash = {r["hash"]: r for r in ev if r["hash"]}
        for tr in train:
            if not tr["ngrams"]:
                continue
            for er in ev:
                if not er["ngrams"]:
                    continue
                if tr["hash"] and tr["hash"] == er["hash"]:
                    continue
                score = jaccard(tr["ngrams"], er["ngrams"])
                if score >= fuzzy_threshold:
                    reviews.append(
                        {
                            "train_id": tr["id"],
                            "eval_id": er["id"],
                            "jaccard": round(score, 4),
                            "train_hash": tr["hash"],
                            "eval_hash": er["hash"],
                        }
                    )
        reviews.sort(key=lambda x: -x["jaccard"])
        fuzzy[name] = reviews[:50]
        _ = eval_by_hash
    excluded_ids = sorted({x for hits in exact_id.values() for x in hits})
    excluded_hashes = sorted({x for hits in exact_hash.values() for x in hits})
    return {
        "ok": not hard_fail,
        "hard_fail": hard_fail,
        "n_train": len(train_rows),
        "exact_id_overlap": exact_id,
        "exact_hash_overlap": exact_hash,
        "fuzzy_review": fuzzy,
        "excluded_ids": excluded_ids,
        "excluded_hashes": excluded_hashes,
        "at": datetime.now(timezone.utc).isoformat(),
    }


def load_exclusion(manifest_path: Path) -> Tuple[Set[str], Set[str]]:
    if not manifest_path.is_file():
        return set(), set()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return set(data.get("excluded_ids") or []), set(data.get("excluded_hashes") or [])


def row_excluded(row: Dict[str, Any], excluded_ids: Set[str], excluded_hashes: Set[str]) -> bool:
    rid = row_id(row)
    if rid and rid in excluded_ids:
        return True
    qh = question_hash(row_question(row))
    return bool(qh and qh in excluded_hashes)


def write_manifest(path: Path, report: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    payload = dict(report)
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--eval-file", dest="eval_files", action="append", default=[], help="name=path")
    p.add_argument("--heldout", type=Path, default=ROOT / "data/rl/heldout_eval.jsonl")
    p.add_argument("--hipho", type=Path, default=Path("/slow_share/jinjianhan/workspace/benchmarks/hipho/hipho_text_only.jsonl"))
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--fail-on-exact", action="store_true", default=True)
    p.add_argument("--no-fail-on-exact", dest="fail_on_exact", action="store_false")
    args = p.parse_args()

    eval_sets: Dict[str, List[Dict[str, Any]]] = {}
    if args.heldout.is_file():
        eval_sets["heldout"] = _load_jsonl(args.heldout)
    if args.hipho.is_file():
        eval_sets["hipho_to"] = _load_jsonl(args.hipho)
    for item in args.eval_files:
        if "=" not in item:
            raise SystemExit(f"--eval-file must be name=path, got {item}")
        name, path_s = item.split("=", 1)
        eval_sets[name] = _load_jsonl(Path(path_s))

    train_rows = _load_jsonl(args.train)
    report = audit(train_rows, eval_sets)
    report["train"] = str(args.train)
    report["eval_files"] = {k: "loaded" for k in eval_sets}
    write_manifest(args.manifest, report, extra={"frozen": True})
    print(json.dumps({k: report[k] for k in ("ok", "hard_fail", "n_train", "exact_id_overlap", "exact_hash_overlap") if k in report}, ensure_ascii=False, indent=2))
    if report["hard_fail"] and args.fail_on_exact:
        print("[error] exact eval leakage; rebuild train data from stable question hashes", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
