#!/usr/bin/env python3
"""Build slime-compatible RL prompt jsonl from PhysicsVerifier data sources."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCES = [
    ROOT / "data/evaluation_sample_300.json",
    ROOT / "data/evaluation_sample_1000_expansion.json",
    ROOT / "data/evaluation_sample_3000_expansion.json",
    ROOT / "data/physics_rubric_data_1000.json",
]

SYSTEM_PROMPT = (
    "You are an expert physics competition solver. "
    "Show clear step-by-step reasoning and put the final answer in \\boxed{}."
)


def _norm_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _ngrams(text: str, n: int = 5) -> Set[str]:
    words = _norm_text(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _parse_answer(answer: Any) -> List[str]:
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(x) for x in answer if x]
    text = str(answer).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
        except Exception:
            pass
    return [text]


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            for item in data["samples"]:
                if isinstance(item, dict):
                    yield item
        else:
            yield data


def _extract_qa(record: Dict[str, Any]) -> Optional[Tuple[str, List[str], str]]:
    question = (
        record.get("question")
        or record.get("prompt")
        or record.get("input")
        or ""
    )
    if isinstance(question, list):
        parts = []
        for msg in question:
            if isinstance(msg, dict) and msg.get("content"):
                parts.append(str(msg["content"]))
        question = "\n".join(parts)
    question = str(question).strip()
    if not question:
        return None

    answer = (
        record.get("answer")
        or record.get("reference_answer")
        or record.get("label")
        or record.get("ground_truth")
    )
    labels = _parse_answer(answer)
    if not labels:
        return None

    sample_id = str(record.get("id") or record.get("sample_id") or record.get("rollout_id") or "")
    return question, labels, sample_id


def _load_bench_questions(bench_paths: List[Path]) -> List[Set[str]]:
    out: List[Set[str]] = []
    for path in bench_paths:
        for rec in _iter_records(path):
            qa = _extract_qa(rec)
            if qa is None:
                q = rec.get("problem") or rec.get("Problem") or rec.get("question")
                if q:
                    out.append(_ngrams(str(q)))
                continue
            out.append(_ngrams(qa[0]))
    return out


def _is_contaminated(question: str, bench_ngrams: List[Set[str]], threshold: float) -> bool:
    q_ng = _ngrams(question)
    for b in bench_ngrams:
        if _jaccard(q_ng, b) >= threshold:
            return True
    return False


def _to_slime_record(
    question: str,
    labels: List[str],
    sample_id: str,
    source: str,
) -> Dict[str, Any]:
    return {
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "label": labels,
        "metadata": {
            "question": question,
            "rm_type": "remote_rm",
            "source": source,
            "sample_id": sample_id,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/rl/rl_prompts_raw.jsonl")
    parser.add_argument("--sources", nargs="*", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--bench-paths", nargs="*", type=Path, default=[])
    parser.add_argument("--contamination-threshold", type=float, default=0.6)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bench_ngrams = _load_bench_questions(args.bench_paths)

    seen_questions: Set[str] = set()
    kept = 0
    skipped_dup = 0
    skipped_contaminated = 0

    with args.output.open("w", encoding="utf-8") as out:
        for src in args.sources:
            for rec in _iter_records(src):
                qa = _extract_qa(rec)
                if qa is None:
                    continue
                question, labels, sample_id = qa
                norm_q = _norm_text(question)
                if norm_q in seen_questions:
                    skipped_dup += 1
                    continue
                if bench_ngrams and _is_contaminated(question, bench_ngrams, args.contamination_threshold):
                    skipped_contaminated += 1
                    continue
                seen_questions.add(norm_q)
                out.write(
                    json.dumps(
                        _to_slime_record(question, labels, sample_id, source=str(src.name)),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
                if args.max_samples and kept >= args.max_samples:
                    break
            if args.max_samples and kept >= args.max_samples:
                break

    stats = {
        "kept": kept,
        "skipped_duplicate": skipped_dup,
        "skipped_contaminated": skipped_contaminated,
        "output": str(args.output),
        "bench_paths": [str(p) for p in args.bench_paths],
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
