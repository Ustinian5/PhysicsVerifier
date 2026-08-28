#!/usr/bin/env python3
"""Official SciYu/HiPhO data contract: provenance, fields, and Text-Only filtering."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

INTERNAL_SOURCE_MARKERS = (
    "evaluation_sample_",
    "_expansion.json",
    "heldout_eval",
    "internal150",
)
TEXT_ONLY_MODALITIES = {"text-only", "text only", "text", "to", "hipho-to"}
FIGURE_MODALITIES = {
    "text+illustration figure",
    "text+variable figure",
    "text+data figure",
    "ti",
    "tv",
    "td",
    "text+figure",
}
REQUIRED_ROW_FIELDS = ("id", "exam", "question", "modality", "full_mark")


class OfficialHiPhOError(ValueError):
    """Raised when a dataset is not official SciYu/HiPhO."""


AWARD_PT_RE = re.compile(
    r"Award\s+([0-9]+(?:\.[0-9]+)?)\s*(?:pt|pts|point|points)\b",
    re.IGNORECASE,
)


def _criterion_from_text(text: str, index: int) -> Optional[Dict[str, Any]]:
    src = _as_text(text)
    if not src:
        return None
    match = AWARD_PT_RE.search(src)
    weight = float(match.group(1)) if match else None
    if weight is None:
        return None
    return {
        "id": f"c{index}",
        "description": src,
        "weight": weight,
    }


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _as_text(value).casefold()


def is_internal_expansion_source(value: Any) -> bool:
    blob = _lower(value)
    return any(marker in blob for marker in INTERNAL_SOURCE_MARKERS)


def is_internal_expansion_row(row: Dict[str, Any]) -> bool:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    fields = [
        row.get("source"),
        row.get("origin"),
        row.get("dataset"),
        row.get("id"),
        row.get("sample_id"),
        row.get("question_id"),
        meta.get("source"),
        meta.get("sample_id"),
        meta.get("origin"),
        (row.get("provenance") or {}).get("source") if isinstance(row.get("provenance"), dict) else None,
    ]
    return any(is_internal_expansion_source(x) for x in fields if x is not None)


def modality_of(row: Dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return _lower(row.get("modality") or meta.get("modality") or "")


def has_figure_assets(row: Dict[str, Any]) -> bool:
    images = row.get("image_question") or row.get("images") or row.get("figure") or []
    if isinstance(images, str):
        return bool(images.strip())
    if isinstance(images, list):
        return any(bool(str(x).strip()) for x in images)
    return bool(images)


def is_official_text_only(row: Dict[str, Any]) -> bool:
    """Keep official Text-Only items only. Never strip figures from TI/TV/TD rows."""
    if is_internal_expansion_row(row):
        return False
    modality = modality_of(row)
    if modality in FIGURE_MODALITIES:
        return False
    if has_figure_assets(row):
        return False
    if modality and modality not in TEXT_ONLY_MODALITIES:
        return False
    return True


def _first(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return value
    return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _float_list(value: Any) -> List[float]:
    out: List[float] = []
    for item in _as_list(value):
        if item is None or item == "":
            continue
        out.append(float(item))
    return out


def normalize_marking_schemes(raw: Any) -> List[Dict[str, Any]]:
    """Normalize official marking into a list of schemes with weighted criteria.

    Official SciYu/HiPhO JSON stores marking as a list of schemes, where each
    scheme is a list of 'Award X pt if ...' strings. Dict criteria are also accepted.
    """
    if not raw:
        return []
    if isinstance(raw, dict) and "criteria" in raw:
        raw = [raw]
    if isinstance(raw, dict) and any(k in raw for k in ("marking_scheme", "schemes", "solutions")):
        raw = raw.get("marking_scheme") or raw.get("schemes") or raw.get("solutions")
    schemes_in = _as_list(raw)
    schemes: List[Dict[str, Any]] = []
    for idx, scheme in enumerate(schemes_in):
        if isinstance(scheme, dict) and "criteria" in scheme:
            criteria_in = _as_list(scheme.get("criteria"))
            name = _as_text(scheme.get("name") or scheme.get("id") or f"scheme_{idx}")
        elif isinstance(scheme, list):
            criteria_in = scheme
            name = f"scheme_{idx}"
        elif isinstance(scheme, str):
            criteria_in = [scheme]
            name = f"scheme_{idx}"
        elif isinstance(scheme, dict):
            criteria_in = [scheme]
            name = _as_text(scheme.get("name") or scheme.get("id") or f"scheme_{idx}")
        else:
            continue
        criteria: List[Dict[str, Any]] = []
        for j, crit in enumerate(criteria_in):
            parsed: Optional[Dict[str, Any]] = None
            if isinstance(crit, str):
                parsed = _criterion_from_text(crit, j)
            elif isinstance(crit, dict):
                weight = crit.get("weight", crit.get("points", crit.get("full_mark", crit.get("score"))))
                if weight is None and crit.get("description"):
                    parsed = _criterion_from_text(str(crit.get("description")), j)
                elif weight is not None:
                    parsed = {
                        "id": _as_text(crit.get("id") or crit.get("criterion_id") or f"c{j}"),
                        "description": _as_text(
                            crit.get("description") or crit.get("text") or crit.get("criterion") or crit.get("item")
                        ),
                        "weight": float(weight),
                    }
            if parsed:
                criteria.append(parsed)
        if criteria:
            schemes.append({"name": name or f"scheme_{idx}", "criteria": criteria})
    return schemes


def normalize_official_row(
    raw: Dict[str, Any],
    *,
    exam_name: str = "",
    extra_context: str = "",
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    exam = _as_text(
        _first(raw.get("exam"), raw.get("source"), meta.get("exam"), exam_name)
    )
    question = _as_text(_first(raw.get("question"), raw.get("problem"), meta.get("question")))
    context = "\n".join(x for x in [_as_text(extra_context), _as_text(raw.get("context"))] if x)
    answers = [str(x) for x in _as_list(_first(raw.get("answer"), raw.get("answers"), raw.get("ground_truth")))]
    points = _float_list(_first(raw.get("points"), raw.get("full_mark"), raw.get("full_marks"), meta.get("points")))
    if not points and answers:
        points = [1.0] * len(answers)
    full_mark = float(sum(points)) if points else float(raw.get("full_mark") or 0.0)
    marking = _first(
        raw.get("marking_schemes"),
        raw.get("marking_scheme"),
        raw.get("marking"),
        meta.get("marking"),
        meta.get("marking_scheme"),
    )
    row = {
        "id": _as_text(_first(raw.get("id"), raw.get("problem_id"), raw.get("sample_id"), meta.get("sample_id"))),
        "exam": exam,
        "question": question,
        "context": context,
        "answer": answers,
        "answer_type": [str(x) for x in _as_list(raw.get("answer_type") or meta.get("answer_type"))],
        "unit": _as_list(raw.get("unit")),
        "points": points,
        "full_mark": full_mark,
        "modality": _as_text(_first(raw.get("modality"), meta.get("modality"))),
        "field": _as_text(_first(raw.get("field"), meta.get("field"))),
        "subquestion": raw.get("subquestion") if raw.get("subquestion") is not None else list(range(len(answers))),
        "marking_schemes": normalize_marking_schemes(marking),
        "image_question": _as_list(raw.get("image_question") or raw.get("images")),
        "source": "SciYu/HiPhO",
        "provenance": dict(provenance or {"source": "SciYu/HiPhO"}),
    }
    if context and question and context not in question:
        row["question_with_context"] = f"{context}\n\n{question}"
    else:
        row["question_with_context"] = question
    return row


def validate_official_row(row: Dict[str, Any], *, index: int = 0) -> List[str]:
    errors: List[str] = []
    if is_internal_expansion_row(row):
        errors.append(f"row[{index}] internal expansion source is not official HiPhO")
    for field in REQUIRED_ROW_FIELDS:
        if not row.get(field) and row.get(field) != 0:
            errors.append(f"row[{index}] missing required field {field}")
    if not row.get("exam"):
        errors.append(f"row[{index}] missing exam")
    if row.get("full_mark") is None:
        errors.append(f"row[{index}] missing full_mark")
    return errors


def validate_official_rows(rows: Sequence[Dict[str, Any]], *, require_text_only: bool = False) -> None:
    if not rows:
        raise OfficialHiPhOError("official HiPhO dataset is empty")
    errors: List[str] = []
    for i, row in enumerate(rows):
        errors.extend(validate_official_row(row, index=i))
        if require_text_only and not is_official_text_only(row):
            errors.append(f"row[{i}] is not official Text-Only")
    if errors:
        preview = "; ".join(errors[:12])
        raise OfficialHiPhOError(f"official HiPhO contract failed ({len(errors)} issues): {preview}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_exam_json(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    exam_name = path.stem
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("problems") or payload.get("data") or [payload]
    context_parts: List[str] = []
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not item.get("id") and not item.get("question") and not item.get("problem"):
            info = item.get("information") or item.get("context") or item.get("note")
            if info:
                context_parts.append(str(info))
            continue
        rows.append(
            normalize_official_row(
                item,
                exam_name=exam_name,
                extra_context="\n".join(context_parts),
            )
        )
    return exam_name, rows


def iter_official_exam_files(repo_dir: Path) -> List[Path]:
    data_dir = repo_dir / "data"
    if not data_dir.is_dir():
        return []
    return sorted(p for p in data_dir.glob("*.json") if p.is_file())


def build_manifest(
    *,
    repo_dir: Path,
    jsonl_path: Path,
    n_all: int,
    n_text_only: int,
    exams: Sequence[str],
    git_commit: str = "",
    hf_revision: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "dataset": "SciYu/HiPhO",
        "subset": "HiPhO-TO",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "hf_revision": hf_revision,
        "repo_dir": str(repo_dir),
        "jsonl": str(jsonl_path),
        "jsonl_sha256": sha256_file(jsonl_path) if jsonl_path.is_file() else "",
        "n_problems_all": int(n_all),
        "n_text_only": int(n_text_only),
        "exams": list(exams),
        "n_exams": len(list(exams)),
        "notes": "Official Text-Only subset. Do not treat as the full multimodal HiPhO leaderboard.",
    }
    if extra:
        payload.update(extra)
    return payload


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise OfficialHiPhOError(f"missing official HiPhO manifest: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("dataset") != "SciYu/HiPhO":
        raise OfficialHiPhOError(f"manifest is not SciYu/HiPhO: {path}")
    if not data.get("git_commit") and not data.get("hf_revision"):
        raise OfficialHiPhOError("manifest lacks git commit and HF revision")
    if int(data.get("n_text_only") or 0) <= 0:
        raise OfficialHiPhOError("manifest n_text_only is missing or zero")
    return data


def sample_count_from_manifest(path: Path) -> int:
    return int(load_manifest(path)["n_text_only"])


_WHITESPACE_RE = re.compile(r"\s+")
_LATEX_SPACE_RE = re.compile(r"(?<!\\)[ \t]+")


def normalize_question_text(text: str) -> str:
    src = (text or "").replace("\u00a0", " ")
    src = src.replace("\\\\", "\\")
    src = _WHITESPACE_RE.sub(" ", src).strip()
    src = _LATEX_SPACE_RE.sub(" ", src)
    return src.casefold()
