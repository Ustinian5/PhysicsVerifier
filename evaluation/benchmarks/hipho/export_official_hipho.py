#!/usr/bin/env python3
"""Export official SciYu/HiPhO JSON exams into a frozen Text-Only jsonl + manifest."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from evaluation.benchmarks.hipho.hipho_contract import (
    OfficialHiPhOError,
    build_manifest,
    is_internal_expansion_row,
    is_official_text_only,
    iter_official_exam_files,
    load_exam_json,
    validate_official_rows,
    write_jsonl,
)

DEFAULT_HF_REVISION = "8e196c09a71e4e68b75c422defa512473359e0e5"


def _git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return ""


def export_official_hipho(
    repo_dir: Path,
    out_jsonl: Path,
    manifest_path: Path,
    *,
    hf_revision: str = DEFAULT_HF_REVISION,
) -> Dict[str, Any]:
    exam_files = iter_official_exam_files(repo_dir)
    if not exam_files:
        raise OfficialHiPhOError(f"no official exam JSON files under {repo_dir}/data")

    all_rows: List[Dict[str, Any]] = []
    exams: List[str] = []
    file_hashes: Dict[str, str] = {}
    from evaluation.benchmarks.hipho.hipho_contract import sha256_file

    provenance = {
        "source": "SciYu/HiPhO",
        "hf_revision": hf_revision,
        "git_commit": _git_commit(repo_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    for path in exam_files:
        exam_name, rows = load_exam_json(path)
        exams.append(exam_name)
        file_hashes[str(path)] = sha256_file(path)
        for row in rows:
            row["provenance"] = dict(provenance)
            row["exam_file"] = path.name
            if is_internal_expansion_row(row):
                raise OfficialHiPhOError(f"refusing internal expansion row in {path}")
            all_rows.append(row)

    validate_official_rows(all_rows, require_text_only=False)
    text_only = [row for row in all_rows if is_official_text_only(row)]
    if not text_only:
        raise OfficialHiPhOError("official export produced zero Text-Only problems")
    validate_official_rows(text_only, require_text_only=True)

    n = write_jsonl(out_jsonl, text_only)
    manifest = build_manifest(
        repo_dir=repo_dir,
        jsonl_path=out_jsonl,
        n_all=len(all_rows),
        n_text_only=n,
        exams=exams,
        git_commit=provenance["git_commit"],
        hf_revision=hf_revision,
        extra={
            "exam_file_sha256": file_hashes,
            "n_problems_all": len(all_rows),
            "rejected_figure_problems": len(all_rows) - n,
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-dir", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    args = p.parse_args()
    manifest = export_official_hipho(
        args.repo_dir,
        args.out_jsonl,
        args.manifest,
        hf_revision=args.hf_revision,
    )
    print(json.dumps({"ok": True, **{k: manifest[k] for k in ("n_text_only", "n_exams", "jsonl")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
