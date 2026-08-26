#!/usr/bin/env python3
"""Validate and fingerprint the external OpenRLHF training implementation.

The four-GPU Qwen3-8B training entrypoint uses a variance-filter extension that is
not part of the official OpenRLHF 0.8.2 command-line contract.  This module
fails before Ray/GPU allocation when that extension is missing and records the
exact external source identity used by an experiment.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set


REQUIRED_TRAIN_FLAGS = (
    "--dynamic_filtering_mode",
    "--dynamic_filtering_min_spread",
    "--dynamic_filtering_min_std",
    "--dynamic_filtering_max_gen_batches",
    "--dynamic_filtering_max_candidate_samples",
    "--dynamic_filtering_budget_exhausted",
)

REQUIRED_FILTER_SYMBOLS = (
    "FilterConfig",
    "MODE_MEAN_RANGE",
    "MODE_REWARD_VARIANCE",
    "decide_group",
    "simulate_filter_rates",
)
EXPECTED_OPENRLHF_VERSION = "0.8.2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload: Dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.is_file(),
    }
    if resolved.is_file():
        stat = resolved.stat()
        payload.update({"size_bytes": stat.st_size, "sha256": sha256_file(resolved)})
    return payload


def _run_git(root: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def git_identity(root: Path) -> Dict[str, Any]:
    """Return commit plus a content hash for any uncommitted tracked patch."""
    top = _run_git(root, "rev-parse", "--show-toplevel")
    if not top:
        return {"available": False, "root": str(root.expanduser().resolve())}
    top_path = Path(top).resolve()
    status = _run_git(top_path, "status", "--porcelain=v1", "--untracked-files=all") or ""
    try:
        diff = subprocess.check_output(
            ["git", "-C", str(top_path), "diff", "--binary", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        diff = b""
    return {
        "available": True,
        "root": str(top_path),
        "commit": _run_git(top_path, "rev-parse", "HEAD") or "",
        "describe": _run_git(top_path, "describe", "--always", "--dirty", "--tags") or "",
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _installed_package_root() -> Optional[Path]:
    spec = importlib.util.find_spec("openrlhf")
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return Path(locations[0]).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    return None


def _resolve_package_root(source_root: Optional[Path]) -> Optional[Path]:
    if source_root is None:
        return _installed_package_root()
    root = source_root.expanduser().resolve()
    if (root / "openrlhf").is_dir():
        return root / "openrlhf"
    if root.name == "openrlhf" and root.is_dir():
        return root
    return None


def _top_level_symbols(path: Path) -> Set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets: Iterable[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def inspect_openrlhf_contract(
    source_root: Optional[Path] = None,
    *,
    expected_commit: str = "",
    require_clean: bool = False,
) -> Dict[str, Any]:
    errors = []
    warnings = []
    package_root = _resolve_package_root(source_root)
    if package_root is None:
        return {
            "ok": False,
            "errors": ["cannot locate the installed or configured openrlhf package"],
            "warnings": [],
        }

    checkout_root = package_root.parent
    train_path = package_root / "cli" / "train_ppo_ray.py"
    filter_path = package_root / "trainer" / "ppo_utils" / "dynamic_filter.py"

    missing_flags = list(REQUIRED_TRAIN_FLAGS)
    if train_path.is_file():
        train_source = train_path.read_text(encoding="utf-8")
        missing_flags = [flag for flag in REQUIRED_TRAIN_FLAGS if flag not in train_source]
    else:
        errors.append(f"missing training entrypoint: {train_path}")
    if missing_flags:
        errors.append("missing required training flags: " + ", ".join(missing_flags))

    missing_symbols = list(REQUIRED_FILTER_SYMBOLS)
    if filter_path.is_file():
        try:
            symbols = _top_level_symbols(filter_path)
        except (OSError, SyntaxError) as exc:
            symbols = set()
            errors.append(f"cannot parse dynamic_filter.py: {type(exc).__name__}: {exc}")
        missing_symbols = [name for name in REQUIRED_FILTER_SYMBOLS if name not in symbols]
    else:
        errors.append(f"missing variance filter implementation: {filter_path}")
    if missing_symbols:
        errors.append("missing required dynamic-filter symbols: " + ", ".join(missing_symbols))

    git = git_identity(checkout_root)
    actual_commit = str(git.get("commit") or "")
    if expected_commit and actual_commit != expected_commit:
        errors.append(
            f"OpenRLHF commit mismatch: expected {expected_commit}, found {actual_commit or '<unknown>'}"
        )
    if git.get("dirty"):
        message = "OpenRLHF checkout has uncommitted changes; their tracked diff hash is recorded"
        if require_clean:
            errors.append(message)
        else:
            warnings.append(message)

    try:
        package_version = importlib.metadata.version("openrlhf")
    except importlib.metadata.PackageNotFoundError:
        package_version = ""
    if package_version and package_version != EXPECTED_OPENRLHF_VERSION:
        errors.append(
            "OpenRLHF package version mismatch: "
            f"expected {EXPECTED_OPENRLHF_VERSION}, found {package_version}"
        )
    elif not package_version:
        warnings.append("OpenRLHF distribution version is unavailable")

    return {
        "ok": not errors,
        "contract": "physicsverifier_openrlhf_variance_filter_v1",
        "expected_package_version": EXPECTED_OPENRLHF_VERSION,
        "package_version": package_version,
        "package_root": str(package_root),
        "git": git,
        "train_entrypoint": file_identity(train_path),
        "dynamic_filter": file_identity(filter_path),
        "required_train_flags": list(REQUIRED_TRAIN_FLAGS),
        "missing_train_flags": missing_flags,
        "required_filter_symbols": list(REQUIRED_FILTER_SYMBOLS),
        "missing_filter_symbols": missing_symbols,
        "expected_commit": expected_commit,
        "require_clean": bool(require_clean),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the patched OpenRLHF training contract")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = inspect_openrlhf_contract(
        args.source_root,
        expected_commit=str(args.expected_commit or "").strip(),
        require_clean=bool(args.require_clean),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result.get("ok") is True else 2)


if __name__ == "__main__":
    main()
