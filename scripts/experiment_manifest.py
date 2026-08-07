from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


MANIFEST_SCHEMA_VERSION = 1
SOURCE_DIRECTORIES = ("core", "rules", "rule_framework", "symbolic", "scripts")
SOURCE_SUFFIXES = {".py", ".sh"}
SOURCE_ROOT_FILES = ("environment.yml", "pyproject.toml")
SENSITIVE_ENV_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def fingerprint_file(
    path_value: str | Path,
    *,
    project_root: Path,
    required: bool = False,
    json_count: bool = False,
) -> Dict[str, Any]:
    raw = str(path_value or "")
    path = Path(raw).expanduser() if raw else Path()
    if raw and not path.is_absolute():
        path = project_root / path

    if not raw or not path.exists() or not path.is_file():
        result: Dict[str, Any] = {"path": raw, "exists": False}
        if required:
            raise FileNotFoundError(f"Required file does not exist: {raw}")
        return result

    result = {
        "path": _portable_path(path, project_root),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.is_symlink():
        result["symlink_target"] = os.readlink(path)
    if json_count:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, (list, dict)):
            result["json_count"] = len(payload)
        else:
            result["json_count"] = None
            result["json_type"] = type(payload).__name__
    return result


def fingerprint_files(
    paths: Mapping[str, str | Path],
    *,
    project_root: Path,
    json_count: bool = False,
) -> Dict[str, Dict[str, Any]]:
    return {
        name: fingerprint_file(
            path,
            project_root=project_root,
            required=False,
            json_count=json_count,
        )
        for name, path in paths.items()
    }


def validate_unified_catalog(path_value: str | Path, *, project_root: Path) -> Dict[str, Any]:
    fingerprint = fingerprint_file(path_value, project_root=project_root, required=True)
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unified catalog is not valid JSON: {path_value}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Unified catalog must be a JSON object: {path_value}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("catalog_type") != "unified_rules_v2":
        raise ValueError(
            "Unified catalog must declare metadata.catalog_type='unified_rules_v2': "
            f"{path_value}"
        )
    domains = payload.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError(f"Unified catalog must contain a non-empty domains array: {path_value}")
    fingerprint["catalog_type"] = metadata.get("catalog_type")
    fingerprint["schema_profile"] = metadata.get("schema_profile")
    fingerprint["total_domains"] = metadata.get("total_domains", len(domains))
    fingerprint["topics_with_rules"] = metadata.get("topics_with_rules")
    fingerprint["total_scenario_clusters"] = metadata.get("total_scenario_clusters")
    fingerprint["total_executable_rules"] = metadata.get("total_executable_rules")
    return fingerprint


def _git_bytes(project_root: Path, args: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed with {result.returncode}")
    return result.stdout


def capture_git_state(project_root: Path) -> Dict[str, Any]:
    try:
        head = _git_bytes(project_root, ["rev-parse", "HEAD"]).decode().strip()
        branch = _git_bytes(project_root, ["rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        status_bytes = _git_bytes(
            project_root,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        )
        diff_bytes = _git_bytes(
            project_root,
            ["diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        )
    except (OSError, RuntimeError) as exc:
        return {"available": False, "error": str(exc)[:1000]}

    status_text = status_bytes.decode("utf-8", errors="replace")
    return {
        "available": True,
        "head": head,
        "branch": branch,
        "dirty": bool(status_bytes),
        "status_entries": [line for line in status_text.splitlines() if line],
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
    }


def _iter_source_files(project_root: Path) -> Iterable[Path]:
    for relative in SOURCE_ROOT_FILES:
        path = project_root / relative
        if path.is_file():
            yield path
    for directory in SOURCE_DIRECTORIES:
        root = project_root / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES and "__pycache__" not in path.parts:
                yield path


def fingerprint_source_tree(project_root: Path) -> Dict[str, Any]:
    files: Dict[str, Dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    for path in sorted(set(_iter_source_files(project_root)), key=lambda item: item.as_posix()):
        relative = path.resolve().relative_to(project_root.resolve()).as_posix()
        file_hash = sha256_file(path)
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": file_hash}
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode("ascii"))
        aggregate.update(b"\0")
    return {
        "sha256": aggregate.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def capture_source_state(project_root: Path) -> Dict[str, Any]:
    return {
        "git": capture_git_state(project_root),
        "source_tree": fingerprint_source_tree(project_root),
    }


def resolve_python_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    located = shutil.which(value)
    if located:
        return str(Path(located).resolve())
    raise FileNotFoundError(f"Python interpreter does not exist: {value}")


def probe_python_runtime(python_executable: str, *, require_conda: bool = True) -> Dict[str, Any]:
    probe = """
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

packages = sorted(
    f"{dist.metadata.get('Name') or ''}=={dist.version}"
    for dist in importlib.metadata.distributions()
)
payload = {
    "executable": str(Path(sys.executable).resolve()),
    "python_version": sys.version.split()[0],
    "prefix": str(Path(sys.prefix).resolve()),
    "base_prefix": str(Path(sys.base_prefix).resolve()),
    "is_conda": (Path(sys.prefix) / "conda-meta").is_dir(),
    "conda_env": Path(sys.prefix).name,
    "package_count": len(packages),
    "package_set_sha256": hashlib.sha256("\\n".join(packages).encode("utf-8")).hexdigest(),
}
print(json.dumps(payload, sort_keys=True))
"""
    result = subprocess.run(
        [python_executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Unable to inspect Python interpreter {python_executable}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        runtime = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Python interpreter probe returned invalid JSON: {python_executable}") from exc
    if require_conda and not bool(runtime.get("is_conda")):
        raise RuntimeError(
            "Evaluation commands must use a conda interpreter; got "
            f"{runtime.get('executable') or python_executable}"
        )
    return runtime


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _redact_message(message: str) -> str:
    redacted = message
    for env_name in SENSITIVE_ENV_NAMES:
        secret = os.getenv(env_name)
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted[:2000]


class ExperimentManifest:
    def __init__(
        self,
        path: Path,
        payload: Dict[str, Any],
        *,
        project_root: Path,
        artifact_groups: Optional[Mapping[str, Mapping[str, str | Path]]] = None,
    ) -> None:
        self.path = path
        self.project_root = project_root
        self.payload = dict(payload)
        self.artifact_groups = {
            section: dict(paths) for section, paths in (artifact_groups or {}).items()
        }
        self.payload["schema_version"] = MANIFEST_SCHEMA_VERSION
        self.payload.setdefault("created_at_utc", utc_now())
        self.payload.setdefault("status", "prepared")
        self.payload.setdefault("stages", [])
        self.payload.setdefault("current_stage", "prepared")
        self._write()

    def _write(self) -> None:
        self.payload["updated_at_utc"] = utc_now()
        atomic_write_json(self.path, self.payload)

    def refresh_artifacts(self) -> None:
        for section, paths in self.artifact_groups.items():
            self.payload[section] = fingerprint_files(
                paths,
                project_root=self.project_root,
                json_count=section == "datasets",
            )

    def start_stage(self, name: str) -> None:
        self.payload["status"] = "running"
        self.payload["current_stage"] = name
        self.payload["stages"].append(
            {"name": name, "status": "running", "started_at_utc": utc_now()}
        )
        self._write()

    def complete_stage(self, name: str) -> None:
        self.refresh_artifacts()
        self.payload["current_stage"] = name
        self.payload["stages"].append(
            {"name": name, "status": "completed", "completed_at_utc": utc_now()}
        )
        self._write()

    def set_value(self, key: str, value: Any) -> None:
        self.payload[key] = value
        self._write()

    def complete(self) -> None:
        self.refresh_artifacts()
        self.payload["status"] = "completed"
        self.payload["current_stage"] = "completed"
        self.payload["completed_at_utc"] = utc_now()
        self._write()

    def fail(self, exc: BaseException) -> None:
        if self.payload.get("status") == "completed":
            return
        self.refresh_artifacts()
        self.payload["status"] = "failed"
        self.payload["failed_at_utc"] = utc_now()
        self.payload["failure"] = {
            "stage": self.payload.get("current_stage"),
            "type": type(exc).__name__,
            "message": _redact_message(str(exc)),
        }
        self._write()
