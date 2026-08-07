from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rule_framework.incremental_validation import incremental_artifact_binding

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    import openai
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenAI package not found. Please run 'pip install openai'.") from exc


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _build_client() -> Any:
    if load_dotenv:
        load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def _normalize(vector: Iterable[float]) -> List[float]:
    values = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 0:
        return values
    return [item / norm for item in values]


def _cosine(left: List[float], right: List[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _embedding_fingerprint(rule: Dict[str, Any]) -> str:
    text = _text(rule.get("embedding_text") or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _valid_vector(raw: Any) -> List[float] | None:
    if not isinstance(raw, list) or not raw:
        return None
    try:
        vector = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in vector):
        return None
    return vector


def _load_valid_cached_embeddings(
    *,
    cache: Dict[str, Any],
    model: str,
    rules: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    if _text(cache.get("embedding_model") or "") != model:
        return {}
    raw_embeddings = cache.get("embeddings")
    if not isinstance(raw_embeddings, dict):
        return {}
    fingerprints = {
        str(rule.get("rule_id") or ""): _embedding_fingerprint(rule)
        for rule in rules
    }
    valid: Dict[str, List[float]] = {}
    for rule_id, raw_entry in raw_embeddings.items():
        if not isinstance(raw_entry, dict):
            continue
        if _text(raw_entry.get("fingerprint") or "") != fingerprints.get(str(rule_id)):
            continue
        vector = _valid_vector(raw_entry.get("vector"))
        if vector is not None:
            valid[str(rule_id)] = vector
    return valid


def _cache_payload(
    *,
    model: str,
    rules: List[Dict[str, Any]],
    embeddings: Dict[str, List[float]],
) -> Dict[str, Any]:
    rule_index = {str(rule.get("rule_id") or ""): rule for rule in rules}
    return {
        "schema_version": 2,
        "embedding_model": model,
        "embeddings": {
            rule_id: {
                "fingerprint": _embedding_fingerprint(rule_index[rule_id]),
                "vector": vector,
            }
            for rule_id, vector in embeddings.items()
            if rule_id in rule_index
        },
    }


def _validate_embedding_input(rules: List[Dict[str, Any]]) -> None:
    rule_ids = [str(rule.get("rule_id") or "") for rule in rules]
    if any(not rule_id for rule_id in rule_ids):
        raise ValueError("Embedding input contains an empty rule_id.")
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Embedding input contains duplicate rule_id values.")
    if any(not _text(rule.get("topic_key") or "") for rule in rules):
        raise ValueError("Embedding input contains an empty topic_key.")
    if any(not _text(rule.get("embedding_text") or "") for rule in rules):
        raise ValueError("Embedding input contains empty embedding_text.")


def _connected_components(vectors: List[List[float]], *, threshold: float) -> List[List[int]]:
    parent = list(range(len(vectors)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if _cosine(vectors[i], vectors[j]) >= threshold:
                union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(vectors)):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda item: (-len(item), item[0]))


def _cluster_topic_rules(
    rules: List[Dict[str, Any]],
    embeddings: Dict[str, List[float]],
    *,
    threshold: float,
    min_cluster_size: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    indexed = [rule for rule in rules if rule.get("rule_id") in embeddings]
    missing_embedding_ids = [
        str(rule.get("rule_id") or "")
        for rule in rules
        if str(rule.get("rule_id") or "") not in embeddings
    ]
    vectors = [_normalize(embeddings[str(rule["rule_id"])]) for rule in indexed]
    components = _connected_components(vectors, threshold=threshold) if vectors else []

    clusters: List[Dict[str, Any]] = []
    residual_rule_ids: List[str] = list(missing_embedding_ids)
    for index, component in enumerate(components, start=1):
        rule_ids = [str(indexed[i]["rule_id"]) for i in component]
        if len(rule_ids) < min_cluster_size:
            residual_rule_ids.extend(rule_ids)
            continue
        exemplar_rules = [indexed[i] for i in component[:5]]
        clusters.append(
            {
                "cluster_id": f"embedding_cluster_{index:02d}",
                "rule_ids": rule_ids,
                "size": len(rule_ids),
                "representative_rules": [
                    {
                        "rule_id": str(rule.get("rule_id") or ""),
                        "title": _text(rule.get("title") or ""),
                        "summary": _text(rule.get("summary") or ""),
                        "trigger": _text(rule.get("trigger") or ""),
                    }
                    for rule in exemplar_rules
                ],
            }
        )
    return clusters, residual_rule_ids


def _embed_rules(
    *,
    client: Any,
    model: str,
    rules: List[Dict[str, Any]],
    batch_size: int,
    existing: Dict[str, List[float]],
    on_progress: Any = None,
) -> Dict[str, List[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    embeddings = dict(existing)
    pending = [rule for rule in rules if str(rule.get("rule_id") or "") not in embeddings]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [_text(rule.get("embedding_text") or "") for rule in batch]
        result = client.embeddings.create(model=model, input=texts)
        response_items = list(result.data)
        if len(response_items) != len(batch):
            raise RuntimeError(
                f"Embedding API returned {len(response_items)} vectors for {len(batch)} inputs."
            )
        vectors_by_index: Dict[int, List[float]] = {}
        for position, item in enumerate(response_items):
            index = int(getattr(item, "index", position))
            vector = _valid_vector(getattr(item, "embedding", None))
            if index < 0 or index >= len(batch) or index in vectors_by_index or vector is None:
                raise RuntimeError("Embedding API returned invalid or duplicate indexed vectors.")
            vectors_by_index[index] = vector
        dimensions = {len(vector) for vector in vectors_by_index.values()}
        if len(dimensions) != 1:
            raise RuntimeError("Embedding API returned inconsistent vector dimensions.")
        for index, rule in enumerate(batch):
            embeddings[str(rule["rule_id"])] = vectors_by_index[index]
        if on_progress:
            on_progress(embeddings)
        print(f"[embedding] {min(start + batch_size, len(pending))}/{len(pending)} new embeddings")
    return embeddings


def run_embedding_clustering(
    *,
    input_path: Path,
    output_path: Path,
    cache_path: Path,
    embedding_model: str,
    similarity_threshold: float,
    min_cluster_size: int,
    batch_size: int,
    incremental_manifest_path: Path | None = None,
    incremental_stage: str = "embedding",
) -> Dict[str, Any]:
    payload = _load_json(input_path)
    rules = [item for item in payload.get("rules", []) if isinstance(item, dict)]
    _validate_embedding_input(rules)
    cache = _load_json(cache_path) if cache_path.exists() else {"embeddings": {}}
    if not isinstance(cache, dict):
        cache = {}
    existing = _load_valid_cached_embeddings(
        cache=cache,
        model=embedding_model,
        rules=rules,
    )

    client = _build_client()
    def _save_cache(current: Dict[str, List[float]]) -> None:
        _write_json(
            cache_path,
            _cache_payload(model=embedding_model, rules=rules, embeddings=current),
        )

    embeddings = _embed_rules(
        client=client,
        model=embedding_model,
        rules=rules,
        batch_size=batch_size,
        existing=existing,
        on_progress=_save_cache,
    )
    _save_cache(embeddings)
    missing_after_embedding = [
        str(rule.get("rule_id") or "")
        for rule in rules
        if str(rule.get("rule_id") or "") not in embeddings
    ]
    if missing_after_embedding:
        raise RuntimeError(
            f"Embedding output is incomplete; missing {len(missing_after_embedding)} rules."
        )
    dimensions = {len(vector) for vector in embeddings.values()}
    if len(dimensions) > 1:
        raise RuntimeError("Embedding cache/output contains inconsistent vector dimensions.")

    by_topic: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        by_topic[str(rule.get("topic_key") or "")].append(rule)

    topics = []
    for topic_key, topic_rules in sorted(by_topic.items()):
        clusters, residual_rule_ids = _cluster_topic_rules(
            topic_rules,
            embeddings,
            threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
        )
        assigned_rule_ids = [
            rule_id
            for cluster in clusters
            for rule_id in (cluster.get("rule_ids") or [])
        ] + residual_rule_ids
        expected_rule_ids = [str(rule.get("rule_id") or "") for rule in topic_rules]
        if len(assigned_rule_ids) != len(set(assigned_rule_ids)) or set(assigned_rule_ids) != set(expected_rule_ids):
            raise RuntimeError(f"Embedding clustering lost or duplicated rules in topic {topic_key!r}.")
        domain, _, topic = topic_key.partition("::")
        topics.append(
            {
                "domain": domain,
                "topic": topic,
                "topic_key": topic_key,
                "rule_count": len(topic_rules),
                "cluster_count": len(clusters),
                "clusters": clusters,
                "residual_rule_ids": residual_rule_ids,
            }
        )

    result = {
        "metadata": {
            "generator": "topic_local_rule_embedding_clustering_v1",
            "embedding_model": embedding_model,
            "similarity_threshold": similarity_threshold,
            "min_cluster_size": min_cluster_size,
            "batch_size": batch_size,
            "rule_count": len(rules),
            "topic_count": len(topics),
            "cache_hit_count": len(existing),
            "embedded_rule_count": len(rules) - len(existing),
        },
        "topics": topics,
    }
    if incremental_manifest_path is not None:
        result["metadata"].update(
            incremental_artifact_binding(
                incremental_manifest_path,
                stage=incremental_stage,
                input_paths={"rule_input": input_path},
            )
        )
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed cleaned rules and cluster them within each topic.")
    parser.add_argument("--input", default="results/unified_rules_3000/rule_embedding_input.json")
    parser.add_argument("--output", default="results/unified_rules_3000/rule_embedding_clusters.json")
    parser.add_argument("--cache", default="results/unified_rules_3000/rule_embedding_cache.json")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true", help="Kept for command stability; cache is always reused if present.")
    parser.add_argument("--incremental-manifest", default="")
    parser.add_argument("--incremental-stage", default="embedding")
    args = parser.parse_args()

    result = run_embedding_clustering(
        input_path=Path(args.input),
        output_path=Path(args.output),
        cache_path=Path(args.cache),
        embedding_model=args.embedding_model,
        similarity_threshold=float(args.similarity_threshold),
        min_cluster_size=int(args.min_cluster_size),
        batch_size=int(args.batch_size),
        incremental_manifest_path=(
            Path(args.incremental_manifest) if args.incremental_manifest else None
        ),
        incremental_stage=str(args.incremental_stage),
    )
    print(json.dumps({"metadata": result["metadata"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
