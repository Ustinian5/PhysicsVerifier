"""Rule fields in the reference-guided verifier release."""

SCHEMA = "verifier_rules_v2"
LEAF_FIELDS = ("rule_id", "title", "summary", "trigger", "check_logic", "error_type")


def current(metadata):
    schema = metadata.get("rule_schema")
    if schema not in (None, SCHEMA):
        raise ValueError(f"Unknown verifier rule schema: {schema}")
    return schema == SCHEMA


def without_symbolic_hint(value):
    """Project structured model output without touching quoted student or reference text."""
    if isinstance(value, dict):
        return {key: without_symbolic_hint(item) for key, item in value.items() if key != "symbolic_hint"}
    if isinstance(value, list):
        return [without_symbolic_hint(item) for item in value]
    return value
