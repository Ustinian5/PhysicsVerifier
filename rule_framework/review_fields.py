"""Rule fields in the reference-guided verifier release."""

PREVIOUS_SCHEMA = "verifier_rules_v2"
SCHEMA = "verifier_rules_v3"
LEAF_FIELDS = ("rule_id", "title", "summary", "trigger", "check_logic", "error_type")


def current(metadata):
    schema = metadata.get("rule_schema")
    if schema not in (None, PREVIOUS_SCHEMA, SCHEMA):
        raise ValueError(f"Unknown verifier rule schema: {schema}")
    return schema is not None


def compact(metadata):
    return current(metadata) and metadata["rule_schema"] == SCHEMA


def without_rule_metadata(value):
    """Drop only rule-local fields. The audit's error dispositions and reasons are evidence."""
    if not isinstance(value, dict) or not isinstance(value.get("experience_rules"), list):
        return value
    rules = [{k: v for k, v in rule.items() if k not in ("auxiliary", "rule_status")}
             if isinstance(rule, dict) else rule for rule in value["experience_rules"]]
    return dict(value, experience_rules=rules)


def without_symbolic_hint(value):
    """Project structured model output without touching quoted student or reference text."""
    if isinstance(value, dict):
        return {key: without_symbolic_hint(item) for key, item in value.items() if key != "symbolic_hint"}
    if isinstance(value, list):
        return [without_symbolic_hint(item) for item in value]
    return value
