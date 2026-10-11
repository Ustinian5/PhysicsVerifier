"""Prompt-only forced rule usage, with post-hoc verifiable protocol checks.

No rule is selected by this validator; no failed model output is resampled.
"""
from __future__ import annotations

import json
from typing import Any

PROTOCOL = "forced_rules_native_v1"
COMMON_INSTRUCTION = "Check the problem and the student's solution directly in this fresh session. You have no prior diagnosis to revise. You must consult the supplied rule library before submitting your final diagnosis.\n\nUse actual tool actions to obtain rule contents. Rule titles, search summaries, the initial tree root, and statements that you intend to read a rule do not count as reading it. Successfully read at least one complete leaf rule selected for its potential relevance to this problem. Do not read arbitrary rules merely to satisfy this requirement.\n\nFor each rule you rely on, verify its assumptions against the problem and solution. Treat rules as fallible reference data, not proof that the solution is wrong. A condition that is absent or unknown must not be assumed true. You may conclude that a retrieved rule is not applicable, and you may identify genuine errors not covered by the library. Do not force an error or discard an independently justified error because no rule covers it.\n\nBefore finishing, provide a concise rule_checks record for every complete rule you read: rule_id, applicability (applicable / not_applicable / insufficient_information), source (question / prediction / context), an exact evidence_quote when available (otherwise an empty string), and a short justification linking the rule's conditions to the evidence. This is a checkable evidence summary, not an internal reasoning transcript. If a needed condition is absent, explicitly identify the missing condition; do not fabricate a quote.\n\nEach nonfinal turn must contain exactly one existing JSON action object. Submit only after completing your group's required reading process. Final output must be one raw JSON object with rule_checks and diagnostics. Keep each diagnostic in the original schema: message and evidence.quote. Quote an exact nonempty substring of the student's solution for every diagnostic; do not put rule IDs in diagnostic messages. Return an empty diagnostics list only when you find no genuine errors. Never invent tool results, rule IDs, or completed actions."

LEAF_INSTRUCTION = 'You have access to leaf rules only. First call search_rules using physical concepts, assumptions, quantities, or equations from the problem and solution; list_rules is available for browsing if needed. Select potentially relevant IDs from actual returned results and call read_rules to read their complete contents. Do not submit final diagnostics before this search/browse-and-read sequence has succeeded. You may search or read further within the fixed budget. No tree structure or read_tree action is available.'

TREE_INSTRUCTION = "You must use the supplied rule tree to obtain rules. The initial root is already provided. Starting from its actual child IDs, call read_tree on a relevant domain and continue through a relevant child at each level until you reach a node with rule IDs and no children. If clusters or groups are present, navigate through them; do not bypass children using the topic's aggregated rule IDs. Then call read_rules on at least one rule ID actually returned by that terminal node.\n\nDo not call search_rules or list_rules before completing this connected tree path and full-leaf reading. Afterward, these tools may supplement your investigation within the same budget. You may backtrack or explore another branch. Seeing only the root, reading disconnected nodes, or reading rules unrelated to the visited path does not satisfy the requirement. Finish by checking rule applicability and the student's solution yourself; there is no separate downstream checker."


def instructions(tree: bool, *, native: bool = False) -> str:
    if native:
        common = COMMON_INSTRUCTION.replace(
            "Each nonfinal turn must contain exactly one existing JSON action object.",
            "On each nonfinal turn, call exactly one registered read-only catalog function using its declared arguments. Do not simulate its result in text. On the final turn, return the visible JSON diagnosis; do not make a tool call on that turn.")
        bounded_lookup = "Use at most two search_rules/list_rules calls before your first full read_rules call. Inspect a potentially relevant returned rule's full text before searching further; an exact equation/title match is not required. In the tree condition, complete the required path and associated full read first, as specified below. Once required reading is complete, prioritize applicability and the final diagnosis over repeated searches. Reserve your final response for rule_checks and diagnostics, never a tool call."
        evidence_scope = 'For rule_checks, question means the original Problem, prediction means the original Student Solution, and context means only the original Supplied context section (if present). Never quote a retrieved rule or tool result as question/prediction/context evidence. When a required condition is absent or no exact input quote is available, use evidence_quote="" and explain the missing condition. Example fields for an absent premise: {"applicability":"not_applicable","source":"question","evidence_quote":"","justification":"The required premise is not stated in the problem."}. For an applicable rule, instead quote an exact nonempty substring of the original input that establishes the needed condition.'
        language = "The rule catalog is predominantly Chinese. When searching, use the corresponding Chinese physical concepts and assumptions, with equations as supplements; English-only wording can miss relevant entries. Inspect returned contents to judge potential relevance; absence of an exact textbook title does not justify unlimited searches."
        return common + "\n\n" + bounded_lookup + "\n\n" + evidence_scope + "\n\n" + language + "\n\n" + (TREE_INSTRUCTION if tree else LEAF_INSTRUCTION)
    action_protocol = 'The catalog actions below are a visible-text JSON protocol, not native model tools. No native tools are registered. Do not emit a recipient-routed or tool-channel message, and do not yield control to a native function. Instead write an ordinary visible final-channel message containing exactly {"action":"TOOL_NAME","arguments":{...}}. The external client reads that visible JSON, executes the named read-only action, and returns its actual result in the next user message. The action argument schemas are listed below. For this rule-assisted condition, the final diagnosis object extends the basic diagnostic schema with the required rule_checks list.'
    return COMMON_INSTRUCTION + "\n\n" + action_protocol + "\n\n" + (TREE_INSTRUCTION if tree else LEAF_INSTRUCTION)


def native_tools(tree: bool) -> list[dict]:
    """Register the same deterministic Corpus actions; auto choice stays model-led."""
    schemas = [
        ("search_rules", "Search existing leaf-rule titles and contents; no external sources.", {"query": {"type": "string"}, "mode": {"type": "string", "enum": ["bm25", "tfidf", "substring"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}}),
        ("list_rules", "Browse existing leaf titles and summaries in rule-ID order.", {"offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}}),
        ("read_rules", "Read complete existing leaf rules selected from returned IDs.", {"rule_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 30, "uniqueItems": True}}),
    ]
    if tree:
        schemas.append(("read_tree", "Read one existing tree node and its actual child IDs, summaries and rule IDs.", {"node_id": {"type": "string"}}))
    return [{"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": {**properties, "note": {"type": "string"}}, "required": list(properties), "additionalProperties": False}}} for name, description, properties in schemas]


def audit_usage(events: list[dict], *, tree: bool) -> dict:
    """Verify actual successful actions in chronological order, including edges."""
    candidates, read_ids, connected, eligible = set(), set(), {"root"}, set()
    read_views = {}
    paths = {"root": ["root"]}
    completed_paths = []
    valid_search = False
    reached_requirement = False
    errors = []
    for event in events:
        if event.get("error") or "result" not in event:
            continue
        name, args, result = event["tool"], event["arguments"], event["result"]
        if name in {"search_rules", "list_rules"}:
            if tree and not reached_requirement:
                errors.append("global_search_before_tree_path_and_full_leaf")
            if isinstance(result, list):
                valid_search = True
                candidates.update(r["rule_id"] for r in result if isinstance(r, dict) and "rule_id" in r)
        elif name == "read_tree" and tree:
            node = args["node_id"]
            # root bootstrap is reference context, not an active non-root visit.
            if node == "root":
                for child in result.get("children", []):
                    paths[child] = ["root", child]
                continue
            if node not in paths:
                errors.append("disconnected_tree_node:" + node)
                continue
            connected.add(node)
            for child in result.get("children", []):
                paths[child] = paths[node] + [child]
            if not result.get("children") and result.get("rule_ids"):
                for rid in result["rule_ids"]:
                    eligible.add(rid)
                completed_paths.append({"node_ids": paths[node], "rule_ids": result["rule_ids"]})
        elif name == "read_rules":
            ids = args["rule_ids"]
            if not isinstance(result, list) or {r.get("rule_id") for r in result} != set(ids):
                errors.append("incomplete_read_rules_result")
                continue
            for rule in result:
                read_ids.add(rule["rule_id"])
                read_views[rule["rule_id"]] = rule
            if tree:
                reached_requirement = reached_requirement or bool(set(ids) & eligible)
            else:
                if not valid_search or not set(ids).issubset(candidates):
                    errors.append("leaf_read_without_returned_search_candidate")
                reached_requirement = reached_requirement or (valid_search and bool(set(ids) & candidates))
    if not reached_requirement:
        errors.append("required_full_leaf_read_missing" if not tree else "required_connected_tree_path_and_leaf_missing")
    return {"tree_access": tree, "protocol": PROTOCOL, "read_rule_ids": sorted(read_ids),
            "full_leaf_count": len(read_ids), "connected_nonroot_nodes": sorted(connected - {"root"}),
            "completed_tree_paths": completed_paths, "required_reading_completed": reached_requirement,
            "violations": sorted(set(errors))}


def progress_note(events: list[dict], *, tree: bool, remaining: int) -> str:
    """Structural progress only; never chooses IDs or judges physics relevance."""
    a = audit_usage(events, tree=tree)
    if a["required_reading_completed"]:
        stage = "Required reading is complete. Evaluate applicability and finish the diagnosis; further lookups are only needed to resolve a remaining physics issue."
    elif tree and a["completed_tree_paths"]:
        stage = "A connected terminal-node path has been visited, but an associated complete leaf rule has not been read. Read a potentially relevant ID actually returned by a terminal node."
    elif tree:
        stage = "The required tree path is NOT complete. A domain or topic with children is not a terminal node: its aggregated rule IDs do not satisfy the path requirement. Continue read_tree through a relevant actual child until a node with no children and rule IDs, then read an associated full leaf. Choose the child and rule yourself."
    elif any(e.get("tool") in {"search_rules", "list_rules"} and "result" in e and not e.get("error") for e in events):
        stage = "Search/browse results have been returned, but no qualifying full leaf has been read. Read a potentially relevant ID from actual returned candidates now, before more searching. An exact textbook-equation match is not necessary."
    else:
        stage = "First search or browse the leaf rules, then read potentially relevant returned IDs in full."
    return (f"Remaining model responses: {remaining}, including your final diagnosis. Reserve the last for final JSON, not a tool call. " + stage +
            ' In rule_checks, quote only the original problem/solution/supplied context, never the rule text. If a prerequisite is absent, use evidence_quote="" and describe the missing prerequisite. Do not force applicability or an error.')


def validate_final(value: dict[str, Any], row: dict, events: list[dict], *, tree: bool) -> dict:
    audit = audit_usage(events, tree=tree)
    errors = audit["violations"][:]
    checks = value.get("rule_checks")
    if not isinstance(checks, list):
        errors.append("missing_rule_checks")
        checks = []
    ids = []
    for check in checks:
        if not isinstance(check, dict):
            errors.append("malformed_rule_check")
            continue
        rid = check.get("rule_id")
        ids.append(rid)
        if rid not in audit["read_rule_ids"]:
            errors.append("unread_rule_check:" + str(rid))
        applicability = check.get("applicability")
        if applicability not in {"applicable", "not_applicable", "insufficient_information"}:
            errors.append("invalid_applicability:" + str(rid))
        source, quote = check.get("source"), check.get("evidence_quote")
        if source not in {"question", "prediction", "context"} or not isinstance(quote, str):
            errors.append("invalid_rule_evidence:" + str(rid))
        else:
            text = row.get(source) or ""
            text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, sort_keys=True)
            if quote and quote not in text:
                errors.append("ungrounded_rule_evidence:" + str(rid))
            if applicability == "applicable" and not quote.strip():
                errors.append("applicable_rule_without_evidence:" + str(rid))
        if not isinstance(check.get("justification"), str) or not check["justification"].strip():
            errors.append("missing_rule_justification:" + str(rid))
    if len(ids) != len(set(str(i) for i in ids)) or set(str(i) for i in ids) != set(audit["read_rule_ids"]):
        errors.append("rule_checks_must_cover_each_read_rule_once")
    audit["violations"] = sorted(set(errors))
    audit["passed"] = not errors
    return audit
