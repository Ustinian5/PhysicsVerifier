"""C4: independent full-solution assessment followed by targeted rule assistance."""
from scripts import continuous_context_v2 as previous

PROTOCOL = "continuous_assessment_then_rules_v3"
ARMS = ("C0p", "C4")
BASE = previous.BASE
RULES = '''The rule library is an external knowledge aid in this same physics-checking conversation. Complete your own assessment of the entire solution first: identify the quantities, important relations, assumptions, and any independently established contradictions. Keep this assessment in the current context. There is no separate checker and no requirement to publish an intermediate answer.

Before the final diagnostics, make at least ONE actual search_rules call to check a concrete physical relation or applicability condition from the solution. This is an operational instruction to consult evidence, not a quota of errors or applicable rules. Use a short Chinese concept plus a distinguishing physical condition, mode="bm25", limit=6; use note for a concise statement of what relation is being checked. For example, distinguish 恒定功率 阻力 极值 from 恒定推力, or 球壳 感应电流 from 单一线圈. These are generic lookup examples, not answers to this problem. Do not write a simulated tool result.

Inspect returned titles/summaries. When a candidate could resolve the question, call read_rules for its actual ID and compare its trigger, check_logic and symbolic_hint with the objects, variable definitions and boundary conditions here. A title or search score alone supplies no physical evidence. If two searches for the same issue add no relevant evidence, use read_tree to navigate an actual child of the supplied root, or leave that issue and continue checking. read_tree accepts node IDs; read_rules accepts returned leaf IDs. Do not invent identifiers or read unrelated leaves to satisfy a count. At most six distinct full leaves may be read.

Use the rule review for two purposes: test an existing suspect relation against its actual premises, and look for a missed independent contradiction in a related part of the answer. Retain independently verified errors when the catalog lacks a rule, a lookup fails, or the retrieved rule concerns another situation. Remove a provisional allegation only when a concrete derivation or an applicable physical relation shows that allegation was wrong. Library coverage, failure to cite, a recommended method, or mere doubt does not settle correctness. A rule is fallible; verify it rather than treating its presence as proof.

Resolve any newly suggested allegation against the original problem and solution before reporting it. Reject applications with different subsystems, force laws, reference conventions or unknown essential premises. Accept equivalent valid methods and do not count propagation of an existing root as another mistake. After the lookup, revisit the entire original solution and combine the supported findings into ONE final JSON under the common schema. Copy minimal continuous original quotations exactly. Put only actually read and relevant IDs into rule_refs, which may be empty for independent findings. Preserve sufficient budget for the final JSON and the common quote-only correction. A genuine empty result is valid. If the response budget reaches its final turn, submit the findings already supported; do not sacrifice the final output to another lookup.'''


def messages(row, *, rules, tree_root):
    # Only the rule instruction changes; the pure-model input stays byte-identical.
    result = previous.messages(row, rules=False, tree_root=tree_root)
    if rules:
        result[0]["content"] = BASE + "\n\n" + RULES
    return result


invalid_quotes = previous.invalid_quotes
repair_message = previous.repair_message
apply_quote_patches = previous.apply_quote_patches
maybe_start_repair = previous.maybe_start_repair


def audit_final(value, row, events, *, rules):
    result = previous.audit_final(value, row, events, rules=rules)
    searches = [e for e in events if e.get("tool") == "search_rules"]
    result.update(targeted_search_attempts=len(searches),
                  search_instruction_followed=bool(searches) if rules else None)
    return result  # Non-use is measured, never a content reroll or scoring gate.


def progress_note(events, *, tree, remaining, remaining_output):
    note = previous.progress_note(events, tree=tree, remaining=remaining,
                                  remaining_output=remaining_output)
    if remaining > 1 and not any(e.get("tool") == "search_rules" for e in events):
        note += " Before final, use search_rules once for a concrete relation in this solution, then inspect any relevant full leaf."
    return note


def finish(row, arm, model, signature, state, corpus, value, sample_dir, *, consume_response, content=""):
    from scripts.run_gpt_oss_selfcheck import write_json
    result = previous.finish(row, arm, model, signature, state, corpus, value, sample_dir,
                             consume_response=consume_response, content=content)
    result["protocol_audit"] = audit_final(value, row, corpus.events, rules=arm == "C4")
    write_json(sample_dir / "result.json", result)
    return result
