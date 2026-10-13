"""Prepare, run and score the frozen 50-question C0p/C3 comparison.

prepare is offline. run is server-local and waits for confirm-detached from a
new SSH session. No model/server deployment, parameter search or automatic retry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import urlparse
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_gpt_oss_selfcheck as r
from scripts import continuous_context_v2 as protocol
from scripts import gpt_oss_harmony_transport as transport
from scripts.gpt_oss_rule_protocol import native_tools

REFERENCE = ROOT / "results/retrieval_comparison_50_20260920"
INPUT = REFERENCE / "frozen50/inputs.json"
CATALOG = ROOT / "catalogs/rules_unified_3000.json"
# First-round formal manifest: keep model configuration/template/weight index.
MODEL_FILES = {
    "config.json": "933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566",
    "tokenizer_config.json": "9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070",
    "model.safetensors.index.json": "ede2655fdc05008561983b6e0829c600727c28d591e071077377059f03a6c00e",
    "chat_template.jinja": "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146",
}
LIMITS = {"backend": "vllm", "review_protocol": protocol.PROTOCOL,
          "max_requests_per_sample": 12, "max_tokens_per_response": 16384,
          "output_tokens_per_sample": 16384, "input_tokens_per_sample": 600000,
          "context_window": 131072, "wall_seconds_per_sample": 1800,
          "request_timeout_seconds": 1200, "tool_returned_chars": 500000,
          "max_distinct_full_leaves": 6}
CONTROLS = [
    {"id": "control_arithmetic", "question": "质量2 kg的物体受到6 N合外力，求加速度。",
     "prediction": "由牛顿第二定律 a=F/m=6/2=12 m/s²。"},
    {"id": "control_equivalent", "question": "无摩擦水平面上质量m物体受恒力F，从静止运动t秒，求位移。",
     "prediction": "动量定理给出 v=Ft/m，恒定加速度下平均速度为 v/2，因此位移为 Ft²/(2m)。"},
    {"id": "control_missing", "question": "质点沿平面曲线运动，已知瞬时速率v及曲率半径R，求瞬时加速度。",
     "prediction": "法向加速度为 v²/R；切向加速度取决于速率变化，现有信息不能确定总加速度。"},
    {"id": "control_source_copy", "question": "对任意常数a与非零b，比较两个表达式。",
     "prediction": r"第一式为 \(z = \frac{a}{b}\)，第二式为 \(z=ab^{-1}\)。两式等价。"},
    {"id": "control_power_extremum", "question": "飞行器阻力D=A/v²+Bv²，A、B为正常数，求v>0时使功率最小的速度。",
     "prediction": "功率P=Dv=A/v+Bv³，令导数为零得到-A/v²+3Bv²=0，因此最小功率速度v=(A/(3B))^(1/2)。"},
]
SOURCE_FILES = ["scripts/run_continuous_context_v2.py", "scripts/continuous_context_v2.py",
                "scripts/run_gpt_oss_selfcheck.py", "scripts/continuous_context_prompts.py",
                "scripts/gpt_oss_harmony_transport.py", "scripts/gpt_oss_rule_protocol.py",
                "scripts/run_retrieval_comparison.py", "core/rule_catalog_retrieval.py",
                "scripts/evaluate_physics_eval_sets.py"]
SCORING_FILES = ("metrics_automatic_proxy_location_unreviewed.json", "review/blind_review.json",
                 "review/operator_map.json")


def source_hashes():
    return {name: r.sha256_file(ROOT / name) for name in SOURCE_FILES}


def frozen_rows():
    if r.sha256_file(INPUT) != r.EXPECTED_INPUT_SHA256:
        raise ValueError("Frozen 50 inputs changed")
    if r.sha256_file(CATALOG) != r.EXPECTED_CATALOG_SHA256:
        raise ValueError("Frozen catalog changed")
    rows = json.loads(INPUT.read_text())
    if (len(rows) != 50 or len({x["id"] for x in rows}) != 50 or
            any(set(x) - {"id", "question", "prediction", "context"} for x in rows)):
        raise ValueError("Wrong sample count, duplicate ID or leaked reference field")
    return rows


def configuration():
    rows = frozen_rows()
    reference = json.loads((REFERENCE / SCORING_FILES[0]).read_text())
    r.frozen_reference_dataset(REFERENCE, reference)
    return {"protocol": protocol.PROTOCOL, "arms": list(protocol.ARMS),
            "model": r.EXPECTED_MODEL, "model_files": MODEL_FILES, "reasoning_effort": "medium",
            "temperature": 0, "top_p": 1, "api_base": "http://127.0.0.1:18083/v1",
            "input_sha256": r.sha256_file(INPUT), "catalog_sha256": r.sha256_file(CATALOG),
            "source_sha256": source_hashes(), "limits": LIMITS,
            "scoring_source_sha256": {name: r.sha256_file(REFERENCE / name) for name in SCORING_FILES},
            "prompts": {"base": protocol.BASE, "rules": protocol.RULES},
            "tools": native_tools(True), "controls": CONTROLS,
            "sample_ids": [x["id"] for x in rows],
            "quote_repair": "one quote-only patch within the original caps, no GT, no diagnosis edits; retain unresolved",
            "failure_policy": "no automatic retry; successful/empty outputs immutable; stop on first technical failure",
            "score": "unchanged legacy location proxy; physics validity requires independent review"}


def prepare(output):
    config = configuration()
    signature = r.sha256_value(config)
    output.mkdir(parents=True, exist_ok=True)
    with r.run_lock(output / "active.lock"):
        path = output / "plan.json"
        if path.exists():
            if json.loads(path.read_text())["configuration_sha256"] != signature:
                raise ValueError("Plan already frozen under different code/config; use a distinct result directory")
        else:
            r.write_json(path, {"configuration": config, "configuration_sha256": signature,
                                "created_unix": time.time(), "git": r.git_state(ROOT)})
    print(json.dumps({"state": "prepared_no_model_calls", "planned_formal": 100,
                      "planned_controls": 10, "native_io_probe": 1,
                      "configuration_sha256": signature, "output": str(output)}, ensure_ascii=False))


def validate_plan(output):
    plan = json.loads((output / "plan.json").read_text())
    if plan["configuration"] != configuration() or plan["configuration_sha256"] != r.sha256_value(plan["configuration"]):
        raise ValueError("Prepared inputs, prompts, tools, budgets or source hashes changed")
    return plan


def process_identity(pid):
    # /proc start time prevents acceptance of an unrelated process reusing a PID.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]


def confirm_detached(output):
    ready = json.loads((output / "disconnect_ready.json").read_text())
    os.kill(ready["pid"], 0)
    if process_identity(ready["pid"]) != ready["process_start"]:
        raise ValueError("PID reused")
    status = json.loads((output / "execution_status.json").read_text())
    if status["state"] != "waiting_disconnect_acceptance" or status["token"] != ready["token"]:
        raise ValueError("Runner is not waiting for this acceptance")
    if time.time() - ready["unix"] < 5:
        raise ValueError("Reconnect from a new SSH session after at least five seconds")
    if ready["configuration_sha256"] != validate_plan(output)["configuration_sha256"]:
        raise ValueError("Code/config changed after launch")
    r.write_json(output / "disconnect_acceptance.json", dict(ready, passed=True, confirmed_unix=time.time()))
    print("Detached runner verified; engineering acceptance will precede formal generation.")


def wait_detached(output, signature):
    ready = {"pid": os.getpid(), "process_start": process_identity(os.getpid()),
             "configuration_sha256": signature, "token": uuid.uuid4().hex, "unix": time.time()}
    r.write_json(output / "disconnect_ready.json", ready)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        r.write_json(output / "execution_status.json",
                     dict(ready, state="waiting_disconnect_acceptance", updated_unix=time.time()))
        receipt = output / "disconnect_acceptance.json"
        if receipt.exists():
            accepted = json.loads(receipt.read_text())
            if accepted.get("passed") and all(accepted.get(k) == v for k, v in ready.items()):
                return
        time.sleep(2)
    raise RuntimeError("No fresh SSH disconnect acceptance; no generation started")


def audit_result(result):
    if result.get("model") != r.EXPECTED_MODEL:
        raise ValueError("Wrong model in result")
    if result.get("status") == "completed" and not result.get("trace", {}).get("response_evidence"):
        raise ValueError("Completed result has no model response evidence")
    if result.get("first_diagnosis_path"):
        if r.sha256_file(Path(result["first_diagnosis_path"])) != result["first_diagnosis_sha256"]:
            raise ValueError("First diagnosis changed")
    seen = set()
    for item in result.get("trace", {}).get("response_evidence", []):
        if item["turn"] in seen:
            raise ValueError("Duplicate response evidence")
        seen.add(item["turn"])
        for key, hashkey in (("request_path", "request_sha256"), ("response_path", "response_sha256"),
                             ("raw_sse_path", "raw_sse_sha256")):
            if r.sha256_file(Path(item[key])) != item[hashkey]:
                raise ValueError("Raw artifact changed: " + item[key])
        if item["model_ids"] != [r.EXPECTED_MODEL]:
            raise ValueError("Wrong model in result")
        response = json.loads(Path(item["response_path"]).read_text())
        if response.get("model_ids") != item["model_ids"] or response.get("error"):
            raise ValueError("Response identity/error audit failed")


def healthy_result(result):
    return result["status"] == "completed" and result.get("quote_repair", {}).get("status") != "technical_failure"


def native_io_probe(client, tokenizer, catalog, output, config):
    """Explicit engineering-only forced leaf IO; never uses a formal benchmark item."""
    folder = output / "native_io"
    receipt = folder / "acceptance.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if not saved.get("passed") or saved["configuration_sha256"] != r.sha256_value(config):
            raise ValueError("Native IO acceptance configuration changed")
        for name, sha in saved["artifacts"].items():
            if r.sha256_file(folder / name) != sha:
                raise ValueError("Native IO evidence changed")
        return
    if (folder / "request.json").exists():
        raise RuntimeError("Prior native IO attempt preserved; inspect it, do not resend blindly")
    corpus = r.Corpus(catalog, tree=True)
    rid = corpus.ids[0]
    messages = protocol.messages(CONTROLS[1], rules=True, tree_root=corpus.call(
        "read_tree", {"node_id": "root", "note": "Initial tree context"}, max_chars=500000))
    messages.append({"role": "user", "content": f"Engineering IO probe only: call read_rules with rule_ids=[\"{rid}\"]. This is not a physics evaluation."})
    response = transport.chat_response(client, endpoint=config["api_base"] + "/chat/completions",
        model=r.EXPECTED_MODEL, messages=messages, max_tokens=4096, timeout=1200,
        raw_path=folder / "raw.json", response_path=folder / "response.json", request_path=folder / "request.json",
        native_tools=native_tools(True), force_tool="read_rules")
    action = json.loads(response["content"])
    if action["action"] != "read_rules" or action["arguments"].get("rule_ids") != [rid]:
        raise ValueError("Forced engineering leaf IO did not execute")
    returned = corpus.call("read_rules", action["arguments"], max_chars=500000)
    call = response["tool_calls"][0]
    messages.extend([{"role": "assistant", "content": response["reasoning_content"], "tool_calls": [call]},
                     {"role": "tool", "tool_call_id": call["id"], "name": "read_rules", "content": r.canonical(returned)}])
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=native_tools(True))
    if (protocol.RULES not in rendered or rid not in rendered or returned[0]["check_logic"] not in rendered or
            (response["reasoning_content"] and response["reasoning_content"] not in rendered)):
        raise ValueError("Actual template lost instructions, rule text or exposed tool-call context")
    r.write_json(folder / "history.json", {"messages": messages, "rendered_sha256": r.sha256_value(rendered)})
    r.write_json(receipt, {"passed": True, "configuration_sha256": r.sha256_value(config),
                          "usage": response["usage"], "physics_accuracy_not_gate": True,
                          "artifacts": {n: r.sha256_file(folder / n) for n in ("request.json", "raw.json", "response.json", "history.json")}})


def collect(root, rows, signature):
    groups = {}
    for arm in protocol.ARMS:
        found = []
        for row in rows:
            prefix = r.hashlib.sha256(str(row["id"]).encode()).hexdigest()[:16]
            path = root / "samples" / arm / prefix / "result.json"
            if path.exists():
                result = json.loads(path.read_text())
                if (result["configuration_sha256"] != signature or result["id"] != row["id"] or
                        result["verifier"] != arm):
                    raise ValueError("Stored task configuration mismatch")
                audit_result(result)
                found.append(result)
        r.write_json(root / arm / "results.json", found)
        groups[arm] = found
    return groups


def run_batch(output):
    if sys.platform != "linux" or "huhengyuan" not in Path(os.environ.get("CONDA_PREFIX", "")).parts:
        raise ValueError("Run on the laboratory server in an explicit personal conda environment")
    if "huhengyuan" not in output.parts:
        raise ValueError("Use personal persistent storage")
    plan = validate_plan(output)
    config, signature = plan["configuration"], plan["configuration_sha256"]
    if urlparse(config["api_base"]).hostname != "127.0.0.1":
        raise ValueError("Server-local loopback service required")
    rows = frozen_rows()
    catalog = json.loads(CATALOG.read_text())
    with r.run_lock(output / "active.lock"), httpx.Client(trust_env=False) as client:
        (output / "runner.pid").write_text(str(os.getpid()))
        exit_code, phase, error = 1, "preflight", None
        try:
            for command in (["df", "-h", str(output)], ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv"]):
                print(r.subprocess.check_output(command, text=True), flush=True)
            service = client.get(config["api_base"] + "/models", timeout=10)
            service.raise_for_status()
            if not any(x["id"] == r.EXPECTED_MODEL and x.get("max_model_len", 0) >= 131072 for x in service.json()["data"]):
                raise ValueError("Existing service identity/context mismatch; do not launch another model")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(r.EXPECTED_MODEL, local_files_only=True, trust_remote_code=False)
            runtime = {"files": {n: r.sha256_file(Path(r.EXPECTED_MODEL) / n) for n in
                       ("config.json", "tokenizer_config.json", "model.safetensors.index.json", "chat_template.jinja")},
                       "packages": {n: r.package_version(n) for n in ("vllm", "transformers", "torch", "httpx")}}
            if runtime["files"] != config["model_files"]:
                raise ValueError("Model configuration/template/weight index differ from frozen first round")
            manifest = dict(plan, runtime=runtime)
            if (output / "manifest.json").exists():
                previous = json.loads((output / "manifest.json").read_text())
                if previous["configuration_sha256"] != signature or previous["runtime"] != runtime:
                    raise ValueError("Runtime changed since initial execution")
            else:
                r.write_json(output / "manifest.json", manifest)
            wait_detached(output, signature)
            phase = "engineering"
            r.write_json(output / "execution_status.json", {"state": phase, "unix": time.time()})
            native_io_probe(client, tokenizer, catalog, output, config)
            for name, samples in (("controls", CONTROLS), ("formal", rows)):
                phase = name
                root = output / name
                r.write_json(output / "execution_status.json", {"state": phase, "unix": time.time()})
                for i, row in enumerate(samples):
                    order = protocol.ARMS[i % 2:] + protocol.ARMS[:i % 2]
                    for arm in order:
                        if (output / "STOP").exists():
                            raise RuntimeError("STOP requested; successful outputs retained")
                        result = r.run_sample(row, arm=arm, catalog=catalog, tokenizer=tokenizer, client=client,
                            endpoint=config["api_base"] + "/chat/completions", model=r.EXPECTED_MODEL,
                            root=root, signature=signature, limits=dict(LIMITS))
                        groups = collect(root, samples, signature)
                        r.write_json(root / "progress.json", {"processed": sum(map(len, groups.values())),
                            "planned": len(samples) * 2, "last_id": row["id"], "last_arm": arm,
                            "counts": {a: dict(Counter(x["status"] for x in v)) for a, v in groups.items()},
                            "unix": time.time()})
                        print(name, arm, row["id"], result["status"], result.get("quote_repair", {}).get("status"), flush=True)
                        if not healthy_result(result):
                            raise RuntimeError("Technical failure preserved; inspect artifacts before any continuation")
                if name == "controls":
                    passed = all(x["protocol_audit"]["passed"] for v in groups.values() for x in v)
                    r.write_json(root / "acceptance.json", {"passed": passed, "configuration_sha256": signature,
                        "physics_accuracy_not_gate": True, "cases": len(samples), "conditions": len(samples) * 2})
                    if not passed:
                        raise RuntimeError("Engineering quote/protocol acceptance failed; formal run not started")
            phase = "scoring"
            score(output)
            phase, exit_code = "finished", 0
        except BaseException as exc:
            error = f"{type(exc).__name__}: {str(exc)[:1500]}"
            raise
        finally:
            r.write_json(output / "execution_status.json", {"state": phase if exit_code == 0 else "stopped_" + phase,
                         "exit_code": exit_code, "error": error, "unix": time.time(),
                         "configuration_sha256": signature})
            (output / "runner.pid").unlink(missing_ok=True)


def score(output):
    plan = validate_plan(output)
    rows = frozen_rows()
    groups = collect(output / "formal", rows, plan["configuration_sha256"])
    source = REFERENCE / "metrics_automatic_proxy_location_unreviewed.json"
    dataset = r.frozen_reference_dataset(REFERENCE, json.loads(source.read_text()))
    scores, before_repair = {}, {}
    with tempfile.TemporaryDirectory(prefix="continuous-v2-score-") as tmp:
        for arm, results in groups.items():
            if len(results) == 50 and all(healthy_result(x) for x in results):
                scores[arm] = r.location_score(dataset, results, Path(tmp), arm)
                originals = []
                for item in results:
                    original = json.loads(Path(item["first_diagnosis_path"]).read_text())
                    originals.append(dict(item, diagnostics=original["diagnostics"]))
                before_repair[arm] = r.location_score(dataset, originals, Path(tmp), arm + "_before_repair")
    report = {"status": "automatic_location_proxy_unreviewed", "reference_sha256": r.sha256_file(source),
              "scores_all_50": scores, "scores_before_quote_repair": before_repair,
              "counts": {a: dict(Counter(x["status"] for x in v)) for a, v in groups.items()},
              "not_scored_arms": [a for a in protocol.ARMS if a not in scores],
              "quote_repair": {a: dict(Counter(x.get("quote_repair", {}).get("status", "none") for x in v)) for a, v in groups.items()},
              "unresolved_quotes": {a: sum(len(x.get("protocol_audit", {}).get("quote_invalid_indices", [])) for x in v) for a, v in groups.items()},
              "usage": {a: {"completed_responses": sum(x["trace"]["model_requests"] for x in v),
                            "input_token_estimate": sum(x["trace"]["input_tokens"] for x in v),
                            "output_used_or_conservatively_reserved": sum(x["trace"]["output_token_limit_reserved"] for x in v),
                            "full_leaf_read_questions": sum(bool(x["trace"]["read_rule_ids"]) for x in v)}
                        for a, v in groups.items()},
              "interpretation": "Location proxy only; exact quote repair does not establish physical correctness",
              "scorer_known_limitations": [
                  "Legacy precision excludes unlocatable findings",
                  "Legacy offset normalization maps a zero start offset to -1; preserved for common scoring",
                  "Location overlap does not verify the physical reason"]}
    if len(scores) == 2:
        rule_arm = next(a for a in protocol.ARMS if a != "C0p")
        report["changes_from_C0p"] = r.matched_claim_differences(scores["C0p"], scores[rule_arm])
        report["all_three_point_estimates_not_lower"] = all(
            scores[rule_arm]["summary"][k] >= scores["C0p"]["summary"][k] for k in ("precision", "recall", "f1"))
    r.write_json(output / "metrics_automatic_proxy_unreviewed.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "confirm-detached", "score"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CONDA_PREFIX"):
        parser.error("Use conda physicsverifier locally, or the personal server conda environment")
    functions = {"prepare": prepare, "run": run_batch, "confirm-detached": confirm_detached, "score": score}
    functions[args.mode](args.output.resolve())


if __name__ == "__main__":
    main()
