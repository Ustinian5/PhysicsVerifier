"""Trace the 50 paired retrieval outcomes to the material actually present in each CHECK."""
import argparse
from collections import Counter
from pathlib import Path

from scripts import score_timed_semantic_retrieval as scored

vr, require = scored.vr, scored.require


def inspect_check(check, expected_messages):
    files = check['files']
    request_path, = [Path(p) for p in files if Path(p).name == 'request.json']
    raw_path = request_path.with_name('raw.json')
    for path in (request_path, raw_path):
        require(vr.sha(path) == files[str(path)], f'CHECK file changed: {path}')
    request = vr.read(request_path)['request']
    require(request.pop('messages') == expected_messages, f'Unexpected CHECK messages: {request_path}')
    raw = vr.read(raw_path)
    choice, = raw['choices']
    tokens = raw['prompt_token_ids']
    used = raw['usage']['completion_tokens']
    require(raw['model'] == request['model'] and used == len(choice['token_ids'])
            and raw['usage']['prompt_tokens'] == len(tokens) == check['effective_input']['token_count'],
            f'Unexpected model or token count: {raw_path}')
    receipt = dict(request_sha256=files[str(request_path)], raw_sha256=files[str(raw_path)],
                   max_output=request.pop('max_tokens'), used_output=used, finish=choice['finish_reason'])
    return receipt, request, tokens


def locate_material(old, trace, delivered):
    fused, chosen = set(trace['fused']), set(trace['selected'])
    before, after = set(old), set(delivered)
    require(len(before) == len(old) and len(after) == len(delivered) and after <= chosen <= fused,
            'Inconsistent delivery trace')
    removed = before - after
    return dict(old=len(before), new=len(after), retained=len(before & after), added=len(after - before),
                lost_before_rerank=len(removed - fused), lost_in_rerank=len((removed & fused) - chosen),
                lost_in_packing=len(removed & chosen), selected=len(chosen), not_packed=len(chosen - after),
                delivered_semantic_only=len(after & (set(trace['semantic']) - set(trace['lexical']))))


def summarize(rows):
    totals, by_count = Counter(), {}
    for row in rows:
        counts = row['material']
        totals.update(counts)
        direction = 'more' if counts['new'] > counts['old'] else 'fewer' if counts['new'] < counts['old'] else 'equal'
        by_count.setdefault(direction, Counter()).update([row['change']])
    return dict(material=dict(totals), changes_by_delivered_count=by_count,
                nonempty_bm25=sum(r['material']['old'] > 0 for r in rows),
                nonempty_fusion=sum(r['material']['new'] > 0 for r in rows),
                reduced_check_cap=sum(r['fusion']['max_output'] < r['bm25']['max_output'] for r in rows),
                identical_prompt_questions=[r['position'] for r in rows if r['same_prompt_tokens']])


def audit(root):
    trial = root / scored.cohort.RESULT
    dest = trial / scored.timing.NAME
    metrics_path = dest / 'metrics_50.json'
    metrics = vr.read(metrics_path)
    require(vr.read(dest / 'scoring_50_attempt_2.json')['metrics_sha256'] == vr.sha(metrics_path)
            and vr.sha(metrics_path) == vr.read(dest / 'paired_comparison_50.json')['metrics_sha256'],
            'Unbound published scores')
    registry = vr.read(trial / 'cohort.json')
    input_path = trial / 'inputs.json'
    label_path = root / scored.cohort.BENCH / 'labels_private.json'
    inputs = vr.read(input_path)
    ids = [row['id'] for row in inputs]
    require(metrics['cohort_sha256'] == vr.sha(trial / 'cohort.json') and registry['inputs_sha256'] == vr.sha(input_path)
            and registry['labels_sha256'] == vr.sha(label_path) and len(ids) == 50 and ids == metrics['ids'],
            'Frozen inputs or labels changed')
    labels = vr.read(label_path)
    controls, scores = scored.baseline_scores(trial, ids, {q: labels[q]['verdict'] for q in ids})
    require(scores == metrics['bm25'], 'Different BM25 control scores')
    replacement, = vr.read(trial / 'bm25_aligned/metrics.json')['changed']
    report = {}
    for arm in scored.cohort.ARMS:
        index = scored.timing.base.load_index(root, arm)
        group = metrics['groups'][arm]
        changes = {q: key for key, questions in group['same_question_changes'].items() for q in questions}
        rows = []
        for n, item in enumerate(inputs, 1):
            q = item['id']
            control = controls[arm][q]
            if arm == 'rule_tree' and q == replacement:
                path = trial / 'bm25_aligned/rows' / q / 'check/result.json'
                require(vr.sha(path) == control['file_sha256'], 'Different aligned tree CHECK')
                old_check = vr.read(path)
            else:
                old_check = scored.timing.base.archived_check(arm, q, control, root)
            folder = dest / arm / 'rows' / q
            row = vr.read(folder / 'result.json')
            require(vr.sha(folder / 'result.json') == group['results_sha256'][q]
                    and vr.sha(folder / 'selection.json') == row['files']['selection']
                    and vr.sha(folder / 'check/result.json') == row['files']['check'], 'Changed fusion row: ' + q)
            selection = vr.read(folder / 'selection.json')
            delivered = row['provided']
            require(selection['provided'] == delivered, 'Delivery mismatch: ' + q)
            def inspect(ids_sent, check, output):
                require(check['status'] == output['status'] and check['value'] == output['value'], 'Verdict mismatch: ' + q)
                packet = [index.store.view(i) for i in ids_sent]
                messages = scored.b.final.p.continuation_messages(item, {'queries': selection['query']}, packet, arm=arm)
                return inspect_check(check, messages)

            old, old_settings, old_tokens = inspect(control['delivered'], old_check, control)
            new, new_settings, new_tokens = inspect(delivered, vr.read(folder / 'check/result.json'), row)
            same_prompt = old_tokens == new_tokens
            require(old_settings == new_settings and same_prompt == (control['delivered'] == delivered),
                    'Unexplained CHECK input or decoding difference: ' + q)
            rows.append(dict(position=n, id=q, gold=labels[q]['verdict'], change=changes[q],
                             same_prompt_tokens=same_prompt, bm25=old, fusion=new,
                             material=locate_material(control['delivered'], selection['trace'], delivered)))
        report[arm] = dict(summary=summarize(rows), rows=rows)
    return dict(metrics_sha256=vr.sha(metrics_path), script_sha256=vr.sha(__file__), model_calls=0,
                unit='one material ID per question per arm', groups=report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    path = root / scored.cohort.RESULT / scored.timing.NAME / 'material_delivery_50.json'
    scored.b.final.immutable(path, audit(root))
