"""Print the error-positive P/R/F1 and coverage from a published final-answer result."""
import argparse
from pathlib import Path

from rule_framework import verified_resume as vr
from scripts.audit_final_answer_metrics import ARMS, divide, measure


def report_row(group):
    c = group['counts']
    tp, fp, fn, tn = (c[k] for k in ('TP', 'FP', 'FN', 'TN'))
    failed = c['failed_positive'] + c['failed_negative']
    abstained = c['uncertain_positive'] + c['uncertain_negative']
    valid = tp + fp + fn + tn
    wrong = tp + fn + c['failed_positive'] + c['uncertain_positive']
    if (any(type(v) is not int or v < 0 for v in c.values())
            or valid != group['valid_binary'] or valid + failed + abstained != group['n']
            or wrong != group['labeled_incorrect'] or group['n'] - wrong != group['labeled_correct']):
        raise ValueError('Inconsistent confusion matrix or coverage')
    metrics = group['original_binary_view']
    expected = dict(precision=divide(tp, tp + fp), recall=divide(tp, tp + fn), f1=divide(2 * tp, 2 * tp + fp + fn))
    if metrics != expected:
        raise ValueError('Published P/R/F1 differ from the error-positive confusion matrix')
    return dict(**metrics, valid=valid, total=group['n'], abstained=abstained, failed=failed,
                all_wrong_detected=divide(tp, wrong))


def tables(report):
    if 'positive_class' in report and report['positive_class'] != 'incorrect':
        raise ValueError('Expected incorrect as the positive class')
    if 'sizes' in report:
        return [(f'评测子集 {level["n"]} 题', level['groups']) for level in report['sizes']]
    groups = report['groups']
    if 'revised_labels_same_subset' in groups['pure']:
        return [('旧输出同题复算（修订标签）', {arm: groups[arm]['revised_labels_same_subset'] for arm in ARMS})]
    if report.get('positive_class') != 'incorrect':
        raise ValueError('Expected a published final-answer report with incorrect as the positive class')
    return [('题目级正式成绩', groups)]


def paired_records(path, report):
    from scripts.analyze_verifier_knowledge_groups import checked_json

    root = path.parent
    release_path = root / 'bench_release.json'
    release = checked_json(release_path, {str(release_path): report['benchmark_sha256']})
    manifest = report['files']
    predictions = {arm: {} for arm in ARMS}
    if path.name == 'archived_rescore.json':
        label_path = root / 'dispositions.json'
        gold = checked_json(label_path, {str(label_path): release['dispositions_sha256']})
        files = (Path(name) for name in manifest)
    elif path.name == 'metrics.json':
        label_path = root / 'labels_private.json'
        gold = checked_json(label_path, {str(label_path): release['labels_sha256']})
        files = (root / 'formal' / arm / vr.digest(sid)[:24] / 'result.json' for arm in ARMS for sid in gold)
    else:
        raise ValueError('Paired comparisons require the revised archive or the current formal release')
    for file in files:
        result = checked_json(file, manifest)
        arm, sid = result['arm'], result['id']
        if (arm not in ARMS or file.parent.parent.name != arm or file.parent.name not in {sid, vr.digest(sid)[:24]}
                or sid in predictions[arm] or gold[sid]['state'] != 'eligible'):
            raise ValueError('Unexpected, repeated, or ineligible answer')
        predictions[arm][sid] = result
    labels = {sid: gold[sid]['verdict'] for sid in predictions['pure']}
    groups = tables(report)[0][1]
    binary = {}
    for arm, rows in predictions.items():
        stats = measure(labels, rows)
        if stats['counts'] != groups[arm]['counts'] or len(rows) != groups[arm]['n']:
            raise ValueError('Paired records differ from the published confusion matrix')
        binary[arm] = {sid: r['value']['verdict'] for sid, r in rows.items()
                       if r['status'] == 'complete' and r['value']['verdict'] in {'incorrect', 'correct'}}
    return labels, binary


def paired_changes(labels, binary):
    for arm in ARMS[1:]:
        common = binary['pure'].keys() & binary[arm].keys()
        changes = []
        for gold_class in ('incorrect', 'correct'):
            changed = [sid for sid in common if labels[sid] == gold_class and binary[arm][sid] != binary['pure'][sid]]
            gained = sum(binary[arm][sid] == gold_class for sid in changed)
            changes.extend([gained, len(changed) - gained])
        yield arm, len(common), *changes


def best_existing_output(labels, binary):
    common = {sid: gold for sid, gold in labels.items() if all(sid in binary[arm] for arm in ARMS)}
    pure, best = {}, {}
    for sid, gold in common.items():
        baseline = binary['pure'][sid]
        choice = gold if any(binary[arm][sid] == gold for arm in ARMS) else baseline
        pure[sid] = dict(status='complete', value=dict(verdict=baseline))
        best[sid] = dict(status='complete', value=dict(verdict=choice))
    return measure(common, pure), measure(common, best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path, help='已发布的计分、同题复算或样本量曲线 JSON')
    parser.add_argument('--paired', action='store_true', help='在保存原件的节点复核同题相对纯组的判定变化')
    args = parser.parse_args()
    if not args.file.is_file():
        parser.exit(1, f'还没有该发布文件：{args.file}\n')
    path = args.file.resolve()
    published = vr.read(path)
    for title, groups in tables(published):
        if set(groups) != set(ARMS):
            raise ValueError('Expected all four verifier arms')
        print(f'\n{title}\n')
        print('| 组别 | Precision | Recall | F1 | 有效/总数 | 弃权 | 内容失败 | 全体标错题检出率 |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|')
        for arm in ARMS:
            row = report_row(groups[arm])
            print('| {arm} | {precision:.2%} | {recall:.2%} | {f1:.2%} | {valid}/{total} | '
                  '{abstained} | {failed} | {all_wrong_detected:.2%} |'.format(arm=arm, **row))
    if args.paired:
        labels, binary = paired_records(path, published)
        print('\n与纯组在同题均给出有效二分类时比较\n')
        print('| 组别 | 共同有效 | 新检出错答 | 丢失错答 | 消除误报 | 新增误报 |')
        print('|---|---:|---:|---:|---:|---:|')
        for row in paired_changes(labels, binary):
            print('| ' + ' | '.join(map(str, row)) + ' |')
        pure, best = best_existing_output(labels, binary)
        print(f"\n四组共同有效 {best['n']} 题。纯组 F1 {report_row(pure)['f1']:.2%}，"
              f"已知标签并限于逐题从四份已有判定中择一时，F1 上限 {report_row(best)['f1']:.2%}。"
              f"四组均漏检 {best['counts']['FN']}/{best['labeled_incorrect']} 份错答。")


if __name__ == '__main__':
    main()
