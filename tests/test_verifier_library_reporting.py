import pytest

from rule_framework import verified_resume as vr
from scripts import audit_verifier_library_capacity as capacity
from scripts import probe_verifier_rerank as rerank
from scripts import report_final_answer_metrics as report
from scripts.audit_final_answer_metrics import measure


def test_published_f1_excludes_abstention_and_failure_while_whole_detection_keeps_both():
    labels = dict(a='incorrect', b='incorrect', c='incorrect', d='correct')
    predictions = {k: dict(status='complete', value=dict(verdict=v))
                   for k, v in dict(a='incorrect', b='uncertain', d='incorrect').items()}
    predictions['c'] = dict(status='content_failure', value=None)
    raw = measure(labels, predictions)
    row = report.report_row(raw)
    assert row == dict(precision=.5, recall=1, f1=2 / 3, valid=2, total=4,
                       abstained=1, failed=1, all_wrong_detected=1 / 3)
    raw['original_binary_view'] = raw['whole_cohort_view']
    with pytest.raises(ValueError, match='error-positive'):
        report.report_row(raw)
    with pytest.raises(ValueError, match='positive class'):
        report.tables(dict(positive_class='correct', sizes=[]))
    revisions = {arm: dict(old_labels_same_subset=None, revised_labels_same_subset=arm) for arm in report.ARMS}
    assert report.tables(dict(groups=revisions)) == [('旧输出同题复算（修订标签）', {a: a for a in report.ARMS})]


def test_development_rerank_can_abstain_and_only_orders_distinct_candidate_ids():
    allowed = {'one', 'two'}
    for ids in [[], ['two', 'one']]:
        assert rerank.selected_ids(dict(ordered_ids=ids), allowed, 2) == ids
    for invalid in [['other'], ['one', 'one'], ['one', 'two', 'one'], [None]]:
        with pytest.raises(ValueError, match='material IDs'):
            rerank.selected_ids(dict(ordered_ids=invalid), allowed, 2)


def test_oracle_is_restricted_to_available_outputs_on_exactly_the_common_rows():
    labels = dict(a='incorrect', b='incorrect', c='correct', d='correct')
    binary = {arm: dict(a='correct', b='correct', c='incorrect', d='correct') for arm in report.ARMS}
    binary['case_rag']['a'] = 'incorrect'
    binary['flat_skills']['c'] = 'correct'
    del binary['rule_tree']['d']
    assert next(report.paired_changes(labels, binary)) == ('case_rag', 4, 1, 0, 0, 0)
    pure, best = report.best_existing_output(labels, binary)
    assert pure['n'] == best['n'] == 3 and report.report_row(pure)['f1'] == 0
    assert best['counts']['FN'] == 1 and report.report_row(best)['f1'] == 2 / 3


def test_current_paired_report_binds_frozen_labels_and_individual_arm_files(tmp_path):
    labels = {'a': dict(state='eligible', verdict='incorrect')}
    labels_path = tmp_path / 'labels_private.json'
    vr.atomic(labels_path, labels)
    release = tmp_path / 'bench_release.json'
    vr.atomic(release, dict(labels_sha256=vr.sha(labels_path)))
    files, groups = {}, {}
    for arm in report.ARMS:
        file = tmp_path / 'formal' / arm / vr.digest('a')[:24] / 'result.json'
        row = dict(id='a', arm=arm, status='complete', value=dict(verdict='incorrect'))
        vr.atomic(file, row)
        files[str(file)] = vr.sha(file)
        groups[arm] = measure({'a': 'incorrect'}, {'a': row})
    published = dict(benchmark_sha256=vr.sha(release), positive_class='incorrect', files=files, groups=groups)
    args = (tmp_path / 'metrics.json', published)
    gold, binary = report.paired_records(*args)
    assert list(report.paired_changes(gold, binary)) == [(arm, 1, 0, 0, 0, 0) for arm in report.ARMS[1:]]
    vr.atomic(file, dict(row, status='content_failure'))
    with pytest.raises(ValueError, match='evidence changed'):
        report.paired_records(*args)


def test_source_subsets_stay_nested_with_whole_families_and_exclude_reserve():
    families = [dict(id='x', members=['a', 'b']), dict(id='y', members=['c']), dict(id='z', members=['d'])]
    build = {'a', 'b', 'c', 'd'}
    levels = list(capacity.family_levels(build, {'e'}, families, (2, 4), 14))
    assert [n for n, _ in levels] == [2, 4]
    assert len(levels[0][1]) == 2 and levels[0][1] < levels[1][1] == build
    assert ('a' in levels[0][1]) == ('b' in levels[0][1])
    families[0]['members'].append('e')
    with pytest.raises(ValueError, match='crosses splits'):
        list(capacity.family_levels(build, {'e'}, families, (2, 4), 14))
