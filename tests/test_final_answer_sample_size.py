import pytest

from scripts import analyze_final_answer_sample_size as curve


def test_balanced_paired_subsets_match_native_scores_and_preserve_failures(monkeypatch):
    monkeypatch.setattr(curve, 'SUBSET_DRAWS', 40)
    monkeypatch.setattr(curve, 'BOOTSTRAP_DRAWS', 150)
    labels = {f'q{i}': 'correct' if i % 2 else 'incorrect' for i in range(600)}
    predictions = {arm: {} for arm in curve.binary.ARMS}
    for i, (sid, gold) in enumerate(labels.items()):
        for arm, cases in predictions.items():
            verdict = gold if i % 5 else ('correct' if gold == 'incorrect' else 'incorrect')
            if arm == 'case_rag' and i % 7 == 0:
                cases[sid] = dict(status='content_failure', value=None)
            else:
                value = 'uncertain' if arm == 'flat_skills' and i % 3 == 0 else verdict
                cases[sid] = dict(status='complete', value=dict(verdict=value))
    report = curve.analyze(labels, predictions)
    order = report['subset_order']
    assert curve.cohort_order(dict(reversed(labels.items()))) == order
    previous = set()
    for n, level in zip(curve.SIZES, report['sizes'], strict=True):
        cohort = {k: labels[k] for label in curve.CLASSES for k in order[label][:n // 2]}
        assert previous < set(cohort) and len(cohort) == n == level['n']
        previous = set(cohort)
        for arm in curve.binary.ARMS:
            expect = curve.binary.measure(cohort, {k: predictions[arm][k] for k in cohort})
            assert level['groups'][arm] == expect
        assert level['versus_pure']['rule_tree']['paired_bootstrap_ci95_pp'] == [0, 0]
        assert level['versus_pure']['rule_tree']['balanced_subsets']['tied'] == curve.SUBSET_DRAWS
    end = report['sizes'][-1]
    assert end['groups']['flat_skills']['valid_binary'] == 400
    assert end['groups']['case_rag']['counts']['failed_positive'] > 0
    for v in end['versus_pure'].values():
        assert max(v['balanced_subsets'].values()) == curve.SUBSET_DRAWS
    low, high = end['versus_pure']['flat_skills']['paired_bootstrap_ci95_pp']
    assert low < high


def test_unpublished_benchmark_produces_no_result(tmp_path):
    with pytest.raises(ValueError, match='not been published'):
        curve.run(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_reduced_cohort_ends_at_actual_size(monkeypatch):
    monkeypatch.setattr(curve, 'SUBSET_DRAWS', 3)
    monkeypatch.setattr(curve, 'BOOTSTRAP_DRAWS', 12)
    labels = {str(i): curve.CLASSES[i % 2] for i in range(478)}
    predictions = {k: dict(status='complete', value=dict(verdict=label)) for k, label in labels.items()}
    report = curve.analyze(labels, {arm: predictions for arm in curve.binary.ARMS})
    assert [level['n'] for level in report['sizes']] == [100, 200, 400, 478]
    assert all(group['n'] == 478 and group['labeled_incorrect'] == 239 and group['valid_binary'] == 478
               for group in report['sizes'][-1]['groups'].values())


def test_changed_published_prediction_blocks_analysis(tmp_path, monkeypatch):
    vr = curve.final.vr
    plan = dict(configuration_sha256='test-config', configuration=dict(quota_per_class=1))
    monkeypatch.setattr(curve.final, 'execution_plan', lambda root: plan)
    rows = [dict(id=sid, question='Question', prediction='Answer', context='') for sid in ('a', 'b')]
    labels = dict(a=dict(state='eligible', verdict='correct'), b=dict(state='eligible', verdict='incorrect'))
    for name, value in (('inputs', rows), ('candidates', rows), ('dispositions', labels), ('labels_private', labels)):
        vr.atomic(tmp_path / f'{name}.json', value)
    release = dict(configuration_sha256='test-config', files={}, inputs_sha256=vr.sha(tmp_path / 'inputs.json'),
                   labels_sha256=vr.sha(tmp_path / 'labels_private.json'),
                   dispositions_sha256=vr.sha(tmp_path / 'dispositions.json'))
    vr.atomic(tmp_path / 'bench_release.json', release)
    arm = curve.binary.ARMS[0]
    path = tmp_path / 'formal' / arm / vr.digest('a')[:24] / 'result.json'
    record = dict(id='a', arm=arm, configuration_sha256='test-config', status='complete', value=dict(verdict='correct'))
    vr.atomic(path, record)
    vr.atomic(tmp_path / 'metrics.json', dict(positive_class='incorrect', files={str(path): vr.sha(path)},
                                             benchmark_sha256=vr.sha(tmp_path / 'bench_release.json')))
    vr.atomic(path, dict(record, value=dict(verdict='incorrect')))
    with pytest.raises(ValueError, match='Archived evidence changed'):
        curve.run(tmp_path)
    assert not (tmp_path / 'sample_size_stability.json').exists()
