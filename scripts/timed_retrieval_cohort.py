"""Bind the 50-question timing trial to the released 30B inputs and BM25 controls."""
import argparse
from collections import Counter
from itertools import cycle
from pathlib import Path

from scripts import run_reviewed_bm25 as original

vr = original.vr
require = original.require
RESULT = Path('retrieval_fusion_dev/timed50')
BENCH = Path('aligned_old99/final_revision_balanced')
BM25 = Path('library_source_rerun_20261124/bm25_n2000')
FAMILIES = Path('corrected/source_screen/family_components.json')
FAMILIES_SHA = '57d4296ea30bcfef3d37e4ca176e50a7d1b69fd1b7876298efb43890fe04feff'
CURRENT = '32ca1b1da11186e5f4c76ee1a101caa64dd6290e864bc5085be143cebd49094a'
SOURCES = '2558df2ba5685ed607d5bec6492f67a4c4811b90a4fe369e53301772c573d966'
IDS = '084801740b876ac09b3499d847722ec3f73f17ce8c564b5f8d82c290a02c6ba8'
ARMS = ('case_rag', 'flat_skills', 'rule_tree')


def freeze(root):
    bench = root / BENCH
    out = root / RESULT
    release = vr.read(bench / 'bench_release.json')
    require(vr.sha(bench / 'bench_release.json') == original.BENCH
            and vr.sha(bench / 'inputs.json') == release['inputs_sha256']
            and vr.sha(bench / 'labels_private.json') == release['labels_sha256']
            and vr.sha(bench / 'dispositions.json') == release['dispositions_sha256']
            and vr.sha(root / FAMILIES) == FAMILIES_SHA, 'Frozen answering or family release changed')
    rows = vr.read(bench / 'inputs.json')
    labels = vr.read(bench / 'labels_private.json')
    require(len(rows) == len(labels) == 474 and Counter(x['verdict'] for x in labels.values())
            == {'incorrect': 237, 'correct': 237}, 'Wrong formal benchmark')
    family = {q: group['id'] for group in vr.read(root / FAMILIES)['components'] for q in group['members']}
    eligible = [row for row in rows if row['id'] not in original.sources.FINAL_HOLD]
    require(len(eligible) == 468, 'Changed formal exclusions')
    order = sorted(eligible, key=lambda row: vr.digest(['timed-semantic-50', 20261124, row['id']]))
    streams = {v: iter([row for row in order if labels[row['id']]['verdict'] == v])
               for v in ('incorrect', 'correct')}
    chosen, used_families = [], set()
    for label in cycle(streams):
        row = next(row for row in streams[label] if family[row['id']] not in used_families)
        chosen.append(row)
        used_families.add(family[row['id']])
        if len(chosen) == 50:
            break
    ids = [row['id'] for row in chosen]
    require(vr.digest(ids) == IDS and len(used_families) == 50, 'Changed nested question prefixes')
    require(all(set(row) == {'id','context','question','prediction'} for row in chosen), 'Unexpected model-visible field')
    original.final.immutable(out / 'inputs.json', chosen)
    original.final.immutable(out / 'cohort.json', dict(
        population=474, eligible=468, seed=20261124, unique_families=len(used_families), n=50,
        classes={'incorrect':25, 'correct':25}, prefixes=[5,10,50], positive='incorrect', human_gold=False,
        ids=ids, order='Hash-order within label; alternate starting with incorrect; skip previously selected families.',
        inputs_sha256=vr.sha(out/'inputs.json'), parent_inputs_sha256=release['inputs_sha256'],
        labels_sha256=release['labels_sha256'], family_registry_sha256=FAMILIES_SHA))
    return chosen


def baseline(root):
    rows = freeze(root)
    current = root / BM25 / 'final'
    require(vr.sha(current / 'metrics.json') == CURRENT and vr.sha(root/BM25/'sources.json') == SOURCES,
            'Wrong published BM25 controls or materials')
    receipt = vr.read(current/'metrics.json')
    require(receipt['common_questions'] == 468 and receipt['positive_class'] == 'incorrect', 'Wrong score convention')
    labels = vr.read(root/BENCH/'labels_private.json')
    truth = {row['id']: labels[row['id']]['verdict'] for row in rows}
    by_arm, scores = {}, {}
    for arm in original.ARMS:
        results = {}
        for q in truth:
            path = current/'formal'/arm/vr.digest(q)[:24]/'result.json'
            require(vr.sha(path) == receipt['files'][str(path)], 'Unbound BM25 control')
            data = vr.read(path)
            require(data['arm'] == arm and data['id'] == q and
                    data['configuration_sha256'] == receipt['configuration_sha256'], 'Mismatched control')
            if data['source'] == 'frozen_baseline':
                require(vr.sha(data['original_result']) == data['original_sha256'], 'Parent control drift')
            results[q] = dict(status=data['status'], value=data['value'], delivered=data['provided'],
                              file_sha256=vr.sha(path), source=data['source'])
        by_arm[arm] = results
        scores[arm] = original.final.measure(truth, results)
    report = dict(groups=scores, rows=by_arm, source_sha256=CURRENT, cohort=IDS, human_gold=False)
    original.final.immutable(root/RESULT/'bm25_baseline.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','baseline'))
    parser.add_argument('--root', type=Path, default=original.RESULTS)
    args = parser.parse_args()
    (freeze if args.action=='freeze' else baseline)(args.root)
