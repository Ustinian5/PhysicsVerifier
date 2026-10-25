"""Question-only, immutable-source preparation for the cumulative 5,000 cohort.

No inference here. TextbookReasoning's answers/references never enter the export.
New-source selection precedes generation; outcomes cannot change the cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from rule_framework import oss120b_construction as p
from scripts import build_120b_rules as b

DATASET = 'MegaScience/TextbookReasoning'
REVISION = 'ca7ecbec76d01bff2e99f3dc17735b02f87d4e96'
PARQUET_SHA256 = 'f27c71763b21db22cb58c75a8495decb2fe5e2a83efc16f4b6c004fda606e543'
SOURCE_URL = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/train-00000-of-00001.parquet'
SEED = 20261025
MISSING_CONTEXT = re.compile(
    r'\b(?:fig(?:ure)?\.?|diagram|table)\s*(?:\d|\([a-z0-9])|'
    r'\b(?:shown|depicted|illustrated)\s+(?:in|below|above)|'
    r'\b(?:previous|preceding|following)\s+(?:problem|exercise|figure|diagram|table)|'
    r'\b(?:eq(?:uation)?\.?)\s*\(?\d+\.\d+|'
    r'!\[[^\]]*\]\(|<img|\\includegraphics|\[IMAGE', re.I)


class QuestionIndex:
    """Exact existing lexical thresholds, indexed without embedding calls.

    Embedded checks use question text only during preparation. Generated text
    is checked with the original full-field isolation before construction.
    """
    def __init__(self, rows=()):
        self.refs = []; self.postings = defaultdict(set)
        self.keys = defaultdict(set); self.templates = defaultdict(set)
        for row in rows:
            self.add(row)

    def add(self, row):
        key, template = p.question_key(row['question']), p.question_template(row['question'])
        grams = p.shingles(row['question']); i = len(self.refs)
        self.refs.append((row['id'], key, template, grams))
        self.keys[key].add(i); self.templates[template].add(i)
        for gram in grams:
            self.postings[gram].add(i)

    def matches(self, question):
        key, template = p.question_key(question), p.question_template(question)
        grams = p.shingles(question); common = Counter()
        for gram in grams:
            common.update(self.postings.get(gram, ()))
        found = {}
        for i in self.keys.get(key, ()):
            found[i] = 'normalized_exact'
        for i in self.templates.get(template, ()):
            found.setdefault(i, 'numeric_template')
        for i, count in common.items():
            rg = self.refs[i][3]
            if min(len(grams), len(rg)) >= 40 and (
                count / max(1, len(grams) + len(rg) - count) >= .5 or
                count / max(1, min(len(grams), len(rg))) >= .8):
                found.setdefault(i, 'lexical_family')
        for i, (_, rk, rt, _) in enumerate(self.refs):
            if (len(rk) >= 80 and rk in key) or (len(rt) >= 80 and rt in template):
                found.setdefault(i, 'embedded_benchmark_question')
        return [{'reference_id': self.refs[i][0], 'reason': reason} for i, reason in sorted(found.items())]


def export_source(parquet, output):
    import pyarrow.parquet as pq  # Optional preprocessing dependency, not model runtime.
    p.require(b.r.sha256_file(parquet) == PARQUET_SHA256, 'Unexpected upstream parquet bytes')
    output.mkdir(parents=True, exist_ok=True)
    rows, counts = [], Counter(); index = 0
    for batch in pq.ParquetFile(parquet).iter_batches(columns=['question', 'subject']):
        for raw in batch.to_pylist():
            row_index = index; index += 1; counts['all_train_rows'] += 1
            if raw['subject'] != 'physics':
                continue
            counts['physics_rows'] += 1
            q = raw['question']
            if not isinstance(q, str) or not q.strip():
                counts['empty_question'] += 1; continue
            if '<|' in q or len(q) > 24000:
                counts['control_or_length'] += 1; continue
            if MISSING_CONTEXT.search(q):
                counts['explicit_missing_context_risk'] += 1; continue
            rows.append({'id': f'tbr_{row_index:06d}', 'question': q})
    manifest = {'dataset': DATASET, 'revision': REVISION, 'source_url': SOURCE_URL,
        'split': 'train', 'subject': 'physics', 'license': 'CC-BY-NC-SA-4.0',
        'source_parquet_sha256': PARQUET_SHA256, 'source_parquet_bytes': 532362586,
        'export_fields': ['id', 'question'], 'answers_or_references_exported': False,
        'upstream_curation': 'Textbook questions extracted with Llama3.3-70B and refined with DeepSeek-V3; not authored by this project',
        'upstream_row_index_in_id': True, 'counts': dict(counts), 'exported': len(rows),
        'question_pool_sha256': p.digest(rows), 'all_missing_context_detected': False,
        'source_card': f'https://huggingface.co/datasets/{DATASET}/blob/{REVISION}/README.md'}
    b.write_once(output/'question_pool.json', rows)
    b.write_once(output/'source_manifest.json', manifest)
    return manifest


def select_new(pool, heldout, previous_questions, *, limit, seed=SEED):
    p.require(limit > 0 and heldout and previous_questions, 'Missing target or isolation cohorts')
    p.require(len({x['id'] for x in pool}) == len(pool), 'Duplicate source IDs')
    bench = QuestionIndex(heldout); seen = QuestionIndex(previous_questions)
    kept, rejected, counts = [], [], Counter()
    for row in sorted(pool, key=lambda x: p.digest([seed, x['id']])):
        p.require(set(row) == {'id', 'question'}, 'Only id/question may leave source preparation')
        p.text_fields(row, ('id', 'question'))
        counts['examined'] += 1
        if MISSING_CONTEXT.search(row['question']) or '<|' in row['question'] or len(row['question']) > 24000:
            rejected.append({'id': row['id'], 'reason': 'unsupported_input'}); counts['unsupported_input'] += 1
            continue
        matches = bench.matches(row['question'])
        reason = 'benchmark_overlap'
        if not matches:
            matches = seen.matches(row['question']); reason = 'previous_or_new_source_duplicate'
            # Source-source containment is symmetric, unlike embedded bench screening.
            if not matches:
                k, t = p.question_key(row['question']), p.question_template(row['question'])
                matches = [{'reference_id': sid, 'reason': 'reverse_embedded_source'} for sid, rk, rt, _ in seen.refs
                           if (len(k) >= 80 and k in rk) or (len(t) >= 80 and t in rt)]
        if matches:
            rejected.append({'id': row['id'], 'reason': reason, 'matches': matches}); counts[reason] += 1
            continue
        kept.append(row); seen.add(row)
        if len(kept) == limit:
            break
    p.require(len(kept) == limit, f'Insufficient new independent candidates: {len(kept)}/{limit}; no fallback sampling')
    return kept, {'counts': dict(counts), 'selected': len(kept), 'excluded': rejected,
        'order': 'SHA256(seed, original row ID); fixed before model outcomes',
        'policy': p.ISOLATION_POLICY, 'cross_source_near_duplicate_screen': True,
        'complete_semantic_independence_proven': False}


def prepare(source, base_corpus, parent, output, target=5000, reserve=4000):
    p.require(target == 5000, 'This experiment is frozen to a cumulative target of 5000')
    manifest = b.read(source/'source_manifest.json'); pool = b.read(source/'question_pool.json')
    p.require(manifest['dataset'] == DATASET and manifest['revision'] == REVISION and
              manifest['split'] == 'train' and manifest['question_pool_sha256'] == p.digest(pool) and
              manifest['answers_or_references_exported'] is False, 'Unverified source export')
    baseline = b.read(base_corpus/'answers.json'); generation_release = b.read(base_corpus/'release_manifest.json')
    p.require(b.r.sha256_file(base_corpus/'answers.json') == generation_release['files']['answers.json'], 'Changed base answers')
    p.require(len(baseline) == generation_release['counts']['generation']['complete'], 'Base cohort shortened')
    previous = b.read(base_corpus/'inputs.json')
    p.require(b.r.sha256_file(base_corpus/'inputs.json') == generation_release['files']['inputs.json'], 'Changed base attempted questions')
    plan = b.read(parent/'plan.json'); c = plan['configuration']
    p.require(c['protocol'] == 'experience_first_120b_v5_candidate_collection' and
              p.digest(c) == plan['configuration_sha256'], 'Unexpected base construction protocol')
    parent_rows = b.read(parent/'inputs.json')
    p.require(p.digest(parent_rows) == c['selected_sha256'] and
              {x['id']: x for x in parent_rows} == {x['id']: x for x in baseline}, 'Parent is not the full base cohort')
    heldout, cohorts = b.benchmark_questions()
    needed = target-len(baseline)
    p.require(0 < needed <= reserve, 'Invalid expansion size/reserve')
    selected, isolation = select_new(pool, heldout, previous, limit=reserve)
    source_files = {str(path.resolve()): b.r.sha256_file(path) for path in [source/'source_manifest.json', source/'question_pool.json']}
    bindings = {str(path.resolve()): b.r.sha256_file(path) for path in [
        base_corpus/'release_manifest.json', base_corpus/'answers.json', base_corpus/'inputs.json',
        parent/'plan.json', parent/'inputs.json', parent/'topic_skeleton.json', parent/'source_release.json']}
    selection = {'target_questions': target, 'baseline_complete_questions': len(baseline), 'new_questions_needed': needed,
        'reserve_questions': reserve, 'seed': SEED, 'base_corpus': str(base_corpus.resolve()),
        'parent_build': str(parent.resolve()), 'source_files': source_files, 'base_bindings': bindings,
        'benchmark_files': cohorts, 'bench_sha256': p.digest(heldout), 'reserve_sha256': p.digest(selected),
        'isolation_sha256': p.digest(isolation), 'human_gold': False,
        'selection_after_domain_only': True, 'no_replacement_after_generation': True,
        'comparison_scope': 'size plus new training source; not a pure same-distribution size ablation'}
    for name, value in [('reserve_questions', selected), ('bench_questions', heldout),
                        ('isolation', isolation), ('selection', selection)]:
        b.write_once(output/(name+'.json'), value)
    return {k: selection[k] for k in ('target_questions','baseline_complete_questions','new_questions_needed','reserve_questions')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    exp = sub.add_parser('export-source'); exp.add_argument('--parquet',type=Path,required=True); exp.add_argument('--output',type=Path,required=True)
    prep = sub.add_parser('prepare')
    for name in ('source','base-corpus','parent','output'):
        prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--target',type=int,default=5000);prep.add_argument('--reserve',type=int,default=4000)
    a = parser.parse_args()
    result = export_source(a.parquet,a.output) if a.mode=='export-source' else prepare(a.source,a.base_corpus,a.parent,a.output,a.target,a.reserve)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
