import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import score_step_semantic as s


def case():
    return {'id': 'q', 'question': 'test', 'context': '',
            'student_solution': 'Wrong premise. Later wrong result.',
            'reference': {'errors': [{'root_id': 'E1', 'status': 'confirmed', 'quote': 'Wrong premise.'},
                                     {'root_id': 'E2', 'status': 'confirmed', 'quote': ''}]},
            'candidates': [{'candidate_id': 'C1', 'message': 'bad premise', 'quote': 'Later wrong result.'},
                           {'candidate_id': 'C2', 'message': 'same bad premise', 'quote': ''}]}


def judgment():
    return {'reference_checks': [{'root_id': e, 'verdict': 'accepted', 'reason': 'verified physics'} for e in ('E1', 'E2')],
            'judgments': [{'candidate_id': c, 'verdict': 'matched', 'compatible_root_ids': ['E1'],
                           'reason': 'same root at another statement', 'evidence_quote': 'Later wrong result.'}
                          for c in ('C1', 'C2')]}


def mapping():
    return {'q': {'a': {'items': [{'candidate_id': 'C1', 'published_index': 0},
                                 {'candidate_id': 'C2', 'published_index': 1}],
                        'status': 'completed', 'execution_failure': False},
                  'b': {'items': [{'candidate_id': 'C1', 'published_index': 0}],
                        'status': 'completed', 'execution_failure': False}}}


class SemanticScoringTests(unittest.TestCase):
    def test_server_evidence_relocation_requires_same_hash_and_containment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'raw.json').write_text('{}')
            sha = s.sha256_file(root / 'raw.json'); sources = {}
            s.freeze_mirrored_file('/server/run/raw.json', sha, Path('/server/run'), root, sources)
            self.assertEqual(sources[str((root / 'raw.json').resolve())], sha)
            with self.assertRaises(ValueError):
                s.freeze_mirrored_file('/server/other/raw.json', sha, Path('/server/run'), root, sources)
            with self.assertRaises(ValueError):
                s.freeze_mirrored_file('/server/run/raw.json', 'changed', Path('/server/run'), root, sources)

    def test_dependency_wait_refuses_dead_incomplete_label_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); labels = root / 'labels'; labels.mkdir(); out = root / 'judge'; out.mkdir()
            (labels / 'inputs.json').write_text('[{"id":"q"}]')
            with self.assertRaisesRegex(ValueError, 'stopped incomplete'):
                s.await_annotations(out, labels)
            for stage in s.teacher.STAGES:
                folder = labels / ('arm_' + stage) / 'q'; folder.mkdir(parents=True)
                (folder / 'result.json').write_text('{}')
            s.await_annotations(out, labels)  # readiness only; prepare still validates every raw record

    def test_duplicate_diagnoses_one_root_and_unlocated_reference_retained(self):
        result = s.calculate([case()], mapping(), {'q': judgment()}, ['a', 'b'])
        self.assertTrue(result['primary_metrics_ready'])
        self.assertEqual(result['scores']['a'], {'precision': .5, 'recall': .5, 'f1': .5})
        self.assertEqual(result['scores']['b']['precision'], 1)
        self.assertEqual(result['groups']['a']['reference'], 2)

    def test_maximum_matching_not_greedy_and_one_diagnosis_one_hit(self):
        self.assertEqual(s.maximum_matching({0: ['E1', 'E2'], 1: ['E1']}), [(0, 'E2'), (1, 'E1')])
        self.assertEqual(len(s.maximum_matching({0: ['E1', 'E2']})), 1)

    def test_wrong_physics_does_not_hit_even_when_same_quote(self):
        c, j = case(), judgment()
        j['judgments'] = [dict(x, verdict='non_error', compatible_root_ids=[], evidence_quote='Wrong premise.') for x in j['judgments']]
        result = s.calculate([c], mapping(), {'q': j}, ['a', 'b'])
        self.assertEqual(result['scores']['a'], {'precision': 0, 'recall': 0, 'f1': 0})

    def test_reference_gaps_and_uncertainty_block_final_scores(self):
        for verdict in ('reference_gap', 'unresolved'):
            with self.subTest(verdict=verdict):
                j = judgment(); j['judgments'][0].update(verdict=verdict, compatible_root_ids=[])
                result = s.calculate([case()], mapping(), {'q': j}, ['a', 'b'])
                self.assertIsNone(result['scores'])
                self.assertEqual(len(result['adjudication_queue']), 1)

    def test_reference_dispute_not_silently_removed_from_recall_denominator(self):
        j = judgment(); j['reference_checks'][1]['verdict'] = 'invalid'
        result = s.calculate([case()], mapping(), {'q': j}, ['a', 'b'])
        self.assertIsNone(result['scores'])
        self.assertEqual(result['groups']['a']['reference'], 2)

    def test_empty_outputs_need_no_matching_calls(self):
        c = case(); c['candidates'] = []
        m = mapping(); m['q']['a']['items'] = []; m['q']['b']['items'] = []
        result = s.calculate([c], m, {'q': s.empty_judgment(c)}, ['a', 'b'])
        self.assertIsNone(result['scores']['a']['precision'])
        self.assertEqual(result['scores']['a']['recall'], 0)

    def test_technical_failure_not_successful_empty(self):
        m = mapping(); m['q']['a']['execution_failure'] = True
        result = s.calculate([case()], m, {'q': judgment()}, ['a', 'b'])
        self.assertFalse(result['primary_metrics_ready'])
        self.assertEqual(result['groups']['a']['technical_failures'], 1)

    def test_candidate_coverage_and_evidence_validation(self):
        for mutation in ('missing', 'duplicate', 'fake_quote', 'unknown_root', 'unaccepted_root'):
            with self.subTest(mutation=mutation):
                j = judgment()
                if mutation == 'missing': j['judgments'].pop()
                elif mutation == 'duplicate': j['judgments'][1]['candidate_id'] = 'C1'
                elif mutation == 'fake_quote': j['judgments'][0]['evidence_quote'] = 'invented!'
                elif mutation == 'unknown_root': j['judgments'][0]['compatible_root_ids'] = ['E3']
                else: j['reference_checks'][0]['verdict'] = 'invalid'
                with self.assertRaises(ValueError): s.validate_judgment(j, case())

    def test_pooling_hides_method_but_retains_duplicate_denominators(self):
        rows = [{'id': 'q', 'question': 'p', 'context': '', 'prediction': 'solution'}]
        diag = {'message': 'claim', 'evidence': {'quote': 'solution'}, 'reference_ids': ['private-rule']}
        predictions = {'a': {'q': {'diagnostics': [diag, diag], 'status': 'completed'}},
                       'b': {'q': {'diagnostics': [diag], 'status': 'completed'}}}
        with patch.object(s.checks_module.base.runtime, 'healthy_result', return_value=True):
            cases, operator = s.blind_cases(rows, {'q': {'errors': []}}, predictions, ['a', 'b'])
        self.assertEqual(len(cases[0]['candidates']), 1)
        self.assertNotIn('private-rule', json.dumps(cases))
        self.assertEqual(len(operator['q']['a']['items']), 2)

    def test_wrong_model_effort_or_request_cannot_be_replayed(self):
        body = s.body_for(case(), 'match', None)
        request = {'model': s.teacher.MODEL, 'reasoning': {'effort': 'high'}, 'max_output_tokens': 16384,
                   'store': False, 'instructions': s.SYSTEM, 'input': s.canonical(body),
                   'text': {'format': {'type': 'json_object'}}, 'stream': True}
        record = {'endpoint': '/responses', 'request': request, 'response': {
            'id': 'r1', 'model': s.teacher.MODEL, 'reasoning': {'effort': 'high'},
            'status': 'completed', 'usage': {'input_tokens': 10, 'output_tokens': 10},
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(judgment())}]}]}}
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'request_001.json'
            p.write_text(json.dumps(record))
            self.assertEqual(s.judgment_from_raw(p, body, case()), judgment())
            for change in ('model', 'effort', 'body'):
                changed = copy.deepcopy(record)
                if change == 'model': changed['response']['model'] = 'different'
                elif change == 'effort': changed['response']['reasoning']['effort'] = 'low'
                else: changed['request']['input'] = '{}'
                p.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): s.judgment_from_raw(p, body, case())

    def test_completed_invalid_judgment_is_not_rerolled_on_resume(self):
        c = case(); body = s.body_for(c, 'match', None)
        request = {'model': s.teacher.MODEL, 'reasoning': {'effort': 'high'}, 'max_output_tokens': 16384,
                   'store': False, 'instructions': s.SYSTEM, 'input': s.canonical(body),
                   'text': {'format': {'type': 'json_object'}}, 'stream': True}
        record = {'endpoint': '/responses', 'request': request, 'started_unix': 1,
                  'response': {'id': 'r1', 'model': s.teacher.MODEL, 'reasoning': {'effort': 'high'},
                   'status': 'completed', 'usage': {'input_tokens': 10, 'output_tokens': 10},
                   'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{}'}]}]}}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); folder = out / 'arm_match' / 'q'; folder.mkdir(parents=True)
            (folder / 'request_001.json').write_text(json.dumps(record))
            (out / 'blind_inputs.json').write_text(json.dumps([c]))
            manifest = {'limits': {'requests': 12, 'input_tokens': 100000, 'output_tokens': 100000},
                        'platform_quota_floor': 100, 'tasks': 2}
            quota = {'unlimited_quota': False, 'total_available': 1000}
            with patch.object(s, 'verify', return_value=manifest), \
                 patch.object(s.teacher, 'await_startup_quota', return_value=quota), \
                 patch.object(s, 'Transport') as transport:
                with self.assertRaisesRegex(ValueError, 'Missing complete judgment'):
                    s.run(out, 'https://example.invalid', 'not-a-real-key')
                transport.assert_not_called()
            self.assertEqual(s.read(out / 'progress.json')['state'], 'stopped')
            self.assertFalse((folder / 'result.json').exists())
            self.assertFalse((out / 'runner.pid').exists())


if __name__ == '__main__': unittest.main()
