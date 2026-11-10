import copy
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from scripts import resume_corrected_annotations as recovery
from scripts.resume_corrected_annotations import known_window_limit


class WindowLimitTests(unittest.TestCase):
    def test_extension_is_fixed_to_two_tasks_and_preserves_attempt_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'binding.json').write_text('{}')
            tasks = []
            for folder in ('arm_blind/cl_258_182363', 'arm_review/trainphysics_7865'):
                (out/folder).mkdir(parents=True)
                records = []
                for n in range(1, 7):
                    p = out/folder/f'request_{n:03d}.json'
                    p.write_text(json.dumps(dict(request={'model':'gpt-6.1-sol', 'stream':True},
                        http_status=200, stream_error={'code':'responses_startup_timeout', 'type':'timeout_error'})))
                    records.append(dict(path=str(p.relative_to(out)), sha256=recovery.runner.sha(p)))
                tasks.append(dict(folder=folder, original_limit=6, additional_attempts=6, records=records))
            receipt = dict(binding_sha256=recovery.runner.sha(out/'binding.json'),
                authorization='user_two_tasks_plus_six_2026_11_10', tasks=tasks)
            path = out/'attempt_extension.json'
            path.write_text(json.dumps(receipt))
            overrides = recovery.approved_attempt_extensions(out)
            policy = recovery.TaskAttemptPolicy(recovery.runtime.POLICY, overrides)
            policy.folder = tasks[0]['folder']
            self.assertEqual(policy['max_stream_attempts_per_task'], 12)
            self.assertFalse(11 >= policy['max_stream_attempts_per_task'])
            self.assertTrue(12 >= policy['max_stream_attempts_per_task'])
            p = out/tasks[0]['records'][0]['path']
            records = [(p, recovery.runner.read(p))] * 12
            self.assertEqual(recovery.runtime.provider_stream_attempts(records), 12)
            policy.folder = 'arm_final/cl_258_182363'
            self.assertEqual(policy['max_stream_attempts_per_task'], 6)
            # A restart does not add another six and historical failures cannot drift.
            self.assertEqual(recovery.approved_attempt_extensions(out), overrides)
            receipt['tasks'][0]['additional_attempts'] = 12
            path.write_text(json.dumps(receipt))
            with self.assertRaises((ValueError, RuntimeError)):
                recovery.approved_attempt_extensions(out)
            receipt['tasks'][0]['additional_attempts'] = 6
            path.write_text(json.dumps(receipt))
            p.write_text('{}')
            with self.assertRaises((ValueError, RuntimeError)):
                recovery.approved_attempt_extensions(out)

    def test_route_receipt_does_not_authorize_new_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'binding.json').write_text('{}')
            record = dict(http_status=429, started_unix=1, request={'model':'gpt-6.1-sol'},
                          response={'error':{'code':'model_not_found','type':'invalid_request_error',
                          'message':'unknown provider for model gpt-6.1-sol'}})
            (out / 'request.json').write_text(json.dumps(record))
            receipt = dict(binding_sha256=recovery.runner.sha(out/'binding.json'),
                authorization='user_requested_repair_and_resume_2026_11_09',
                records=[dict(path='request.json', sha256=recovery.runner.sha(out/'request.json'))])
            (out/'route_recovery.json').write_text(json.dumps(receipt))
            allowed = recovery.approved_route_records(out)
            self.assertIn(recovery.transport.digest(record), allowed)
            record['started_unix'] = 2
            self.assertNotIn(recovery.transport.digest(record), allowed)
            (out/'request.json').write_text(json.dumps(record))
            with self.assertRaises((ValueError, RuntimeError)):
                recovery.approved_route_records(out)

    def test_recovery_entry_locks_binds_and_restores_classifiers(self):
        original = recovery.runtime.failure_kind
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'binding.json').write_text('{}')
            with patch.object(recovery.runner, 'verify'), patch.object(recovery.runner, 'run') as run:
                recovery.run(out, out / 'unused.env')
                run.assert_called_once_with(out, out / 'unused.env')
            self.assertTrue((out / 'window_limit_recovery.json').exists())
        self.assertIs(recovery.runtime.failure_kind, original)

    def test_only_exact_provider_rejection_is_allowed(self):
        record = dict(http_status=429, response=dict(error=dict(
            code='account_rate_limit_exceeded', type='rate_limit_error',
            message='The selected upstream account has reached its rolling 60-second request limit.')))
        self.assertTrue(known_window_limit(record))
        for key, value in [('code', 'insufficient_quota'), ('code', 'rate_limit_exceeded'),
                           ('type', 'authentication_error'), ('message', 'Too many requests')]:
            changed = copy.deepcopy(record)
            changed['response']['error'][key] = value
            self.assertFalse(known_window_limit(changed))
        self.assertFalse(known_window_limit({}))
        record['http_status'] = 403
        self.assertFalse(known_window_limit(record))


if __name__ == '__main__':
    unittest.main()
