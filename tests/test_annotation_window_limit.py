import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts import resume_corrected_annotations as recovery
from scripts.resume_corrected_annotations import known_window_limit


class WindowLimitTests(unittest.TestCase):
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
