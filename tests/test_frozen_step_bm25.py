import unittest
from types import SimpleNamespace

from scripts.run_frozen_step_bm25 import unoffered_tool_output


def parsed(*messages, unfinished=''):
    return SimpleNamespace(current_content=unfinished, messages=[
        SimpleNamespace(channel=channel, recipient=recipient) for channel, recipient in messages])


class FrozenStepContentFailureTests(unittest.TestCase):
    def test_unoffered_tool_with_no_final_is_a_failed_output(self):
        self.assertTrue(unoffered_tool_output(parsed(('analysis', None), ('analysis', 'repo_browser.search'))))

    def test_does_not_discard_a_final_partial_reply_or_ambiguous_route(self):
        for response in (
            parsed(('final', None)),
            parsed(('analysis', 'repo_browser.search'), ('final', None)),
            parsed(('final', None), ('analysis', 'repo_browser.search')),
            parsed(('analysis', 'repo_browser.search'), unfinished='partial'),
            parsed(('final', '<|constrain|>diagnostics')),
            parsed(('analysis', 'functions.search<|channel|>analysis')),
            parsed(('analysis', 'tools.first'), ('analysis', 'tools.second')),
            parsed(),
        ):
            with self.subTest(response=response):
                self.assertFalse(unoffered_tool_output(response))
