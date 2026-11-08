import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from scripts import run_final_answer_labels as label


class LabelTests(unittest.TestCase):
    def test_controls_do_not_leak_expected_answers(self):
        for row in label.CONTROLS:
            messages=label.protocol.label_messages(row,row['reference'])
            self.assertEqual(set(json.loads(messages[1]['content'])),
                {'question','student_answer','original_reference_answers'})

    def test_raw_response_replay_and_no_content_reroll(self):
        row=dict(id='a',question='Q',prediction='A')
        config=dict(auditor='test',max_tokens=16384,fixed_harmony_date='2026-11-07')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rawfile=root/'response.json';rawfile.write_text('{}')
            journal=SimpleNamespace(config=config,call=Mock(return_value=(
                {'usage':{}},{'file':'response.json','sha256':label.sha(rawfile)})))
            with patch.object(label,'validate_effective_input',return_value={}),patch.object(
                    label.corpus.harmony,'decode_response',return_value={'content':'invalid json'}):
                first=label.judge(root,journal,row,['answer'],None)
                self.assertEqual(first['state'],'content_failure')
                self.assertIsNone(first['judgment'])
                self.assertEqual(label.judge(root,journal,row,['answer'],None),first)
                self.assertEqual(journal.call.call_count,1)
                with self.assertRaises(ValueError):label.judge(root,journal,row,['changed'],None)

    def test_partial_reference_is_not_silently_stripped(self):
        with self.assertRaises(ValueError):
            label.protocol.label_messages({'question':'Q','prediction':'A'},['known',''])


if __name__=='__main__':unittest.main()
