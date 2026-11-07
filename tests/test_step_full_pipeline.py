import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from scripts import run_step_full_pipeline as p


class PipelineTests(unittest.TestCase):
    def test_checker_continuation_requires_same_confirmed_coordinator(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);ready={'pid':321,'process_start':'birth','configuration_sha256':'pipeline','token':'once'}
            p.vr.atomic(out/'pipeline/disconnect_ready.json',ready)
            p.vr.atomic(out/'pipeline/disconnect_acceptance.json',dict(ready,passed=True))
            with patch.object(p,'verify',return_value={'full_configuration_sha256':'full'}), \
                 patch.object(p.os,'getpid',return_value=321), \
                 patch.object(p.full.base.runtime,'process_identity',return_value='birth'):
                p.stage_acceptance(out,'full')
                with self.assertRaises(ValueError):p.stage_acceptance(out,'other')
                p.vr.atomic(out/'pipeline/disconnect_acceptance.json',dict(ready,passed=True,token='old'))
                with self.assertRaises(ValueError):p.stage_acceptance(out,'full')

    def test_failed_generator_cannot_start_later_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'runner.exit_code').write_text('1')
            with patch.object(p,'alive',return_value=False),patch.object(p.full,'export') as export:
                with self.assertRaisesRegex(ValueError,'failure'):p.wait_answers(out,{'generator':{}})
                export.assert_not_called()

    def test_completed_answers_survive_later_checker_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'runner.exit_code').write_text('1')
            p.vr.atomic(out/'generation_summary.json',{'complete':100,'failures':[]})
            with patch.object(p,'alive',return_value=False),patch.object(p.full,'verify',return_value={}), \
                 patch.object(p.full,'export',return_value=[{}]*100) as export:
                p.wait_answers(out,{'generator':{}})
                export.assert_called_once()

    def test_batch_stop_propagates_to_api_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);target=out/'annotations';target.mkdir()
            p.link_stop(out,target)
            self.assertFalse((target/'STOP').exists())
            (out/'STOP').write_text('operator stop')
            self.assertEqual((target/'STOP').read_text(),'operator stop')

    def test_busy_gpu_is_not_taken_or_retried(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(p.subprocess,'check_output',return_value='20000'), \
             patch.object(p,'launch') as launch:
            with self.assertRaisesRegex(ValueError,'unavailable'):p.start_checker_service(Path(tmp))
            launch.assert_not_called()

    def test_recycled_service_pid_never_killed(self):
        with patch.object(p.full.base.runtime,'process_identity',return_value='new'),patch.object(p.os,'killpg') as kill:
            p.release_service({'pid':123,'start':'old'},'qwen')
            kill.assert_not_called()

    def test_pipeline_waits_labels_and_scores_only_after_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);target=out/'pipeline';target.mkdir()
            p.vr.atomic(target/'plan.json',{})
            calls=[]
            plan={'endpoint_sha256':p.vr.digest('https://example.invalid/v1'),
                  'python_executable':p.sys.executable,'runtime_pythonpath':'isolated',
                  'qwen_service':{'pid':1,'start':'one'}}
            labels=Mock(pid=99,returncode=0);labels.poll.return_value=0
            def scored(folder):
                calls.append('score');p.vr.atomic(folder/'metrics.json',{'primary_metrics_ready':False})
            with patch.dict(p.os.environ,{'OPENAI_BASE_URL':'https://example.invalid/v1','OPENAI_API_KEY':'test-only',
                                         'PYTHONPATH':'isolated'}), \
                 patch.object(p.Path,'exists',autospec=True,side_effect=lambda x: str(x)=='/proc' or x.is_file() or x.is_dir()), \
                 patch.object(p,'verify',return_value=plan), \
                 patch.object(p.full.base.runtime,'wait_detached'), \
                 patch.object(p,'wait_answers',side_effect=lambda *a:calls.append('answers')), \
                 patch.object(p,'start_labels',return_value=labels), \
                 patch.object(p,'release_service'), \
                 patch.object(p,'start_checker_service',return_value={'pid':2,'start':'two'}), \
                 patch.object(p.full,'run_checks',side_effect=lambda *a:calls.append('checks')), \
                 patch.object(p.semantic,'prepare',side_effect=lambda *a:calls.append('blind_prepare')), \
                 patch.object(p,'link_stop'), \
                 patch.object(p.semantic,'run',side_effect=lambda *a:calls.append('semantic')), \
                 patch.object(p.semantic,'score',side_effect=scored):
                p.run(out)
            self.assertEqual(calls,['answers','checks','blind_prepare','semantic','score'])
            self.assertEqual(p.vr.read(target/'execution_status.json')['state'],'completed_needs_adjudication')


if __name__=='__main__':unittest.main()
