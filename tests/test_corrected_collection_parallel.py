import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, MagicMock
from scripts import run_corrected_collection_parallel as runner


class ParallelTests(unittest.TestCase):
    def exercise(self, fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);tasks=[{'role':'build' if i%2==0 else 'heldout','row':{'id':str(i),'question':'q'}} for i in range(12)]
            runner.base.write(out/'tasks.json',tasks)
            config={'configuration_sha256':'frozen','configuration':{'generator_config':{'generator':'model','generator_api':'http://unused'}}}
            lock=threading.Lock();active=0;peak=0;calls=[];barrier=threading.Barrier(4)
            def generate(journal,row):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active);calls.append(row['id'])
                if int(row['id'])<4:barrier.wait(timeout=3)
                time.sleep(.03 if row['id']=='0' else .06)
                with lock:active-=1
                if fail and row['id']=='0':raise RuntimeError('fixture failure')
                return {'state':'complete','prediction':'answer '+row['id']}
            client=MagicMock();client.__enter__.return_value.get.return_value.json.return_value={'data':[{'id':'model','max_model_len':32768,'root':'/slow_share2/huhengyuan/models/Qwen3-30B-A3B-Instruct-2507'}]}
            with patch.object(runner,'Path',side_effect=lambda v:Path(tmp) if v=='/proc' else Path(v)),patch.object(runner.base,'verify',return_value=config),patch.object(runner.base.corpus.runtime,'wait_detached'),patch.object(runner.base.corpus,'generate',side_effect=generate),patch.object(runner.httpx,'Client',return_value=client):
                if fail:
                    with self.assertRaises(ValueError):runner.run(out,4,8)
                else:runner.run(out,4,8)
            self.assertEqual(peak,4);self.assertEqual(active,0)
            self.assertFalse((out/'runner.pid').exists())
            if fail:
                self.assertEqual(set(calls),{'0','1','2','3'})
                self.assertEqual(json.loads((out/'execution_status.json').read_text())['state'],'stopped')
            else:
                self.assertEqual(set(calls),set(map(str,range(8))))
                self.assertEqual(json.loads((out/'execution_status.json').read_text())['state'],'scheduling_pilot_complete')
                self.assertEqual([r['id'] for r in json.loads((out/'build_inputs.json').read_text())],['0','2','4','6'])
    def test_concurrency_limit_population_and_order(self):self.exercise()
    def test_failure_drains_without_more_dispatch(self):self.exercise(True)

if __name__=='__main__':unittest.main()
