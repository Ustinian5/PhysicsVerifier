"""Replay every published navigation request from the small library's own rule bundle on CPU."""
import argparse
from pathlib import Path
import tempfile

from scripts import run_verifier_library_scale as scale


vr, tree = scale.vr, scale.construction.tree


class RecordedResponses:
    def __init__(self, out, manifest, config):
        self.out = out
        self.manifest = manifest
        self.config = config
        self.seen = set()

    def call(self, stage, key, body):
        original = vr.read(self.out/'decoded'/stage/(vr.digest(key)+'.json'))
        ev = original['evidence']
        relative = Path(ev['file'])
        digest = scale.construction.c.corpus.p.digest
        scale.require(relative.parent == Path('raw')/stage/digest(key)[:24] and
                      self.manifest.get(str(relative)) == ev['sha256'] == vr.sha(self.out/relative),
                      'Navigation response is not bound to this job and release')
        saved = vr.read(self.out/relative)
        scale.require(saved['fingerprint']==digest([key,body,self.config]) and saved['request']==body and
                      saved['status']=='received' and saved['http_status']==200 and
                      saved['response']['model']==self.config['model'], 'Original navigation request differs')
        self.seen.add(str(relative))
        return saved['response'], ev


def audit(out):
    from transformers import AutoTokenizer

    c = scale.verify(out)['configuration']
    accepted = vr.read(out/'construction_acceptance.json')
    scale.require(accepted['passed'] and all(vr.sha(out/'materials'/n)==h for n,h in accepted['materials'].items()),
                  'Published library changed')
    # Read the deployed endpoint from its receipt, including after a resumed run.
    deployments = [vr.read(f) for f in (out/'attempts').glob('*/deployment.json')]
    ports = {d['port'] for d in deployments if d['configuration_sha256']==accepted['configuration_sha256']}
    scale.require(len(ports)==1, 'Published library has no unique matching service endpoint')
    config = dict(c['transport'], auditor_api=f'http://127.0.0.1:{ports.pop()}/v1')
    journal = RecordedResponses(out, accepted['response_files'], config)
    tokenizer = AutoTokenizer.from_pretrained(config['model'],local_files_only=True,trust_remote_code=False)
    bundle = vr.read(out/'tree_bundle.json')
    with tempfile.TemporaryDirectory(dir=out, prefix='.replay-') as temp:
        scratch = Path(temp)
        def call(stage,key,payload):
            name = 'tree_'+stage
            value, _ = scale.construction.model_json(scratch,journal,tokenizer,name,key,tree.messages(stage,payload))
            relative = Path('decoded')/name/(vr.digest(key)+'.json')
            scale.require(vr.read(scratch/relative)==vr.read(out/relative), 'Effective tokens or decoded output differ')
            return value
        progress = vr.Progress(scratch/'state.json', accepted['configuration_sha256'],tree.initial_state())
        organizer = tree.Organizer(bundle,progress,call)
        while organizer.state['phase']!='done':
            organizer.step()
        products = organizer.products()
    scale.require(journal.seen==set(accepted['response_files']) and products['catalog']==vr.read(out/'materials/catalog.json')
                  and {k:v for k,v in products.items() if k!='catalog'}==vr.read(out/'materials/tree_report.json'),
                  'Raw requests do not reproduce the published catalog and decisions')
    scale.immutable(out/'tree_replay.json',dict(passed=True, model_calls=0, reproduced_responses=len(journal.seen),
                    counts=products['counts'], acceptance_sha256=vr.sha(out/'construction_acceptance.json'),
                    code_sha256=vr.sha(Path(__file__))))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit(args.output.resolve())
