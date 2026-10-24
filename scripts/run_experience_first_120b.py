"""Frozen, resumable server execution of the original per-answer construction flow.

prepare verifies the complete released 30B cohort, run first pauses for review of
12 deterministically selected real cases, then replays them into the full build.
No old judgments, leaves, references, external models or embeddings are used.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import httpx

from scripts import build_120b_rules as b
from scripts import unified_rules_pipeline as flow
from scripts import generate_experience_rules as extraction
from rule_framework import released_corpus, oss120b_construction as p
r = b.r
ROOT = b.ROOT
VERSION = extraction.EXPERIENCE_120B_VERSION
SOURCES = sorted(set(b.SOURCES + [__file__.replace(str(ROOT) + '/', ''),
    'scripts/unified_rules_pipeline.py', 'scripts/generate_experience_rules.py',
    'scripts/build_unified_catalog.py', 'scripts/analyze_semantic_experience_run.py',
    'scripts/prepare_rules_for_cluster.py', 'scripts/generate_cluster_proposals.py',
    'scripts/refine_cluster_blueprints.py', 'scripts/analyze_rule_embedding_clusters.py',
    'scripts/evaluate_unified_rules_quality.py', 'scripts/check_server_run_inputs.py']))


def hashes():
    return {s: r.sha256_file(ROOT / s) for s in SOURCES}


def prepare(output, release):
    binding, rows = released_corpus.load(release, release.parent / 'answers.json',
                                        full=True, complete_answers=True)
    heldout, cohorts = b.benchmark_questions()
    eligible, excluded = p.isolate_sources(rows, heldout)
    # New overlap is a protocol decision; do not silently discard source rows.
    p.require(not excluded, 'New overlap in complete answers; mainline must review before freezing')
    templates = [p.question_template(x['question']) for x in eligible]
    p.require(len(set(templates)) == len(templates), 'Duplicate question templates')
    rows = sorted(eligible, key=lambda x: p.digest([20261024, x['id']]))
    skeleton = flow.experience_topic_skeleton(b.read(ROOT / 'catalogs/rules_catalog_top_down.json'))
    config = dict(protocol=VERSION, model=r.EXPECTED_MODEL, model_files=b.existing.MODEL_FILES,
        api_base='http://127.0.0.1:18083/v1', reasoning_effort='medium', max_tokens=16384,
        max_request_input_bytes=400000, attempts_per_job=3, batch_size=12, max_rules_per_sample=2,
        max_requests=3 * (len(rows) + 2 * len(rows) + 2 * len(skeleton['domains']) + 24),
        distill_max_requests=0, pilot_size=min(12, len(rows)), seed=20261024,
        corpus_release=str(release.resolve()), corpus_release_sha256=r.sha256_file(release),
        source_binding_sha256=p.digest(binding), selected_sha256=p.digest(rows),
        benchmark_files=cohorts, bench_questions_sha256=p.digest(heldout),
        topic_skeleton_sha256=p.digest(skeleton), source_hashes=hashes(),
        isolation=p.ISOLATION_POLICY, old_rules_imported=False, old_judgments_imported=False,
        external_teacher=None, embedding_model=None, human_gold=False,
        json_policy='literal_invalid_backslashes_only_with_evidence_no_content_repair',
        source_prediction_model=binding['generator'])
    output.mkdir(parents=True, exist_ok=True)
    with r.run_lock(output / 'active.lock'):
        for name,value in [('inputs',rows),('bench_questions',heldout),('topic_skeleton',skeleton),
            ('source_release',binding),('isolation',{'policy':p.ISOLATION_POLICY,
             'source_count':len(rows),'excluded':excluded,'selected_overlap_count':0})]:
            b.write_once(output / (name+'.json'),value)
        b.write_once(output/'plan.json',{'configuration':config,'configuration_sha256':p.digest(config)})
    print(json.dumps({'state':'prepared','samples':len(rows),'pilot':config['pilot_size'],'requests_cap':config['max_requests']}),flush=True)


def verify(output, source_check=False):
    plan=b.read(output/'plan.json');c=plan['configuration']
    p.require(p.digest(c)==plan['configuration_sha256'], 'Plan changed')
    p.require(c['protocol']==VERSION and c['source_hashes']==hashes(), 'Frozen implementation changed')
    p.require(c['model']==r.EXPECTED_MODEL and c['reasoning_effort']=='medium' and
              c['api_base']=='http://127.0.0.1:18083/v1', 'Model/service changed')
    for name,key in [('inputs','selected_sha256'),('bench_questions','bench_questions_sha256'),
                     ('topic_skeleton','topic_skeleton_sha256'),('source_release','source_binding_sha256')]:
        p.require(p.digest(b.read(output/(name+'.json')))==c[key],name+' changed')
    heldout,cohorts=b.benchmark_questions()
    p.require(cohorts==c['benchmark_files'] and p.digest(heldout)==c['bench_questions_sha256'],'Benchmark registry changed')
    release=Path(c['corpus_release'])
    p.require(r.sha256_file(release)==c['corpus_release_sha256'],'Corpus release changed')
    if source_check:
        binding,rows=released_corpus.load(release,release.parent/'answers.json',complete_answers=True)
        p.require(p.digest(binding)==c['source_binding_sha256'],'Generation provenance changed')
        p.require(sorted(rows,key=lambda x:p.digest([c['seed'],x['id']]))==b.read(output/'inputs.json'),'Source cohort changed')
        _,excluded=p.isolate_sources(rows,heldout)
        p.require(not excluded,'Benchmark overlap')
    return plan


class Journal(b.Journal):
    def supported_stages(self):
        return {'taxonomy','extract_experience','cluster_rules'}

    def messages(self, stage, payload):
        return flow.experience_120b_messages(stage,payload)

    def decode_record(self,path,stage):
        record=b.read(path);raw=record.get('response') or {}
        p.require(not record.get('error') and raw.get('model')==self.config['model'] and raw.get('id'),
                  'Failed/unknown model response')
        text=b.harmony.decode_response(raw,model=self.config['model'],
            max_tokens=self.config['max_tokens'],native_tools=None)['content'].strip()
        if text.startswith('```json\n') and text.endswith('```'): text=text[8:-3].strip()
        value,normalized,positions=b.parse_json_escapes(text,repair=True)
        p.require(isinstance(value,dict),'Expected JSON object')
        if positions:
            b.write_once(path.with_name('json_escape_view_'+path.stem+'.json'),{
                'request_sha256':r.sha256_file(path),'policy':self.config['json_policy'],
                'literal_backslash_positions':positions,'value_sha256':p.digest(value),
                'normalized_sha256':p.digest(normalized),'physical_content_edited':False})
        return value


def accept_pilot(output):
    verify(output)
    summary=output/'pilot'/'counts.json';accept=output/'pilot'/'acceptance.json'
    p.require(summary.exists() and accept.exists(),'Pilot needs Codex assistant content review')
    receipt=b.read(accept)
    p.require(receipt.get('actor')=='Codex assistant' and receipt.get('passed') is True and
              receipt.get('counts_sha256')==r.sha256_file(summary) and
              receipt.get('experience_sha256')==r.sha256_file(output/'pilot'/'semantic_experience.json'),
              'Pilot review does not bind current evidence')


def run(output, phase):
    plan=verify(output);c=plan['configuration']
    p.require(sys.platform=='linux' and 'huhengyuan' in Path(os.getenv('CONDA_PREFIX','')).parts,
              'Use isolated personal server environment')
    p.require('huhengyuan' in output.resolve().parts,'Personal persistent output required')
    for name in ('NO_PROXY','no_proxy'):os.environ[name]='127.0.0.1,localhost'
    with r.run_lock(output/'active.lock'):
        if (output/'release_manifest.json').exists():
            for name,sha in b.read(output/'release_manifest.json')['files'].items():
                p.require(r.sha256_file(output/name)==sha,'Published evidence changed')
            print('Already completed; zero new requests.');return
        if phase=='full':accept_pilot(output)
        (output/'runner.pid').write_text(str(os.getpid()))
        try:
            with httpx.Client(trust_env=False) as client:
                reply=client.get(c['api_base']+'/models',timeout=10);reply.raise_for_status()
                p.require(any(x['id']==c['model'] and x.get('max_model_len',0)>=131072 for x in reply.json()['data']),
                          'Model identity/context mismatch')
            p.require({n:r.sha256_file(Path(c['model'])/n) for n in c['model_files']}==c['model_files'],
                      'Checkpoint configuration changed')
            b.wait_detached(output,plan['configuration_sha256'])
            b.write_once(output/'runtime.json',{'python':sys.version,'model_files':c['model_files'],
                'packages':{n:r.package_version(n) for n in ('vllm','transformers','torch','httpx')},
                'checkpoint_validation':'configuration/tokenizer/template/index hashes; not all weight shards'})
            journal=Journal(output,c,True)
            pilot_journal=Journal(output/'pilot',c,True)
            journal.budget = b.BatchBudget(output, journal.budget.limits, prior_runs=(output/'pilot',), stop_file=output/'STOP')
            pilot_journal.budget = journal.budget
            def call(stage,key,payload):
                selected=pilot_journal if phase=='pilot' and stage=='cluster_rules' else journal
                # Only documented transient failures may be retried, within the
                # same frozen per-job cap. Semantic/identity faults stop here.
                while True:
                    try:return selected(stage,key,payload)
                    except Exception:
                        if (output/'STOP').exists():
                            raise
                        folder=selected.output/('arm_'+stage)/p.digest(key)[:24]
                        paths=sorted(folder.glob('request_*.json'))
                        if not paths or len(paths)>=c['attempts_per_job'] or not b.retryable_transport_failure(b.read(paths[-1])):raise
                        last=b.read(paths[-1]);deadline=max(b.retry_after_deadline(last),
                            last.get('started_unix',0)+last.get('elapsed_seconds',0)+300)
                        while time.time()<deadline:
                            p.require(not (output/'STOP').exists(),'STOP requested')
                            r.write_json(output/'execution_status.json',{'state':'waiting_transient','stage':stage,'key':key,
                                'ready_unix':deadline,'updated_unix':time.time()})
                            time.sleep(min(20,max(.1,deadline-time.time())))
            rows=b.read(output/'inputs.json')
            if phase=='pilot':rows=rows[:c['pilot_size']]
            built=flow.build_120b_experience_catalog(rows,b.read(output/'bench_questions.json'),call,
                topic_skeleton=b.read(output/'topic_skeleton.json'),max_rules_per_sample=c['max_rules_per_sample'],batch_size=c['batch_size'])
            verify(output,source_check=phase == "full")
            target=output/'pilot' if phase=='pilot' else output
            for name,value in built.items():b.write_once(target/(name+'.json'),value)
            if phase=='pilot':
                r.write_json(output/'execution_status.json',{'state':'pilot_awaiting_content_review','counts':built['counts'],'updated_unix':time.time()})
                return
            paths=[x for x in output.rglob('*.json') if x.name not in {'execution_status.json','release_manifest.json','disconnect_ready.json','disconnect_acceptance.json'}]
            b.write_once(output/'release_manifest.json',{'protocol':VERSION,'configuration_sha256':plan['configuration_sha256'],
                'status':'candidate_catalog_complete_not_default_release','human_gold':False,
                'usable_for_runtime':built['counts']['rules']>0,'independent_test_ready':False,
                'files':{str(x.relative_to(output)):r.sha256_file(x) for x in sorted(paths)},'counts':built['counts'],
                'data_isolation':{'passed':True,'selected_overlap_count':0,'complete_semantic_family_isolation_proven':False}})
            r.write_json(output/'execution_status.json',{'state':'completed','counts':built['counts'],'updated_unix':time.time()})
        except BaseException as exc:
            r.write_json(output/'execution_status.json',{'state':'stopped','phase':phase,'error_type':type(exc).__name__,
                'error':str(exc)[:1000],'updated_unix':time.time()});raise
        finally:(output/'runner.pid').unlink(missing_ok=True)


def confirm(output):
    plan=verify(output);ready=b.read(output/'disconnect_ready.json');status=b.read(output/'execution_status.json')
    p.require(b.existing.process_identity(ready['pid'])==ready['process_start'],'PID changed')
    p.require(time.time()-ready['unix']>=5 and status.get('state')=='waiting_disconnect_acceptance' and
              status['token']==ready['token'] and ready['configuration_sha256']==plan['configuration_sha256'],'Not waiting for this reconnect')
    r.write_json(output/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify','run','confirm-detached'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corpus-release',type=Path)
    parser.add_argument('--phase',choices=['pilot','full'],default='pilot')
    args=parser.parse_args();output=args.output.resolve()
    if args.mode=='prepare':
        p.require(args.corpus_release is not None,'Provide corpus release')
        prepare(output,args.corpus_release.resolve())
    elif args.mode=='verify':print(json.dumps(verify(output,source_check=True)['configuration_sha256']))
    elif args.mode=='confirm-detached':confirm(output)
    else:run(output,args.phase)


if __name__=='__main__':main()
