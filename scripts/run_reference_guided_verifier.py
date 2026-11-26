"""Run the frozen reference-guided construction and blind evaluations on node14."""
import argparse
from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path
import time

import httpx
import tempfile

from scripts import reuse_step_baseline as baseline
from scripts import reference_review_pool as pool
from scripts import run_aligned_step_evaluation as step
from scripts import run_corrected_extraction as extract
from scripts import run_corrected_pipeline as materials
from scripts import run_final_answer_revision as final
from scripts import serve_frozen_bm25 as node
from scripts import verifier_build_scope as scope

vr, require = final.vr, final.require
RESULTS = node.host.HOT / 'results/qwen30b_step_verifier'
WORK = RESULTS / 'english_blind_verifier'


def configuration(policy, gpus):
    require(gpus and len(gpus) == len(set(gpus)) and set(gpus) <= {4, 5, 6, 7}, 'Invalid review GPU list')
    names = ('scripts/run_reference_guided_verifier.py', 'scripts/verifier_build_scope.py', 'scripts/reference_review_pool.py',
             'scripts/run_corrected_extraction.py', 'scripts/run_corrected_pipeline.py',
             'scripts/reuse_step_baseline.py', 'scripts/repair_step_judge_locations.py',
             'scripts/run_aligned_step_evaluation.py', 'scripts/run_step_match_judge.py',
             'scripts/serve_frozen_bm25.py', 'scripts/run_final_answer_revision.py',
             'rule_framework/build_review_prompt.txt')
    if policy == 'lines_once':
        names += ('rule_framework/step_match_lines.py', 'scripts/run_step_line_judge.py', 'scripts/step_judge_handoff.py')
    return dict(source_scope_sha256=vr.sha(scope.SCOPE_FILE), source_audit_sha256=vr.sha(WORK/'build_scope_audit.json'),
                judge_location_policy=policy, review_gpus=list(gpus),
                pure_metrics_sha256=baseline.PURE_METRICS, model=final.base.r.EXPECTED_MODEL,
                build_date='2026-11-07', check_date=node.CHECK_DATE, registered_build=2000,
                registered_step=100, formal_step=99, registered_final=474, formal_final=467,
                code={n:vr.sha(scope.PROJECT/n) for n in names})


def bind(policy, scope_sha, gpus):
    node.node()
    require(policy in baseline.POLICIES and scope_sha == scope.SCOPE_SHA, 'Confirm the exact source register and judge policy')
    cfg = configuration(policy, gpus)
    require(cfg['source_scope_sha256'] == scope_sha, 'Scope manifest changed')
    receipt = dict(configuration=cfg, configuration_sha256=vr.digest(cfg))
    final.immutable(WORK/'workflow.json', receipt)
    return receipt


def verify_plan():
    plan = vr.read(WORK/'workflow.json')
    cfg = plan['configuration']
    require(cfg == configuration(cfg['judge_location_policy'], cfg['review_gpus']) and cfg['judge_location_policy'] in baseline.POLICIES
            and cfg['source_scope_sha256'] == scope.SCOPE_SHA and vr.digest(cfg) == plan['configuration_sha256'],
            'Workflow code, configuration or approved source scope changed')
    return plan


def stage(name, **details):
    vr.atomic(WORK/'workflow_status.json', dict(stage=name, unix=time.time(), **details))
    print(name, flush=True)


def connect_stop(folder, root):
    """Make each stage observe the same stop file, including between model calls."""
    folder.mkdir(parents=True, exist_ok=True)
    local = folder / 'STOP'
    target = root / 'STOP'
    require(not target.exists() and not local.exists(), 'STOP requested')
    if local.is_symlink():
        require(local.resolve() == target.resolve(), 'Different stage stop file')
    else:
        local.symlink_to(target)


def run(*, adopted=False):
    node.node()
    with final.base.r.run_lock(WORK/'workflow.lock'):
        if adopted:
            from scripts import step_judge_handoff as handoff
            plan = handoff.verify()
            handoff.take_boundary(plan)
            supervisor = handoff.ROOT
        else:
            plan, supervisor = verify_plan(), WORK
        sig, cfg = plan['configuration_sha256'], plan['configuration']
        gpu = cfg['review_gpus'][0]
        code = 1
        try:
            if not adopted:
                final.base.runtime.wait_detached(supervisor, sig)

            def attached_to_supervisor(folder, signature):
                ready = vr.read(supervisor/'disconnect_acceptance.json')
                require(ready['passed'] and ready['configuration_sha256'] == sig and ready['pid'] == os.getpid()
                        and ready['process_start'] == final.base.runtime.process_identity(os.getpid()),
                        'Stage is outside the accepted detached process')
                require(not (WORK/'STOP').exists() and not (folder/'STOP').exists(), 'STOP requested')
                vr.atomic(folder/'workflow_supervisor.json', dict(workflow=sig, configuration_sha256=signature,
                                                                  pid=ready['pid'], token=ready['token']))

            @contextmanager
            def inference(folder, date, signature):
                attached_to_supervisor(folder, signature)
                with node.service(folder, gpu, WORK/'STOP', signature, fixed_date=date) as endpoint:
                    require(endpoint == 'http://127.0.0.1:18083/v1', 'Unexpected inference endpoint')
                    yield

            pure = WORK/'step_pure'
            stage('verifying_frozen_baseline')
            baseline.baseline(pure)
            build = WORK/'construction'
            connect_stop(build, WORK)
            collection = RESULTS/'corrected/collection'
            stage('binding_frozen_build_inputs')
            if not (build/'plan.json').exists():
                extract.prepare(build, collection, scope.PROJECT/'data/combined_language_only.json', results_root=RESULTS)
            build_plan = extract.verify(build)
            approved = vr.read(build/'build_scope.json')
            audit = vr.read(WORK/'build_scope_audit.json')
            bc = build_plan['configuration']
            require(approved == scope.source_scope(RESULTS) and bc['fixed_date'] == cfg['build_date']
                    and audit['source_manifest_sha256'] == cfg['source_scope_sha256']
                    and audit['generator'] == extract.c.corpus.MODEL and audit['registered'] == 2000
                    and bc['source_plan_sha256'] == 'bf8d9d5d024c201e12cebff2ac209200488db1e494b6fef53b466d3586c392c6'
                    and audit['inputs_digest'] == bc['inputs_sha256']
                    and audit['source_evidence_digest'] == bc['evidence_sha256'],
                    'First-review inputs differ from the verified frozen 30B source audit')
            build_status = vr.read(build/'execution_status.json')
            if build_status['state'] != 'extraction_complete':
                stage('first_review', eligible=len(vr.read(build/'inputs.json')))
                with inference(build, cfg['build_date'], build_plan['configuration_sha256']):
                    extract.run(build, wait_detached=attached_to_supervisor,
                                reviews=partial(pool.ordered_reviews, gpus=cfg['review_gpus']))

            library = WORK/'libraries'
            connect_stop(library, WORK)
            stage('preparing_libraries')
            if not (library/'plan.json').exists():
                materials.prepare(library, WORK)
            lp = materials.verify(library)
            lc = lp['configuration']
            require(lc.get('scope') == 'materials_only' and lc['transport']['fixed_date'] == cfg['build_date'],
                    'Different material protocol')
            with final.base.r.run_lock(library/'active.lock'):
                if not (library/'construction_acceptance.json').is_file():
                    if not all((library/'materials'/n).is_file() for n in ('catalog.json','tree_report.json','skills.json','skills_report.json','cases.json')):
                        from transformers import AutoTokenizer
                        stage('organizing_libraries')
                        tok = AutoTokenizer.from_pretrained(node.host.HOT/'models/gpt-oss-120b',local_files_only=True)
                        with inference(library, cfg['build_date'], lp['configuration_sha256']):
                            materials.build_materials(library,lc,tok)
                    stage('verifying_material_isolation')
                    materials.accept_materials(library,lc)
            step.released_libraries(library,{})

            evaluated = WORK/'step'
            connect_stop(evaluated, WORK)
            stage('step_knowledge_arms')
            if not (evaluated/'plan.json').is_file():
                with final.base.r.run_lock(evaluated/'active.lock'):
                    step.prepare(evaluated, library, benchmark=RESULTS/'aligned_old99/evaluation',
                                 reuse_pure=pure, judge_location_policy=cfg['judge_location_policy'])
            cp = step.verify(evaluated)
            require(cp['configuration']['reuse_pure'] == str(pure) and cp['configuration']['parent'] == str(library)
                    and cp['configuration']['judge_location_policy'] == cfg['judge_location_policy'], 'Different step run')
            citations_pending = False
            first_pass = evaluated/'step_judge_first_pass/first_pass.json'
            measure_lines = cfg['judge_location_policy'] == 'lines_once'
            try:
                if not (evaluated/'metrics.json').is_file() and not (measure_lines and first_pass.is_file()):
                    status_file = evaluated/'step_judge/status.json'
                    if status_file.is_file():
                        status = vr.read(status_file)
                        if status['state'] == 'incomplete' and not status['missing'] and status['content_failures'] and set(
                                status['content_failures'].values()) <= baseline.location.QUOTE_ERRORS:
                            baseline.evaluate(evaluated, cp)
                    step.run(evaluated, gpu=gpu, wait_detached=attached_to_supervisor)
                report = step.score(evaluated)
                scores = report['provisional_scores'] if measure_lines else report['scores']
                require(all(n == 99 for n in report['recorded_tasks'].values()) and len(scores) == 4,
                        'Incomplete step comparison')
            except baseline.CitationDecisionPending:
                require(cfg['judge_location_policy'] == 'stop', 'Unexpected pending citation protocol')
                citations_pending = True
                stage('step_citations_awaiting_decision', status_sha256=vr.sha(evaluated/'step_judge/status.json'))

            dest = WORK/'final'
            connect_stop(dest, WORK)
            stage('final_answer_comparison', step_citations_pending=citations_pending, step_release_criterion_pending=measure_lines)
            with final.base.r.run_lock(dest/'active.lock'):
                if not (dest/'plan.json').is_file():
                    final.prepare(dest,RESULTS,material_parent=library)
                fp = final.verify(dest)
                fc = fp['configuration']
                require(fc['registered_questions'] == 474 and len(fc['eligible_ids']) == 467
                        and fc['online_reference_visible'] is False and fc['fixed_harmony_date'] == cfg['check_date']
                        and fc['released_files']['catalog'] == vr.sha(library/'materials/catalog.json')
                        and fc['released_files']['skills'] == vr.sha(library/'materials/skills.json')
                        and fc['released_files']['cases'] == vr.sha(library/'materials/cases.json'), 'Different final comparison')
                if not (dest/'metrics.json').is_file():
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(node.host.HOT/'models/gpt-oss-120b',local_files_only=True)
                    with inference(dest, cfg['check_date'], fp['configuration_sha256']), httpx.Client(trust_env=False) as client:
                        if not (dest/'pilot_acceptance.json').is_file():
                            final.run_checks(dest,fp,tok,client,pilot=True)
                        final.run_checks(dest,fp,tok,client,pilot=False)
                final.score(dest)
            if measure_lines:
                stage('final_complete_step_first_pass_measured', step_first_pass_sha256=vr.sha(first_pass),
                      final_metrics_sha256=vr.sha(dest/'metrics.json'), step_release_criterion_pending=True)
                code = 2
            elif citations_pending:
                stage('final_complete_step_citations_pending', step_status_sha256=vr.sha(evaluated/'step_judge/status.json'),
                      final_metrics_sha256=vr.sha(dest/'metrics.json'))
                code = 2
            else:
                stage('complete', step_metrics_sha256=vr.sha(evaluated/'metrics.json'), final_metrics_sha256=vr.sha(dest/'metrics.json'))
                code = 0
        except BaseException as exc:
            stage('stopped', reason=str(exc))
            raise
        finally:
            vr.atomic(WORK/'workflow_exit.json', dict(code=code, unix=time.time(), configuration_sha256=sig))
        return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('audit','prepare','run','confirm','adopt-lines'))
    parser.add_argument('--approved-scope-sha256')
    parser.add_argument('--judge-location-policy', choices=baseline.POLICIES, default='lines_once')
    parser.add_argument('--gpus',type=int,nargs='+',choices=(4,5,6,7),default=[4])
    args=parser.parse_args()
    if args.command=='audit':
        node.node()
        tempfile.tempdir = str(node.host.HOT / 'tmp')
        with final.base.r.run_lock(WORK/'source_audit.lock'):
            report = scope.audit(RESULTS)
            final.immutable(WORK/'build_scope_audit.json', report)
            print(report['registered'], report['reviewable'], report['dispositions'])
    elif args.command=='prepare':
        with final.base.r.run_lock(WORK/'workflow.lock'):
            bind(args.judge_location_policy,args.approved_scope_sha256,args.gpus)
    elif args.command == 'adopt-lines':
        from scripts import step_judge_handoff as handoff
        handoff.prepare()
    elif args.command=='confirm':
        node.node()
        from scripts import step_judge_handoff as handoff
        if handoff.ADOPTION.is_file():
            node.confirm(handoff.ROOT, handoff.verify()['configuration_sha256'])
        else:
            node.confirm(WORK,verify_plan()['configuration_sha256'])
    else:
        from scripts import step_judge_handoff as handoff
        if handoff.ADOPTION.is_file():
            raise SystemExit(handoff.resume())
        raise SystemExit(run())


if __name__ == '__main__':
    main()
