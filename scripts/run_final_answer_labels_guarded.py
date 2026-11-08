"""Run the unchanged labeler with an exact, offline 300/300 feasibility guard.

This adds scheduling protection only. It never changes labels, generation,
sampling, prompts, or the frozen labeler's configuration and replay signatures.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import threading
import time

from scripts import run_final_answer_labels as labels


def quota_status(out):
    refs = labels.read(out / 'references_private.json')
    ids = {r['id'] for r in refs}
    labels.require(len(ids) == len(refs), 'Duplicate reference IDs')
    eligible = {r['id'] for r in refs if r['reference_complete']}
    counts = Counter()
    observed = set()
    bindings = {}
    for folder in ('labels', 'generation_failures'):
        for path in sorted((out / folder).glob('*.json')):
            value = labels.read(path)
            sid = value['id']
            labels.require(sid in eligible and sid not in observed and path.stem == sid,
                           'Unexpected or duplicate terminal candidate')
            observed.add(sid)
            bindings[str(path.relative_to(out))] = labels.sha(path)
            if folder == 'generation_failures':
                counts['generation_failure'] += 1
            elif value['state'] == 'complete':
                verdict = labels.protocol.validate_verdict(value['judgment'])['verdict']
                counts[verdict] += 1
            else:
                labels.require(value['state'] == 'content_failure', 'Unknown terminal label state')
                counts['content_failure'] += 1
    pending = len(eligible) - len(observed)
    bounds = {k: counts[k] + pending for k in ('correct', 'incorrect')}
    unreachable = [k for k in bounds if bounds[k] < 300]
    return dict(candidates=len(ids), reference_incomplete=len(ids - eligible),
                labelable_candidates=len(eligible), counts=dict(counts), pending=pending,
                maximum_possible=bounds, required_per_class=300,
                quota_met=all(counts[k] >= 300 for k in bounds),
                quota_impossible=bool(unreachable), unreachable_classes=unreachable,
                interpretation='Bounds only; pending candidates are not predicted labels.',
                configuration_sha256=labels.read(out / 'plan.json')['configuration_sha256'],
                terminal_record_hashes=bindings, updated_unix=time.time())


def run(out):
    # The frozen runner still owns both original locks and disconnect acceptance.
    with labels.corpus.r.run_lock(out / 'quota_guard.lock'):
        labels.verify(out)
        done = threading.Event()
        failures = []

        def monitor():
            while not done.wait(1):
                pid = out / 'runner.pid'
                if not pid.exists() or pid.read_text().strip() != str(os.getpid()):
                    continue
                try:
                    status = quota_status(out)
                    labels.write(out / 'quota_status.json', status)
                    if status['quota_impossible']:
                        labels.write(out / 'quota_stop.json', status)
                        (out / 'STOP').touch()
                        return
                except Exception as exc:
                    failures.append(str(exc))
                    labels.write(out / 'quota_guard_error.json', dict(error=str(exc), unix=time.time()))
                    (out / 'STOP').touch()
                    return
                if done.wait(9):
                    return

        labels.write(out / 'quota_guard_binding.json', dict(
            script_sha256=labels.sha(Path(__file__)), required_correct=300,
            required_incorrect=300, pid=os.getpid(), started_unix=time.time(),
            policy='Stop new label requests only when remaining candidates cannot meet either quota; preserve in-flight output.'))
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            labels.run(out, full=True)
        finally:
            done.set()
            thread.join(timeout=15)
            labels.write(out / 'quota_status.json', quota_status(out))
        labels.require(not failures, 'Quota monitoring failed: ' + '; '.join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect', 'run'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.mode == 'run':
        run(out)
    else:
        import json
        print(json.dumps(quota_status(out), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
