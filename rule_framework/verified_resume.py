"""Content-bound receipts and durable progress for long offline constructions.

Metadata is only an invalidation signal after a SHA256 check, never the source
of an isolation verdict. Force full hashing at import and publication. Normal
resumes stat immutable evidence and hash changed files, without parsing the
historical model transcripts or re-running the quadratic isolation algorithm.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    """Same-directory replace, file + directory fsync; one lock-owning writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


def seal(path, value):
    atomic(path, {'value': value, 'sha256': digest(value)})


def unseal(path):
    data = read(path)
    if data['sha256'] != digest(data['value']):
        raise ValueError('Damaged checkpoint: ' + str(path))
    return data['value']


def stamp(path):
    st = Path(path).stat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError('Evidence is not a regular file: ' + str(path))
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


class EvidenceCache:
    def __init__(self, path):
        self.path = Path(path)

    def check(self, files, *, full=False, workers=16):
        """Every expected SHA comes from a bound manifest, not from this cache."""
        started = time.monotonic()
        try:
            prior = unseal(self.path) if self.path.exists() else {}
        except (ValueError, KeyError):
            prior = {}  # Broken optimization: verify content again, never trust it.

        def check_one(pair):
            name, expected = pair
            before = stamp(name)
            old = prior.get(name)
            checked = full or old != {'sha256': expected, 'stamp': before}
            if checked:
                if sha(name) != expected:
                    raise ValueError('Bound evidence changed: ' + name)
                if stamp(name) != before:
                    raise ValueError('Evidence changed during verification: ' + name)
            return name, {'sha256': expected, 'stamp': before}, checked

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(check_one, sorted(files.items())))
        seal(self.path, {name: record for name, record, _ in results})
        return {'files': len(files), 'files_rehashed': sum(x[2] for x in results),
                'full': full, 'seconds': round(time.monotonic() - started, 4)}


class Progress:
    """A complete small state snapshot; no replay of completed model calls."""
    def __init__(self, path, binding, initial):
        self.path, self.binding = Path(path), binding
        if self.path.exists():
            saved = unseal(self.path)
            if saved['binding'] != binding:
                raise ValueError('Checkpoint belongs to a different protocol/materials')
            self.state = saved['state']
        else:
            self.state = initial
            self.save()

    def save(self):
        seal(self.path, {'binding': self.binding, 'state': self.state})


def resumed_budget(budget_class, output, limits, prior):
    """Use the existing budget guards without re-reading every raw response.

    Single-process use under the runner lock. Pending reservations from a crash
    remain fully charged and counted as unknown, even if replay later finds a
    response; this intentionally overestimates instead of clearing old usage.
    """
    import threading

    class DurableBudget(budget_class):
        def __init__(self):
            self.output, self.limits = Path(output), limits
            self.quota_probe, self.quota_floor = None, 500000
            self.retry_overload = self.defer_transient = False
            self.deadline_unix = None
            self.stop_file = self.output / 'STOP'
            self.lock = threading.Lock()
            self.platform_quota = self.quota_error = None
            self.next_quota_check = self.retry_after_until = 0.0
            self.reservations = {}
            path = self.output / 'budget_checkpoint.json'
            saved = unseal(path) if path.exists() else None
            if saved and (saved['limits'] != limits or saved['prior_sha256'] != digest(prior)):
                raise ValueError('Budget limits or inherited usage changed')
            source = saved or prior
            self.requests = source['requests_started']
            self.input_tokens = source['accounted_input_tokens'] + source.get('reserved_input_tokens', 0)
            self.output_tokens = source['accounted_output_tokens'] + source.get('reserved_output_tokens', 0)
            self.unknown_usage = source['requests_without_usage'] + source.get('in_flight', 0)
            self.stopped = source.get('stop_reason', '') if saved else ''
            # Only an explicit operator STOP is cleared on an authorized restart.
            if self.stopped == 'operator_stop_requested' and not self.stop_file.exists():
                self.stopped = ''
            self._save()

        def _save(self):
            saved = {'limits': self.limits, 'prior_sha256': digest(prior),
                     'requests_started': self.requests,
                     'accounted_input_tokens': self.input_tokens,
                     'accounted_output_tokens': self.output_tokens,
                     'requests_without_usage': self.unknown_usage,
                     'in_flight': len(self.reservations),
                     'reserved_input_tokens': sum(x[0] for x in self.reservations.values()),
                     'reserved_output_tokens': sum(x[1] for x in self.reservations.values()),
                     'stop_reason': self.stopped}
            seal(self.output / 'budget_checkpoint.json', saved)
            atomic(self.output / 'budget.json', dict(saved, updated_unix=time.time(),
                unknown_usage_policy='retain orphaned reservations; no historical usage reset'))

    return DurableBudget()
