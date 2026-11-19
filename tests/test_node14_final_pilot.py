from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

from scripts import run_node14_final_pilot as pilot


def test_idle_card_excludes_compute_pids_utilization_and_allocations():
    def gpu(index, used, utilization):
        return [str(index), f'gpu-{index}', 'NVIDIA A800-SXM4-80GB', '81920', str(used), str(utilization)]

    snapshot = dict(gpus=[gpu(0, 3, 0), gpu(1, 3, 1), gpu(2, 100, 0)], compute_processes=[['gpu-0', '15']])
    assert pilot.idle_card(snapshot) is None
    snapshot['gpus'].append(gpu(3, 3, 0))
    assert pilot.idle_card(snapshot) == ('3', 'gpu-3')


def test_node14_launcher_changes_only_four_storage_paths():
    root = Path(__file__).resolve().parents[1]
    source = root / pilot.LAUNCHER
    frozen = '5a6972d1fbe2a12d113477b42527d13c6d44933a51c957d634f5ca44d034b7b5'
    with patch.object(pilot, 'ROOT', root):
        script, paths = pilot.service_script(dict(source_hashes={pilot.LAUNCHER: frozen}))
    assert len(paths) == 4
    for before, after in paths.items():
        script = script.replace(after + '\n', before + '\n')
    assert script == source.read_text()


def test_shutdown_reads_both_gauges_across_engines():
    metrics = ('# HELP vllm:num_requests_running Running requests\n'
               'vllm:num_requests_running{engine="0"} 0.0\n'
               'vllm:num_requests_running{engine="1"} 1.0\n'
               'vllm:num_requests_waiting{engine="0"} 2.0\n')
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text=metrics))
    with httpx.Client(transport=transport) as client:
        assert pilot.request_counts(client, 'http://127.0.0.1') == dict(running=1, waiting=2)


def test_shutdown_tracks_vllm_worker_by_session_with_an_empty_environ():
    with patch.object(pilot.os, 'getpgid', return_value=33), patch.object(pilot.os, 'getsid', return_value=33), \
            patch.object(Path, 'read_bytes', return_value=b''):
        assert pilot.owns_gpu_process('34', 33, Path('/private/service'))
        assert not pilot.owns_gpu_process('34', 44, Path('/private/service'))


def test_launcher_exit_does_not_release_gpu_while_recorded_worker_lives(tmp_path):
    service = Mock(pid=33, returncode=0)
    service.poll.return_value = 0
    worker = {'34': '1234'}
    busy = dict(compute_processes=[['gpu', '34']])
    free = dict(compute_processes=[])
    with patch.object(pilot, 'session_workers', side_effect=[worker.copy(), worker, {}]), \
            patch.object(pilot, 'resources', side_effect=[busy, free]), \
            patch.object(pilot, 'sockets', return_value=''), patch.object(pilot, 'owns_gpu_process', return_value=True), \
            patch.object(pilot.final.base.runtime, 'process_identity', return_value='1234'), \
            patch.object(pilot.time, 'monotonic', side_effect=[0, 46]), patch.object(pilot.time, 'sleep'), \
            patch.object(pilot.os, 'killpg') as terminate, patch.object(pilot.os, 'kill') as force:
        pilot.release(service, dict(gpu_uuid='gpu'), tmp_path, False, '')
    terminate.assert_called_once_with(33, pilot.signal.SIGTERM)
    force.assert_called_once_with(34, pilot.signal.SIGKILL)
    result = pilot.vr.read(tmp_path / 'release.json')
    assert result['forced_pids'] == worker and result['owned_gpu_pids'] == [] and result['session_workers'] == {}


def test_hot_storage_preserves_served_id_and_rechecks_weight_bytes(tmp_path):
    hot = tmp_path / 'hot'
    local = hot / 'models/gpt-oss-120b'
    local.mkdir(parents=True)
    folder = tmp_path / 'attempt'
    folder.mkdir()
    index = local / 'model.safetensors.index.json'
    shard = local / 'shard.safetensors'
    pilot.vr.atomic(index, {'weight_map': {'layer.weight': shard.name}})
    shard.write_bytes(b'original')
    files = {p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=pilot.vr.sha(p))
             for p in (index, shard)}
    served_id = '/shared/original-gpt-oss'
    receipt = tmp_path / 'weights.json'
    pilot.vr.atomic(receipt, dict(verified=True, source=served_id, local=str(local), shards=[shard.name], files=files))
    config = dict(model=served_id)
    with patch.object(pilot, 'HOT', hot), patch.object(pilot, 'CONTROL', tmp_path), \
            patch.object(pilot.final.base.runtime, 'MODEL_FILES', {index.name: pilot.vr.sha(index)}):
        path = pilot.weight_location(config, folder, pilot.vr.sha(receipt))
        assert pilot.vr.read(path) == {served_id: str(local)}
        stat = shard.stat()
        shard.write_bytes(b'modified')
        pilot.os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with pytest.raises(ValueError, match='Bound evidence changed'):
            pilot.weight_location(config, folder, pilot.vr.sha(receipt))
