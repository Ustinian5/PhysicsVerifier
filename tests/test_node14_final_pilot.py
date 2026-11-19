from pathlib import Path
from unittest.mock import patch

import httpx

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
