"""Measure warm-index ranking on the frozen BM25F cohort, excluding prompt packing."""
from time import perf_counter_ns

from scripts import run_bm25f_retrieval as replay


def run():
    indexes = {name: replay.final.stores(replay.structured.OUT) for name in ('bm25', 'bm25f')}
    for arm, store in indexes['bm25f'].items():
        if arm in replay.ARMS:
            flat = store.memory if arm == 'rule_tree' else store
            flat.search = replay.FieldSearch(flat).search
    samples = []
    for source, ids in replay.vr.read(replay.OUT / 'cohort.json').items():
        for i, sid in enumerate(ids):
            for arm in replay.ARMS:
                folder = replay.OUT / source / sid / arm
                receipt = replay.vr.read(folder / 'receipt.json')
                if 'retained_failure' in receipt:
                    continue
                packet = replay.vr.read(folder / 'packet.json')['retrieval']
                original = replay.vr.read(receipt['original_result'].rsplit('/', 1)[0] + '/reference_packet.json')
                baseline = original.get('packet', original)['retrieval']
                assert packet['query'] == baseline['query']
                for mode in (('bm25f', 'bm25') if i % 2 else ('bm25', 'bm25f')):
                    store = indexes[mode][arm]
                    start = perf_counter_ns()
                    ranked = store.retrieve(packet['query'])[0] if arm == 'rule_tree' else store.search(packet['query'])
                    elapsed = perf_counter_ns() - start
                    expected = packet if mode == 'bm25f' else baseline
                    assert [d['id'] for d in ranked] == [d['id'] for d in expected['ranked']]
                    samples.append(dict(source=source, id=sid, arm=arm, mode=mode, nanoseconds=elapsed))
    replay.final.immutable(replay.OUT / 'retrieval_timing.json', dict(scope='warm_index_ranking', samples=samples))


if __name__ == '__main__':
    run()
