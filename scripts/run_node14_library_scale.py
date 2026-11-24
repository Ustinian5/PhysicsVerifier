"""Locate the accepted materials for a registered node14 scale run."""
import argparse
from pathlib import Path
import socket
import sys

from scripts import run_node14_final_pilot as host
from scripts import run_verifier_library_scale as scale


require = host.require
RESULTS = host.HOT/'results/qwen30b_step_verifier'


def available_card(snapshot, gpu):
    require(gpu in (4,5,6,7), 'GPU outside authorized indices')
    return host.idle_card(dict(snapshot, gpus=[g for g in snapshot['gpus'] if int(g[0])==gpu]))


def service_port(size, seed):
    size_index = scale.BUILD_SIZES.index(size)
    if seed in scale.RANDOM_SEEDS:
        return 18160 + 10 * size_index + scale.RANDOM_SEEDS.index(seed)
    return 18100 + 10 * size_index + seed % 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run','confirm'))
    parser.add_argument('--seed', type=int, required=True, choices=(20261122, *scale.RANDOM_SEEDS))
    parser.add_argument('--size', type=int, choices=scale.BUILD_SIZES, default=250)
    parser.add_argument('--gpu', type=int, required=True, choices=range(4,8))
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    require(socket.gethostname().split('.')[0]=='node14' and Path(sys.prefix)==host.ENV and args.attempt>0,
            'Unexpected host, conda environment or attempt')
    from rule_framework.shared_experience import reviewed_flat_library
    print(reviewed_flat_library(RESULTS, f'n{args.size}_seed{args.seed}') / 'materials')


if __name__=='__main__':
    main()
