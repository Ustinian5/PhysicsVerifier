"""Verify and plot construction yields for the three preregistered random seeds."""
import argparse
from itertools import combinations
from pathlib import Path
from statistics import mean

from rule_framework import shared_experience as shared
from scripts import run_verifier_library_scale as scale


vr, require = scale.vr, scale.require
METRICS = ('cases', 'skills', 'leaves')
TITLES = ('RAG 错误案例（篇）', '平铺 skills（条）', '规则树叶（条）')


def summarize(rows, sizes, repetitions):
    result = {}
    for n in sizes:
        subset = [r for r in rows if r['source_questions'] == n]
        require(len(subset) == len({r['seed'] for r in subset}) <= repetitions, 'Duplicate draw at this size')
        result[str(n)] = dict(completed=len(subset), planned=repetitions)
        if len(subset) == repetitions:
            for key in METRICS:
                values = [r[key] for r in subset]
                result[str(n)][key] = dict(min=min(values), max=max(values), mean=mean(values))
    return result


def collect_shared(root):
    points, draws = [], {n: {} for n in scale.BUILD_SIZES}
    for item in scale.RANDOM_REGISTRATION['draws']:
        seed, n = item['seed'], item['source_questions']
        directory = shared.library_path(root.parent, f'n{n}_seed{seed}')
        accepted = vr.read(directory / 'construction_acceptance.json')
        sources = vr.read(directory / 'sources.json')
        ids = set(sources['draw'])
        require(len(ids) == n and vr.digest(sorted(ids)) == item['source_ids_sha256'], 'Registered draw changed')
        require(all(smaller < ids for size, other in draws.items() if size < n
                    for s, smaller in other.items() if s == seed), 'Non-nested source draws')
        draws[n][seed] = ids
        points.append(dict(seed=seed, source_questions=n, **accepted['counts'],
                           acceptance_sha256=vr.sha(directory / 'construction_acceptance.json')))
    overlap = {str(n): [dict(seeds=[a,b], shared_questions=len(sets[a] & sets[b])) for a,b in combinations(sets,2)]
               for n, sets in draws.items()}
    return dict(protocol=shared.PROTOCOL, seed_manifest_sha256=scale.SEED_MANIFEST_HASH,
                shared_release_sha256=vr.sha(root / 'release.json'), sizes=list(scale.BUILD_SIZES),
                seeds=list(scale.RANDOM_SEEDS), scope='Shared first-review library yields',
                complete=True, points=points, pending=[], between_seed_overlap=overlap,
                by_size=summarize(points, scale.BUILD_SIZES, len(scale.RANDOM_SEEDS)))


def plot(report, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator

    for family in ('Noto Sans CJK SC', 'Source Han Sans SC', 'PingFang SC', 'Arial Unicode MS', 'STHeiti'):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        break
    else:
        raise ValueError('A Simplified Chinese font is required to render the construction curves')
    sizes = report['sizes']
    points = {(r['seed'], r['source_questions']):r for r in report['points']}
    styles = (('#214e85','o','-'), ('#ab4d10','s','--'), ('#19745e','^','-.'))
    with plt.rc_context({'font.family':family, 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False}):
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3), sharex=True)
        for ax, key, title in zip(axes, METRICS, TITLES):
            for seed, (color, marker, style) in zip(report['seeds'], styles):
                values = [points.get((seed, n), {}).get(key, float('nan')) for n in sizes]
                ax.plot(sizes, values, color=color, marker=marker, linestyle=style, linewidth=1.7,
                        markersize=6, markerfacecolor='white', markeredgewidth=1.5, label=str(seed))
            ax.set(title=title, xlabel='抽取的建库题数', xticks=sizes, xlim=(205,1045))
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
            ax.grid(axis='y', linewidth=0.5, color='#cbd2da')
            ax.set_axisbelow(True)
        title = '同一批错题经验下的三库产量' if report.get('protocol') == shared.PROTOCOL else '不同随机种子下的独立建库产量'
        if not report['complete']:
            title += f"（已验收 {len(points)}/{len(scale.RANDOM_REGISTRATION['draws'])} 档）"
        fig.suptitle(title, y=.97, fontsize=16)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='题族抽样种子', loc='lower center', ncol=3, frameon=False)
        fig.subplots_adjust(left=.065, right=.98, top=.78, bottom=.28, wspace=.32)
        fig.savefig(target, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='shared_experience or qwen30b_step_verifier result directory')
    parser.add_argument('--allow-partial', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if root.name == 'library_scale':
        root = root.parent / shared.DIRECTORY
    elif root.name != shared.DIRECTORY:
        root = root / shared.DIRECTORY
    summary = collect_shared(root)
    vr.atomic(root/'random_seed_yields.json', summary)
    if args.plot:
        plot(summary, root/'random_seed_yields.png')
