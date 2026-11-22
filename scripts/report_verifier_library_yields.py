"""Verify and plot construction yields for the three preregistered random seeds."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

from scripts import run_verifier_library_scale as scale


vr, require = scale.vr, scale.require
METRICS = ('cases', 'skills', 'leaves')
TITLES = ('RAG 错误案例（篇）', '平铺 skills（条）', '规则树叶（条）')


def point(folder, registered):
    seed, n = registered['seed'], registered['source_questions']
    cfg = scale.verify(folder)['configuration']
    sources = vr.read(folder/'sources.json')
    ids = set(sources['original_draw'])
    require((cfg['seed'], cfg['source_count']) == (seed, n) and len(ids) == n == len(sources['original_draw'])
            and vr.digest(sorted(ids)) == cfg['source_ids_sha256'] == registered['source_ids_sha256']
            and set(sources['dispositions']) == ids, 'Draw differs from the registered families')
    status = sources['dispositions']
    counts = dict(Counter(status.values()))
    require(counts == cfg['source_counts'] and set(sources['new_hold']) == ids & scale.SOURCE_HOLD
            and {sid for sid, disposition in status.items() if disposition == 'new_source_hold'} == set(sources['new_hold']),
            'Incorrect source accounting')
    receipt = folder/'construction_acceptance.json'
    accepted, replay = vr.read(receipt), vr.read(folder/'tree_replay.json')
    tree = vr.read(folder/'materials/tree_report.json')
    counts_tree = tree['counts']
    requests = len(accepted['response_files'])
    require(replay['passed'] and replay['acceptance_sha256'] == vr.sha(receipt) and replay['model_calls'] == 0
            and replay['reproduced_responses'] == requests == accepted['tree_responses']
            and accepted['tree_counts'] == counts_tree == replay['counts'] and accepted['source_counts'] == counts,
            'Published materials lack a matching full replay')
    require(counts_tree['all_candidates_preserved'] and cfg['candidates'] == counts_tree['input_candidates']
            == counts_tree['placed_rules'] + counts_tree['unplaced_rules'] and tree['validation']['valid']
            and not vr.read(folder/'material_isolation.json')['tree_matches'], 'Invalid tree or held-out match')
    flat = {}
    for name in ('cases', 'skills'):
        docs = vr.read(folder/'materials'/(name+'.json'))
        require(len(docs) == cfg['flat_counts'][name] == accepted[name], 'Flat material count changed')
        require(all(len(doc['source_ids']) == 1 and status.get(doc['source_ids'][0]) == 'error_found'
                    for doc in docs), 'Source outside the eligible draw')
        flat[name] = {doc['id']: vr.digest(doc) for doc in docs}
        require(len(flat[name]) == len(docs), 'Duplicate flat material ID')
    rules = {r['rule_id']: set(r['sample_ids']) for r in vr.read(folder/'tree_bundle.json')['rules']}
    require(len(rules) == cfg['candidates'] and all(support and all(status.get(s) == 'error_found' for s in support)
                                                    for support in rules.values()), 'Invalid rule support')
    result = dict(seed=seed, source_questions=n, source_ids_sha256=cfg['source_ids_sha256'], source_counts=counts,
                  cases=accepted['cases'], skills=accepted['skills'], candidates=cfg['candidates'],
                  leaves=counts_tree['placed_rules'], unplaced=counts_tree['unplaced_rules'], clusters=counts_tree['clusters'],
                  reproduced_requests=requests, acceptance_sha256=vr.sha(receipt), replay_sha256=vr.sha(folder/'tree_replay.json'))
    return result, ids, flat, rules


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


def collect(root, allow_partial):
    rows, missing = [], []
    draws_by_size = {n: {} for n in scale.BUILD_SIZES}
    for seed in scale.RANDOM_SEEDS:
        previous_ids, previous_flat, previous_rules = set(), {}, {}
        for draw in (r for r in scale.RANDOM_REGISTRATION['draws'] if r['seed'] == seed):
            n = draw['source_questions']
            folder = root/f'n{n}_seed{seed}'
            if not (folder/'tree_replay.json').is_file():
                missing.append(dict(seed=seed, source_questions=n))
                continue
            row, ids, flat, rules = point(folder, draw)
            require(previous_ids < ids and all(previous_flat.get(key, {}).items() <= values.items()
                    for key, values in flat.items()) and all(support <= rules.get(rid, set())
                    for rid, support in previous_rules.items()), 'Nested material or rule support changed')
            previous_ids, previous_flat, previous_rules = ids, flat, rules
            draws_by_size[n][seed] = ids
            rows.append(row)
    require(rows and (allow_partial or not missing), 'Some registered draws lack construction and replay receipts')
    overlap = {str(n): [dict(seeds=[a,b], shared_questions=len(sets[a] & sets[b])) for a,b in combinations(sets,2)]
               for n, sets in draws_by_size.items()}
    return dict(seed_manifest_sha256=scale.SEED_MANIFEST_HASH, sizes=list(scale.BUILD_SIZES), seeds=list(scale.RANDOM_SEEDS),
                scope='Construction yields per original source draw', complete=not missing, points=rows, pending=missing,
                between_seed_overlap=overlap, by_size=summarize(rows, scale.BUILD_SIZES, len(scale.RANDOM_SEEDS)))


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
        title = '不同随机种子下的独立建库产量'
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
    parser.add_argument('--root', type=Path, required=True, help='library_scale result directory')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    summary = collect(args.root, args.allow_partial)
    vr.atomic(args.root/'random_seed_yields.json', summary)
    if args.plot:
        plot(summary, args.root/'random_seed_yields.png')
