from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
SESSIONS = ('01', '02', '03', '04', '05')

def seed_mean(block: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.mean(np.asarray(block['correct_by_seed'], dtype=float), axis=0), np.asarray(block['n'], dtype=float))

def interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

def refit_contrast(path: Path, rng, n_boot: int) -> dict:
    d = json.loads(path.read_text())['conditions']
    draws, point = ({}, {})
    for s in SESSIONS:
        counts = {c: seed_mean(d[c]['folds'][s]['dialogues']) for c in ('fixed', 'refit')}
        n = counts['fixed'][1]
        idx = rng.integers(0, len(n), size=(n_boot, len(n)))
        for c, (correct, nn) in counts.items():
            point.setdefault(c, {})[s] = correct.sum() / nn.sum()
            draws.setdefault(c, {})[s] = correct[idx].sum(axis=1) / nn[idx].sum(axis=1)
    out = {}
    for c in ('fixed', 'refit'):
        out[c] = float(np.mean([point[c][s] for s in SESSIONS]) - point[c]['05'])
        out[c + '_boot'] = np.mean([draws[c][s] for s in SESSIONS], axis=0) - draws[c]['05']
    diff = out['fixed_boot'] - out['refit_boot']
    return {'fixed_difference': out['fixed'], 'refit_difference': out['refit'], 'leak_estimate': out['fixed'] - out['refit'], 'ci95': interval(diff), 'fixed_ci95': interval(out['fixed_boot']), 'refit_ci95': interval(out['refit_boot'])}

def fusion_contrast(base: Path, added: list[Path], rng, n_boot: int) -> dict:
    cells = {p.stem: json.loads(p.read_text())['cells'] for p in [base] + added}
    out = {}
    for ctx in ('no-context', 'context'):
        ref_correct, n = seed_mean(cells[base.stem][f'text/{ctx}']['dialogues'])
        idx = rng.integers(0, len(n), size=(n_boot, len(n)))
        ref_point = ref_correct.sum() / n.sum()
        ref_boot = ref_correct[idx].sum(axis=1) / n[idx].sum(axis=1)
        for p in [base] + added:
            correct, nn = seed_mean(cells[p.stem][f'fused/{ctx}']['dialogues'])
            point = correct.sum() / nn.sum() - ref_point
            boot = correct[idx].sum(axis=1) / nn[idx].sum(axis=1) - ref_boot
            out[f'{p.stem}:{ctx}'] = {'difference': float(point), 'ci95': interval(boot)}
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--refit', type=Path)
    ap.add_argument('--fusion', type=Path, nargs='*', default=[])
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=Path('results/paired_dialogue_ci.json'))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    result = {'n_boot': args.n_boot, 'unit': 'dialogue'}
    if args.refit:
        r = refit_contrast(args.refit, rng, args.n_boot)
        result['refit_control'] = r
        print(f"fixed-fit difference {r['fixed_difference'] * 100:+6.2f} {[round(v * 100, 2) for v in r['fixed_ci95']]}")
        print(f"refit     difference {r['refit_difference'] * 100:+6.2f} {[round(v * 100, 2) for v in r['refit_ci95']]}")
        print(f"attributable to the fitting schedule {r['leak_estimate'] * 100:+6.2f} {[round(v * 100, 2) for v in r['ci95']]}")
    if args.fusion:
        base, added = (args.fusion[0], list(args.fusion[1:]))
        f = fusion_contrast(base, added, rng, args.n_boot)
        result['fusion'] = f
        for key, rec in f.items():
            print(f"fusion {key:46s} {rec['difference'] * 100:+6.2f} {[round(v * 100, 2) for v in rec['ci95']]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f'written {args.out}')
if __name__ == '__main__':
    main()
