from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
SESSIONS = ('01', '02', '03', '04', '05')

def seed_averaged(block: dict) -> tuple[np.ndarray, np.ndarray]:
    correct = np.mean(np.asarray(block['correct_by_seed'], dtype=float), axis=0)
    return (correct, np.asarray(block['n'], dtype=float))

def draw(n, rng, n_boot):
    return rng.integers(0, len(n), size=(n_boot, len(n)))

def accuracy_at(correct, n, idx):
    return correct[idx].sum(axis=1) / n[idx].sum(axis=1)

def resampled_accuracy(correct, n, rng, n_boot):
    return accuracy_at(correct, n, draw(n, rng, n_boot))

def interval(values) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

def gap_intervals(path: Path, rng, n_boot) -> dict:
    d = json.loads(path.read_text())
    blocks = {name: (seed_averaged(b['session4']), seed_averaged(b['session5'])) for name, b in d['dialogues'].items()}
    (_, n4), (_, n5) = next(iter(blocks.values()))
    idx4, idx5 = (draw(n4, rng, n_boot), draw(n5, rng, n_boot))
    point, boot = ({}, {})
    for name, ((c4, n4), (c5, n5)) in blocks.items():
        point[name] = c4.sum() / n4.sum() - c5.sum() / n5.sum()
        boot[name] = accuracy_at(c4, n4, idx4) - accuracy_at(c5, n5, idx5)
    ref = next((k for k in blocks if k.startswith('audio')), None)
    out = {}
    for name in blocks:
        out[name] = {'gap': float(point[name]), 'ci95': interval(boot[name]), 'dialogues_session4': len(n4), 'dialogues_session5': len(n5)}
        if ref is not None and name != ref:
            out[name]['gap_minus_audio'] = float(point[name] - point[ref])
            out[name]['gap_minus_audio_ci95'] = interval(boot[name] - boot[ref])
    return out

def loso_intervals(path: Path, rng, n_boot) -> dict:
    d = json.loads(path.read_text())
    point, boot = ({}, {})
    for s in SESSIONS:
        c, n = seed_averaged(d['folds'][s]['dialogues'])
        point[s] = c.sum() / n.sum()
        boot[s] = resampled_accuracy(c, n, rng, n_boot)
    loso = np.mean([point[s] for s in SESSIONS])
    loso_boot = np.mean([boot[s] for s in SESSIONS], axis=0)
    return {'loso_mean': float(loso), 'standard': float(point['05']), 'inflation': float(loso - point['05']), 'inflation_ci95': interval(loso_boot - boot['05']), 'loso_mean_ci95': interval(loso_boot), 'standard_ci95': interval(boot['05']), 'dialogues_per_fold': {s: len(d['folds'][s]['dialogues']['n']) for s in SESSIONS}}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gap', type=Path, nargs='*', default=[])
    ap.add_argument('--loso', type=Path, nargs='*', default=[])
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=Path('results/dialogue_bootstrap.json'))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    result = {'n_boot': args.n_boot, 'unit': 'dialogue', 'session_gap': {}, 'loso': {}}
    for p in args.gap:
        for name, rec in gap_intervals(p, rng, args.n_boot).items():
            result['session_gap'][f'{p.stem}:{name}'] = rec
            lo, hi = rec['ci95']
            extra = ''
            if 'gap_minus_audio' in rec:
                a, b = rec['gap_minus_audio_ci95']
                extra = f"  minus audio {rec['gap_minus_audio'] * 100:+6.2f} [{a * 100:+6.2f}, {b * 100:+6.2f}]"
            print(f"gap  {p.stem:28s} {name:34s} {rec['gap'] * 100:+6.2f} [{lo * 100:+6.2f}, {hi * 100:+6.2f}]{extra}")
    for p in args.loso:
        key = f'{p.parent.name}/{p.stem}'
        rec = loso_intervals(p, rng, args.n_boot)
        result['loso'][key] = rec
        lo, hi = rec['inflation_ci95']
        print(f"loso {key:40s} {rec['loso_mean'] * 100:6.2f} vs {rec['standard'] * 100:6.2f}  inflation {rec['inflation'] * 100:+6.2f} [{lo * 100:+6.2f}, {hi * 100:+6.2f}]")
    args.out.write_text(json.dumps(result, indent=2))
    print(f'written {args.out}')
if __name__ == '__main__':
    main()
