from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
_spec = importlib.util.spec_from_file_location('loso', Path(__file__).with_name('iemocap_loso.py'))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
DEFAULT = 'released-text,released-video,released-audio,released-tav,bert'

def random_folds(vids: list[str], k: int, seed: int) -> list[list[str]]:
    order = np.random.default_rng(seed).permutation(len(vids))
    return [[vids[i] for i in order[j::k]] for j in range(k)]

def random_cv(raw, blocks, vids, session, seeds, k=5) -> list[dict]:
    maxlen = max((len(raw['lab'][v]) for v in vids))
    runs = []
    for sd in seeds:
        hits = {'old': [0, 0], 'new': [0, 0]}
        for te in random_folds(vids, k, sd):
            held = set(te)
            tr = [v for v in vids if v not in held]
            xt, yt, mt = L.pack(raw, blocks, tr, maxlen)
            xe, ye, me = L.pack(raw, blocks, te, maxlen)
            xt, xe = L.H.standardise(xt, xe, mt)
            out = L.H.run_one(torch.from_numpy(xt), torch.from_numpy(yt), torch.from_numpy(mt), torch.from_numpy(xe), torch.from_numpy(ye), torch.from_numpy(me), 6, sd)
            pred = out[1].argmax(-1).numpy()
            y = ye[me > 0]
            ses = np.concatenate([[session[v]] * len(raw['lab'][v]) for v in te])
            for key, sel in (('new', ses == '05'), ('old', ses != '05')):
                hits[key][0] += int((pred[sel] == y[sel]).sum())
                hits[key][1] += int(sel.sum())
        n = hits['old'][1] + hits['new'][1]
        runs.append({'seed': sd, 'acc': (hits['old'][0] + hits['new'][0]) / n, 'acc_sessions_1_to_4_utterances': hits['old'][0] / hits['old'][1], 'acc_session_5_utterances': hits['new'][0] / hits['new'][1], 'share_sessions_1_to_4': hits['old'][1] / n})
        print(f"    seed {sd}: all {runs[-1]['acc'] * 100:6.2f}  ses1-4 utts {runs[-1]['acc_sessions_1_to_4_utterances'] * 100:6.2f}  ses5 utts {runs[-1]['acc_session_5_utterances'] * 100:6.2f}", flush=True)
    return runs

def speaker_cv(raw, blocks, vids, session) -> dict:
    x = np.concatenate([np.concatenate([np.asarray(b[v], dtype=np.float32) for b in blocks], 1) for v in vids])
    y = np.concatenate([np.asarray(raw['lab'][v]) for v in vids])
    spk = np.concatenate([[f'{session[v]}{g}' for g in raw['spk'][v]] for v in vids])
    folds = {}
    for s in sorted(set(spk)):
        tr, te = (spk != s, spk == s)
        mu, sd = (x[tr].mean(0), x[tr].std(0))
        sd[sd < 1e-06] = 1.0
        clf = LogisticRegression(max_iter=3000).fit((x[tr] - mu) / sd, y[tr])
        folds[s] = float((clf.predict((x[te] - mu) / sd) == y[te]).mean())
        print(f'    hold out speaker {s}: acc={folds[s] * 100:6.2f}', flush=True)
    old = [a for s, a in folds.items() if not s.startswith('05')]
    new = [a for s, a in folds.items() if s.startswith('05')]
    return {'folds': folds, 'mean_acc': float(np.mean(list(folds.values()))), 'mean_acc_sessions_1_to_4_speakers': float(np.mean(old)), 'mean_acc_session_5_speakers': float(np.mean(new))}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=('random', 'speaker'), required=True)
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--configs', default=DEFAULT)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out-dir', type=Path, default=Path('results/iemocap_protocols'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    raw = L.R.load_raw(args.pickle)
    vids = raw['tr'] + raw['te']
    session = {v: L.R.PARSE(v)[0] for v in vids}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.configs.split(','):
        print(f'  {name} ({args.protocol})', flush=True)
        blocks = L.blocks_for(raw, L.CONFIGS[name], args.cache_dir)
        if args.protocol == 'random':
            runs = random_cv(raw, blocks, vids, session, list(range(args.seeds)))
            result = {'runs': runs, 'summary': {k: float(np.mean([r[k] for r in runs])) for k in ('acc', 'acc_sessions_1_to_4_utterances', 'acc_session_5_utterances', 'share_sessions_1_to_4')}}
        else:
            result = speaker_cv(raw, blocks, vids, session)
        (args.out_dir / f'{args.protocol}_{name}.json').write_text(json.dumps({'config': name, 'protocol': args.protocol, 'blocks': L.CONFIGS[name], 'pickle': str(args.pickle), **result}, indent=2))
if __name__ == '__main__':
    main()
