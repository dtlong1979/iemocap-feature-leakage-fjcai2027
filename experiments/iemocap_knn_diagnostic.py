from __future__ import annotations
import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import numpy as np
_spec = importlib.util.spec_from_file_location('loso', Path(__file__).with_name('iemocap_loso.py'))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

def knn_by_session(x, y, session, dialogue, k: int) -> dict:
    x = (x - x.mean(0)) / np.where(x.std(0) < 1e-06, 1.0, x.std(0))
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    out = {}
    for s in L.SESSIONS:
        idx = np.flatnonzero(session == s)
        sim = x[idx] @ x[idx].T
        sim[dialogue[idx][:, None] == dialogue[idx][None, :]] = -np.inf
        nn = np.argpartition(-sim, k, axis=1)[:, :k]
        votes = y[idx][nn]
        pred = np.array([np.bincount(row, minlength=6).argmax() for row in votes])
        out[s] = float((pred == y[idx]).mean())
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--mmgcn-pickle', type=Path, required=True)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--out', type=Path, default=Path('results/iemocap_knn_diagnostic.json'))
    args = ap.parse_args()
    raw = L.R.load_raw(args.pickle)
    mm = L.R.load_raw(args.mmgcn_pickle)
    vids = raw['tr'] + raw['te']
    blocks = {'released text': raw['txt'], 'released visual': raw['vis'], 'released audio': raw['aud'], 'MMGCN audio': mm['aud'], 'MMGCN visual': mm['vis']}
    for name, tag in L.CACHES.items():
        with open(args.cache_dir / f'iemocap_text_frozen_{tag}.pkl', 'rb') as fh:
            blocks[f'frozen {name}'] = pickle.load(fh)
    y = np.concatenate([np.asarray(raw['lab'][v]) for v in vids])
    session = np.concatenate([[L.R.PARSE(v)[0]] * len(raw['lab'][v]) for v in vids])
    dialogue = np.concatenate([[i] * len(raw['lab'][v]) for i, v in enumerate(vids)])
    result = {}
    for name, b in blocks.items():
        x = np.concatenate([np.asarray(b[v], dtype=np.float32) for v in vids])
        per = knn_by_session(x, y, session, dialogue, args.k)
        gap = float(np.mean([per[s] for s in L.SESSIONS[:4]]) - per['05'])
        result[name] = {'per_session': per, 'sessions_1_to_4_minus_5': gap}
        print(f'  {name:18s} ' + ' '.join((f'{per[s] * 100:6.2f}' for s in L.SESSIONS)) + f'   gap {gap * 100:+6.2f}', flush=True)
    args.out.write_text(json.dumps({'k': args.k, 'blocks': result}, indent=2))
    print(f'written {args.out}')
if __name__ == '__main__':
    main()
