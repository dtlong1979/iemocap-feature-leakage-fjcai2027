from __future__ import annotations
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score
_spec = importlib.util.spec_from_file_location('loso', Path(__file__).with_name('iemocap_loso.py'))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

def project(blocks, train_vids, all_vids, n_components: int, seed: int=0):
    out = []
    for b in blocks:
        dim = np.asarray(b[all_vids[0]]).shape[1]
        if dim <= n_components:
            out.append(b)
            continue
        pca = PCA(n_components=n_components, random_state=seed)
        pca.fit(np.concatenate([np.asarray(b[v], dtype=np.float32) for v in train_vids]))
        out.append({v: pca.transform(np.asarray(b[v], dtype=np.float32)).astype(np.float32) for v in all_vids})
    return out

def score(raw, blocks, tr, te, maxlen, seeds):
    xt, yt, mt = L.pack(raw, blocks, tr, maxlen)
    xe, ye, me = L.pack(raw, blocks, te, maxlen)
    xt, xe = L.H.standardise(xt, xe, mt)
    y_true = ye[me > 0]
    rows = []
    for sd in seeds:
        out = L.H.run_one(torch.from_numpy(xt), torch.from_numpy(yt), torch.from_numpy(mt), torch.from_numpy(xe), torch.from_numpy(ye), torch.from_numpy(me), 6, sd)
        pred = out[1].argmax(-1).numpy()
        rows.append({'acc': float(accuracy_score(y_true, pred)), 'w_f1': float(f1_score(y_true, pred, average='weighted')), 'macro_f1': float(f1_score(y_true, pred, average='macro')), 'uar': float(recall_score(y_true, pred, average='macro'))})
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--configs', default='released-text,bert')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--project', type=int, default=0)
    ap.add_argument('--out-dir', type=Path, default=Path('results/iemocap_matched_gap'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    seeds = list(range(args.seeds))
    raw = L.R.load_raw(args.pickle)
    vids = raw['tr'] + raw['te']
    session = {v: L.R.PARSE(v)[0] for v in vids}
    maxlen = max((len(raw['lab'][v]) for v in vids))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.configs.split(','):
        base = L.blocks_for(raw, L.CONFIGS[name], args.cache_dir)
        held = {}
        for s in L.SESSIONS:
            te = [v for v in vids if session[v] == s]
            others = [o for o in L.SESSIONS if o != s]
            subsets = []
            for combo in itertools.combinations(others, 3):
                tr = [v for v in vids if session[v] in combo]
                blocks = project(base, tr, vids, args.project) if args.project else base
                rows = score(raw, blocks, tr, te, maxlen, seeds)
                subsets.append({'train_sessions': list(combo), 'runs': rows, **{m: float(np.mean([r[m] for r in rows])) for m in ('acc', 'w_f1', 'macro_f1', 'uar')}})
                print(f"  {name:14s} test {s} train {'+'.join(combo)}: acc={subsets[-1]['acc'] * 100:6.2f} uar={subsets[-1]['uar'] * 100:6.2f}", flush=True)
            held[s] = {m: float(np.mean([sub[m] for sub in subsets])) for m in ('acc', 'w_f1', 'macro_f1', 'uar')}
            held[s]['sd_acc'] = float(np.std([r['acc'] for sub in subsets for r in sub['runs']], ddof=1))
            held[s]['subsets'] = subsets
        summary = {}
        for m in ('acc', 'w_f1', 'macro_f1', 'uar'):
            seen = float(np.mean([held[s][m] for s in L.SESSIONS[:4]]))
            summary[m] = {'sessions_1_to_4': seen, 'session_5': held['05'][m], 'gap': seen - held['05'][m]}
        print(f"  {name:14s} matched gap (acc) {summary['acc']['gap'] * 100:+.2f} points\n", flush=True)
        (args.out_dir / f"{name}{('_pca' + str(args.project) if args.project else '')}.json").write_text(json.dumps({'config': name, 'blocks': L.CONFIGS[name], 'projection': args.project, 'train_sessions_per_fold': 3, 'seeds': seeds, 'pickle': str(args.pickle), 'held_out': held, 'summary': summary}, indent=2))
if __name__ == '__main__':
    main()
