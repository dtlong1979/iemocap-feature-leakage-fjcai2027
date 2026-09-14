from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
_spec = importlib.util.spec_from_file_location('loso', Path(__file__).with_name('iemocap_loso.py'))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
DEFAULT = 'released-text,released-audio,released-video,bert,roberta,deberta'

def utterances(raw, blocks, vids):
    x = np.concatenate([np.concatenate([np.asarray(b[v], dtype=np.float32) for b in blocks], 1) for v in vids])
    y = np.concatenate([np.asarray(raw['lab'][v]) for v in vids])
    return (x, y)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--configs', default=DEFAULT)
    ap.add_argument('--out-dir', type=Path, default=Path('results/iemocap_loso_linear'))
    args = ap.parse_args()
    raw = L.R.load_raw(args.pickle)
    vids = raw['tr'] + raw['te']
    session = {v: L.R.PARSE(v)[0] for v in vids}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.configs.split(','):
        blocks = L.blocks_for(raw, L.CONFIGS[name], args.cache_dir)
        folds = {}
        for s in L.SESSIONS:
            xt, yt = utterances(raw, blocks, [v for v in vids if session[v] != s])
            xe, ye = utterances(raw, blocks, [v for v in vids if session[v] == s])
            mu, sd = (xt.mean(0), xt.std(0))
            sd[sd < 1e-06] = 1.0
            clf = LogisticRegression(max_iter=3000)
            clf.fit((xt - mu) / sd, yt)
            pred = clf.predict((xe - mu) / sd)
            folds[s] = {'acc': float(accuracy_score(ye, pred)), 'w_f1': float(f1_score(ye, pred, average='weighted')), 'macro_f1': float(f1_score(ye, pred, average='macro')), 'uar': float(recall_score(ye, pred, average='macro'))}
            print(f"  {name:15s} hold out session {s}: acc={folds[s]['acc'] * 100:6.2f}", flush=True)
        accs = [folds[s]['acc'] for s in L.SESSIONS]
        summary = {'loso_mean_acc': float(np.mean(accs)), 'standard_split_acc': folds['05']['acc'], 'folds_1_to_4_mean_acc': float(np.mean(accs[:4])), 'inflation_acc': float(np.mean(accs) - folds['05']['acc'])}
        print(f"  {name:15s} 5-fold mean {summary['loso_mean_acc'] * 100:.2f}  vs standard split {summary['standard_split_acc'] * 100:.2f}\n", flush=True)
        (args.out_dir / f'{name}.json').write_text(json.dumps({'config': name, 'reader': 'multinomial logistic regression, per utterance', 'pickle': str(args.pickle), 'folds': folds, 'summary': summary}, indent=2))
if __name__ == '__main__':
    main()
