from __future__ import annotations
import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import numpy as np
import torch

def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
H = _load('headroom_main', 'mosi_mosei_headroom.py')
R = _load('reextract', 'iemocap_text_reextract.py')
SESSIONS = ('01', '02', '03', '04', '05')
EMOTIONS = ('happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated')
CACHES = {'bert': 'bert-base-uncased', 'roberta': 'roberta-base', 'deberta': 'microsoft_deberta-v3-base'}
CONFIGS = {'released-text': ['txt'], 'released-audio': ['aud'], 'released-video': ['vis'], 'released-tav': ['txt', 'aud', 'vis'], 'bert': ['bert'], 'roberta': ['roberta'], 'deberta': ['deberta'], 'bert+audio': ['bert', 'aud']}

def blocks_for(raw: dict, names: list[str], cache_dir: Path) -> list[dict]:
    out = []
    for n in names:
        if n in CACHES:
            with open(cache_dir / f'iemocap_text_frozen_{CACHES[n]}.pkl', 'rb') as fh:
                out.append(pickle.load(fh))
        else:
            out.append(raw[n])
    return out

def pack(raw: dict, blocks: list[dict], vids: list[str], maxlen: int):
    dims = [np.asarray(b[vids[0]]).shape[1] for b in blocks]
    x = np.zeros((len(vids), maxlen, sum(dims)), dtype=np.float32)
    y = np.zeros((len(vids), maxlen), dtype=np.int64)
    m = np.zeros((len(vids), maxlen), dtype=np.float32)
    for i, v in enumerate(vids):
        n = len(raw['lab'][v])
        x[i, :n] = np.concatenate([np.asarray(b[v], dtype=np.float32) for b in blocks], 1)
        y[i, :n] = raw['lab'][v]
        m[i, :n] = 1.0
    return (x, y, m)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--configs', default=','.join(CONFIGS))
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out-dir', type=Path, default=Path('results/iemocap_loso'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    seeds = list(range(args.seeds))
    raw = R.load_raw(args.pickle)
    vids = raw['tr'] + raw['te']
    session = {v: R.PARSE(v)[0] for v in vids}
    assert {session[v] for v in raw['te']} == {'05'}
    maxlen = max((len(raw['lab'][v]) for v in vids))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.configs.split(','):
        blocks = blocks_for(raw, CONFIGS[name], args.cache_dir)
        folds = {}
        for s in SESSIONS:
            te = [v for v in vids if session[v] == s]
            tr = [v for v in vids if session[v] != s]
            xt, yt, mt = pack(raw, blocks, tr, maxlen)
            xe, ye, me = pack(raw, blocks, te, maxlen)
            xt, xe = H.standardise(xt, xe, mt)
            outs = [H.run_one(torch.from_numpy(xt), torch.from_numpy(yt), torch.from_numpy(mt), torch.from_numpy(xe), torch.from_numpy(ye), torch.from_numpy(me), 6, sd) for sd in seeds]
            folds[s] = H.aggregate([o[0] for o in outs])
            y_true = ye[me > 0]
            folds[s]['recall'] = {emo: float(np.mean([(o[1].argmax(-1).numpy()[y_true == k] == k).mean() for o in outs])) for k, emo in enumerate(EMOTIONS) if (y_true == k).any()}
            print(f"  {name:15s} hold out session {s}: acc={folds[s]['acc']['mean'] * 100:6.2f} wF1={folds[s]['w_f1']['mean'] * 100:6.2f}", flush=True)
        accs = [folds[s]['acc']['mean'] for s in SESSIONS]
        summary = {'loso_mean_acc': float(np.mean(accs)), 'standard_split_acc': folds['05']['acc']['mean'], 'folds_1_to_4_mean_acc': float(np.mean(accs[:4])), 'inflation_acc': float(np.mean(accs) - folds['05']['acc']['mean'])}
        print(f"  {name:15s} 5-fold mean {summary['loso_mean_acc'] * 100:.2f}  vs standard split {summary['standard_split_acc'] * 100:.2f}  (folds 1-4 mean {summary['folds_1_to_4_mean_acc'] * 100:.2f})\n", flush=True)
        (args.out_dir / f'{name}.json').write_text(json.dumps({'config': name, 'blocks': CONFIGS[name], 'pickle': str(args.pickle), 'seeds': seeds, 'folds': folds, 'summary': summary}, indent=2))
if __name__ == '__main__':
    main()
