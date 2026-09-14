from __future__ import annotations
import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
_spec = importlib.util.spec_from_file_location('linear', Path(__file__).with_name('iemocap_loso_linear.py'))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
L = M.L
SPLITS = (('04', ('01', '02', '03')), ('05', ('01', '02', '03', '04')))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--configs', default='released-text,released-audio,released-video,bert')
    ap.add_argument('--out', type=Path, default=Path('results/iemocap_session_gap_linear.json'))
    args = ap.parse_args()
    raw = L.R.load_raw(args.pickle)
    vids = raw['tr'] + raw['te']
    session = {v: L.R.PARSE(v)[0] for v in vids}
    blocks_out = {}
    for name in args.configs.split(','):
        start = time.perf_counter()
        blocks = L.blocks_for(raw, L.CONFIGS[name], args.cache_dir)
        acc = {}
        for test, train in SPLITS:
            xt, yt = M.utterances(raw, blocks, [v for v in vids if session[v] in train])
            xe, ye = M.utterances(raw, blocks, [v for v in vids if session[v] == test])
            mu, sd = (xt.mean(0), xt.std(0))
            sd[sd < 1e-06] = 1.0
            clf = LogisticRegression(max_iter=3000).fit((xt - mu) / sd, yt)
            acc[test] = float((clf.predict((xe - mu) / sd) == ye).mean())
        seconds = time.perf_counter() - start
        blocks_out[name] = {'acc_session4': acc['04'], 'acc_session5': acc['05'], 'gap': acc['04'] - acc['05'], 'seconds': seconds, 'dim': int(np.asarray(blocks[0][vids[0]]).shape[1]) if len(blocks) == 1 else None}
        print(f"  {name:15s} a4={acc['04'] * 100:6.2f} a5={acc['05'] * 100:6.2f} g={(acc['04'] - acc['05']) * 100:+6.2f}  {seconds:5.1f}s", flush=True)
    ref = blocks_out.get('released-audio', {}).get('gap')
    if ref is not None:
        for rec in blocks_out.values():
            rec['gap_minus_audio'] = rec['gap'] - ref
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'reader': 'multinomial logistic regression, per utterance', 'utterances': int(sum((len(raw['lab'][v]) for v in vids))), 'cpu': platform.processor(), 'blocks': blocks_out}, indent=2))
    print(f'written {args.out}')
if __name__ == '__main__':
    main()
