from __future__ import annotations
import argparse
import builtins
import hashlib
import importlib
import json
import pickle
from pathlib import Path
import numpy as np
FILES = {'bc-LSTM': 'IEMOCAP_features_raw.pkl', 'DialogueGCN': 'iemocap_features_dialoguegcn.pkl', 'MMGCN': 'iemocap_features_mmgcn.pkl'}
BLOCKS = {'text': 3, 'audio': 4, 'video': 5}
ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'), ('__builtin__', 'set')}

class _Safe(pickle.Unpickler):

    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f'refused {module}.{name}')
        if module == '__builtin__':
            return getattr(builtins, name)
        return getattr(importlib.import_module(module), name)

def load(path: Path):
    with open(path, 'rb') as fh:
        return _Safe(fh, encoding='latin1').load()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def compare(a, b, block: int) -> dict:
    first = next(iter(a[block]))
    dims = (np.asarray(a[block][first]).shape[1], np.asarray(b[block][first]).shape[1])
    if dims[0] != dims[1]:
        return {'identical': False, 'dims': dims}
    same, n, worst = (0, 0, 0.0)
    for v in a[block]:
        d = np.abs(np.asarray(a[block][v], dtype=np.float64) - np.asarray(b[block][v], dtype=np.float64)).max(axis=1)
        same += int((d < 1e-06).sum())
        n += len(d)
        worst = max(worst, float(d.max()))
    return {'identical': same == n, 'dims': dims, 'utterances_identical': same, 'utterances': n, 'max_abs_diff': worst}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('results/iemocap_feature_provenance.json'))
    args = ap.parse_args()
    data = {k: load(args.dir / f) for k, f in FILES.items()}
    ref = data['bc-LSTM']
    out = {'files': {k: {'file': f, 'sha256': sha256(args.dir / f)} for k, f in FILES.items()}, 'structure': {}, 'blocks': {}}
    for k in ('DialogueGCN', 'MMGCN'):
        d = data[k]
        out['structure'][k] = {'same_dialogues': set(ref[0]) == set(d[0]), 'same_utterance_ids': all((list(ref[0][v]) == list(d[0][v]) for v in ref[0])), 'same_labels': all((list(ref[2][v]) == list(d[2][v]) for v in ref[2])), 'same_split': sorted(ref[7]) == sorted(d[7]) and sorted(ref[8]) == sorted(d[8])}
        out['blocks'][k] = {name: compare(ref, d, i) for name, i in BLOCKS.items()}
    for k in ('DialogueGCN', 'MMGCN'):
        print(f"{k} vs bc-LSTM: {out['structure'][k]}")
        for name, r in out['blocks'][k].items():
            detail = f"{r['utterances_identical']}/{r['utterances']} identical" if 'utterances' in r else f"different dims {r['dims']}"
            print(f"    {name:5s} {('IDENTICAL' if r['identical'] else 'differs')}: {detail}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f'written to {args.out}')
if __name__ == '__main__':
    main()
