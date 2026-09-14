from __future__ import annotations
import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
_spec = importlib.util.spec_from_file_location('reextract', Path(__file__).with_name('iemocap_text_reextract.py'))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
EMOTIONS = ('hap', 'sad', 'neu', 'ang', 'exc', 'fru')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('results/iemocap_session_stats.json'))
    args = ap.parse_args()
    raw = R.load_raw(args.pickle)
    per = {}
    for v in raw['tr'] + raw['te']:
        s = R.PARSE(v)[0]
        d = per.setdefault(s, {'dialogues': 0, 'labels': Counter()})
        d['dialogues'] += 1
        d['labels'].update((int(y) for y in raw['lab'][v]))
    out = {}
    print(f"{'session':>7} {'dial.':>6} {'utt.':>6}  " + ' '.join((f'{e:>6}' for e in EMOTIONS)) + '   majority')
    for s in sorted(per):
        c, n = (per[s]['labels'], sum(per[s]['labels'].values()))
        shares = {EMOTIONS[k]: c[k] / n for k in range(len(EMOTIONS))}
        top = max(shares, key=shares.get)
        out[s] = {'dialogues': per[s]['dialogues'], 'utterances': n, 'shares': shares, 'majority_class': top, 'majority_share': shares[top]}
        print(f"{s:>7} {per[s]['dialogues']:6d} {n:6d}  " + ' '.join((f'{shares[e] * 100:6.1f}' for e in EMOTIONS)) + f'   {top} {shares[top] * 100:.1f}%')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f'written to {args.out}')
if __name__ == '__main__':
    main()
