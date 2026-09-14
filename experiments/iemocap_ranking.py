from __future__ import annotations
import json
from pathlib import Path
from scipy.stats import kendalltau
ROWS = {'released text': 'results/iemocap_loso/released-text.json', 'released visual': 'results/iemocap_loso/released-video.json', 'released audio': 'results/iemocap_loso/released-audio.json', 'released text+audio+visual': 'results/iemocap_loso/released-tav.json', 'MMGCN audio': 'results/iemocap_loso_mmgcn/released-audio.json', 'MMGCN visual': 'results/iemocap_loso_mmgcn/released-video.json', 'MMGCN text+audio+visual': 'results/iemocap_loso_mmgcn/released-tav.json', 'frozen BERT': 'results/iemocap_loso/bert.json', 'frozen RoBERTa': 'results/iemocap_loso/roberta.json', 'frozen DeBERTa-v3': 'results/iemocap_loso/deberta.json', 'frozen BERT + released audio': 'results/iemocap_loso/bert+audio.json'}

def main() -> None:
    rows = {}
    for name, path in ROWS.items():
        s = json.loads(Path(path).read_text())['summary']
        rows[name] = (s['loso_mean_acc'], s['standard_split_acc'])
    names = list(rows)
    loso = [rows[n][0] for n in names]
    std = [rows[n][1] for n in names]
    tau, p = kendalltau(loso, std)
    by_loso = sorted(names, key=lambda n: -rows[n][0])
    by_std = sorted(names, key=lambda n: -rows[n][1])
    print(f"{'configuration':30s} {'LOSO':>6s} {'rank':>4s} {'std':>6s} {'rank':>4s}")
    for n in by_loso:
        print(f'{n:30s} {rows[n][0] * 100:6.2f} {by_loso.index(n) + 1:4d} {rows[n][1] * 100:6.2f} {by_std.index(n) + 1:4d}')
    print(f'\nKendall tau = {tau:+.3f} (p = {p:.3f}); winner under LOSO: {by_loso[0]}; under the standard split: {by_std[0]}')
    out = {'rows': {n: {'loso_mean_acc': rows[n][0], 'standard_split_acc': rows[n][1], 'rank_loso': by_loso.index(n) + 1, 'rank_standard': by_std.index(n) + 1} for n in names}, 'kendall_tau': tau, 'p': p}
    Path('results/iemocap_ranking.json').write_text(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
