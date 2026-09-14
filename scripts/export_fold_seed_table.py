from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
FIELDS = ('experiment', 'reader', 'features', 'test', 'train', 'seed', 'metric', 'value')

def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding='utf-8'))

def loso_bilstm(rows):
    for sub, source in (('iemocap_loso', 'bc-LSTM file'), ('iemocap_loso_mmgcn', 'MMGCN file')):
        for p in sorted((R / sub).glob('*.json')):
            d = load(p)
            for s, fold in d['folds'].items():
                for metric in ('acc', 'w_f1', 'macro_f1', 'kappa', 'mcc'):
                    if metric not in fold:
                        continue
                    for seed, v in zip(d['seeds'], fold[metric]['runs']):
                        rows.append(('loso', 'bi-LSTM', f"{source}: {d['config']}", f'session {s}', 'other four sessions', seed, metric, v))

def session_gap(rows):
    for p in sorted(R.glob('iemocap_session_gap_*.json')):
        d = load(p)
        if 'dialogues' not in d:
            continue
        for name, blocks in d['dialogues'].items():
            for key, test, train in (('session4', 'session 04', 'sessions 01-03'), ('session5', 'session 05', 'sessions 01-04')):
                b = blocks[key]
                n = sum(b['n'])
                for seed, correct in zip(d['seeds'], b['correct_by_seed']):
                    rows.append(('session_gap', 'bi-LSTM', f'{p.stem}: {name}', test, train, seed, 'acc', sum(correct) / n))

def matched_gap(rows):
    for p in sorted((R / 'iemocap_matched_gap').glob('*.json')):
        d = load(p)
        for s, fold in d['held_out'].items():
            for subset in fold['subsets']:
                train = 'sessions ' + ','.join(subset['train_sessions'])
                for seed, run in zip(d['seeds'], subset['runs']):
                    for metric, v in run.items():
                        rows.append(('matched_gap', 'bi-LSTM', p.stem, f'session {s}', train, seed, metric, v))

def loso_linear(rows):
    for p in sorted((R / 'iemocap_loso_linear').glob('*.json')):
        d = load(p)
        for s, fold in d['folds'].items():
            for metric, v in fold.items():
                rows.append(('loso', 'linear probe', d['config'], f'session {s}', 'other four sessions', '', metric, v))

def protocols(rows):
    for p in sorted((R / 'iemocap_protocols').glob('speaker_*.json')):
        d = load(p)
        for spk, v in d['folds'].items():
            rows.append(('leave_one_speaker_out', 'linear probe', d['config'], f'speaker {spk}', 'other nine speakers', '', 'acc', v))
    for p in sorted((R / 'iemocap_protocols').glob('random_*.json')):
        d = load(p)
        for i, run in enumerate(d['runs']):
            for metric in ('acc', 'acc_sessions_1_to_4_utterances', 'acc_session_5_utterances'):
                rows.append(('random_5fold', 'bi-LSTM', d['config'], 'all five folds', 'dialogue-level random folds', run.get('seed', i), metric, run[metric]))

def dialoguecrn(rows):
    for p in sorted((R / 'dialoguecrn_loso').glob('*.json')):
        d = load(p)
        for s, fold in d.get('folds', {}).items():
            for seed, run in enumerate(fold['runs']):
                for metric in ('acc', 'w_f1', 'valid', 'epoch'):
                    rows.append(('loso', 'DialogueCRN', d['config'], f'session {s}', 'other four sessions', seed, metric, run[metric]))

def knn(rows):
    for p in sorted(R.glob('iemocap_knn_diagnostic*.json')):
        d = load(p)
        k = d.get('k', 10)
        for block, rec in d['blocks'].items():
            for s, v in rec['per_session'].items():
                rows.append(('knn_label_agreement', f'{k}-NN', block, f'session {s}', 'same session, other dialogues', '', 'agreement', v))

def fusion(rows):
    for name, added in (('iemocap_context_bert', 'BERT text + released audio and visual'), ('iemocap_context_bert_text+audio', 'BERT text + released audio'), ('iemocap_context_bert_text+video', 'BERT text + released visual')):
        d = load(R / f'{name}.json')
        for cell, metrics in d['cells'].items():
            features = 'BERT text' if cell.startswith('text/') else added
            for metric, rec in metrics.items():
                if isinstance(rec, dict) and 'runs' in rec:
                    for seed, v in enumerate(rec['runs']):
                        rows.append(('fusion_standard_split', cell.split('/')[1], features, 'session 05', 'sessions 01-04', seed, metric, v))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=R / 'fjcai_fold_seed.csv')
    args = ap.parse_args()
    rows: list[tuple] = []
    for part in (loso_bilstm, session_gap, matched_gap, loso_linear, protocols, dialoguecrn, knn, fusion):
        part(rows)
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        w.writerows(rows)
    print(f'written {args.out}: {len(rows)} rows')
if __name__ == '__main__':
    main()
