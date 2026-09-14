from __future__ import annotations
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
_spec = importlib.util.spec_from_file_location('loso', Path(__file__).with_name('iemocap_loso.py'))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
HP = dict(base_model='LSTM', base_layer=2, hidden_size=100, dropout=0.2, reason_steps=[3, 4], lr=0.0001, l2=0.0002, batch_size=32, gamma=0)
CLASS_WEIGHTS = [1 / 0.087178797, 1 / 0.145836136, 1 / 0.229786089, 1 / 0.148392305, 1 / 0.140051123, 1 / 0.24875555]

def dialogues(raw, blocks, vids):
    out = []
    for v in vids:
        x = np.concatenate([np.asarray(blocks[i][v], dtype=np.float32) for i in range(len(blocks))], 1)
        spk = np.array([[1.0, 0.0] if s == 'M' else [0.0, 1.0] for s in raw['spk'][v]], dtype=np.float32)
        out.append((x, np.asarray(raw['lab'][v], dtype=np.int64), spk))
    return out

def standardise(train, *others):
    flat = np.concatenate([x for x, _, _ in train])
    mu, sd = (flat.mean(0), flat.std(0))
    sd[sd < 1e-06] = 1.0
    out = [[((x - mu) / sd).astype(np.float32), y, s] for x, y, s in train]
    return [out] + [[[((x - mu) / sd).astype(np.float32), y, s] for x, y, s in o] for o in others]

def batches(data, size, rng=None):
    idx = list(range(len(data)))
    if rng is not None:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        chunk = [data[j] for j in idx[i:i + size]]
        lengths = [len(c[1]) for c in chunk]
        m, b, d = (max(lengths), len(chunk), chunk[0][0].shape[1])
        u = np.zeros((m, b, d), dtype=np.float32)
        q = np.zeros((m, b, 2), dtype=np.float32)
        y = np.zeros((b, m), dtype=np.int64)
        for j, (x, lab, spk) in enumerate(chunk):
            n = len(lab)
            u[:n, j], q[:n, j], y[j, :n] = (x, spk, lab)
        yield (torch.from_numpy(u), torch.from_numpy(q), torch.from_numpy(y), lengths)

def evaluate(model, data, loss_f, size, device):
    model.eval()
    preds, labels, losses = ([], [], [])
    with torch.no_grad():
        for u, q, y, lengths in batches(data, size):
            u, q, y = (u.to(device), q.to(device), y.to(device))
            log_prob = model(u, q, lengths)
            flat = torch.cat([y[j][:lengths[j]] for j in range(len(lengths))])
            losses.append(float(loss_f(log_prob, flat)))
            preds.append(log_prob.argmax(1).cpu().numpy())
            labels.append(flat.cpu().numpy())
    p, t = (np.concatenate(preds), np.concatenate(labels))
    return (float(np.mean(losses)), float(accuracy_score(t, p)), float(f1_score(t, p, average='weighted')))

def run_fold(M, F, train, valid, test, seed, args, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = M.DialogueCRN(base_model=HP['base_model'], base_layer=HP['base_layer'], input_size=train[0][0].shape[1], hidden_size=HP['hidden_size'], n_speakers=2, n_classes=6, dropout=HP['dropout'], cuda_flag=device.type == 'cuda', reason_steps=HP['reason_steps']).to(device)
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    loss_f = F.FocalLoss(gamma=HP['gamma'], alpha=weights)
    opt = torch.optim.Adam(model.parameters(), lr=HP['lr'], weight_decay=HP['l2'])
    rng = random.Random(seed)
    best = {'valid': -1.0, 'acc': 0.0, 'w_f1': 0.0, 'epoch': -1}
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        for u, q, y, lengths in batches(train, HP['batch_size'], rng):
            opt.zero_grad()
            u, q, y = (u.to(device), q.to(device), y.to(device))
            log_prob = model(u, q, lengths)
            flat = torch.cat([y[j][:lengths[j]] for j in range(len(lengths))])
            loss = loss_f(log_prob, flat)
            loss.backward()
            opt.step()
        _, _, v_f1 = evaluate(model, valid, loss_f, HP['batch_size'], device)
        if v_f1 > best['valid']:
            _, acc, w_f1 = evaluate(model, test, loss_f, HP['batch_size'], device)
            best = {'valid': v_f1, 'acc': acc, 'w_f1': w_f1, 'epoch': epoch}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    return best

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--code-dir', type=Path, required=True)
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--configs', default='released-text,bert')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--valid-rate', type=float, default=0.1)
    ap.add_argument('--no-standardise', dest='standardise', action='store_false')
    ap.add_argument('--sessions', default=','.join(L.SESSIONS))
    ap.add_argument('--out-dir', type=Path, default=Path('results/dialoguecrn_loso'))
    args = ap.parse_args()
    sys.path.insert(0, str(args.code_dir))
    try:
        import torch_scatter
    except ImportError:
        import types
        from torch_geometric.utils import scatter as _scatter
        shim = types.ModuleType('torch_scatter')
        shim.scatter_add = lambda src, index, dim=0, dim_size=None: _scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
        sys.modules['torch_scatter'] = shim
    M = importlib.import_module('model')
    F = importlib.import_module('loss')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    print(f'device {device}, model from {args.code_dir}', flush=True)
    raw = L.R.load_raw(args.pickle)
    vids = raw['tr'] + raw['te']
    session = {v: L.R.PARSE(v)[0] for v in vids}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.configs.split(','):
        blocks = L.blocks_for(raw, L.CONFIGS[name], args.cache_dir)
        folds = {}
        for s in args.sessions.split(','):
            te = [v for v in vids if session[v] == s]
            tr = [v for v in vids if session[v] != s]
            rng = random.Random(0)
            rng.shuffle(tr)
            cut = max(1, int(args.valid_rate * len(tr)))
            train = dialogues(raw, blocks, tr[cut:])
            valid = dialogues(raw, blocks, tr[:cut])
            test = dialogues(raw, blocks, te)
            if args.standardise:
                train, valid, test = standardise(train, valid, test)
            runs = []
            for seed in range(args.seeds):
                t0 = time.time()
                r = run_fold(M, F, train, valid, test, seed, args, device)
                r['seconds'] = round(time.time() - t0, 1)
                runs.append(r)
                print(f"  {name:14s} session {s} seed {seed}: acc={r['acc'] * 100:6.2f} wF1={r['w_f1'] * 100:6.2f} (epoch {r['epoch']}, {r['seconds']}s)", flush=True)
            folds[s] = {'acc': float(np.mean([r['acc'] for r in runs])), 'w_f1': float(np.mean([r['w_f1'] for r in runs])), 'runs': runs}
        accs = [folds[s]['acc'] for s in folds]
        contaminated = [folds[s]['acc'] for s in folds if s != '05']
        summary = {'loso_mean_acc': float(np.mean(accs)), 'standard_split_acc': folds.get('05', {}).get('acc'), 'folds_1_to_4_mean_acc': float(np.mean(contaminated)) if contaminated else None}
        if summary['standard_split_acc'] is not None:
            summary['inflation_acc'] = summary['loso_mean_acc'] - summary['standard_split_acc']
        print(f'  {name:14s} summary {json.dumps(summary)}\n', flush=True)
        (args.out_dir / f'{name}.json').write_text(json.dumps({'config': name, 'blocks': L.CONFIGS[name], 'model': 'DialogueCRN (ACL 2021)', 'hyperparameters': HP, 'standardised': args.standardise, 'valid_rate': args.valid_rate, 'epochs': args.epochs, 'patience': args.patience, 'pickle': str(args.pickle), 'folds': folds, 'summary': summary}, indent=2))
if __name__ == '__main__':
    main()
