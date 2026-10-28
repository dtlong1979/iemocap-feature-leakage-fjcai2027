from __future__ import annotations
import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
H = _load('headroom_main', 'mosi_mosei_headroom.py')
L = _load('loso', 'iemocap_loso.py')
SESSIONS = ('01', '02', '03', '04', '05')
TOKEN = re.compile("[a-z']+")

class TextCNN(nn.Module):

    def __init__(self, vocab: int, num_classes: int, emb: int=300, maps: int=50, widths: tuple[int, ...]=(3, 4, 5), feat: int=100, dropout: float=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, maps, w, padding=w - 1) for w in widths])
        self.drop = nn.Dropout(dropout)
        self.dense = nn.Linear(maps * len(widths), feat)
        self.out = nn.Linear(feat, num_classes)

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(ids).transpose(1, 2)
        h = torch.cat([torch.relu(c(h)).max(dim=2).values for c in self.convs], dim=1)
        return torch.relu(self.dense(self.drop(h)))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(self.features(ids)))

def build_vocab(raw: dict, vids: list[str], min_count: int=2) -> dict[str, int]:
    counts = Counter((w for v in vids for s in raw['sent'][v] for w in TOKEN.findall(str(s).lower())))
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, n in counts.items():
        if n >= min_count:
            vocab[word] = len(vocab)
    return vocab

def encode_ids(raw: dict, vids: list[str], vocab: dict[str, int], maxlen: int=40):
    ids, labels, index = ([], [], [])
    for v in vids:
        for i, s in enumerate(raw['sent'][v]):
            toks = [vocab.get(w, 1) for w in TOKEN.findall(str(s).lower())][:maxlen]
            ids.append(toks + [0] * (maxlen - len(toks)))
            labels.append(int(raw['lab'][v][i]))
            index.append((v, i))
    return (np.asarray(ids, dtype=np.int64), np.asarray(labels, dtype=np.int64), index)

def train_extractor(ids, labels, vocab_size, num_classes, seed, device, epochs=20, patience=4, batch=64, valid_rate=0.1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    cut = max(1, int(valid_rate * len(order)))
    va, tr = (order[:cut], order[cut:])
    model = TextCNN(vocab_size, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.from_numpy(ids).to(device)
    y = torch.from_numpy(labels).to(device)
    best, best_state, bad = (-1.0, None, 0)
    for _ in range(epochs):
        model.train()
        for i in range(0, len(tr), batch):
            sel = torch.from_numpy(tr[i:i + batch]).to(device)
            opt.zero_grad()
            loss_fn(model(x[sel]), y[sel]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sel = torch.from_numpy(va).to(device)
            acc = (model(x[sel]).argmax(-1) == y[sel]).float().mean().item()
        if acc > best:
            best, bad = (acc, 0)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return (model, best)

@torch.no_grad()
def extract(model, ids, index, raw, device, batch=256):
    out = {}
    x = torch.from_numpy(ids).to(device)
    feats = torch.cat([model.features(x[i:i + batch]).cpu() for i in range(0, len(ids), batch)])
    for (v, i), row in zip(index, feats.numpy()):
        out.setdefault(v, np.zeros((len(raw['lab'][v]), feats.shape[1]), dtype=np.float32))
        out[v][i] = row
    return out

def read_with_bilstm(raw, blocks, train_vids, test_vids, maxlen, seeds):
    outs = []
    for seed, block in zip(seeds, blocks):
        xt, yt, mt = L.pack(raw, [block], train_vids, maxlen)
        xe, ye, me = L.pack(raw, [block], test_vids, maxlen)
        xt, xe = H.standardise(xt, xe, mt)
        outs.append(H.run_one(torch.from_numpy(xt), torch.from_numpy(yt), torch.from_numpy(mt), torch.from_numpy(xe), torch.from_numpy(ye), torch.from_numpy(me), 6, seed))
    fold = H.aggregate([o[0] for o in outs])
    y_true = ye[me > 0]
    dlg = np.concatenate([[i] * len(raw['lab'][v]) for i, v in enumerate(test_vids)])
    fold['dialogues'] = {'ids': list(test_vids), 'n': [int(len(raw['lab'][v])) for v in test_vids], 'correct_by_seed': [np.bincount(dlg, weights=o[1].argmax(-1).numpy() == y_true, minlength=len(test_vids)).astype(int).tolist() for o in outs]}
    return fold

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', type=Path, default=Path('results/iemocap_refit_control.json'))
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    seeds = list(range(args.seeds))
    raw = L.R.load_raw(args.pickle)
    vids = raw['tr'] + raw['te']
    session = {v: L.R.PARSE(v)[0] for v in vids}
    maxlen = max((len(raw['lab'][v]) for v in vids))
    vocab = build_vocab(raw, vids)
    ids, labels, index = encode_ids(raw, vids, vocab)
    pos = {v: [k for k, (vv, _) in enumerate(index) if vv == v] for v in vids}
    print(f'device {device}; {len(vocab)} word types, {len(ids)} utterances', flush=True)
    result = {'conditions': {}, 'vocabulary': len(vocab), 'seeds': seeds, 'seed_varies': 'extractor and reader', 'extractor': 'text CNN, 300-d learned embeddings, widths 3/4/5 with 50 maps each, 100-d dense, six-way softmax, Adam 1e-3, batch 64, <=20 epochs, patience 4 on 10% of the fitting utterances'}
    for condition in ('fixed', 'refit'):
        folds, extractor_acc = ({}, {})
        if condition == 'fixed':
            train_idx = np.concatenate([pos[v] for v in vids if session[v] != '05'])
            blocks = []
            for seed in seeds:
                model, acc = train_extractor(ids[train_idx], labels[train_idx], len(vocab), 6, seed, device)
                blocks.append(extract(model, ids, index, raw, device))
                extractor_acc[f'sessions 01-04, seed {seed}'] = acc
        for s in SESSIONS:
            te = [v for v in vids if session[v] == s]
            tr = [v for v in vids if session[v] != s]
            if condition == 'refit':
                train_idx = np.concatenate([pos[v] for v in tr])
                blocks = []
                for seed in seeds:
                    model, acc = train_extractor(ids[train_idx], labels[train_idx], len(vocab), 6, seed, device)
                    blocks.append(extract(model, ids, index, raw, device))
                    extractor_acc[f'without session {s}, seed {seed}'] = acc
            folds[s] = read_with_bilstm(raw, blocks, tr, te, maxlen, seeds)
            print(f"  {condition:5s} hold out session {s}: acc={folds[s]['acc']['mean'] * 100:6.2f}", flush=True)
        accs = [folds[s]['acc']['mean'] for s in SESSIONS]
        result['conditions'][condition] = {'folds': folds, 'extractor_validation_acc': extractor_acc, 'summary': {'loso_mean_acc': float(np.mean(accs)), 'standard_split_acc': folds['05']['acc']['mean'], 'folds_1_to_4_mean_acc': float(np.mean(accs[:4])), 'difference_acc': float(np.mean(accs) - folds['05']['acc']['mean'])}}
        print(f"  {condition:5s} LOSO {np.mean(accs) * 100:.2f} vs standard {folds['05']['acc']['mean'] * 100:.2f}\n", flush=True)
    d = {c: result['conditions'][c]['summary']['difference_acc'] * 100 for c in ('fixed', 'refit')}
    result['leak_estimate_points'] = d['fixed'] - d['refit']
    print(f"fixed {d['fixed']:+.2f}, refit {d['refit']:+.2f}, difference {d['fixed'] - d['refit']:+.2f} points")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f'written {args.out}')
if __name__ == '__main__':
    main()
