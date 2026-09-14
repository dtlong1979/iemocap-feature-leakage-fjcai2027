from __future__ import annotations
import argparse
import pickle
import re
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_hp = importlib.util.spec_from_file_location('headroom_main', Path(__file__).with_name('mosi_mosei_headroom.py'))
H = importlib.util.module_from_spec(_hp)
_hp.loader.exec_module(H)
PARSE = lambda v: re.match('Ses(\\d+)([FM])_(impro|script)(\\w*)', v).groups()

def load_raw(path: Path):
    with open(path, 'rb') as fh:
        ids, spk, lab, txt, aud, vis, sent, tr, te = pickle.load(fh, encoding='latin1')
    return dict(ids=ids, spk=spk, lab=lab, txt=txt, aud=aud, vis=vis, sent=sent, tr=sorted(tr), te=sorted(te))

@torch.no_grad()
def encode(raw: dict, model_name: str, device: str, batch: int=64) -> dict:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == 'cuda':
        model = model.half()
    order = raw['tr'] + raw['te']
    flat, index = ([], [])
    for v in order:
        for i, s in enumerate(raw['sent'][v]):
            flat.append(str(s))
            index.append((v, i))
    vecs = []
    for i in range(0, len(flat), batch):
        enc = tok(flat[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        out = model(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).to(out.dtype)
        vecs.append(((out * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu())
        if i // batch % 20 == 0:
            print(f'    {i:6d}/{len(flat)}', flush=True)
    vecs = torch.cat(vecs).numpy()
    per_video = {v: np.zeros((len(raw['sent'][v]), vecs.shape[1]), dtype=np.float32) for v in order}
    for (v, i), row in zip(index, vecs):
        per_video[v][i] = row
    print(f'    encoded {len(flat)} utterances -> {vecs.shape[1]}-d')
    return per_video

def pack(raw: dict, text_src: dict, vids, maxlen: int, channel: str):
    src = {'text': text_src, 'audio': raw['aud'], 'video': raw['vis']}[channel]
    dim = np.asarray(src[vids[0]]).shape[1]
    x = np.zeros((len(vids), maxlen, dim), dtype=np.float32)
    y = np.zeros((len(vids), maxlen), dtype=np.int64)
    m = np.zeros((len(vids), maxlen), dtype=np.float32)
    for i, v in enumerate(vids):
        n = len(raw['lab'][v])
        x[i, :n] = np.asarray(src[v], dtype=np.float32)
        y[i, :n] = raw['lab'][v]
        m[i, :n] = 1.0
    return (x, y, m)

def evaluate(raw, text_src, channel, seeds):
    maxlen = max((len(raw['lab'][v]) for v in raw['tr'] + raw['te']))
    fit4 = [v for v in raw['tr'] if PARSE(v)[0] != '04']
    val4 = [v for v in raw['tr'] if PARSE(v)[0] == '04']

    def run(tr_v, te_v):
        xt, yt, mt = pack(raw, text_src, tr_v, maxlen, channel)
        xe, ye, me = pack(raw, text_src, te_v, maxlen, channel)
        xt, xe = H.standardise(xt, xe, mt)
        y_true = ye[me > 0]
        dlg = np.concatenate([[i] * len(raw['lab'][v]) for i, v in enumerate(te_v)])
        accs, correct = ([], [])
        for s in seeds:
            out = H.run_one(torch.from_numpy(xt), torch.from_numpy(yt), torch.from_numpy(mt), torch.from_numpy(xe), torch.from_numpy(ye), torch.from_numpy(me), 6, s)
            accs.append(out[0]['acc'])
            hit = out[1].argmax(-1).numpy() == y_true
            correct.append(np.bincount(dlg, weights=hit, minlength=len(te_v)).astype(int).tolist())
        detail = {'ids': list(te_v), 'n': [int(len(raw['lab'][v])) for v in te_v], 'correct_by_seed': correct}
        return (float(np.mean(accs)), detail)
    (a4, d4), (a5, d5) = (run(fit4, val4), run(raw['tr'], raw['te']))
    return (a4, a5, {'session4': d4, 'session5': d5})

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', type=Path, required=True)
    ap.add_argument('--model', default='bert-base-uncased')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--cache', type=Path, default=Path('data-features/iemocap_text_frozen.pkl'))
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    raw = load_raw(args.pickle)
    seeds = list(range(args.seeds))
    print(f'device: {device}' + (f'  ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))
    if args.cache.exists():
        with open(args.cache, 'rb') as fh:
            new_text = pickle.load(fh)
        print(f'loaded cached features from {args.cache}')
    else:
        print(f'encoding transcripts with a frozen {args.model} ...')
        new_text = encode(raw, args.model, device)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache, 'wb') as fh:
            pickle.dump(new_text, fh)
        print(f'cached to {args.cache}')
    dim_old = np.asarray(raw['txt'][raw['tr'][0]]).shape[1]
    dim_new = new_text[raw['tr'][0]].shape[1]
    print(f'\ntext features: released CNN {dim_old}-d, frozen {args.model} {dim_new}-d')
    print(f'{args.seeds} seeds, same architecture and protocol for both\n')
    print(f"{'text features':>34} {'hold out ses.4':>15} {'hold out ses.5':>15} {'gap':>8}")
    rows, details = ({}, {})
    for name, src in (('released CNN (2017, label-fit)', raw['txt']), (f'frozen {args.model}', new_text)):
        a4, a5, details[name] = evaluate(raw, src, 'text', seeds)
        rows[name] = (a4, a5)
        print(f'{name:>34} {a4 * 100:14.2f}% {a5 * 100:14.2f}% {(a4 - a5) * 100:+7.2f}')
    for channel, label in (('audio', 'audio (unsupervised, reference)'), ('video', 'video (released 3D-CNN)')):
        a4, a5, details[label] = evaluate(raw, raw['txt'], channel, seeds)
        rows[label] = (a4, a5)
        print(f'{label:>34} {a4 * 100:14.2f}% {a5 * 100:14.2f}% {(a4 - a5) * 100:+7.2f}')
    old_gap = rows['released CNN (2017, label-fit)'][0] - rows['released CNN (2017, label-fit)'][1]
    new_gap = rows[f'frozen {args.model}'][0] - rows[f'frozen {args.model}'][1]
    print(f'\n  gap {old_gap * 100:+.2f} -> {new_gap * 100:+.2f} points  ({100 * (1 - new_gap / old_gap):.0f}% of it removed)' if old_gap else '')
    print("  If the leak explanation holds, the gap collapses toward audio's; if it")
    print('  survives, something other than supervised feature extraction causes it.')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(__import__('json').dumps({'model': args.model, 'seeds': seeds, 'pickle': str(args.pickle), 'hold_out_session_4_vs_5': {k: {'acc_session4': v[0], 'acc_session5': v[1], 'gap': v[0] - v[1]} for k, v in rows.items()}, 'dialogues': details}, indent=2))
        print(f'written to {args.out}')
if __name__ == '__main__':
    main()
