# DialogueCRN leakage stress test

A stress test of the released text block with a published architecture, not a replication
of the published result.

## Source
- Repository: https://github.com/zerohd4869/DialogueCRN
- Commit: 2c2070d07ff190be25082804c13521aac8d25844 (2023-10-24), MIT License
- Used from the release: `code/model.py` (model and reasoning modules), `code/loss.py`,
  and the IEMOCAP hyper-parameters of `script/run_train_ie.sh`
- Feature file shipped by the release: MD5 identical to the DialogueGCN file
  (`results/iemocap_feature_files_md5.json`)

## Environment
- Python 3.11.9, PyTorch 2.6.0+cu124 (CUDA 12.4), torch_geometric 2.8.0.post1,
  NumPy 2.4.6, scikit-learn 1.9.1
- GPU: NVIDIA GeForce RTX 3050, 6 GB; Windows 11

## Settings
- Hyper-parameters (release values): base_model LSTM, base_layer 2, hidden_size 100,
  dropout 0.2, reason_steps (3, 4), lr 1e-4, l2 2e-4, batch_size 32, gamma 0
- Epochs 100, early-stopping patience 20
- Seeds 0, 1, 2 (model initialisation and batch order); validation dialogues drawn once
  with `random.Random(0)`
- Feature blocks: released text (100-d) and frozen BERT-base text (768-d)

## Differences from the release (applied to both feature blocks)
1. Folds: five-fold leave-one-session-out; the release hard-codes the session-5 split.
2. Standardisation: each input block is standardised with statistics of the training
   dialogues of the fold. Without it the 768-d BERT states barely train; on the released
   text it changes standard-split accuracy by 0.1 points (58.47 vs 58.35).
3. Epoch selection: on 10% of the training dialogues; the release's default
   (`valid_rate 0`) selects the best epoch on the test fold.
4. Training loop: our own loop around the released model and loss.
5. `torch_scatter.scatter_add`, which has no Windows wheel, is supplied by the equivalent
   `torch_geometric.utils.scatter(..., reduce="sum")`; nothing else in the model changes.
6. No search over reasoning steps; the release's shell script reports the best of 25
   combinations.

## Results (accuracy, %, mean of three seeds)

| Held-out session | Released text | Frozen BERT |
|---|---|---|
| 1 | 87.98 | 56.86 |
| 2 | 90.71 | 64.97 |
| 3 | 88.02 | 57.91 |
| 4 | 90.81 | 56.77 |
| 5 (standard split) | 57.84 | 61.72 |
| LOSO mean | 83.07 | 59.65 |
| LOSO mean minus standard | +25.23 | -2.07 |

Per-seed values, weighted F1 and selected epochs: `results/dialoguecrn_loso/*.json` and
`results/fjcai_fold_seed.csv`.

## Published figure
The release reports 66.05 accuracy for text-only IEMOCAP. With the settings above on the
standard split we obtain 58.47 (standardised) and 58.35 (raw), one seed
(`results/dialoguecrn_repro/`). The numbers above therefore compare the two feature blocks
with each other, not with the published figure.
