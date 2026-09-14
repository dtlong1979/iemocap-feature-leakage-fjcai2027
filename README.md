# iemocap-feature-leakage-fjcai2027

- `scripts/fetch_features.py`: released MOSI, MOSEI and IEMOCAP feature files, with MD5 check
- `experiments/mosi_mosei_headroom.py`: bi-LSTM reader; MOSI and MOSEI held-out versus test
- `experiments/iemocap_text_reextract.py`: frozen-encoder text features; session-4 versus session-5 gap
- `experiments/iemocap_session_stats.py`: class balance per session
- `experiments/iemocap_loso.py`: leave-one-session-out with the bi-LSTM, per-emotion recall
- `experiments/iemocap_loso_linear.py`: leave-one-session-out with a linear probe
- `experiments/iemocap_matched_gap.py`: session gap with three training sessions per fold, PCA projection
- `experiments/iemocap_protocols.py`: leave-one-speaker-out and random dialogue-level cross-validation
- `experiments/iemocap_knn_diagnostic.py`: nearest-neighbour label agreement per session
- `experiments/iemocap_gap_linear.py`: session gap with a linear probe, timed
- `experiments/dialogue_bootstrap.py`: dialogue-level bootstrap intervals for the session gap and LOSO inflation
- `scripts/export_fold_seed_table.py`: one row per experiment, fold, seed and metric
- `experiments/iemocap_ranking.py`: rank agreement between protocols
- `experiments/dialoguecrn_loso.py`: DialogueCRN leakage stress test under leave-one-session-out
- `experiments/iemocap_feature_provenance.py`: utterance-level comparison of the released feature files
- `experiments/feature_file_md5.py`: MD5 digests of the feature files
- `experiments/iemocap_context.py`: frozen text fused with released audio and visual blocks
- `src/uacmd/__init__.py`: package
- `src/uacmd/metrics.py`: accuracy, weighted and macro F1, kappa, MCC, majority-class baseline
- `results/`: 57 result files, per fold and per seed
- `docs/usage_audit.md`
- `docs/dialoguecrn_stress_test.md`
- `docs/mosi_mosei_heldout.md`
- https://github.com/zerohd4869/DialogueCRN: DialogueCRN release (commit 2c2070d, MIT License), read by experiments/dialoguecrn_loso.py

## Environment

- bi-LSTM, linear-probe and nearest-neighbour experiments: CPU, Intel Core i7-12700, Windows 11, Python 3.11.9, PyTorch 2.13.0+cpu, NumPy 2.4.6, scikit-learn 1.9.0, SciPy 1.17.1, 4 threads
- frozen-encoder feature extraction and DialogueCRN: NVIDIA GeForce RTX 3050 (6 GB), CUDA 12.4, Python 3.11.9, PyTorch 2.6.0+cu124, torch_geometric 2.8.0.post1
