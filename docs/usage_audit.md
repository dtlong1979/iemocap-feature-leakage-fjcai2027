# Usage audit of twenty MERC systems (inspected 2026-09-14)

A purposive code audit, not a systematic review. Candidates are systems published since
MMGCN that cite MMGCN or DialogueGCN, with their linked code releases. For each repository
the HEAD commit was taken with `git ls-remote`, that commit was cloned, and the IEMOCAP
data loader and model input sizes were read; line numbers refer to that commit. Whether
two feature files are identical was decided by their git blob SHA (`git ls-tree -l`):
declare-lab/conv-emotion at 6128ca20e9c7 has `DialogueGCN/IEMOCAP_features/IEMOCAP_features.pkl`
= blob `48a47d73` and `bc-LSTM-pytorch/IEMOCAP_features/IEMOCAP_features_raw.pkl` = blob
`a2150100`; MMGCN's `IEMOCAP_features.pkl` = blob `889532ca`. A repository is taken as
official when its owner is an author account or its README says so; exceptions are noted.
Every system that loads a released block evaluates on the file's own split (train
sessions 1-4, test session 5): `trainVid`/`testVid` of the pickle.

Groups: shared text = loads the 100-d text block of the bc-LSTM/DialogueGCN/MMGCN files;
released A/V + own text = own text encoder with the bc-LSTM file's 100-d audio and 512-d
visual blocks; MMGCN A/V + own text = own text encoder with MMGCN's 1,582-d audio and
342-d visual blocks; other or unverified = neither, or the audio/visual source cannot be
verified from the release; no code release = no official code found.

| Group | Systems |
|---|---|
| shared text | 5 |
| released A/V + own text | 3 |
| MMGCN A/V + own text | 7 |
| other or unverified | 2 |
| no code release | 3 |

| System | Venue | Group | Repository URL | Commit (12) | File:line | Quoted line | Notes |
|---|---|---|---|---|---|---|---|
| MMGCN (source) | ACL-IJCNLP 2021 | source | github.com/hujingwen6666/MMGCN | 85732d984f70 | dataloader.py:12; train.py:336 | `pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb')…` / `D_text = feat2dim['textCNN'] if args.Dataset=='IEMOCAP'…` | textCNN=100 (train.py:333). Ships the file (blob 889532ca). |
| MM-DFN | ICASSP 2022 | shared text | github.com/zerohd4869/MM-DFN | da9703660692 | code/run_train_erc.py:249; :362 | `default='../data/iemocap/IEMOCAP_features.pkl'` / `D_text = feat2dim['textCNN'] if args.dataset == 'IEMOCAP'…` | Bundled data/iemocap/IEMOCAP_features.pkl is byte-identical to MMGCN's (same blob 889532ca). |
| GraphCFC | IEEE TMM 2024 | shared text | github.com/lijfrank/GraphCFC | 23ba8f6bdc61 | run.py:28; :128 | `IEMOCAP_path = '/home/lijfrank/code/dataset/IEMOCAP_features/IEMOCAP_features.pkl'` / `D_text = feat2dim['textCNN']…` | dataloader.py returns `videoText`. README:26 thanks MMGCN for its datasets. |
| GraphMFT | Neurocomputing 2023 | shared text | github.com/lijfrank/GraphMFT | 2cebeae30578 | run.py:28; :131 | same `IEMOCAP_path` / `D_text = feat2dim['textCNN']…` | |
| GA2MIF | IEEE TAC 2024 | shared text | github.com/lijfrank/GA2MIF | 56104c3e5b7e | run.py:29; :132 | same `IEMOCAP_path` / `D_text = feat2dim['textCNN']…` | |
| DialogueCRN | ACL 2021 | shared text | github.com/zerohd4869/DialogueCRN | 2c2070d07ff1 | code/run_train_ie.py:133; :191 | `default='../data/iemocap/IEMOCAP_features.pkl'` / `input_size = 100` | Bundled file = blob 48a47d73 (DialogueGCN/bc-LSTM file). Default `--feature_type` is `text` (:131); `multi` gives 712 (:189). A RoBERTa script (run_train_bert_ie.py:127) also exists. |
| M3Net | CVPR 2023 | MMGCN A/V + own text | github.com/feiyuchen7/M3NET | 4d0fa88cd4f9 | dataloader.py:12, :15; train.py:343 | `…'./IEMOCAP_features/iemocap_features_roberta.pkl'…` / `D_text = 1024 #feat2dim['textCNN']…` | `__getitem__` returns roberta1–4 plus videoVisual and videoAudio; videoText is never returned. D_audio=1582. |
| CMERC | ACM MM 2024 | MMGCN A/V + own text | github.com/HITSZ-HLT/CMERC | 815c7c060914 | dataloader.py:12; train.py:344 | `pickle.load(open('data/IEMOCAP_features.pkl', 'rb')…` / `D_text = 1024 #…` | Same loader pattern as M3Net (RoBERTa at dataloader.py:15). |
| PCGNet | SIGIR 2024 | MMGCN A/V + own text | github.com/HITSZ-HLT/PCGNet | 1e1af49b37b7 | code/run_train_erc.py:382, :541; code/dataloader.py:21 | `default='../data/iemocap/IEMOCAP_features.pkl'` / `D_text = 1024` | There is an optional `use_multiemo` path (dataloader.py:44). Group confirmed. |
| CFN-ESA | 2023 (arXiv) | MMGCN A/V + own text | github.com/lijfrank/CFN-ESA | 9ad11f8507ae | run_iemocap.py:58; :151 | `…/iemocap_multi_features_mmgcn_cosmic.pkl'` / `embedding_dims = [1024, 342, 1582]` | The file name itself says MMGCN+COSMIC. README:31 credits both. |
| SDT | IEEE TMM 2023 | MMGCN A/V + own text | github.com/butterfliesss/SDT | 60af92182318 | dataloader.py:12; train.py:169 | `pickle.load(open('data/iemocap_multimodal_features.pkl', 'rb')…` / `D_text = 1024` | Its own merged file (A 1582, V 342 at train.py:166). File contents not checked. |
| GS-MCC | AAAI 2025 | MMGCN A/V + own text | github.com/FuchenZhang/GS-MCC (also github.com/yuntaoshou/GS-MCC, fb5a02acff2a) | 38c4038a7738 | dataloader.py:12; train_fourier.py:275 | `…'./IEMOCAP_features/iemocap_multimodal_features.pkl'…` / `feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 1024,…` | The text key is named "textCNN" but set to 1024. Two repos exist: FuchenZhang's (README says official) and yuntaoshou's (identical dataloader.py, no training script). Neither link was found in the arXiv PDF. Code confirms 1024/1582/342. |
| GraphSmile | TPAMI 2025 | MMGCN A/V + own text | github.com/lijfrank/GraphSmile | 80ae44515f4c | run.py:127; :231 | `IEMOCAP_path = ''` / `embedding_dims = [1024, 342, 1582]` | The pickle path is blank, so the group rests on dims only. The IEMOCAP4 variant (:233) uses `[1024, 512, 100]` (bc-LSTM-style A/V dims). Partly verified. |
| COGMEN | NAACL 2022 | released A/V + own text | github.com/Exploration-Lab/COGMEN | 8c00c9bfec34 | preprocess.py:32; cogmen/Dataset.py:47; train.py:261 | `open("./data/iemocap/IEMOCAP_features.pkl", "rb")…` / `zip(s.sbert_sentence_embeddings, s.audio, s.visual)` | Text is SBERT 768; it loads the DialogueGCN/bc-LSTM file itself (bundled blob 48a47d73) and uses that file's audio (100) and visual (512) blocks. |
| CORECT | EMNLP 2023 | released A/V + own text | github.com/leson502/CORECT_EMNLP2023 | f8e3b1699fe3 | preprocess.py:29, :58; train.py:268 | `…"/data/iemocap/IEMOCAP_features.pkl"…` / `"text": sbert_model.encode(video_sentence[vid]),` | Same as COGMEN. Bundled blob 48a47d73; SBERT text; bc-LSTM A 100 / V 512. |
| Joyful | EMNLP 2023 | other or unverified | github.com/wykstc/MERC-main (fork: Clearloveyuan/MERC-main, same HEAD) | 15f08e1b22d0 | JOYFUL/joyful/Dataset.py:53; Sample.py:16 | `zip(s.sbert_sentence_embeddings, s.audio, s.visual)` / `…= sbert_model.encode(sentence)` | The paper links anonymous.4open.science/r/MERC-7F88, which now returns HTTP 410 "repository_expired". This repo's README says it is official; the fork belongs to an author account. Only preprocessed data_iemocap.pkl is shipped, so the A/V source is not verified. Text group confirmed. |
| MultiEMO | ACL 2023 | other or unverified | github.com/TaoShi1998/MultiEMO | e96458b870a3 | Dataset/IEMOCAPDataset.py:18, :24; Train/TrainMultiEMO.py:106 | `…open('Data/IEMOCAP/Speakers.pkl', 'rb')…` / `self.videoText = pickle.load(open('Data/IEMOCAP/TextFeatures.pkl', 'rb'))` | Speakers.pkl is byte-identical to the bc-LSTM IEMOCAP_features_raw.pkl (blob a2150100). Only speakers, labels and splits are kept; the text field is discarded (`_`). Text is EmoBERTa, roberta_dim = 768. |
| EmoCaps | Findings ACL 2022 | no code release | no official repo found | not verified | not verified | not verified | The paper PDF names BERT for text and gives no code link. The only candidate, github.com/Jing-yilin/EmoCaps-FFN (9c7de1db054d), was committed by a non-author (2023-08). Its iemocap_lstm.py:156 `D_m = 456` loads its own iemocap_data.pkl, and comments imply 100-d text. It cannot be attributed to the paper. Group rests on the paper only. |
| DER-GCN | IEEE TNNLS 2024/25 | no code release | no repo found | not verified | not verified | not verified | A web-search summary claimed github.com/yuntaoshou/DER-GCN, but ls-remote says "Repository not found" and it is not in that user's repo list. The arXiv 2312.10579 PDF has no code link and says RoBERTa. Group rests on the paper only, so in practice this is "could not check". |
| RL-EMO | ICASSP 2024 | released A/V + own text | github.com/zyh9929/RL-EMO | 397967fa809e | dataloader.py:14; MMGCN_train.py:408 | `…open('./ERC_features/IEMOCAP_features_raw.pkl', 'rb')…` / `D_text = 1024 #feat2dim['textCNN']…` | Repository found through the DmitryRyumin/ICASSP-2023-24-Papers list, not from the paper; that the owner is an author is not verified. Text is RoBERTa (loader returns roberta1–4); A/V are the bc-LSTM raw file's blocks (D_audio=100 at :406, D_visual=512). |
| AdaIGN | AAAI 2024 | no code release | github.com/HITSZ-HLT/AdaIGN (github.com/TuGengs/AdaIGN redirects here, same HEAD) | 1d4e14d95869 | README.md:2 | `The source code will be released as soon` | The repo has only a README. Consistent with "code not released". |
