# MOSI and MOSEI: held-out training data versus test

Released CMU-MOSI and CMU-MOSEI features from the same release as the IEMOCAP features;
bi-LSTM of `experiments/mosi_mosei_headroom.py`, five seeds. Held-out = the
early-stopping videos, drawn from the training split, at the best epoch.

| Block | MOSI held-out | MOSI test | MOSEI held-out | MOSEI test |
|---|---|---|---|---|
| Text (supervised extraction) | 98.88 | 80.53 | 70.95 | 59.09 |
| Visual (supervised extraction) | 90.80 | 54.65 | 56.10 | 53.22 |
| Audio (unsupervised) | 49.97 | 55.29 | 58.62 | 58.64 |
| Early fusion | 99.18 | 79.47 | 71.19 | 59.95 |
| Late fusion | 97.99 | 78.30 | 65.16 | 60.60 |

Source: `results/headroom_mosi_mosei_fixed.json`. The splits of these corpora are not
organised by session, so this is not a session-gap test; a held-out set drawn from the
training distribution, overfitting or duplicated samples can also produce the pattern.
