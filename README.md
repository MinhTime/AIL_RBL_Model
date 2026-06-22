# ECG Signal Classification — MIT-BIH (Random Forest)

Implementation of the pipeline from *"Comparative Analysis of Machine Learning
Algorithms With Advanced Feature Extraction for ECG Signal Classification"*
(Subba & Chingtham, IEEE Access, 2024).

The pipeline classifies heartbeats from the MIT-BIH Arrhythmia Database into the
5 AAMI classes — **N, SVEB, VEB, FB, Q** — using advanced feature extraction
(wavelet denoising, HOS, FFT) and a Random Forest classifier.

## Project layout

| File | Role |
|------|------|
| `dataset.py` | Load MIT-BIH records, map 15 symbols → 5 AAMI classes, 80/20 split, up-sample minorities to 20,000 |
| `preprocess.py` | 9-level `bior4.4` wavelet denoising + multi-width median baseline removal |
| `feature_extraction.py` | R-peaks, QRS duration, amplitude statistics, HOS (skewness/kurtosis), FFT |
| `trainer.py` | Random Forest training, joblib checkpointing, evaluation (CM, accuracy, sensitivity, specificity, PPV) |
| `main.py` | CLI driver that wires every module into the full pipeline |

## Installation

```bash
pip install -r requirements.txt
```

## Get the data

Download the MIT-BIH Arrhythmia Database (WFDB format) from PhysioNet into a
local folder, e.g. `./mitdb`:

```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='./mitdb')
```

or manually from https://physionet.org/content/mitdb/1.0.0/ . The folder should
contain the `.dat`, `.hea` and `.atr` files (records 100, 101, ... 234).

## Run

Train, checkpoint, and evaluate:

```bash
python main.py train --data-dir ./mitdb --model-out rf_model.joblib --verbose
```

Evaluate independently from a saved checkpoint (uses the same seed, so the
held-out 20% test set is identical):

```bash
python main.py test --data-dir ./mitdb --model-in rf_model.joblib --verbose
```

### Useful options

| Flag | Default | Meaning |
|------|---------|---------|
| `--preprocess-level` | `record` | `record` denoises the full signal (true 9-level decomposition) before segmenting; `beat` denoises each segment |
| `--wavelet-level` | `9` | Wavelet decomposition levels |
| `--cutoff-low` / `--cutoff-high` | `1` / `1` | Number of low/high-frequency wavelet sub-bands to zero |
| `--median-widths` | `71 215` | Median-filter widths (≈200 ms, 600 ms at 360 Hz) for baseline removal |
| `--peak-thres` / `--peak-min-dist` | `0.5` / `100` | `peakutils.indexes` settings |
| `--target-per-minority` | `20000` | Up-sampling target per minority class |
| `--window-size` | `360` | Beat segment length in samples |

## Notes on faithfulness to the paper

* **AAMI mapping** follows the community-standard de Chazal grouping. The paper's
  Table 1 contains a couple of transcription quirks (e.g. it lists "ventricular
  escape" twice); the standard mapping is used here.
* **9-level wavelet** denoising requires a long signal. A single 360-sample beat
  only supports ~5 levels, so the default `--preprocess-level record` applies the
  9-level decomposition to the full record before segmentation, exactly as the
  paper intends.
* **Kurtosis** uses the excess-kurtosis form `Σ(xᵢ-x̄)⁴ / (n·σ⁴) − 3` (paper
  eq. 3). The prompt dropped the `n`; the `n`-normalised version is the correct,
  numerically stable one and matches the paper equation.
* **Up-sampling** brings each *minority* class to exactly 20,000 samples with
  replacement while leaving the majority class (N) untouched, per the prompt.
