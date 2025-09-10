# Total Perspective Vortex

Motor imagery (left vs right hand) classification on the PhysioNet EEG Motor Movement/Imagery dataset. Core pipeline: Filter Bank CSP → (optional scaling + feature selection) → Logistic Regression. Includes visualization, per‑run prediction “replay”, and multi‑subject evaluation.

## 1) Dataset

Public source: physionet-open/eegmmidb (EDF files).

- First run: if `data/` (or path in `.env`) is empty, files are auto‑synced (public S3).
- Path pattern: `data/Sxxx/SxxxRyy.edf` (e.g. `data/S001/S001R03.edf`).

Optional `.env` at project root:

```
DATA_DIR=./data
```

## 2) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) CLI Usage

```bash
python mybci.py [subject] [run] [mode] [options]
# If subject/run omitted -> evaluate all (multi‑experiment)
```

Quick examples:

```bash
# Visualize a run
python mybci.py 1 3 visualize

# Interactive viewers
python mybci.py 1 3 visualize --interactive --viewer mne
python mybci.py 1 3 visualize --interactive --viewer stacked

# Train
python mybci.py 1 3 train

# Predict (including replay)
python mybci.py 1 3 predict

# Predict but force CV replay if no saved model
python mybci.py 1 3 predict --force_cv

# Evaluate all subjects (parallel) - if no workers are defined, it uses the maximum available on your machine
python mybci.py --max_workers 4
```

Arguments:

- `mode`: `train | predict | visualize`
- `--max_workers`: for evaluate‑all (when no subject/run)
- `--show`: open generated figures (non‑interactive)
- `--interactive`: display interactive figures
- `--viewer`: `mne | stacked` (default: `mne`)
- `--force_cv`: in predict mode, perform CV‑based replay when no persisted model

## 4) Modes

- train: Baseline K‑fold CV, then stratified holdout test. Inner `RepeatedStratifiedKFold` grid over CSP components (+ optional feature selection). Logs to `logs/train.log`.
- predict: “Replay” style sequential prediction per epoch using a saved model. If no model and `--force_cv`, builds fold‑held models (out‑of‑fold) and replays per epoch. Logs to `logs/predict.log`.
- visualize: Generates raw/filtered/PSD/band‑power/topomap figures; optional interactive viewers.
- evaluate‑all (omit subject/run): Runs predefined experiments across subjects in parallel; logs to `logs/evaluate.log`.

Predict output format (logger):

```
epoch nb: [prediction] [truth] equal?
epoch 00: [1] [2] False
...
Accuracy: 0.6818
```

## 5) Project Layout

```
config.py                      # BCIConfig + ExperimentConfig
mybci.py                       # CLI / application wiring
pipeline/
	bci_classifier.py            # BCIClassifier (pipeline construction / grid search)
features/
	filter_bank_csp.py           # Filter bank + CSP feature extractor
	custom_csp.py                # CSP implementation (from scratch)
modes/
	base.py, train.py, predict.py, visualize.py, evaluate_all.py
utils/
	data_loader.py               # EDF loading + epoch extraction
	visualization.py, logger.py, split.py, sync_data.py, system_info.py, model.py
logs/, figures/, models/       # generated outputs
```

## 6) Configuration (config.py)

BCIConfig defaults:

- `fmin` / `fmax`: 8.0 / 32.0 Hz (global filter for visualization/bandpower)
- `tmin` / `tmax`: 0.5 / 2.0 s (epoch window)
- `eeg_channels`: ["C3", "C4", "Cz"]
- `sfreq`: 160.0
- `n_csp`: 3
- `use_scaler`: True
- `use_feature_selection`: True
- `feature_selection_percentiles`: (10, 25, 50)
- `cv_folds`: 5
- `inner_repeats`: 5
- `test_size`: 0.3
- `random_state`: 42
- `save_min_test_acc`: 0.0
- `force_predict_cv`: False
- `models_dir`: "models"
- `save_models_in_evaluate_all`: True

ExperimentConfig defaults (`runs`):

```
0: [3, 7, 11]
1: [4, 8, 12]
2: [5, 9, 13]
3: [6, 10, 14]
4: [3, 4, 7, 8, 11, 12]
5: [5, 6, 9, 10, 13, 14]
```

## 7) ML Pipeline

1. Epochs from EDF (keep events 1 & 2 ⇒ labels 0/1)
2. Filter bank (default bands: 7–11, 11–15, 15–19, 19–23, 23–27, 27–31 Hz)
3. CSP per band (log‑variance features) + concat
4. Optional StandardScaler
5. Optional SelectPercentile (mutual_info_classif)
6. LogisticRegression (default C=0.01; grid over C when tuning)

Grid (hyperparameter_search):

- `fbcsp__n_csp`: [2, 3]
- `select__percentile`: values from `feature_selection_percentiles` when enabled
- `clf__C`: [0.001, 0.01, 0.1, 1.0]

## 8) Splitting & Validation

- Baseline report: K‑fold CV over the run.
- Model selection: RepeatedStratifiedKFold (repeats × folds).
- Final: stratified holdout test (`test_size`).

## 9) Predict “Replay”

- If a matching saved model exists: replay per‑epoch sequentially, log line by line, then final Accuracy.
- Else if `--force_cv`: build out‑of‑fold models with `StratifiedKFold` and replay per‑epoch.
- A small sleep (`replay_sleep`, optional attribute) may be used to simulate delay (capped < 2s if set).

## 10) Visualization Outputs

Saved per run:

- 00_raw_all.png
- 01_raw_all_psd.png
- 02_filtered_all.png
- 03_filtered_all_psd.png
- 04_filtered_selected.png
- 05_filtered_selected_psd.png
- 06_band_power.png
- 07_topomap.png (if ≥3 channels)

## 11) Evaluate‑All

Runs all experiments (see `ExperimentConfig.runs`) for all subjects in parallel.  
Logs: per‑subject accuracy per experiment + experiment means; prints a final mean across experiments.

## 12) Model Persistence

Saved by Train mode when `test_acc >= save_min_test_acc`.

```
models/
	subject_004/
		S004_run_14.pkl
		S004_run_14.meta.json
```

Meta contains: subject/run, accuracies, counts, config hash, serialized config, library versions, timestamp. Predict mode loads matching models (config hash check). Set `--force_cv` to bypass saved models for replay benchmarks.
