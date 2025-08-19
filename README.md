# Total Perspective Vortex

Motor imagery (left vs right hand) classification on the PhysioNet EEG Motor Movement/Imagery dataset. Core pipeline: Filter Bank CSP → (optional scaling + feature selection) → Shrinkage LDA. Includes visualization, per‑run prediction simulation, and multi‑subject evaluation.

## 1. Dataset

Public source: physionet-open/eegmmidb (EDF files).  
On first run, if `data/` is empty, files are auto‑synced (public S3, `--no-sign-request`).  
Expected path pattern: `data/Sxxx/SxxxRyy.edf` (example: `data/S001/S001R03.edf`).

## 2. Installation

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Create a `.env` file at the root of the project and copy:

```
DATA_DIR=./data
```

Change `DATA_DIR` to point elsewhere if you store EDF files outside `./data`.

## 3. Quick Start

```bash
# Visualize a run (MNE browser)
python mybci.py 1 3 visualize

# Interactive lightweight (stacked matplotlib instead of MNE browser)
python mybci.py 1 3 visualize --interactive --viewer stacked

# Interactive using MNE (heavy)
python mybci.py 1 3 visualize --interactive --viewer mne

# Train (baseline CV + repeated inner CV grid + stratified holdout test)
python mybci.py 1 3 train

# Predict (per‑epoch CV streaming simulation using fold‑held models)
python mybci.py 1 3 predict

# Predict but force CV simulation even if a saved model exists
python mybci.py 1 3 predict --force_cv

# Evaluate all subjects (parallel)
python mybci.py --max_workers 4
```

General form:

```
python mybci.py [subject] [run] [mode] [options]
# If subject/run omitted -> evaluate all (multi‑experiment)
```

## 4. Modes

- train: Baseline K-fold CV report, then stratified holdout split (test fraction configurable). Inner `RepeatedStratifiedKFold` grid search over CSP components (+ optional feature selection percentiles). Logs to `logs/train.log`.
- predict: Builds one model per outer fold and simulates streaming by predicting epochs sequentially, logs to `logs/predict.log`.
- visualize: Generates raw / filtered / PSD / band power / topomap figures. Optional interactive viewers (`mne` or lightweight stacked) with `--interactive`.
- evaluate-all (omit subject/run): Groups predefined runs per experiment (`ExperimentConfig.runs`) and runs them in parallel.

## 5. Layout

```
config.py                      # BCIConfig + ExperimentConfig
classifier.py                  # Backward-compatible shim importing BCIClassifier from pipeline
mybci.py                       # CLI / application wiring
pipeline/
	bci_classifier.py            # BCIClassifier (pipeline construction / grid search)
	__init__.py                  # exports BCIClassifier
features/
	__init__.py                  # exports FilterBankCSP
	filter_bank_csp.py           # Filter bank + CSP feature extractor
	custom_csp.py                # CSP implementation low-level
modes/                         # Mode classes (train / predict / visualize / evaluate_all)
	__init__.py
	base.py
	train.py
	predict.py
	visualize.py
	evaluate_all.py
utils/                         # Services & helpers
	__init__.py                  # exports stratified_holdout
	data_loader.py               # EDF loading + epoch extraction
	visualization.py             # Figure generation service
	logger.py                    # Async color + file logger + progress logger
	split.py                     # stratified_holdout helper
	sync_data.py                 # S3 sync
	system_info.py               # CPU / optimal workers detection
	model.py                     # ModelRepository (persistence)
logs/                          # (generated)
figures/                       # (generated)
models/                        # Persisted trained models + metadata
data/                          # EEG dataset (synced if absent)
```

## 6. Configuration (BCIConfig)

Key fields (see `config.py`) with defaults:

- fmin / fmax (float): global band-pass for preprocessing & band power plots. Default 8.0 / 32.0 Hz.
- tmin / tmax (float): epoch time window relative to event onset. Default 0.7 / 3.9 s.
- eeg_channels (list[str]): channels selected after filtering. Default ["C3", "C4", "Cz"].
- sfreq (float): expected sampling frequency (epochs must match). Default 160.0 Hz.
- n_csp (int): CSP components per sub‑band (grid explores nearby values). Default 3.
- use_scaler (bool): add StandardScaler. Default True.
- use_feature_selection (bool): enable mutual information percentile selector. Default True.
- feature_selection_percentiles (tuple[int]): percentiles tried in grid. Default (70, 85, 100).
- lda_shrinkage (bool): use shrinkage LDA (solver=lsqr, shrinkage=auto). Default True.
- cv_folds (int): K folds used for baseline CV and as base folds in inner repeated CV. Default 5.
- inner_repeats (int): repeats for RepeatedStratifiedKFold in hyperparameter search. Default 3.
- test_size (float): stratified holdout fraction for final test set. Default 0.2.
- random_state (int): global RNG seed. Default 42.
- save_min_test_acc (float): minimum test accuracy required to persist a model. Default 0.0.
- force_predict_cv (bool): override to force CV simulation even if persisted model present. Default False.
- models_dir (str): directory root for persisted models. Default "models".
- save_models_in_evaluate_all (bool): persist per-experiment models in EvaluateAll. Default True.

Adjust values by editing `config.py` or instantiating a custom `BCIConfig`. Experiments (grouped runs) defined in `ExperimentConfig.runs`.

## 7. Visualization

Non‑interactive output (per run):

- 00_raw_all.png
- 01_raw_all_psd.png
- 02_filtered_all.png
- 03_filtered_all_psd.png
- 04_filtered_selected.png
- 05_filtered_selected_psd.png
- 06_band_power.png
- 07_topomap.png (if ≥3 channels)

Returned report dict from `EEGVisualizer.visualize_subject_run` includes file paths, band power values, sample frequencies.

Interactive examples:

```
python mybci.py 1 3 visualize --interactive --viewer mne
python mybci.py 1 3 visualize --interactive --viewer stacked
```

### Plot reference & usage

| File                         | What it shows                                                                     | Why you look at it                                                                                                            |
| ---------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 00_raw_all.png               | Unfiltered stacked traces (all EEG channels)                                      | Quick data sanity: flat lines (bad channel), large drifts, muscle bursts, eye blinks before any processing.                   |
| 01_raw_all_psd.png           | Power Spectral Density (raw)                                                      | Detect broad artefacts: strong 50/60 Hz line noise, excessive low‑freq drift, decide if band limits (fmin/fmax) are sensible. |
| 02_filtered_all.png          | Band‑passed traces (fmin–fmax, all channels)                                      | Verify filter behaved: drifts removed, high‑freq noise attenuated, morphology preserved.                                      |
| 03_filtered_all_psd.png      | PSD after filtering (all channels)                                                | Confirm energy concentrated inside passband and attenuation outside; inspect residual peaks (e.g. line noise).                |
| 04_filtered_selected.png     | Filtered traces for selected channels (eeg_channels) with per‑channel std scaling | Focus on channels used for classification (e.g. C3/C4/Cz); normalized spacing makes relative modulations clearer.             |
| 05_filtered_selected_psd.png | PSD of selected channels                                                          | Inspect mu (≈8–13 Hz) / beta (≈13–30 Hz) rhythms; look for channel‑specific peaks that may discriminate classes.              |
| 06_band_power.png            | Mean band power per selected channel (averaged over fmin–fmax)                    | Rapid channel ranking; outliers may indicate electrode noise (too high) or poor contact (too low).                            |
| 07_topomap.png               | Spatial topography of mean band power (selected channels mapped via montage)      | Visual lateralization / spatial distribution check; helps validate expected motor cortex focus (C3/C4).                       |

Practical workflow:

1. Start with 00 / 01 to assess raw quality & choose frequency band tweaks.
2. Use 02 / 03 to confirm filtering removed unwanted components.
3. Examine 04 / 05 for rhythm prominence & potential discriminative bands.
4. Check 06 for quick per‑channel comparison; adjust eeg_channels if needed.
5. Use 07 to spot spatial asymmetries consistent with motor imagery tasks.

If a plot is missing:

- Topomap requires ≥3 channels.
- Any earlier failure (exception) logs a warning; re‑run with `--interactive --viewer stacked` for debugging.

## 8. Logging

Async logger (`logger.py`):

- Console (INFO+ with colors)
- File logs under `logs/`
- Extra levels: SUCCESS, PROGRESS, ...

For debug verbosity in custom scripts: call `setup_development_logging()`.

## 9. ML Pipeline

1. Load EDF → channel selection → epoch extraction (codes 1 & 2 → labels 0/1).
2. Band-pass filter full signal to fmin–fmax (default 8–32 Hz in config) for visualization / band power.
3. Filter bank (4‑Hz bands covering ~7–31 Hz by default: (7–11, 11–15, 15–19, 19–23, 23–27, 27–31)) → CSP per band (log-variance features) concatenated.
4. (Optional) Standard scaling.
5. (Optional) Mutual information percentile feature selection.
6. Shrinkage LDA classifier.

Event code mapping: EDF annotations with event codes 1 (T1) and 2 (T2) are retained and mapped to class labels 0 and 1 respectively.

Filter bank details: default bands are generated via list comprehension `[(f, f+4) for f in range(7, 30, 4)]` producing 6 contiguous 4‑Hz bands (upper edge inclusive of 31 Hz).

Grid search space: CSP components (`fbcsp__n_csp`) ∈ {2,3,4} + optional feature percentiles (`select__percentile`) when feature selection enabled.

## 10. Splitting & Validation

- Baseline: simple K‑fold CV over whole run (quick variance gauge).
- Model selection: inner repeated stratified K‑fold (repeats × folds) for robustness.
- Final reporting: stratified single holdout test set (never used in tuning).

This keeps the test set untouched and reduces overfitting risk on tiny epoch counts.

## 11. Parallel Evaluation

`EvaluateAllMode` picks a worker count from system info (Apple Silicon capped). Override via `--max_workers`.

## 12. Model Persistence

Train mode saves the tuned pipeline (per subject + run) when `test_acc >= save_min_test_acc`.

Structure:

```
models/
	subject_004/
		S004_run_14.pkl
		S004_run_14.meta.json
```

Meta file fields: subject, run, inner/test accuracies, feature count, sample counts, config hash, full config, library versions, timestamp.

Predict mode:

- Loads a matching saved model if present and config hash matches (fast direct inference).
- Falls back to cross‑validated streaming simulation if none is found.
- Use `--force_cv` to bypass a saved model intentionally (e.g. benchmarking current code changes).

Altering configuration (e.g. enabling feature selection) changes the hash and automatically invalidates stale models (they are skipped with a log line indicating a config mismatch).
