from pathlib import Path
from typing import Tuple
import warnings

import numpy as np
import mne
from mne.datasets.eegbci import standardize
from mne.channels import make_standard_montage

from config import BCIConfig
from utils.logger import get_logger

class BCIDataLoader:
    """Handles loading and preprocessing of EEG data."""
    def __init__(self, config: BCIConfig):
        self.config = config
        self.logger = get_logger('DataLoader')

    def _prep_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Standardize montage, apply bandpass filter, and select EEG channels.
        """
        standardize(raw)
        raw.set_montage(make_standard_montage('standard_1020'), verbose=False)
        raw.filter(l_freq=self.config.fmin, h_freq=self.config.fmax,
                   fir_design='firwin', verbose=False)
        picks = mne.pick_channels(raw.info['ch_names'], include=self.config.eeg_channels)
        raw.pick(picks)
        return raw

    def load_raw(self, filepath: str) -> mne.io.Raw:
        """Load and preprocess raw EDF file."""
        path = Path(filepath)
        if not path.exists():
            self.logger.error(f'EDF file not found: {path}')
            raise FileNotFoundError(f'EDF file not found: {path}')
        self.logger.info(f'Loading raw EDF file: {path}')
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Limited .*annotation\(s\) that were expanding outside the data range\.",
                category=RuntimeWarning
            )
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
        raw = self._prep_raw(raw)
        return raw

    def load_epochs(self, subject: int, run: int, dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load, preprocess, and extract epochs for a given subject/run."""
        file_path = self._get_file_path(subject, run, dataset_dir)
        raw = self.load_raw(str(file_path))
        return self._extract_epochs(raw)

    def _get_file_path(self, subject: int, run: int, dataset_dir: str) -> Path:
        """Generate file path for given subject and run."""
        subj_str = f'S{subject:03d}'
        run_str = f'{run:02d}'
        file_name = f'{subj_str}R{run_str}.edf'
        return Path(dataset_dir) / subj_str / file_name

    def _extract_epochs(self, raw: mne.io.Raw) -> Tuple[np.ndarray, np.ndarray]:
        """Extract epochs for motor imagery: keep only event codes 1 (T1) and 2 (T2)."""
        events, _ = mne.events_from_annotations(raw, verbose=False)
        if events.size == 0:
            self.logger.warning('No events found; returning empty arrays')
            return np.empty((0, len(raw.ch_names), 0)), np.empty((0,), dtype=int)

        target_codes = {1, 2}
        mask = np.isin(events[:, 2], list(target_codes))
        sel_events = events[mask]

        present = set(sel_events[:, 2])
        if len(present) < 2:
            self.logger.warning(f'Expected both event codes 1 & 2; found {sorted(present) or "none"}')
            return np.empty((0, len(raw.ch_names), 0)), np.empty((0,), dtype=int)

        epochs = mne.Epochs(
            raw,
            sel_events,
            event_id=None,
            tmin=self.config.tmin,
            tmax=self.config.tmax,
            baseline=None,
            detrend=1,
            preload=True,
            verbose=False,
            picks='eeg'
        )
        X = epochs.get_data().astype(np.float64)

        mapping = {1: 0, 2: 1}
        retained_codes = epochs.events[:, 2]
        try:
            y = np.array([mapping[c] for c in retained_codes], dtype=int)
        except KeyError:
            self.logger.warning('Unexpected event code outside {1,2} after epoching; discarding.')
            keep = np.array([c in mapping for c in retained_codes])
            X = X[keep]
            y = np.array([mapping[c] for c in retained_codes[keep]], dtype=int)

        if X.shape[0] != len(y):
            n = min(X.shape[0], len(y))
            X, y = X[:n], y[:n]

        return X, y