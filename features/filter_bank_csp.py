from __future__ import annotations
import numpy as np
import mne
from sklearn.base import BaseEstimator, TransformerMixin
from .custom_csp import CustomCSP


class FilterBankCSP(BaseEstimator, TransformerMixin):
    def __init__(self, freq_bands=None, n_csp=3, sfreq=160):
        if freq_bands is None:
            self.freq_bands = [(f, f + 4) for f in range(7, 30, 4)]
        else:
            self.freq_bands = freq_bands
        self.n_csp = n_csp
        self.sfreq = sfreq
        self.csp_list = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.csp_list = []
        for fmin, fmax in self.freq_bands:
            X_filtered = self._bandpass_filter(X, fmin, fmax)
            csp = CustomCSP(n_components=self.n_csp, log=True)
            csp.fit(X_filtered, y)
            self.csp_list.append(csp)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        features = []
        for idx, (fmin, fmax) in enumerate(self.freq_bands):
            X_filtered = self._bandpass_filter(X, fmin, fmax)
            csp = self.csp_list[idx]
            features.append(csp.transform(X_filtered))
        return np.concatenate(features, axis=1)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _bandpass_filter(self, X, fmin, fmax):
        n_trials, n_channels, n_times = X.shape
        X_reshaped = X.reshape(-1, n_times)
        X_filtered = mne.filter.filter_data(
            X_reshaped, self.sfreq, fmin, fmax, verbose="ERROR"
        )
        return X_filtered.reshape(n_trials, n_channels, n_times)

__all__ = ["FilterBankCSP"]
