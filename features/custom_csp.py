from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

class CustomCSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns transformer for EEG classification."""
    def __init__(self, n_components=4, reg=None, log=True, norm_trace=False):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def _compute_covariance(self, X):
        n_trials, n_channels, _ = X.shape
        cov = np.zeros((n_channels, n_channels))
        for trial in range(n_trials):
            trial_data = X[trial]
            trial_cov = np.cov(trial_data)
            if self.norm_trace:
                trial_cov = trial_cov / np.trace(trial_cov)
            cov += trial_cov
        cov /= n_trials
        if self.reg is not None:
            cov += self.reg * np.eye(n_channels)
        return cov

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        if X.ndim != 3:
            raise ValueError("X should be 3D array")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly 2 classes")
        self.classes_ = classes
        X0 = X[y == classes[0]]; X1 = X[y == classes[1]]
        if len(X0) == 0 or len(X1) == 0:
            raise ValueError("Both classes must have at least one sample")
        C1 = self._compute_covariance(X0); C2 = self._compute_covariance(X1)
        self.cov_1_, self.cov_2_ = C1, C2
        Cc = C1 + C2
        evals_c, evecs_c = eigh(Cc)
        evals_c = np.maximum(evals_c, 1e-10)
        W = evecs_c @ np.diag(evals_c ** -0.5) @ evecs_c.T
        C1_w = W @ C1 @ W.T
        evals, evecs = eigh(C1_w)
        idx = np.argsort(evals)
        evals, evecs = evals[idx], evecs[:, idx]
        self.filters_ = W.T @ evecs
        half = self.n_components // 2
        if self.n_components % 2 == 1:
            sel = np.concatenate([np.arange(half), np.arange(-half - 1, 0)])
        else:
            sel = np.concatenate([np.arange(half), np.arange(-half, 0)])
        self.filters_ = self.filters_[:, sel]
        self.eigenvalues_ = evals[sel]
        self.patterns_ = np.linalg.pinv(self.filters_.T)
        return self

    def transform(self, X):
        X = np.asarray(X)
        if not hasattr(self, "filters_"):
            raise ValueError("CSP must be fitted")
        n_trials, n_channels, n_times = X.shape
        if n_channels != self.filters_.shape[0]:
            raise ValueError("Channel mismatch")
        Xf = np.zeros((n_trials, self.n_components, n_times))
        for i in range(n_trials):
            Xf[i] = self.filters_.T @ X[i]
        feats = np.zeros((n_trials, self.n_components))
        for i in range(n_trials):
            var = np.var(Xf[i], axis=1)
            if self.log:
                feats[i] = np.log(var + 1e-10)
            else:
                feats[i] = var
        return feats

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_spatial_patterns(self):
        if not hasattr(self, "patterns_"):
            raise ValueError("CSP must be fitted")
        return self.patterns_

    def get_spatial_filters(self):
        if not hasattr(self, "filters_"):
            raise ValueError("CSP must be fitted")
        return self.filters_

__all__ = ["CustomCSP"]
