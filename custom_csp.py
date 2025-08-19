import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CustomCSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns transformer for EEG classification.
    
    CSP finds spatial filters that maximize the ratio of variance between
    two classes. This implementation supports regularization and trace
    normalization for improved robustness.
    
    Parameters
    ----------
    n_components : int, default=4
        Number of CSP components to extract.
    reg : float or None, default=None
        Regularization parameter for covariance matrices.
    log : bool, default=True
        Whether to apply log transform to variance features.
    norm_trace : bool, default=False
        Whether to normalize covariance matrices by their trace.
    """
    
    def __init__(self, n_components=4, reg=None, log=True, norm_trace=False):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def _compute_covariance(self, X):
        """Compute sample covariance matrix for a set of trials.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG data for one class.
            
        Returns
        -------
        cov : ndarray, shape (n_channels, n_channels)
            Averaged covariance matrix across trials.
        """
        n_trials, n_channels, _ = X.shape
        cov = np.zeros((n_channels, n_channels))

        for trial in range(n_trials):
            trial_data = X[trial]

            if self.norm_trace:
                trial_cov = np.cov(trial_data)
                trial_cov = trial_cov / np.trace(trial_cov)
            else:
                trial_cov = np.cov(trial_data)

            cov += trial_cov

        cov /= n_trials

        if self.reg is not None:
            cov += self.reg * np.eye(n_channels)

        return cov

    def fit(self, X, y):
        """Fit CSP spatial filters from training data.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG training data.
        y : ndarray, shape (n_trials,)
            Class labels (must have exactly 2 unique values).
            
        Returns
        -------
        self : CustomCSP
            Fitted transformer instance.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(f"X should be 3D array, got {X.ndim}D")

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(classes)}")

        self.classes_ = classes

        X_class_0 = X[y == classes[0]]
        X_class_1 = X[y == classes[1]]

        if len(X_class_0) == 0 or len(X_class_1) == 0:
            raise ValueError("Both classes must have at least one sample")

        C1 = self._compute_covariance(X_class_0)
        C2 = self._compute_covariance(X_class_1)

        self.cov_1_ = C1
        self.cov_2_ = C2

        C_composite = C1 + C2

        eigenvals_comp, eigenvecs_comp = eigh(C_composite)

        if np.any(eigenvals_comp <= 0):
            eigenvals_comp = np.maximum(eigenvals_comp, 1e-10)

        W = eigenvecs_comp @ np.diag(eigenvals_comp**-0.5) @ eigenvecs_comp.T
        C1_white = W @ C1 @ W.T
        eigenvals, eigenvecs = eigh(C1_white)

        sorted_indices = np.argsort(eigenvals)
        eigenvals = eigenvals[sorted_indices]
        eigenvecs = eigenvecs[:, sorted_indices]

        self.filters_ = W.T @ eigenvecs

        n_comp_half = self.n_components // 2
        if self.n_components % 2 == 1:
            indices = np.concatenate([
                np.arange(n_comp_half),
                np.arange(-n_comp_half - 1, 0),
            ])
        else:
            indices = np.concatenate([
                np.arange(n_comp_half),
                np.arange(-n_comp_half, 0),
            ])

        self.filters_ = self.filters_[:, indices]
        self.eigenvalues_ = eigenvals[indices]
        self.patterns_ = np.linalg.pinv(self.filters_.T)

        return self

    def transform(self, X):
        """Transform EEG data using fitted CSP filters.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG data to transform.
            
        Returns
        -------
        features : ndarray, shape (n_trials, n_components)
            CSP features (log-variance of filtered signals).
        """
        X = np.asarray(X)

        if not hasattr(self, "filters_"):
            raise ValueError("CSP must be fitted before transform")

        if X.ndim != 3:
            raise ValueError(f"X should be 3D array, got {X.ndim}D")

        n_trials, n_channels, n_times = X.shape

        if n_channels != self.filters_.shape[0]:
            raise ValueError(
                f"Number of channels mismatch: expected {self.filters_.shape[0]}, got {n_channels}"
            )

        X_filtered = np.zeros((n_trials, self.n_components, n_times))

        for trial in range(n_trials):
            X_filtered[trial] = self.filters_.T @ X[trial]

        features = np.zeros((n_trials, self.n_components))

        for trial in range(n_trials):
            for comp in range(self.n_components):
                variance = np.var(X_filtered[trial, comp, :])

                if self.log:
                    features[trial, comp] = np.log(variance + 1e-10)
                else:
                    features[trial, comp] = variance

        return features

    def fit_transform(self, X, y):
        """Fit CSP filters and transform data in one step.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG training data.
        y : ndarray, shape (n_trials,)
            Class labels.
            
        Returns
        -------
        features : ndarray, shape (n_trials, n_components)
            CSP features.
        """
        return self.fit(X, y).transform(X)

    def get_spatial_patterns(self):
        """Get spatial patterns (inverse of spatial filters).
        
        Returns
        -------
        patterns : ndarray, shape (n_channels, n_components)
            Spatial patterns for visualization.
        """
        if not hasattr(self, "patterns_"):
            raise ValueError("CSP must be fitted before getting patterns")
        return self.patterns_

    def get_spatial_filters(self):
        """Get spatial filters.
        
        Returns
        -------
        filters : ndarray, shape (n_channels, n_components)
            Spatial filters for signal transformation.
        """
        if not hasattr(self, "filters_"):
            raise ValueError("CSP must be fitted before getting filters")
        return self.filters_
