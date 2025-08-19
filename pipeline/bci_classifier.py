from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_classif

from config import BCIConfig
from features import FilterBankCSP


@dataclass
class BCIClassifier:
    """Filter bank CSP + (optional scaler + feature selection) + LDA pipeline wrapper."""
    config: BCIConfig

    def __post_init__(self):
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        steps = [
            ('fbcsp', FilterBankCSP(n_csp=self.config.n_csp, sfreq=self.config.sfreq)),
        ]
        if self.config.use_scaler:
            steps.append(('scaler', StandardScaler()))
        if self.config.use_feature_selection:
            steps.append(('select', SelectPercentile(mutual_info_classif, percentile=100)))
        lda = LDA(
            solver='lsqr' if self.config.lda_shrinkage else 'svd',
            shrinkage='auto' if self.config.lda_shrinkage else None,
        )
        steps.append(('clf', lda))
        return Pipeline(steps)

    def hyperparameter_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: Optional[Any] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Pipeline, float]:
        pipe = self._build_pipeline()
        params: Dict[str, Any] = {'fbcsp__n_csp': [2, 3, 4]}
        if self.config.use_feature_selection:
            params['select__percentile'] = list(self.config.feature_selection_percentiles)
        if extra_params:
            params.update(extra_params)
        gs = GridSearchCV(
            pipe,
            params,
            cv=cv or self.config.cv_folds,
            scoring='accuracy',
            refit=True,
        )
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_score_

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.pipeline.fit(X, y)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.pipeline.score(X, y)

    def predict(self, X: np.ndarray):
        return self.pipeline.predict(X)

__all__ = ['BCIClassifier']
