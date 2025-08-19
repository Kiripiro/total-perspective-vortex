from __future__ import annotations
import time
import numpy as np
from sklearn.model_selection import KFold
from .base import ModeBase
from pipeline import BCIClassifier
from utils.model import ModelRepository


class PredictMode(ModeBase):
    LOG_NAME = 'PredictMode'
    LOG_FILE = 'predict.log'

    def execute(self, subject: int, run: int, dataset_dir: str) -> None:
        self.logger.info(f'Start prediction S{subject:03d} run {run:02d}')
        try:
            X, y = self.data_loader.load_epochs(subject, run, dataset_dir)
            if X.size == 0:
                self.logger.warning('No data')
                return
            self.logger.info(f'Data: {X.shape[0]} epochs')

            if not self.config.force_predict_cv:
                repo = ModelRepository(self.config)
                loaded = repo.load_subject_run(subject, run)
                if loaded:
                    pipe, meta = loaded
                    self.logger.info('Loaded persisted model; direct batch prediction.')
                    preds = pipe.predict(X)
                    acc = (preds == y).mean()
                    self.logger.success(f'Accuracy (saved model): {acc:.4f}')
                    return
                else:
                    self.logger.info('No persisted model (or config mismatch). Proceeding with CV simulation.')

            n = len(y)
            models = [None] * n
            kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            for tr, te in kf.split(X):
                clf = BCIClassifier(self.config)
                clf.fit(X[tr], y[tr])
                for idx in te:
                    models[idx] = clf

            preds = np.empty(n, dtype=y.dtype)
            correct = 0
            worst_latency = 0.0
            for idx in range(n):
                clf = models[idx]
                t0 = time.perf_counter()
                pred = clf.pipeline.predict(X[idx:idx+1])[0]
                dt = time.perf_counter() - t0
                worst_latency = max(worst_latency, dt)
                preds[idx] = pred
                ok = pred == y[idx]
                correct += int(ok)
                self.logger.info(f'epoch {idx:02d}: [{pred}] [{y[idx]}] {ok}')
                time.sleep(0.5)

            acc = correct / n
            self.logger.success(f'Accuracy (CV-held out per epoch): {acc:.4f}')
        finally:
            self._finish()