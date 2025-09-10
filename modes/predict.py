from __future__ import annotations
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
            n = len(y)
            self.logger.info(f'Data: {n} epochs')

            replay_sleep = min(getattr(self.config, 'replay_sleep', 1.0), 1.9)

            repo = ModelRepository(self.config)
            loaded = repo.load_subject_run(subject, run)
            if loaded:
                pipe, _ = loaded
                self.logger.info('Loaded persisted model; replaying per-epoch predictions.')
                self.logger.info("epoch nb: [prediction] [truth] equal?")
                correct = 0
                for idx in range(n):
                    pred = pipe.predict(X[idx:idx+1])[0]
                    ok = pred == y[idx]
                    correct += int(ok)
                    self.logger.info(f"epoch {idx:02d}: [{pred+1}] [{y[idx]+1}] {ok}")
                    time.sleep(replay_sleep)
                acc = correct / n
                self.logger.success(f'Accuracy: {acc:.4f}')
                return

            if getattr(self.config, 'force_predict_cv', False):
                models = {}
                skf = StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=self.config.random_state,
                )
                for tr, te in skf.split(X, y):
                    clf = BCIClassifier(self.config)
                    clf.fit(X[tr], y[tr])
                    for idx in te:
                        models[idx] = clf

                correct = 0
                self.logger.info("epoch nb: [prediction] [truth] equal?")
                for idx in range(n):
                    pred = models[idx].pipeline.predict(X[idx:idx+1])[0]
                    ok = pred == y[idx]
                    correct += int(ok)
                    self.logger.info(f"epoch {idx:02d}: [{pred+1}] [{y[idx]+1}] {ok}")
                    time.sleep(replay_sleep)
                acc = correct / n
                self.logger.success(f'Accuracy: {acc:.4f}')
                return

            self.logger.error('No persisted model. Train first or set --force_cv to run CV-based replay')
        finally:
            self._finish()