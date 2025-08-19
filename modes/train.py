from __future__ import annotations
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from .base import ModeBase
from pipeline import BCIClassifier
from utils.model import ModelRepository
from utils.split import stratified_holdout


class TrainMode(ModeBase):
    LOG_NAME = 'TrainMode'
    LOG_FILE = 'train.log'

    def execute(self, subject: int, run: int, dataset_dir: str) -> None:
        self.logger.info(f'Start training S{subject:03d} run {run:02d}')
        test_acc = None
        level = 'info'
        try:
            X, y = self.data_loader.load_epochs(subject, run, dataset_dir)
            if X.size == 0 or len(np.unique(y)) != 2:
                self.logger.warning('Invalid data: empty or !=2 classes')
                return

            self.logger.info(f'Data: {X.shape[0]} epochs, classes={np.unique(y)}')

            baseline_clf = BCIClassifier(self.config)
            cv_scores = cross_val_score(
                baseline_clf.pipeline, X, y,
                cv=self.config.cv_folds,
                scoring='accuracy'
            )
            self.logger.info(f'baseline CV scores: {np.array2string(cv_scores, precision=4)} (mean={cv_scores.mean():.4f})')

            split = stratified_holdout(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            X_trainval = split['X_trainval']
            y_trainval = split['y_trainval']
            X_test = split['X_test']
            y_test = split['y_test']
            self.logger.info(f'Sizes trainval={len(y_trainval)} test={len(y_test)} (test_size={self.config.test_size})')

            inner_cv = RepeatedStratifiedKFold(
                n_splits=self.config.cv_folds,
                n_repeats=self.config.inner_repeats,
                random_state=self.config.random_state,
            )

            clf = BCIClassifier(self.config)
            best_pipe, best_score = clf.hyperparameter_search(
                X_trainval, y_trainval, cv=inner_cv
            )
            self.logger.info(f'Best inner CV acc={best_score:.4f}')
            test_acc = best_pipe.score(X_test, y_test)
            level = 'success' if test_acc >= 0.6 else 'warning'
            getattr(self.logger, level)(f'Test acc {test_acc:.4f} (thr=0.6)')
            if test_acc is not None and test_acc >= self.config.save_min_test_acc:
                repo = ModelRepository(self.config)
                try:
                    n_features = best_pipe[:-1].transform(X_trainval[:1]).shape[1]
                except Exception:
                    n_features = None
                repo.save_subject_run(
                    subject,
                    run,
                    best_pipe,
                    test_acc=test_acc,
                    inner_cv_acc=best_score,
                    n_trainval=len(y_trainval),
                    test_size=self.config.test_size,
                    n_features=n_features,
                )
                self.logger.info('Model persisted (subject-run).')

        finally:
            self._finish()
            if test_acc is not None:
                getattr(self.logger, level)(f'Final test acc {test_acc:.4f}')
