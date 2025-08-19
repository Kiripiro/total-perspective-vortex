from __future__ import annotations
from pathlib import Path
import json
import hashlib
import time
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import sklearn

from config import BCIConfig


class ModelRepository:
    """Handles saving/loading subject-run and subject-experiment models.

    - Filenames encode scope (subject, run or experiment).
    - Metadata JSON stores config hash to prevent incompatible loads.
    - Write uses temp file rename for basic atomicity.
    """

    SUBJECT_DIR_PATTERN = "subject_{sid:03d}"

    def __init__(self, config: BCIConfig):
        self.config = config
        self.root = Path(config.models_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_subject_run(
        self,
        subject: int,
        run: int,
        pipeline: Any,
        *,
        test_acc: Optional[float],
        inner_cv_acc: Optional[float],
        n_trainval: int,
        test_size: float,
        n_features: Optional[int],
    ) -> Path:
        return self._save(
            scope_id=self._subject_run_id(subject, run),
            subject=subject,
            meta_extra={
                "scope": "subject_run",
                "subject": subject,
                "run": run,
                "test_acc": test_acc,
                "inner_cv_acc": inner_cv_acc,
                "n_trainval": n_trainval,
                "test_size": test_size,
                "n_features": n_features,
            },
            pipeline=pipeline,
        )

    def load_subject_run(
        self, subject: int, run: int, strict_config: bool = True
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        return self._load(self._subject_run_id(subject, run), subject, strict_config)

    def save_subject_experiment(
        self,
        subject: int,
        experiment_id: int,
        pipeline: Any,
        *,
        test_acc: Optional[float],
        n_train: int,
        test_size: float,
    ) -> Path:
        return self._save(
            scope_id=self._subject_experiment_id(subject, experiment_id),
            subject=subject,
            meta_extra={
                "scope": "subject_experiment",
                "subject": subject,
                "experiment_id": experiment_id,
                "test_acc": test_acc,
                "n_train": n_train,
                "test_size": test_size,
            },
            pipeline=pipeline,
        )

    def load_subject_experiment(
        self, subject: int, experiment_id: int, strict_config: bool = True
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        return self._load(
            self._subject_experiment_id(subject, experiment_id), subject, strict_config
        )

    @staticmethod
    def _subject_run_id(subject: int, run: int) -> str:
        return f"S{subject:03d}_run_{run:02d}"

    @staticmethod
    def _subject_experiment_id(subject: int, experiment_id: int) -> str:
        return f"S{subject:03d}_exp_{experiment_id}"

    def _subject_dir(self, subject: int) -> Path:
        d = self.root / self.SUBJECT_DIR_PATTERN.format(sid=subject)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _config_hash(self) -> str:
        items = sorted(self.config.__dict__.items())
        raw = json.dumps(items, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()[:12]

    def _paths(self, subject: int, scope_id: str) -> Tuple[Path, Path]:
        d = self._subject_dir(subject)
        return d / f"{scope_id}.pkl", d / f"{scope_id}.meta.json"

    def _save(
        self,
        scope_id: str,
        subject: int,
        meta_extra: Dict[str, Any],
        pipeline: Any,
    ) -> Path:
        pkl_path, meta_path = self._paths(subject, scope_id)
        tmp = pkl_path.with_suffix('.pkl.tmp')
        joblib.dump(pipeline, tmp)
        tmp.replace(pkl_path)
        meta = {
            **meta_extra,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "config_hash": self._config_hash(),
            "config": self.config.__dict__,
            "versions": {
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return pkl_path

    def _load(
        self, scope_id: str, subject: int, strict_config: bool
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        pkl_path, meta_path = self._paths(subject, scope_id)
        if not pkl_path.exists() or not meta_path.exists():
            return None
        try:
            pipe = joblib.load(pkl_path)
            meta = json.loads(meta_path.read_text())
        except Exception:
            return None
        if strict_config and meta.get("config_hash") != self._config_hash():
            return None
        return pipe, meta


__all__ = ["ModelRepository"]
