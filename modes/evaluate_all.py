from __future__ import annotations
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split

from config import BCIConfig, DEFAULT_EXPERIMENT_CONFIG
from utils.data_loader import BCIDataLoader
from utils.logger import get_logger, setup_file_logging, create_progress_logger
from pipeline import BCIClassifier
from utils.model import ModelRepository
from utils.system_info import get_system_info


class EvaluateAllMode:
    """Runs all experiments for all subjects using multiprocessing."""
    def __init__(self, config: BCIConfig, max_workers: Optional[int] = None):
        self.config = config
        self.data_loader = BCIDataLoader(config)
        self.runs = DEFAULT_EXPERIMENT_CONFIG.runs
        sysinfo = get_system_info()
        avail = sysinfo['available_cores']
        optimal = sysinfo['optimal_workers']
        requested = max_workers
        if requested is None:
            chosen = optimal
            warn_msgs: List[str] = []
        else:
            warn_msgs = []
            if requested < 1:
                chosen = 1
                warn_msgs.append(f"Requested max_workers={requested} < 1; using 1")
            elif requested > avail:
                chosen = avail
                warn_msgs.append(f"Requested max_workers={requested} > available_cores={avail}; capping to {avail}")
            else:
                chosen = requested
        self.max_workers = chosen
        self.logger = get_logger('EvaluateAll')
        setup_file_logging(Path('logs'), 'evaluate.log')
        for msg in warn_msgs:
            self.logger.warning(msg)

    def execute(self, dataset_dir: str) -> None:
        subjects = [d for d in Path(dataset_dir).iterdir() if d.is_dir() and d.name.startswith('S')]
        subject_ids = [int(d.name[1:]) for d in subjects]
        total = len(subjects)
        exp_count = len(self.runs)
        self.logger.info(f'Evaluating {total} subjects × {exp_count} experiments')
        progress = create_progress_logger(total * exp_count, 'Overall')

        self.logger.info('Preloading data...')
        subject_data: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
        for d in subjects:
            sid = int(d.name[1:])
            exp_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for eid, runs in self.runs.items():
                Xs, ys = [], []
                for r in runs:
                    try:
                        Xr, yr = self.data_loader.load_epochs(sid, r, dataset_dir)
                        if Xr.size:
                            Xs.append(Xr); ys.append(yr)
                    except FileNotFoundError:
                        continue
                if Xs:
                    exp_data[eid] = (np.concatenate(Xs), np.concatenate(ys))
            subject_data[sid] = exp_data

        results: List[List[float]] = [[np.nan] * total for _ in range(exp_count)]
        tasks = [(idx, int(d.name[1:])) for idx, d in enumerate(subjects)]

        self.logger.info(f'Starting parallel evaluation with {self.max_workers} workers')
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
                futures = {exe.submit(self._eval_subject, sid, subject_data[sid]): idx for idx, sid in tasks}
                self.logger.info(f'Submitted {len(futures)} tasks')

                for fut in concurrent.futures.as_completed(futures):
                    idx = futures[fut]
                    try:
                        res = fut.result()
                        for eid, acc in res.items():
                            if eid < len(results):
                                results[eid][idx] = acc
                        progress(1)
                    except Exception as e:
                        self.logger.error(f'Worker failed for idx={idx}: {e}')
                        progress(1)
        except Exception as e:
            self.logger.error(f'Parallel evaluation error: {e}')

        self.logger.info('Aggregating results...')
        for eid in sorted(self.runs.keys()):
            for idx, sid in enumerate(subject_ids):
                acc = results[eid][idx]
                if not np.isnan(acc):
                    self.logger.info(f'Experiment {eid}: subject {sid:03d}: accuracy = {acc:.4f}')
            accs = [a for a in results[eid] if not np.isnan(a)]
            if accs:
                self.logger.success(f'Experiment {eid}: mean Test accuracy = {np.mean(accs):.4f}')
            else:
                self.logger.warning(f'Experiment {eid}: no valid results')
        self.logger.success(f'Global mean accuracy: {np.nanmean(results):.4f}')
        time.sleep(0.2)
        self.logger.stop()

    @staticmethod
    def _eval_subject(sid: int, exp_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, float]:
        out: Dict[int, float] = {}
        cfg = BCIConfig()
        repo = ModelRepository(cfg)
        for eid, (Xc, yc) in exp_data.items():
            loaded = repo.load_subject_experiment(sid, eid) if cfg.save_models_in_evaluate_all else None
            if loaded and not cfg.force_predict_cv:
                pipe, meta = loaded
                out[eid] = pipe.score(Xc, yc)
                continue
            X_train, X_test, y_train, y_test = train_test_split(
                Xc, yc, test_size=0.15, stratify=yc, random_state=cfg.random_state
            )
            clf = BCIClassifier(cfg)
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            out[eid] = acc
            if cfg.save_models_in_evaluate_all and acc >= cfg.save_min_test_acc:
                repo.save_subject_experiment(
                    sid,
                    eid,
                    clf.pipeline,
                    test_acc=acc,
                    n_train=len(y_train),
                    test_size=0.15,
                )
        return out

__all__ = ['EvaluateAllMode']
