from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class BCIConfig:
    """Configuration for BCI motor imagery classification.

    NOTE: Some fields appear twice previously due to a merge oversight; duplicates removed.
    """

    fmin: float = 8.0
    fmax: float = 32.0
    tmin: float = 0.7
    tmax: float = 3.9
    eeg_channels: List[str] = None
    sfreq: float = 160.0

    n_csp: int = 3
    use_scaler: bool = True
    use_feature_selection: bool = True
    feature_selection_percentiles: Tuple[int, ...] = (70, 85, 100)
    lda_shrinkage: bool = True

    cv_folds: int = 5
    inner_repeats: int = 3
    test_size: float = 0.2
    random_state: int = 42

    save_min_test_acc: float = 0.0
    force_predict_cv: bool = False
    models_dir: str = "models"
    save_models_in_evaluate_all: bool = True

    def __post_init__(self):
        if self.eeg_channels is None:
            object.__setattr__(self, 'eeg_channels', ["C3", "C4", "Cz"])


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for experiments."""
    
    runs: Dict[int, List[int]] = None
    
    def __post_init__(self):
        if self.runs is None:
            object.__setattr__(self, 'runs', {
                0: [3, 7, 11],
                1: [4, 8, 12],
                2: [5, 9, 13],
                3: [6, 10, 14],
                4: [3, 4, 7, 8, 11, 12],
                5: [5, 6, 9, 10, 13, 14],
            })


DEFAULT_BCI_CONFIG = BCIConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
