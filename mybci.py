import argparse
import logging
from typing import Optional
from pathlib import Path
from utils.sync_data import sync_from_s3

from config import BCIConfig
from utils.data_loader import BCIDataLoader
from utils.logger import get_logger
from modes.train import TrainMode
from modes.predict import PredictMode
from modes.visualize import VisualizeMode
from modes.evaluate_all import EvaluateAllMode


def _load_env(env_path: Path) -> dict:
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip()
    return env


def _ensure_dataset(dataset_dir: Path):
    if not dataset_dir.exists() or not any(dataset_dir.glob('S*')):
        sync_from_s3(str(dataset_dir))


class BCIApplication:
    def __init__(self, max_workers: Optional[int] = None):
        self.config = BCIConfig()
        self.data_loader = BCIDataLoader(self.config)
        self.max_workers = max_workers
        self.logger = get_logger('BCIApp')
        self.mode_map = {
            'train': TrainMode,
            'predict': PredictMode,
            'visualize': VisualizeMode,
        }

    def run(self, args):
        env = _load_env(Path(__file__).parent / '.env')
        dataset_dir = Path(env.get('DATA_DIR', 'data'))
        _ensure_dataset(dataset_dir)
        if args.subject is not None and args.run is not None:
            if not args.mode:
                if args.interactive or args.show:
                    args.mode = 'visualize'
                else:
                    raise ValueError('Mode required (train|predict|visualize) when subject and run are provided')
            mode_cls = self.mode_map.get(args.mode)
            if not mode_cls:
                raise ValueError(f'Unknown mode {args.mode}')
            if args.mode == 'visualize':
                mode_cls(self.config, self.data_loader).execute(
                    args.subject, args.run, str(dataset_dir),
                    show=args.show, interactive=args.interactive, viewer=args.viewer
                )
            else:
                mode_cls(self.config, self.data_loader).execute(args.subject, args.run, str(dataset_dir))
        else:
            EvaluateAllMode(self.config, self.max_workers).execute(str(dataset_dir))


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    parser = argparse.ArgumentParser(description='BCI motor imagery classification')
    parser.add_argument('subject', nargs='?', type=int)
    parser.add_argument('run', nargs='?', type=int)
    parser.add_argument('mode', choices=['train', 'predict', 'visualize'], nargs='?')
    parser.add_argument('--max_workers', type=int)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--viewer', choices=['mne','stacked'], default='mne')
    parser.add_argument('--force_cv', action='store_true', help='Force CV simulation in predict mode even if a saved model exists')
    args = parser.parse_args()
    app = BCIApplication(max_workers=args.max_workers)
    if args.force_cv:
        cfg_dict = {k: getattr(app.config, k) for k in app.config.__dataclass_fields__.keys()}
        cfg_dict['force_predict_cv'] = True
        app.config = BCIConfig(**cfg_dict)
        app.data_loader = BCIDataLoader(app.config)
    app.run(args)


if __name__ == '__main__':
    main()
