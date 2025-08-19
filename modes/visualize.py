from __future__ import annotations
import numpy as np
from .base import ModeBase
from utils.visualization import EEGVisualizer, EEGVisualizationError


class VisualizeMode(ModeBase):
    LOG_NAME = 'VisualizeMode'
    LOG_FILE = 'visualize.log'

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.visualizer = EEGVisualizer(config)

    def execute(self, subject: int, run: int, dataset_dir: str, show: bool=False, interactive: bool=False, viewer: str='mne') -> None:
        try:
            report = self.visualizer.visualize_subject_run(subject, run, dataset_dir, keep_open=False, interactive=interactive, viewer=viewer)
            self.logger.success(f"Visualisation OK -> {report['output_dir']}")
            bp_vals = report.get('band_power_values', {})
            if bp_vals:
                mean_bp = np.mean(list(bp_vals.values()))
                self.logger.info(f'Mean band power: {mean_bp:.4f}')
            if show and not interactive:
                import subprocess, os
                for key in ['raw_plot','raw_psd','filtered_plot','filtered_psd','band_power_plot','topomap']:
                    p = report.get(key)
                    if p and os.path.exists(p):
                        subprocess.Popen(['open', p])
            if interactive:
                import matplotlib.pyplot as plt
                plt.show()
        except FileNotFoundError as e:
            self.logger.error(str(e))
        except EEGVisualizationError as e:
            self.logger.error(f'Visualization error: {e}')
        except Exception as e:
            self.logger.error(f'Unexpected error: {e}')
        finally:
            self._finish()
