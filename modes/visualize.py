from __future__ import annotations
import numpy as np
from .base import ModeBase
from utils.visualization import EEGVisualizer, EEGVisualizationError

# Configure matplotlib backend for cross-platform compatibility
from utils.matplotlib_config import configure_matplotlib_backend
configure_matplotlib_backend()


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
                figures = report.get('figures', [])
                if figures:
                    self.logger.info(f"Displaying {len(figures)} interactive figures...")
                    for i, fig in enumerate(figures):
                        if hasattr(fig, 'show'):
                            fig.show()
                        elif hasattr(fig, 'canvas'):
                            fig.canvas.draw()
                    plt.show(block=True)
                    
                    out_dir = report['output_dir']
                    for key, path in [('raw_all', '00_raw_all.png'), 
                                    ('filtered_all', '02_filtered_all.png'), 
                                    ('filtered_selected', '04_filtered_selected.png'),
                                    ('filtered_selected_psd', '05_filtered_selected_psd.png'),
                                    ('band_power_plot', '06_band_power.png'),
                                    ('topomap', '07_topomap.png')]:
                        if key in report:
                            fig_idx = 0 if key == 'raw_all' else (1 if key == 'filtered_all' else 
                                     (2 if key == 'filtered_selected' else 
                                     (3 if key == 'filtered_selected_psd' else 
                                     (4 if key == 'band_power_plot' else 5))))
                            if fig_idx < len(figures) and hasattr(figures[fig_idx], 'savefig'):
                                from pathlib import Path
                                save_path = Path(out_dir) / path
                                figures[fig_idx].savefig(save_path, dpi=120)
                                self.logger.info(f"Saved figure: {save_path}")
                else:
                    self.logger.warning("No figures to display in interactive mode")
        except FileNotFoundError as e:
            self.logger.error(str(e))
        except EEGVisualizationError as e:
            self.logger.error(f'Visualization error: {e}')
        except Exception as e:
            self.logger.error(f'Unexpected error: {e}')
        finally:
            self._finish()
