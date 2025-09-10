from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Sequence

import numpy as np

from utils.matplotlib_config import configure_matplotlib_backend
configure_matplotlib_backend()

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import mne
from mne.datasets.eegbci import standardize
from mne.channels import make_standard_montage
from scipy.signal import welch

from config import BCIConfig
from utils.logger import get_logger


class EEGVisualizationError(RuntimeError):
    """Domain specific exception for visualization errors."""
    pass


class EEGVisualizer:
    """Service object producing EEG visual artifacts.

    Responsibilities:
    - Load & minimally standardize raw file (no filtering) for raw view.
    - Apply band-pass filtering + channel selection per config.
    - Generate & persist figures (raw traces, PSDs, filtered traces, band power, topomap).
    - Return a lightweight report (dict) summarizing outputs.

    External side-effects (files) isolated here.
    """

    def __init__(self, config: BCIConfig, output_root: Path | str = 'figures', logger=None):
        self.config = config
        self.output_root = Path(output_root)
        self.logger = logger or get_logger('EEGVisualizer')
        if plt is None:
            self.logger.warning('matplotlib not available: visualization disabled')

    def visualize_subject_run(self, subject: int, run: int, dataset_dir: str, keep_open: bool=False, interactive: bool=False, viewer: str='mne') -> Dict[str, Any]:
        """Produce all figures for a subject/run.

        If interactive=True, create simpler matplotlib figures (no MNE raw.plot) to avoid backend crashes.
        Adds: full set raw (all channels), full set filtered (all channels), then selected subset.
        """
        if plt is None:
            raise EEGVisualizationError('matplotlib not installed')
        edf_path = self._edf_path(subject, run, dataset_dir)
        if not edf_path.exists():
            raise FileNotFoundError(f'EDF file not found: {edf_path}')
        out_dir = self.output_root / f'subject_{subject:03d}_run_{run:02d}'
        out_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Visualization output -> {out_dir}')
        raw = self._load_raw_minimal(edf_path)
        if interactive:
            return self._visualize_interactive(raw, out_dir, viewer=viewer)
        artifacts: Dict[str, Any] = {'output_dir': str(out_dir)}
        if keep_open:
            artifacts['fig_objects'] = {}
        artifacts['raw_all'] = self._plot_stacked(raw, out_dir / '00_raw_all.png', title='Raw EEG (all channels)')
        artifacts['raw_all_psd'] = self._plot_psd(raw, out_dir / '01_raw_all_psd.png', fmin=1, fmax=60, keep_open=keep_open, fig_store=artifacts.get('fig_objects'))
        raw_filt_all = self._bandpass(raw)
        artifacts['filtered_all'] = self._plot_stacked(raw_filt_all, out_dir / '02_filtered_all.png', title=f'Filtered {self.config.fmin}-{self.config.fmax} Hz (all)')
        artifacts['filtered_all_psd'] = self._plot_psd(raw_filt_all, out_dir / '03_filtered_all_psd.png', fmin=self.config.fmin, fmax=self.config.fmax, keep_open=keep_open, fig_store=artifacts.get('fig_objects'))
        raw_filt_sel = self._select_channels(raw_filt_all)
        artifacts['filtered_selected'] = self._plot_stacked(raw_filt_sel, out_dir / '04_filtered_selected.png', title='Filtered (selected)', scale_by_std=True)
        artifacts['filtered_selected_psd'] = self._plot_psd(raw_filt_sel, out_dir / '05_filtered_selected_psd.png', fmin=self.config.fmin, fmax=self.config.fmax, keep_open=keep_open, fig_store=artifacts.get('fig_objects'))
        band_power, freqs = self._compute_band_power(raw_filt_sel)
        artifacts['band_power_plot'] = self._plot_band_power(raw_filt_sel.ch_names, band_power, out_dir / '06_band_power.png', keep_open=keep_open, fig_store=artifacts.get('fig_objects'))
        if len(raw_filt_sel.ch_names) >= 3:
            artifacts['topomap'] = self._plot_topomap(band_power, raw_filt_sel, out_dir / '07_topomap.png', keep_open=keep_open, fig_store=artifacts.get('fig_objects'))
        artifacts['band_power_values'] = {ch: float(v) for ch, v in zip(raw_filt_sel.ch_names, band_power)}
        artifacts['freqs_sample'] = freqs[:10].tolist()
        return artifacts

    def _edf_path(self, subject: int, run: int, dataset_dir: str) -> Path:
        return Path(dataset_dir) / f'S{subject:03d}' / f'S{subject:03d}R{run:02d}.edf'

    def _load_raw_minimal(self, edf_path: Path) -> mne.io.Raw:
        self.logger.info(f'Loading raw EDF: {edf_path}')
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        standardize(raw)
        raw.set_montage(make_standard_montage('standard_1005'), verbose=False)
        return raw

    def _bandpass(self, raw: mne.io.Raw) -> mne.io.Raw:
        self.logger.info(f'Filtering (retain all channels) {self.config.fmin}-{self.config.fmax} Hz')
        return raw.copy().filter(l_freq=self.config.fmin, h_freq=self.config.fmax, fir_design='firwin', verbose=False)

    def _select_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        self.logger.info(f'Selecting subset channels {self.config.eeg_channels}')
        rf = raw.copy()
        picks = mne.pick_channels(rf.info['ch_names'], include=self.config.eeg_channels)
        rf.pick(picks)
        return rf

    def _plot_raw(self, raw: mne.io.Raw, out_file: Path, n_channels: int = 10, keep_open: bool=False, fig_store=None) -> Optional[str]:
        try:
            fig = raw.plot(n_channels=min(n_channels, len(raw.ch_names)), duration=10, scalings='auto', show=False)
            fig.savefig(out_file, dpi=120)
            if keep_open and fig_store is not None:
                fig_store['raw_plot'] = fig
            else:
                import matplotlib.pyplot as plt_local
                plt_local.close(fig)
            self.logger.debug(f'Saved raw plot: {out_file}')
            return str(out_file)
        except Exception as e:
            self.logger.warning(f'Raw plot failed: {e}')
            return None

    def _plot_psd(self, raw: mne.io.Raw, out_file: Path, fmin: float, fmax: float, keep_open: bool=False, fig_store=None) -> Optional[str]:
        try:
            psd = raw.compute_psd(fmin=fmin, fmax=fmax)
            fig = psd.plot(show=False)
            fig.savefig(out_file, dpi=120)
            if keep_open and fig_store is not None:
                fig_store['raw_psd' if 'raw' in str(out_file) else 'filtered_psd'] = fig
            else:
                import matplotlib.pyplot as plt_local
                plt_local.close(fig)
            self.logger.debug(f'Saved PSD: {out_file}')
            return str(out_file)
        except Exception as e:
            self.logger.warning(f'PSD plot failed: {e}')
            return None

    def _compute_band_power(self, raw: mne.io.Raw):
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        freqs, pxx = welch(data, fs=sfreq, nperseg=min(256, data.shape[1]))
        mask = (freqs >= self.config.fmin) & (freqs <= self.config.fmax)
        band_power = pxx[:, mask].mean(axis=1)
        return band_power, freqs

    def _plot_band_power(self, ch_names: Sequence[str], band_power: np.ndarray, out_file: Path, keep_open: bool=False, fig_store=None) -> Optional[str]:
        try:
            import matplotlib.pyplot as plt_local
            fig, ax = plt_local.subplots(figsize=(6, 3))
            ax.bar(ch_names, band_power)
            ax.set_title('Mean Band Power')
            ax.set_ylabel('Power (a.u.)')
            ax.set_xlabel('Channel')
            fig.tight_layout()
            fig.savefig(out_file, dpi=120)
            if keep_open and fig_store is not None:
                fig_store['band_power_plot'] = fig
            else:
                plt_local.close(fig)
            self.logger.debug(f'Saved band power: {out_file}')
            return str(out_file)
        except Exception as e:
            self.logger.warning(f'Band power plot failed: {e}')
            return None

    def _plot_topomap(self, band_power: np.ndarray, raw: mne.io.Raw, out_file: Path, keep_open: bool=False, fig_store=None) -> Optional[str]:
        try:
            import matplotlib.pyplot as plt_local
            fig, ax = plt_local.subplots(figsize=(4, 4))
            mne.viz.plot_topomap(band_power, raw.info, axes=ax, show=False)
            ax.set_title('Band Power Topomap')
            fig.tight_layout()
            fig.savefig(out_file, dpi=120)
            if keep_open and fig_store is not None:
                fig_store['topomap'] = fig
            else:
                plt_local.close(fig)
            self.logger.debug(f'Saved topomap: {out_file}')
            return str(out_file)
        except Exception as e:
            self.logger.warning(f'Topomap failed: {e}')
            return None

    def _plot_stacked(self, raw: mne.io.Raw, out_file: Path, title: str, scale_by_std: bool=False) -> Optional[str]:
        try:
            if plt is None:
                return None
            fig = self._make_stacked_figure(raw, title, scale_by_std=scale_by_std)
            fig.savefig(out_file, dpi=120)
            plt.close(fig)
            return str(out_file)
        except Exception as e:
            self.logger.warning(f'Stacked plot failed: {e}')
            return None

    def _make_stacked_figure(self, raw: mne.io.Raw, title: str, scale_by_std: bool=False):
        import matplotlib.pyplot as plt_local
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        t = np.arange(data.shape[1]) / sfreq
        fig, ax = plt_local.subplots(figsize=(10, 0.25 * data.shape[0] + 2))
        if scale_by_std:
            spacings = data.std(axis=1)
            spacing = np.nanmedian(spacings) * 4 or 1.0
        else:
            spacing = (data.std(axis=1).mean()) * 4 or 1.0
        for i in range(len(raw.ch_names)):
            ax.plot(t, data[i] + i * spacing, linewidth=0.6)
        ax.set_yticks([])
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def _visualize_interactive(self, raw: mne.io.Raw, out_dir: Path, viewer: str='mne') -> Dict[str, Any]:
        import matplotlib.pyplot as plt_local
        artifacts: Dict[str, Any] = {'output_dir': str(out_dir), 'figures': []}
        if viewer == 'mne':
            raw_browser = raw.plot(block=False, title='Raw EEG (all channels)')
            artifacts['figures'].append(raw_browser)
        else:
            fig_raw = self._make_stacked_figure(raw, 'Raw EEG (all channels)')
            artifacts['figures'].append(fig_raw)
        raw_filt_all = self._bandpass(raw)
        if viewer == 'mne':
            filt_browser = raw_filt_all.plot(block=False, title=f'Filtered {self.config.fmin}-{self.config.fmax} Hz (all)')
            artifacts['figures'].append(filt_browser)
        else:
            fig_filt = self._make_stacked_figure(raw_filt_all, f'Filtered {self.config.fmin}-{self.config.fmax} Hz (all)')
            artifacts['figures'].append(fig_filt)
        raw_filt_sel = self._select_channels(raw_filt_all)
        fig_sel = self._make_stacked_figure(raw_filt_sel, 'Filtered (selected)')
        artifacts['figures'].append(fig_sel)
        data_sel = raw_filt_sel.get_data()
        sfreq = raw_filt_sel.info['sfreq']
        freqs_sel, pxx_sel = welch(data_sel, fs=sfreq, nperseg=min(512, data_sel.shape[1]))
        fig_sel_psd, ax_sel_psd = plt_local.subplots(figsize=(7, 4))
        for i, ch in enumerate(raw_filt_sel.ch_names):
            ax_sel_psd.semilogy(freqs_sel, pxx_sel[i], linewidth=0.8, label=ch)
        ax_sel_psd.set_xlim(0, min(60, freqs_sel.max()))
        ax_sel_psd.set_title('Selected Channels PSD')
        ax_sel_psd.set_xlabel('Hz')
        ax_sel_psd.set_ylabel('PSD')
        ax_sel_psd.legend(fontsize='x-small')
        fig_sel_psd.tight_layout()
        artifacts['figures'].append(fig_sel_psd)
        mask_sel = (freqs_sel >= self.config.fmin) & (freqs_sel <= self.config.fmax)
        band_power = pxx_sel[:, mask_sel].mean(axis=1)
        fig_bp, ax_bp = plt_local.subplots(figsize=(6, 3))
        ax_bp.bar(raw_filt_sel.ch_names, band_power)
        ax_bp.set_title(f'Mean {self.config.fmin}-{self.config.fmax} Hz Power (selected)')
        fig_bp.tight_layout()
        artifacts['figures'].append(fig_bp)
        if len(raw_filt_sel.ch_names) >= 3:
            fig_topo, ax_topo = plt_local.subplots(figsize=(4, 4))
            mne.viz.plot_topomap(band_power, raw_filt_sel.info, axes=ax_topo, show=False)
            ax_topo.set_title('Band Power Topomap')
            fig_topo.tight_layout()
            artifacts['figures'].append(fig_topo)
        artifacts['band_power_values'] = {ch: float(v) for ch, v in zip(raw_filt_sel.ch_names, band_power)}
        artifacts['freqs_sample'] = freqs_sel[:10].tolist()
        return artifacts

__all__ = ['EEGVisualizer', 'EEGVisualizationError']
