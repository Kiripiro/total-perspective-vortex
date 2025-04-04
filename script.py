import os
import mne
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from mne.time_frequency import psd_array_multitaper
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------- Utilitaires -----------------------------

def clean_channel_name(ch_name):
    return ch_name.replace('.', '').upper()

def sync_files_from_s3(destination):
    s3_command = (
        f"aws s3 sync --no-sign-request "
        f"s3://physionet-open/eegmmidb/1.0.0/ {destination}"
    )
    print("Synchronisation des fichiers depuis S3...")
    try:
        subprocess.run(s3_command, shell=True, check=True)
        print("Synchronisation réussie.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la synchronisation : {e}")
        exit(1)

def get_all_edf_files(data_dir):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(data_dir)
        for file in files
        if file.lower().endswith(".edf")
    ]

def filter_task_files(edf_files, runs=['R03', 'R04', 'R07', 'R08', 'R11', 'R12']):
    return [f for f in edf_files if any(r in f for r in runs)]

# -------------------------- Prétraitement EEG --------------------------

def preprocess_raw(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.set_eeg_reference('average', projection=False)  # directement appliquée
    raw.rename_channels({ch: clean_channel_name(ch) for ch in raw.ch_names})
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    raw.filter(4, 40, fir_design='firwin', verbose=False)
    return raw

def extract_valid_epochs(raw, event_id, tmin=-0.5, tmax=4):
    try:
        events, annot_event_id = mne.events_from_annotations(raw)
        valid_event_id = {k: v for k, v in event_id.items() if k in annot_event_id}
        if not valid_event_id:
            print("Aucune annotation trouvée.")
            return None
        epochs = mne.Epochs(raw, events, event_id=valid_event_id, tmin=tmin, tmax=tmax,
                            picks='eeg', preload=True, baseline=None, verbose=False)
        return epochs
    except Exception as e:
        print(f"Erreur extraction epochs : {e}")
        return None

# --------------------------- Visualisation -----------------------------

def save_psd_visualizations(psds, freqs, labels, raw, output_dir, filename_prefix=""):
    os.makedirs(output_dir, exist_ok=True)

    # PSD C3 vs C4
    try:
        c3_idx = raw.ch_names.index("C3")
        c4_idx = raw.ch_names.index("C4")
        plt.figure(figsize=(8, 4))
        for label_id in np.unique(labels):
            psd_c3 = psds[labels == label_id, c3_idx, :].mean(axis=0)
            psd_c4 = psds[labels == label_id, c4_idx, :].mean(axis=0)
            plt.plot(freqs, psd_c3, label=f"T{label_id} - C3", linestyle='--')
            plt.plot(freqs, psd_c4, label=f"T{label_id} - C4", linestyle='-')
        plt.title("PSD - C3 vs C4")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Puissance (uV²/Hz)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}PSD_C3_vs_C4.png"))
        plt.close()
    except ValueError:
        print("Canaux C3 ou C4 non trouvés.")

    # Heatmaps par classe
    for label_id in np.unique(labels):
        psd_class = psds[labels == label_id].mean(axis=0)
        plt.figure(figsize=(10, 6))
        sns.heatmap(psd_class, xticklabels=10, yticklabels=raw.ch_names, cmap="viridis")
        plt.title(f"Heatmap PSD - Classe T{label_id}")
        plt.xlabel("Fréquences (Hz)")
        plt.ylabel("Canaux EEG")
        xtick_freqs = freqs[::10].astype(int)
        plt.xticks(ticks=np.arange(0, len(freqs), 10), labels=xtick_freqs, rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}heatmap_T{label_id}.png"))
        plt.close()

# ----------------------------- Main Script -----------------------------

def main():
    data_dir = os.path.expanduser("~/sgoinfre/eegmmidb")
    os.makedirs(data_dir, exist_ok=True)

    edf_files = get_all_edf_files(data_dir)

    if not edf_files:
        sync_files_from_s3(data_dir)
        edf_files = get_all_edf_files(data_dir)

    print(f"Nombre total de fichiers EDF : {len(edf_files)}")

    edf_files_tasks = filter_task_files(edf_files)
    print(f"Nombre de fichiers pour les tâches motrices : {len(edf_files_tasks)}")

    X_all = []
    y_all = []

    event_id_input = {'T0': 1, 'T1': 2, 'T2': 3}  # T1 = main gauche, T2 = main droite

    for edf_file in edf_files_tasks:
        print(f"Traitement : {edf_file}")
        raw = preprocess_raw(edf_file)
        epochs = extract_valid_epochs(raw, event_id_input)

        if epochs is None:
            continue

        # Ne garder que les essais T1 et T2 (main gauche et droite)
        epochs = epochs.copy().filter(None, None)  # Juste pour forcer la copie complète
        events = epochs.events[:, 2]
        mask = np.isin(events, [2, 3])
        epochs = epochs[mask]
        labels = events[mask]
        labels_bin = (labels == 3).astype(int)  # 0 = gauche (T1), 1 = droite (T2)

        # Sauvegarde des visualisations PSD classiques
        psds, freqs = psd_array_multitaper(epochs.get_data(), sfreq=epochs.info['sfreq'], fmin=4, fmax=40)
        subject_id = os.path.basename(edf_file).split('.')[0]
        output_dir = os.path.join("outputs", "psd_plots", subject_id)
        save_psd_visualizations(psds, freqs, labels, raw, output_dir, filename_prefix=subject_id + "_")

        X_all.append(epochs.get_data())  # (n_epochs, n_channels, n_times)
        y_all.append(labels_bin)

    # Fusion des fichiers
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    print(f"Forme finale X : {X.shape}, y : {y.shape}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Pipeline CSP + SVM
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    clf = SVC(kernel='linear', C=1)

    pipe = Pipeline([
        ('CSP', csp),
        ('SVM', clf)
    ])

    # Entraînement et test
    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)
    print(f"✅ Précision CSP + SVM (main gauche vs droite) : {accuracy:.4f}")

    # Visualisation des patterns CSP
    print("Affichage des patterns spatiaux CSP...")
    csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (a.u.)', size=1.5)

if __name__ == "__main__":
    main()
