#!/usr/bin/env python3
import os
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt

# Paramètres globaux
subjects = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006']
runs = ['R03', 'R04', 'R05', 'R06', 'R07', 'R08']  # Mouvements exécutés

# Ces listes correspondent aux canaux d'intérêt dans la littérature (informatives)
channels_erd_ers = ['C3', 'C4', 'CZ']
channels_mrcp = ['FC3', 'FCZ', 'FC4','C3', 'C1', 'CZ', 'C2', ' C4']

# Utilisation des annotations T1 (Left) et T2 (Right)
# Après extraction, T1 est généralement associé à la valeur 2 et T2 à 3.
event_id = {'T1': 'Left', 'T2': 'Right'}

def clean_channel_name(ch_name):
    # Supprime les points et met en majuscules.
    return ch_name.replace('.', '').upper()

def load_and_visualize_raw(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    print(f"Chargement de {file_path} : {raw.info['sfreq']} Hz, {len(raw.ch_names)} canaux")
    for ch_name in raw.ch_names:
        raw.rename_channels({ch_name: clean_channel_name(ch_name)})
    montage = mne.channels.make_standard_montage('standard_1020')
    print("Channels avant montage :", raw.info['ch_names'])
    raw.set_montage(montage, match_case=False)
    print("Channels après montage :", raw.info['ch_names'])
    # Re-référencement à la moyenne (important pour l'analyse EEG)
    raw.set_eeg_reference('average', projection=False)
    return raw

# Prétraitement amélioré avec ICA sur l'ensemble des canaux (8 composants)
def preprocess_data(raw):
    # Filtrage passe-bande de 0.5 à 79 Hz pour respecter la contrainte de Nyquist
    raw.filter(0.5, 79, fir_design='firwin', verbose=False)
    # Filtrage Notch pour éliminer la ligne 50 Hz
    raw.notch_filter(50, filter_length='auto', phase='zero', verbose=False)
    
    # Le re-référencement a déjà été fait dans load_and_visualize_raw
    # (vous pouvez aussi le faire ici si nécessaire)
    
    # Application de l'ICA
    ica = ICA(n_components=8, random_state=42, verbose=False)
    ica.fit(raw)
    
    # Détection automatique des artefacts EOG via les canaux frontaux
    try:
        # On utilise FP1, FPZ et FP2 comme canaux représentatifs des mouvements oculaires.
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['FP1', 'FPZ', 'FP2'], threshold=3.0)
        print("Indices EOG détectés :", eog_indices)
        ica.exclude = eog_indices  # Exclusion automatique des composants corrélés aux EOG
    except Exception as e:
        print("Erreur dans la détection EOG :", e)
    
    # Application de l'ICA (avec exclusion des composants détectés)
    raw_clean = ica.apply(raw.copy())
    return raw_clean, ica

# Extraction des epochs sans restriction de canaux (picks=None)
def extract_epochs(raw, event_id, tmin, tmax, picks=None):
    try:
        events = mne.find_events(raw, stim_channel='STI014', verbose=False)
    except Exception:
        print("Canal 'STI014' non trouvé. Extraction des événements à partir des annotations.")
        events, annot_event_id = mne.events_from_annotations(raw)
        event_id = {k: v for k, v in annot_event_id.items() if k in event_id}
        if not event_id:
            raise ValueError("Aucun événement correspondant à event_id trouvé dans les annotations.")
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        picks=picks, preload=True, baseline=None, verbose=False)
    return epochs

# Extraction des caractéristiques : calcule la moyenne (mean), le power (moyenne du PSD) et l'energy (somme du PSD)
def extract_features(epochs, freq_range):
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    mean_val = np.mean(data, axis=2)
    n_times = data.shape[-1]
    sfreq = epochs.info['sfreq']
    n_fft = 2 ** int(np.ceil(np.log2(n_times)))
    n_per_seg = n_times if n_fft > n_times else None
    psd, _ = psd_array_welch(data, sfreq=sfreq, fmin=freq_range[0],
                             fmax=freq_range[1], n_fft=n_fft,
                             n_per_seg=n_per_seg, verbose=False)
    power = np.mean(psd, axis=2)
    energy = np.sum(psd, axis=2)
    return mean_val, power, energy

# Recodage des labels : T1 (valeur 2) -> 0 (Left), T2 (valeur 3) -> 1 (Right)
def recode_labels(y):
    return np.where(y == 2, 0, np.where(y == 3, 1, y))

# Pipeline principal
def run_pipeline(dataset_path, feature_set='P_M_E_X'):
    # feature_set peut être 'P_E_X' ou 'P_M_E_X'
    X_list = []
    y_list = []
    
    for subject in subjects:
        for run in runs:
            file_path = os.path.join(dataset_path, subject, f"{subject}{run}.edf")
            if not os.path.exists(file_path):
                print(f"Fichier manquant : {file_path}")
                continue
            raw = load_and_visualize_raw(file_path)
            raw_clean, ica = preprocess_data(raw)
            
            # Extraction des epochs pour trois intervalles temporels :
            epochs_erd = extract_epochs(raw_clean, event_id=event_id, tmin=-2, tmax=0, picks=None)
            print(epochs_erd)
            epochs_ers = extract_epochs(raw_clean, event_id=event_id, tmin=4.1, tmax=5.1, picks=None)
            print(epochs_ers)
            epochs_mrcp = extract_epochs(raw_clean, event_id=event_id, tmin=-2, tmax=0, picks=None)
            print(epochs_mrcp)
            
            # Transformation par ICA pour obtenir les sources
            sources_erd = ica.get_sources(epochs_erd)
            sources_ers = ica.get_sources(epochs_ers)
            sources_mrcp = ica.get_sources(epochs_mrcp)
            
            # Extraction des caractéristiques sur les sources ICA :
            # Pour ERD et ERS, bande 8-30 Hz, pour MRCP, bande 0.5-3 Hz.
            mean_erd, power_erd, energy_erd = extract_features(sources_erd, [8, 30])
            mean_ers, power_ers, energy_ers = extract_features(sources_ers, [8, 30])
            mean_mrcp, power_mrcp, energy_mrcp = extract_features(sources_mrcp, [0.5, 3])
            
            # Construction des vecteurs de caractéristiques
            if feature_set == 'P_E_X':
                feat_erd = np.hstack([power_erd, energy_erd, np.zeros((power_erd.shape[0], 1))])
                feat_ers = np.hstack([power_ers, energy_ers, np.ones((power_ers.shape[0], 1))])
                feat_mrcp = np.hstack([power_mrcp, energy_mrcp, 2 * np.ones((power_mrcp.shape[0], 1))])
            elif feature_set == 'P_M_E_X':
                feat_erd = np.hstack([power_erd, mean_erd, energy_erd, np.zeros((power_erd.shape[0], 1))])
                feat_ers = np.hstack([power_ers, mean_ers, energy_ers, np.ones((power_ers.shape[0], 1))])
                feat_mrcp = np.hstack([power_mrcp, mean_mrcp, energy_mrcp, 2 * np.ones((power_mrcp.shape[0], 1))])
            else:
                raise ValueError("Choix de feature_set non reconnu")
            
            # Extraction des labels depuis epochs_erd (supposés constants pour le run)
            y_run = recode_labels(epochs_erd.events[:, -1])
            n_erd = feat_erd.shape[0]
            n_ers = feat_ers.shape[0]
            n_mrcp = feat_mrcp.shape[0]
            y_run_combined = np.concatenate([y_run[:n_erd], y_run[:n_ers], y_run[:n_mrcp]])
            X_run = np.vstack([feat_erd, feat_ers, feat_mrcp])
            
            X_list.append(X_run)
            y_list.append(y_run_combined)
    
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    print(f"Forme finale des données : X={X_all.shape}, y={y_all.shape}")
    
    # Normalisation des features par colonne dans l'intervalle [0.1, 0.9]
    X_min = X_all.min(axis=0)
    X_max = X_all.max(axis=0)
    X_all = 0.1 + 0.8 * (X_all - X_min) / (X_max - X_min + 1e-6)
    
    # Définition de la grille de recherche pour le SVM
    param_grid = {
        'kernel': ['poly', 'rbf'],
        'degree': [3, 4, 5],   # Pertinent seulement pour le noyau 'poly'
        'gamma': [0.1, 1, 4, 10],
        'C': [0.1, 1, 10]
    }
    
    # Utilisation de StratifiedKFold pour la validation croisée
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    svm = SVC()
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(svm, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_all, y_all)
    
    print("Meilleurs paramètres trouvés :", grid_search.best_params_)
    print("Meilleure précision en validation croisée :", grid_search.best_score_)
    
    # Évaluation du meilleur estimateur sur chaque fold
    best_svm = grid_search.best_estimator_
    acc_train_list = []
    acc_test_list = []
    
    for train_idx, test_idx in skf.split(X_all, y_all):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        best_svm.fit(X_train, y_train)
        acc_train_list.append(best_svm.score(X_train, y_train))
        acc_test_list.append(best_svm.score(X_test, y_test))
    
    print(f"Précision moyenne entraînement : {np.mean(acc_train_list):.3f} ± {np.std(acc_train_list):.3f}")
    print(f"Précision moyenne test : {np.mean(acc_test_list):.3f} ± {np.std(acc_test_list):.3f}")
    
    plt.show()

if __name__ == "__main__":
    dataset_path = os.path.expanduser("~/sgoinfre/eegmmidb")
    run_pipeline(dataset_path, feature_set='P_M_E_X')
