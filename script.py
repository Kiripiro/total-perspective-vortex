import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib

from mne.preprocessing import ICA
from mne.decoding import CSP
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Fonction d'extraction des epochs ---
def extract_epochs(raw, event_id, tmin, tmax, picks='eeg'):
    """
    Extrait les epochs en utilisant le canal de stimulation 'STI014'
    ou, en cas d'absence, à partir des annotations.
    Utilise par défaut uniquement les canaux EEG.
    """
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

# --- Parcours des fichiers EDF dans le dataset ---
data_dir = os.path.expanduser("~/sgoinfre/eegmmidb")
if not os.path.exists(data_dir):
    raise ValueError(f"Le répertoire {data_dir} n'existe pas.")

edf_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith('.edf'):
            full_path = os.path.join(root, file)
            if os.path.exists(full_path):
                edf_files.append(full_path)
            else:
                print(f"Le fichier {full_path} n'existe pas.")

print(f"Nombre de fichiers EDF trouvés : {len(edf_files)}")

# --- Initialisation des listes pour accumuler les données extraites ---
features_all = []
labels_all = []
groups_all = []  # Pour stocker l'ID du sujet pour chaque epoch

# Paramètres communs
event_id = {'T1': 1, 'T2': 2}   # On exclut T0 (baseline)
tmin, tmax = 0, 2               # Fenêtre d'extraction (à ajuster si besoin)
# On peut aussi ajouter des bandes supplémentaires si nécessaire (ici jusqu'à 40Hz)
freq_bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30), (30, 40)]  
n_csp = 8  # Nombre de composantes CSP par bande

# --- Extraction des features sur l'ensemble des fichiers ---
for file in edf_files:
    print("Traitement de :", file)
    # Extraction de l'ID du sujet à partir du chemin : 
    # Par exemple, pour "/.../S076/S076R02.edf", on récupère "S076"
    subject_id = os.path.basename(os.path.dirname(file))
    
    try:
        raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
    except Exception as e:
        print(f"Erreur lors de la lecture de {file} : {e}")
        continue
    
    # Filtrage entre 1 et 40 Hz
    raw.filter(1, 40, fir_design='firwin', verbose=False)
    
    # Application de l'ICA pour éliminer les artefacts
    try:
        ica = ICA(n_components=20, random_state=97, max_iter=1000,
                  fit_params={'tol': 0.0001}, verbose=False)
        ica.fit(raw)
        raw_clean = raw.copy()
        ica.apply(raw_clean)
    except Exception as e:
        print(f"Erreur lors de l'ICA pour {file} : {e}")
        continue

    # Extraction des epochs (uniquement canaux EEG)
    try:
        epochs = extract_epochs(raw_clean, event_id, tmin, tmax, picks='eeg')
        unique_events = np.unique(epochs.events[:, -1])
        if not any(evt in unique_events for evt in event_id.values()):
            print(f"Fichier {file} semble être un run baseline (uniquement {unique_events}). On le saute.")
            continue
        print(f"Distribution des classes pour {file}: {np.unique(epochs.events[:, -1], return_counts=True)}")
    except Exception as e:
        print(f"Erreur lors de l'extraction des epochs pour {file} : {e}")
        continue

    # Récupération des labels
    labels = epochs.events[:, -1]
    
    # Pour chaque epoch, assigner l'ID du sujet (même valeur pour toutes les epochs du fichier)
    n_epochs = len(labels)
    groups_all.append(np.full(n_epochs, subject_id))
    
    # Extraction des caractéristiques via FBCSP pour chaque bande
    features_list = []
    for band in freq_bands:
        try:
            epochs_band = epochs.copy().filter(band[0], band[1], fir_design='firwin', verbose=False)
            csp = CSP(n_components=n_csp, reg=None, log=True, norm_trace=False)
            X_band = epochs_band.get_data()  # (n_epochs, n_channels, n_times)
            csp.fit(X_band, labels)
            X_features = csp.transform(X_band)
            features_list.append(X_features)
        except Exception as e:
            print(f"Erreur avec CSP pour {file} et la bande {band} : {e}")
            continue
    
    if not features_list:
        print(f"Aucune caractéristique extraite pour {file}")
        continue

    X_file = np.concatenate(features_list, axis=1)
    features_all.append(X_file)
    labels_all.append(labels)

if not features_all:
    raise ValueError("Aucune donnée n'a pu être extraite des fichiers.")

# Création du dataset global
X_total = np.concatenate(features_all, axis=0)
y_total = np.concatenate(labels_all, axis=0)
groups_total = np.concatenate(groups_all, axis=0)

print("Taille totale des données :", X_total.shape)
print("Distribution globale des classes:", np.unique(y_total, return_counts=True))
print("Nombre de sujets distincts :", len(np.unique(groups_total)))

# --- Séparation Train/Test au niveau des sujets ---
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X_total, y_total, groups_total))
X_train, X_test = X_total[train_idx], X_total[test_idx]
y_train, y_test = y_total[train_idx], y_total[test_idx]
groups_train, groups_test = groups_total[train_idx], groups_total[test_idx]

print("Nombre de sujets dans l'entraînement :", len(np.unique(groups_train)))
print("Nombre de sujets dans le test :", len(np.unique(groups_test)))

# --- Grid Search pour optimiser et comparer plusieurs classificateurs ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())
])

param_grid = [
    {
        'clf': [SVC()],
        'clf__kernel': ['linear', 'rbf'],
        'clf__C': [0.1, 1, 10]
    },
    {
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20]
    },
    {
        'clf': [LinearDiscriminantAnalysis()]
    }
]

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Meilleurs paramètres trouvés :", grid.best_params_)
print("Score de validation croisée optimal :", grid.best_score_)

best_model = grid.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Précision sur le jeu de test :", test_score)

# --- Sauvegarde du modèle entraîné ---
joblib.dump(best_model, f"model_eeg_{test_score:.4f}.pkl")
print("Modèle sauvegardé sous 'model_eeg_*.pkl'.")
