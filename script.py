import os
import mne
import numpy as np
from mne.preprocessing import ICA
from mne.decoding import CSP
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# --- Fonction d'extraction des epochs ---
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

# Paramètres communs
event_id = {'T1': 1, 'T2': 2}   # À adapter selon vos annotations
tmin, tmax = 0, 2               # Fenêtre d'extraction des epochs (en secondes)
freq_bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]  # Bandes de fréquences pour FBCSP

# --- Parcours de tous les fichiers EDF ---
for file in edf_files:
    print("Traitement de :", file)
    try:
        raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
    except Exception as e:
        print(f"Erreur lors de la lecture de {file} : {e}")
        continue
    
    # Filtrage entre 1 et 40 Hz
    raw.filter(1, 40, fir_design='firwin', verbose=False)
    
    # Application de l'ICA pour éliminer les artefacts
    try:
        ica = ICA(n_components=20, random_state=97, max_iter=1000, fit_params={'tol': 0.0001}, verbose=False)
        ica.fit(raw)
        raw_clean = raw.copy()
        ica.apply(raw_clean)
    except Exception as e:
        print(f"Erreur lors de l'ICA pour {file} : {e}")
        continue

    # Extraction des epochs avec la fonction fournie
    try:
        epochs = extract_epochs(raw_clean, event_id, tmin, tmax)
        # Vérifier que les epochs contiennent au moins l'un des événements de tâche (T1 ou T2)
        unique_events = np.unique(epochs.events[:, -1])
        if not any(evt in unique_events for evt in event_id.values()):
            print(f"Fichier {file} semble être un run baseline (uniquement {unique_events}). On le saute.")
            continue
    except Exception as e:
        print(f"Erreur lors de l'extraction des epochs pour {file} : {e}")
        continue


    # Récupération des labels depuis la dernière colonne des événements
    labels = epochs.events[:, -1]
    
    # Extraction des caractéristiques via FBCSP pour chaque bande de fréquences
    features_list = []
    for band in freq_bands:
        try:
            # Filtrage pour la bande spécifique
            epochs_band = epochs.copy().filter(band[0], band[1], fir_design='firwin', verbose=False)
            # Mise en place du CSP : extraction de 4 composantes par bande
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
            X_band = epochs_band.get_data()  # Dimensions : (n_epochs, n_channels, n_times)
            csp.fit(X_band, labels)
            X_features = csp.transform(X_band)
            features_list.append(X_features)
        except Exception as e:
            print(f"Erreur avec CSP pour {file} et la bande {band} : {e}")
            continue
    
    if not features_list:
        print(f"Aucune caractéristique extraite pour {file}")
        continue

    # Concaténation des caractéristiques extraites de toutes les bandes
    X_file = np.concatenate(features_list, axis=1)
    features_all.append(X_file)
    labels_all.append(labels)

# Vérifier que des données ont été extraites
if not features_all:
    raise ValueError("Aucune donnée n'a pu être extraite des fichiers.")

# Création du jeu de données global
X_total = np.concatenate(features_all, axis=0)
y_total = np.concatenate(labels_all, axis=0)

print("Taille totale des données :", X_total.shape)

# --- Division du dataset : 75 % pour l'entraînement, 25 % pour le test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_total, y_total, test_size=0.25, random_state=42, stratify=y_total
)

# --- Grid Search pour optimiser et comparer plusieurs classificateurs ---
pipeline = Pipeline([
    ('clf', SVC())  # Placeholder, remplacé par grid search
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
        # LDA a peu d'hyperparamètres à optimiser
    }
]

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Meilleurs paramètres trouvés :", grid.best_params_)
print("Score de validation croisée optimal :", grid.best_score_)

# Évaluation finale sur le jeu de test
best_model = grid.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Précision sur le jeu de test :", test_score)
