import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

import matplotlib.pyplot as plt
"Machine learning tools"
import pickle

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.classification.datasets import Dataset
from src.classification.utils.audio_student import AudioUtil, Feature_vector_DS
from src.classification.utils.plots import (
    plot_decision_boundaries,
    plot_specgram,
    show_confusion_matrix,
)
from src.classification.utils.utils import accuracy
from skimage.transform import resize  # Helper to fix the dimensions
import shutil # Needed to copy files

dataset = Dataset()
classnames = dataset.list_classes()


# -------------------------------------------------------
# Choose classes
# -------------------------------------------------------

dataset.remove_class("background")
dataset.remove_class("handsaw")
dataset.remove_class("birds")
dataset.remove_class("helicopter")
classnames = dataset.list_classes()

print("\n".join(classnames))

fm_dir = "data/feature_matrices/"  # where to save the features matrices
model_dir = "data/models/"  # where to save the models
os.makedirs(fm_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

"Creation of the dataset"
myds = Feature_vector_DS(dataset, Nft=512, nmel=20, duration=950, step=np.inf)

"Some attributes..."
myds.nmel
myds.duration
myds.sr
myds.data_aug
myds.ncol




# -------------------------------------------------------
# Train - Test split
# -------------------------------------------------------
# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
train_aug_list = ["original", "time_shift", "add_echo", "add_bg_fixed_db"]
test_aug_list  = ["original"] 

fm_dir = "data/feature_matrices/"
audio_test_dir = "data/test_audio_v1/"
os.makedirs(fm_dir, exist_ok=True)
os.makedirs(audio_test_dir, exist_ok=True)
# TARGET SHAPE: Must match your Training Data dimensions (20 mel bands x 20 time steps)
# 20 * 20 = 400 features
TARGET_SHAPE = (20, 20)

# -------------------------------------------------------
# HELPER: Feature Extraction with Forced Resize
# -------------------------------------------------------
def get_fixed_feature(dataset, classname, idx):
    # Retrieve the spectrogram (2D array)
    feat2d = dataset[classname, idx]
    
    # If the shape is not 20x20 (e.g. it is 20x107), force resize it
    if feat2d.shape != TARGET_SHAPE:
        # mode='reflect' handles borders smoothly, anti_aliasing prevents artifacts
        feat2d = resize(feat2d, TARGET_SHAPE, mode='reflect', anti_aliasing=True)
        
    # Flatten to 1D vector (length 400)
    return feat2d.reshape(-1)

# -------------------------------------------------------
# LOGIC: Generation Loop
# -------------------------------------------------------

X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []

print(f"🔄 Processing classes. Forcing all features to shape {TARGET_SHAPE} (Vector size: {TARGET_SHAPE[0]*TARGET_SHAPE[1]})...")

for classname in classnames:
    # 1. Identify all valid indices for this class
    n_audio = dataset.naudio[classname]
    indices = np.arange(n_audio)
    
    # 2. Split indices (Prevents Data Leakage)
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    # --- PROCESS TRAINING DATA ---
    myds.mod_data_aug(train_aug_list) 
    
    for s in range(myds.data_aug_factor):
        for idx in train_idx:
            # Use helper to ensure size is 400
            feat_vec = get_fixed_feature(myds, classname, idx)
            X_train_list.append(feat_vec)
            y_train_list.append(classname)
            
    # --- PROCESS TEST DATA ---
    myds.mod_data_aug(test_aug_list) 
    filenames_test_list = [] # Store filenames here
    global_test_counter = 0   # To prefix copied audio files in order
    for idx in test_idx:
        # Use helper here too! This fixes the bug where test data was size 2140
        feat_vec = get_fixed_feature(myds, classname, idx)
        X_test_list.append(feat_vec)
        y_test_list.append(classname)
        # 2. Get Filename & Copy Audio
        # Based on your class definition, myds.dataset[(classname, idx)] returns the file path
        original_filepath = myds.dataset[(classname, idx)]
        filename_only = os.path.basename(original_filepath)
        
        filenames_test_list.append(filename_only)
        
        # Copy to the new folder with a number prefix (e.g., "001_clapping_sound15.wav")
        # This guarantees the order in the folder matches X_test exactly.
        global_test_counter += 1
        new_name = f"{global_test_counter:03d}_{classname}_{filename_only}"
        shutil.copy(original_filepath, os.path.join(audio_test_dir, new_name))

# -------------------------------------------------------
# CONVERT & SAVE
# -------------------------------------------------------
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
X_test  = np.array(X_test_list)
y_test  = np.array(y_test_list)

print("-" * 30)
print(f"Training Set: {X_train.shape} samples") # Should be (N, 400)
print(f"Testing Set : {X_test.shape} samples")  # Should be (M, 400) - MATCHING!
print("-" * 30)

# Verify shapes match before saving
if X_train.shape[1] == X_test.shape[1]:
    np.save(fm_dir + 'X_train.npy', X_train)
    np.save(fm_dir + 'y_train.npy', y_train)
    np.save(fm_dir + 'X_test.npy', X_test)
    np.save(fm_dir + 'y_test.npy', y_test)
    print("✅ Data generated correctly and saved.")
else:
    print(f"❌ ERROR: Dimension mismatch! Train={X_train.shape[1]}, Test={X_test.shape[1]}")

# -------------------------------------------------------
# PARTIE 0 — Répertoires
# -------------------------------------------------------

fm_dir = "data/feature_matrices/"
model_dir = "C:\\Users\\tvang\\LELEC210X\\classification\\"  # where to save the models

os.makedirs(fm_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# -------------------------------------------------------
# PARTIE 1 — Charger les matrices sauvegardées
# -------------------------------------------------------

# feature_file = fm_dir + "feature_matrix_2D_aug.npy"
# label_file   = fm_dir + "labels_aug.npy"

# # Vérifier que les fichiers existent
# if not os.path.exists(feature_file) or not os.path.exists(label_file):
#     raise FileNotFoundError(
#         "\n❌ Les fichiers de features n'existent pas !\n"
#         f"Je cherche :\n - {feature_file}\n - {label_file}\n\n"
#         "➡️ Exécute d'abord la cellule qui crée X_aug et y_aug,\n"
#         "puis sauvegarde-les avec :\n\n"
#         "np.save(fm_dir + 'feature_matrix_2D_aug.npy', X_aug)\n"
#         "np.save(fm_dir + 'labels_aug.npy', y_aug)\n"
#     )

# print("✔ Chargement des features...")
# X = np.load(feature_file)
# y = np.load(label_file, allow_pickle=True)
# print("X shape :", X.shape)
# print("y shape :", y.shape)

print("✔ Chargement des données...")

try:
    X_train = np.load(fm_dir + 'X_train.npy')
    y_train = np.load(fm_dir + 'y_train.npy', allow_pickle=True)
    X_test  = np.load(fm_dir + 'X_test.npy')
    y_test  = np.load(fm_dir + 'y_test.npy', allow_pickle=True)
except FileNotFoundError:
    raise FileNotFoundError("❌ Les fichiers train/test n'existent pas. Lance le script de génération de données corrigé.")

print(f"Train shape : {X_train.shape}")
print(f"Test shape  : {X_test.shape}")
# -------------------------------------------------------
# PARTIE 2 — Split / Normalisation / PCA
# -------------------------------------------------------

# Mélange des données
X_train, y_train = shuffle(X_train, y_train, random_state=1)
X_test, y_test = shuffle(X_test, y_test, random_state=1)

# Normalisation
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.8, random_state=1)  # garde 85 % de la variance
X_train_pca = pca.fit_transform(X_train_norm)
X_test_pca  = pca.transform(X_test_norm)

print("✔ Dimension après PCA :", X_train_pca.shape)


# -------------------------------------------------------
# PARTIE 3 — Entraînement du modèle SVM
# -------------------------------------------------------

model = SVC(kernel="rbf", C=4.281, gamma=0.0002, class_weight="balanced",probability=True, random_state=1)
model.fit(X_train_pca, y_train)

print("✔ Modèle entraîné !")

y_train_test = model.predict(X_train_pca)
print("\n🎯 Accuracy train :", accuracy_score(y_train, y_train_test))
print("\n📊 Classification report train :\n")
print(classification_report(y_train, y_train_test))
show_confusion_matrix(y_train_test, y_train, classnames)

# -------------------------------------------------------
# PARTIE 4 — Sauvegarde du modèle complet
# -------------------------------------------------------

model_filename = "model_audio_svm_test.pickle"

with open(model_dir + model_filename, "wb") as f:
    pickle.dump({"scaler": scaler, "pca": pca, "model": model}, f)

print("✔ Modèle sauvegardé dans :", model_dir + model_filename)

print("Saving to absolute path:", os.path.abspath(model_dir + model_filename))
# -------------------------------------------------------
# PARTIE 5 — Évaluation
# -------------------------------------------------------

y_pred = model.predict(X_test_pca)

print("\n🎯 Accuracy :", accuracy_score(y_test, y_pred))
print("\n📊 Classification report :\n")
print(classification_report(y_test, y_pred))

show_confusion_matrix(y_pred, y_test, classnames)
