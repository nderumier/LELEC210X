import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from src.classification.utils.plots import show_confusion_matrix

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
INPUT_VECTORS_DIR = "classification\\feature_vector" 
FM_DIR = "classification\\data\\feature_matrices"
MODEL_DIR = "classification\\data\\models"
TARGET_SHAPE = (20, 20)

# Classes to exclude (matching your original dataset.remove_class logic)
CLASSES_TO_REMOVE = ["background", "handsaw", "birds", "helicopter"]

os.makedirs(FM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
# LOGIC: Loading .npy files from directory
# -------------------------------------------------------
# ... (rest of your imports and configuration remain the same)

# -------------------------------------------------------
# LOGIC: Loading .npy files from directory
# -------------------------------------------------------
X_all = []
y_all = []

print(f"📂 Scanning directory: {INPUT_VECTORS_DIR}")

if not os.path.exists(INPUT_VECTORS_DIR):
    raise FileNotFoundError(f"❌ Directory not found: {INPUT_VECTORS_DIR}")

all_files = [f for f in os.listdir(INPUT_VECTORS_DIR) if f.endswith('.npy')]

for filename in all_files:
    # 1. Get the part before the first underscore (e.g., "fire1_01.npy" -> "fire1")
    raw_name = filename.split('_')[0]
    
    # 2. Remove any trailing digits from that string (e.g., "fire1" -> "fire")
    # This handles "fire1", "fire2", "chainsaw10", etc.
    classname = raw_name.rstrip('0123456789')
    
    # Apply the class filter
    if classname in CLASSES_TO_REMOVE:
        continue
        
    filepath = os.path.join(INPUT_VECTORS_DIR, filename)
    spec_matrix = np.load(filepath)
    
    if spec_matrix.shape == TARGET_SHAPE:
        X_all.append(spec_matrix.flatten()) 
        y_all.append(classname)
    else:
        print(f"⚠️ Skipping {filename}: Wrong shape {spec_matrix.shape}")

# ... (the rest of the training and evaluation code remains the same)
X_all = np.array(X_all)
y_all = np.array(y_all)

# Get unique class names for the confusion matrix later
classnames = sorted(list(set(y_all)))
print(f"✔ Classes kept: {', '.join(classnames)}")

# -------------------------------------------------------
# TRAIN - TEST SPLIT & SHUFFLE
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)

# Shuffling (Mirroring PARTIE 2 of your original code)
X_train, y_train = shuffle(X_train, y_train, random_state=1)
X_test, y_test = shuffle(X_test, y_test, random_state=1)

print("-" * 30)
print(f"Training Set: {X_train.shape} samples") 
print(f"Testing Set : {X_test.shape} samples")
print("-" * 30)

# Save intermediate matrices
np.save(os.path.join(FM_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(FM_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(FM_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(FM_DIR, 'y_test.npy'), y_test)
print("✅ Data split and saved.")

# -------------------------------------------------------
# PIPELINE: Scaling, PCA, and SVM
# -------------------------------------------------------
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

pca = PCA(n_components=0.8, random_state=1)
X_train_pca = pca.fit_transform(X_train_norm)
X_test_pca = pca.transform(X_test_norm)

print("✔ Dimension après PCA :", X_train_pca.shape)

model = SVC(kernel="rbf", C=4.281, gamma=0.0002, class_weight="balanced", probability=True, random_state=1)
model.fit(X_train_pca, y_train)
print("✔ Modèle entraîné !")

# -------------------------------------------------------
# EVALUATION ON TRAIN (Mirroring PARTIE 3)
# -------------------------------------------------------
y_train_pred = model.predict(X_train_pca)
print("\n🎯 Accuracy train :", accuracy_score(y_train, y_train_pred))
print("\n📊 Classification report train :\n")
print(classification_report(y_train, y_train_pred))

# Confusion matrix for training data
show_confusion_matrix(y_train_pred, y_train, classnames)

# -------------------------------------------------------
# SAVING MODEL (Mirroring PARTIE 4)
# -------------------------------------------------------
model_filename = "model_audio_svm_test.pickle"
model_filepath = os.path.join(MODEL_DIR, model_filename)

with open(model_filepath, "wb") as f:
    pickle.dump({"scaler": scaler, "pca": pca, "model": model}, f)

print("✔ Modèle sauvegardé dans :", model_filepath)
print("Saving to absolute path:", os.path.abspath(model_filepath))

# -------------------------------------------------------
# EVALUATION ON TEST (Mirroring PARTIE 5)
# -------------------------------------------------------
y_test_pred = model.predict(X_test_pca)

print("\n🎯 Accuracy test :", accuracy_score(y_test, y_test_pred))
print("\n📊 Classification report test :\n")
print(classification_report(y_test, y_test_pred))

# Confusion matrix for test data
show_confusion_matrix(y_test_pred, y_test, classnames)