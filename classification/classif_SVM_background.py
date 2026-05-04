import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
INPUT_VECTORS_DIR = "classification\\feature_vector_sud11" 
FM_DIR = "classification\\data\\feature_matrices"
MODEL_DIR = "classification\\data\\models"
TARGET_SHAPE = (20, 20)

# --- DATASET MODE CONFIGURATION ---
# Options: "original_only", "augmented_only", "both"
# Note: This only affects the Training set. Validation & Test sets are ALWAYS augmented.
DATA_MODE = "both" 

# --- PREFIX FILTERING ---
# List the exact prefixes you want to KEEP. 
# If this list is empty [], it will load everything normally.
ALLOWED_PREFIXES = []

os.makedirs(FM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
# PART 1: Data Loading & File-Level Split
# -------------------------------------------------------
print(f"📂 Scanning directory: {INPUT_VECTORS_DIR}")

if not os.path.exists(INPUT_VECTORS_DIR):
    raise FileNotFoundError(f"❌ Directory not found: {INPUT_VECTORS_DIR}")

all_files = [f for f in os.listdir(INPUT_VECTORS_DIR) if f.endswith('.npy')]

# --- STEP A: LOAD ALL BACKGROUND FILES FIRST ---
background_matrices = []
for filename in all_files:
    if filename.startswith('background'):
        filepath = os.path.join(INPUT_VECTORS_DIR, filename)
        bg_matrix = np.load(filepath)
        if bg_matrix.shape == TARGET_SHAPE:
            background_matrices.append(bg_matrix)

print(f"🔊 Found {len(background_matrices)} valid background files for augmentation.")

# --- STEP B: GATHER VALID TARGET FILES AND SPLIT THEM ---
valid_files = []
valid_classes = []

for filename in all_files:
    if filename.startswith('background'): continue
    
    prefix = filename.split('_')[0]
    if len(ALLOWED_PREFIXES) > 0 and prefix not in ALLOWED_PREFIXES: continue
    
    classname = re.sub(r'\d+', '', prefix).lower()
    valid_files.append(filename)
    valid_classes.append(classname)

# 1. Split into 70% Train and 30% Temp (which will be Val + Test)
train_files, temp_files, y_train_files, y_temp_files = train_test_split(
    valid_files, valid_classes, test_size=0.3, random_state=42, stratify=valid_classes
)

# 2. Split Temp 50/50 into Validation and Test (15% overall each)
val_files, test_files = train_test_split(
    temp_files, test_size=0.5, random_state=42, stratify=y_temp_files
)

train_files_set = set(train_files)
val_files_set = set(val_files)
test_files_set = set(test_files)

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

# --- STEP C: PROCESS FILES AND POPULATE LISTS ---
for filename in valid_files:
    prefix = filename.split('_')[0]
    classname = re.sub(r'\d+', '', prefix).lower()
    filepath = os.path.join(INPUT_VECTORS_DIR, filename)
    spec_matrix_orig = np.load(filepath)
    
    # Preprocessing
    spec_matrix_orig = np.log(spec_matrix_orig + 1e-8)
    min_val = spec_matrix_orig.min()
    max_val = spec_matrix_orig.max()
    if max_val > min_val: 
        spec_matrix_orig = (spec_matrix_orig - min_val) / (max_val - min_val)
        
    if spec_matrix_orig.shape == TARGET_SHAPE:
        flat_orig = spec_matrix_orig.flatten()
        
        # Create augmented version
        if len(background_matrices) > 0:
            random_bg = random.choice(background_matrices)
            attenuated_bg = random_bg * 0.1 
            flat_aug = (spec_matrix_orig + attenuated_bg).flatten()
        else:
            flat_aug = flat_orig
            
        # Distribute based on the pre-determined file splits
        if filename in train_files_set:
            if DATA_MODE in ["original_only", "both"]:
                X_train.append(flat_orig)
                y_train.append(classname)
            if DATA_MODE in ["augmented_only", "both"]:
                X_train.append(flat_aug)
                y_train.append(classname)
                
        elif filename in val_files_set:
            X_val.append(flat_aug)
            y_val.append(classname)
            
        elif filename in test_files_set:
            X_test.append(flat_aug)
            y_test.append(classname)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

print(f"✔ Dataset Mode used for Training: {DATA_MODE}")
print(f"✔ Train Set Size: {len(X_train)} | Val Set Size: {len(X_val)} | Test Set Size: {len(X_test)}")

# -------------------------------------------------------
# PART 2: Scale and PCA
# -------------------------------------------------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

pca = PCA(n_components=0.8, random_state=1)
X_train_pca = pca.fit_transform(X_train_sc)
X_val_pca   = pca.transform(X_val_sc)
X_test_pca  = pca.transform(X_test_sc)

print(f"✔ Dimension after PCA : {X_train_pca.shape[1]} features")

# -------------------------------------------------------
# PART 3: Custom Grid Search for C and Gamma
# -------------------------------------------------------
C_values = [3, 4, 5, 6, 7]
gamma_values = [0.3, 0.5, 0.7, 1]

# Matrices to store results for the heatmaps
train_scores = np.zeros((len(C_values), len(gamma_values)))
val_scores = np.zeros((len(C_values), len(gamma_values)))

print("\n🚀 Starting Grid Search...")

best_val_acc = 0
best_params = {}

for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        # Train model
        model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
        model.fit(X_train_pca, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train_pca)
        y_val_pred = model.predict(X_val_pca)
        
        # Calculate Accuracy
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_val = accuracy_score(y_val, y_val_pred)
        
        # Store results
        train_scores[i, j] = acc_train
        val_scores[i, j] = acc_val
        
        # Track the best validation model
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_params = {'C': C, 'gamma': gamma}

print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.4f}")
print(f"🥇 Best Parameters: C = {best_params['C']}, Gamma = {best_params['gamma']}")

# -------------------------------------------------------
# PART 4: Plotting the Accuracy Heatmaps
# -------------------------------------------------------
plt.figure(figsize=(14, 5))

# Plot 1: Training Accuracy
plt.subplot(1, 2, 1)
sns.heatmap(train_scores, annot=True, fmt=".3f", cmap="Blues", 
            xticklabels=gamma_values, yticklabels=C_values)
plt.title("Training Accuracy")
plt.xlabel("Gamma")
plt.ylabel("C")

# Plot 2: Validation Accuracy
plt.subplot(1, 2, 2)
sns.heatmap(val_scores, annot=True, fmt=".3f", cmap="Oranges", 
            xticklabels=gamma_values, yticklabels=C_values)
plt.title("Validation Accuracy (Augmented Data)")
plt.xlabel("Gamma")
plt.ylabel("C")

plt.tight_layout()
plt.show()