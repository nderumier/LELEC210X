import os
import re
import random
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# Ensure you have this utility available in your environment
from src.classification.utils.plots import show_confusion_matrix

# -------------------------------------------------------
# CONFIGURATION & HYPERPARAMETERS
# -------------------------------------------------------
INPUT_VECTORS_DIR = "classification\\feature_vector_sud11" 
FM_DIR = "classification\\data\\feature_matrices"
MODEL_DIR = "classification\\data\\models"
TARGET_SHAPE = (20, 20)

# --- DATASET MODE CONFIGURATION ---
# Options: "original_only", "augmented_only", "both"
# Note: This only affects the Training set and the final 100% Production set.
# The 30% Test set will ALWAYS be augmented.
DATA_MODE = "both" 

# --- SET YOUR OPTIMIZED SVM PARAMETERS HERE ---
SVM_KERNEL = "rbf"   # e.g., "rbf", "poly", "linear"
SVM_C = 5            # Replace with your best C
SVM_GAMMA = 0.7      # Replace with your best Gamma

# --- PREFIX FILTERING ---
ALLOWED_PREFIXES = []  # Example: Only keep files starting with these prefixes. Empty list means keep all.

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

# --- STEP B: GATHER VALID TARGET FILES AND SPLIT THEM (70/30) ---
valid_files = []
valid_classes = []

for filename in all_files:
    if filename.startswith('background'): continue
    
    prefix = filename.split('_')[0]
    if len(ALLOWED_PREFIXES) > 0 and prefix not in ALLOWED_PREFIXES: continue
    
    classname = re.sub(r'\d+', '', prefix).lower()
    valid_files.append(filename)
    valid_classes.append(classname)

# By splitting the FILENAMES rather than the generated matrices, we prevent data leakage.
train_files, test_files = train_test_split(
    valid_files, test_size=0.3, random_state=42, stratify=valid_classes
)
train_files_set = set(train_files) # For faster lookups

# Lists to hold our specific splits
X_train, y_train = [], []
X_test,  y_test  = [], []
X_all,   y_all   = [], [] # Used for the 100% production retraining

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
        
        # Create the augmented version (fallback to original if no bg found)
        if len(background_matrices) > 0:
            random_bg = random.choice(background_matrices)
            attenuated_bg = random_bg * 0.1 
            flat_aug = (spec_matrix_orig + attenuated_bg).flatten()
        else:
            flat_aug = flat_orig
            
        is_train = filename in train_files_set
        
        # 1. POPULATE THE EVALUATION SETS (Train vs Test)
        if is_train:
            # Train set respects DATA_MODE
            if DATA_MODE in ["original_only", "both"]:
                X_train.append(flat_orig)
                y_train.append(classname)
            if DATA_MODE in ["augmented_only", "both"]:
                X_train.append(flat_aug)
                y_train.append(classname)
        else:
            # Test set ALWAYS gets the augmented version, regardless of DATA_MODE
            X_test.append(flat_aug)
            y_test.append(classname)
            
        # 2. POPULATE THE 100% PRODUCTION SET (Respects DATA_MODE)
        if DATA_MODE in ["original_only", "both"]:
            X_all.append(flat_orig)
            y_all.append(classname)
        if DATA_MODE in ["augmented_only", "both"]:
            X_all.append(flat_aug)
            y_all.append(classname)

# Convert all lists to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test   = np.array(X_test),  np.array(y_test)
X_all, y_all     = np.array(X_all),   np.array(y_all)

classnames = sorted(list(set(y_all)))
print(f"✔ Dataset Mode used for Training & Production: {DATA_MODE}")
print(f"✔ Classes kept: {', '.join(classnames)}")
print(f"✔ Train Set Size: {len(X_train)} | Test Set Size (Augmented Only): {len(X_test)}")
print(f"✔ Total Production Set Size: {len(X_all)}")

# -------------------------------------------------------
# PART 2: 70/30 Evaluation (Metrics & Confusion Matrix)
# -------------------------------------------------------
print("\n" + "="*40)
print("🔍 PHASE 1: 70/30 EVALUATION")
print("="*40)

# 1. Scale (Notice we no longer do train_test_split here, it was done in Part 1)
eval_scaler = StandardScaler()
X_train_sc = eval_scaler.fit_transform(X_train)
X_test_sc  = eval_scaler.transform(X_test)

# 2. PCA
eval_pca = PCA(n_components=0.8, random_state=1)
X_train_pca = eval_pca.fit_transform(X_train_sc)
X_test_pca  = eval_pca.transform(X_test_sc)

# 3. Train Eval Model
eval_model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=True, random_state=42)
eval_model.fit(X_train_pca, y_train)

# 4. Predict & Evaluate
y_test_pred = eval_model.predict(X_test_pca)

print(f"\n🎯 Test Accuracy (30% split): {accuracy_score(y_test, y_test_pred):.4f}")
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_test_pred))

# Display Confusion Matrix
show_confusion_matrix(y_test_pred, y_test, classnames)

# -------------------------------------------------------
# PART 3: Production Training (100% of Dataset)
# -------------------------------------------------------
print("\n" + "="*40)
print("🚀 PHASE 2: FULL DATASET RETRAINING")
print("="*40)

# 1. Fit a NEW Scaler on ALL data
prod_scaler = StandardScaler()
X_all_sc = prod_scaler.fit_transform(X_all)

# 2. Fit a NEW PCA on ALL scaled data
prod_pca = PCA(n_components=0.8, random_state=1)
X_all_pca = prod_pca.fit_transform(X_all_sc)

print(f"✔ Final PCA dimensions: {X_all_pca.shape[1]} features representing 80% variance.")

# 3. Train Final Model on ALL PCA data
final_model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=True, random_state=42)
final_model.fit(X_all_pca, y_all)
print("✔ Final model trained on 100% of the dataset!")

# 4. Save the Production Pipeline
model_filename = "final_optimized_svm.pickle"
model_filepath = os.path.join(MODEL_DIR, model_filename)

with open(model_filepath, "wb") as f:
    pickle.dump({
        "scaler": prod_scaler, 
        "pca": prod_pca, 
        "model": final_model
    }, f)

print(f"💾 Production model and pipeline saved to: {os.path.abspath(model_filepath)}")