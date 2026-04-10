import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

# Custom imports (Make sure this path matches your project structure)
from src.classification.utils.plots import plot_decision_boundaries, show_confusion_matrix

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
INPUT_VECTORS_DIR = "classification\\feature_vector" 
FM_DIR = "classification\\data\\feature_matrices"
MODEL_DIR = "classification\\data\\models"
TARGET_SHAPE = (20, 20)

# Classes to exclude
CLASSES_TO_REMOVE = ["background", "handsaw", "birds", "helicopter", "firorks"]

os.makedirs(FM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
# PART 0: LOGIC: Loading .npy files from directory
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

X_all = np.array(X_all)
y_all = np.array(y_all)

# Get unique class names
classnames = sorted(list(set(y_all)))
print(f"✔ Classes kept: {', '.join(classnames)}")
print(f"✔ Total samples loaded: {len(X_all)}")

# -------------------------------------------------------
# PART 1: Train/Test Split
# -------------------------------------------------------
# Reserve 20% of the data for the final, completely unseen test
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# -------------------------------------------------------
# HELPER: EVALUATION METRICS FOR GRID SEARCH
# -------------------------------------------------------
def evaluate_model(X_train_pca, y_train, X_val_pca, y_val, C, kernel, gamma):
    """Trains and evaluates a single SVM configuration."""
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=0)
    model.fit(X_train_pca, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_pca))
    val_acc = accuracy_score(y_val, model.predict(X_val_pca))
    return train_acc, val_acc

def calculate_score(train_acc, val_acc):
    """Custom scoring metric punishing overfitting."""
    val_acc = max(val_acc, 1e-10) # Prevent division by zero
    ratio = train_acc / val_acc
    return val_acc / ratio

# Pre-calculate folds to ensure consistency across all tests
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
folds_indices = list(kf.split(X_train, y_train))

# -------------------------------------------------------
# PHASE 1: PRUNING THE KERNELS
# -------------------------------------------------------
print("\n🚀 PHASE 1: Kernel Screening (Dropping bad kernels early)...")
all_kernels = ["rbf", "poly", "linear", "sigmoid"]
kernel_scores = {}

# Baseline settings for a fair quick test
base_pca = 0.85
base_C = 1.0
base_gamma = 'scale'

for kernel in all_kernels:
    val_accs = []
    
    for idx_learn, idx_val in folds_indices:
        # Build strict folds
        X_fold_train, y_fold_train = X_train[idx_learn], y_train[idx_learn]
        X_fold_val, y_fold_val = X_train[idx_val], y_train[idx_val]
        
        # Scale & PCA
        scaler = StandardScaler()
        X_fold_train_sc = scaler.fit_transform(X_fold_train)
        X_fold_val_sc   = scaler.transform(X_fold_val)
        
        pca = PCA(n_components=base_pca, random_state=1)
        X_fold_train_pca = pca.fit_transform(X_fold_train_sc)
        X_fold_val_pca   = pca.transform(X_fold_val_sc)
        
        # Test
        _, v_acc = evaluate_model(X_fold_train_pca, y_fold_train, X_fold_val_pca, y_fold_val, base_C, kernel, base_gamma)
        val_accs.append(v_acc)
        
    mean_val = np.mean(val_accs)
    kernel_scores[kernel] = mean_val
    print(f"  ↳ Kernel '{kernel}' Baseline Validation Accuracy: {mean_val:.4f}")

# Pruning Logic: Keep kernels within 10% (0.10) of the best performing kernel
best_baseline_acc = max(kernel_scores.values())
threshold = best_baseline_acc - 0.10
surviving_kernels = [k for k, v in kernel_scores.items() if v >= threshold]

print(f"\n✂️ Pruning complete. Keeping kernels: {surviving_kernels}")

# Plot Phase 1 Results
plt.figure(figsize=(8, 4))
colors = ['green' if k in surviving_kernels else 'red' for k in all_kernels]
bars = plt.bar(all_kernels, [kernel_scores[k] for k in all_kernels], color=colors)
plt.axhline(y=threshold, color='black', linestyle='--', label=f'Cutoff Threshold ({threshold:.2f})')
plt.title("Phase 1: Kernel Pruning Baseline")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()

# -------------------------------------------------------
# PHASE 2: FULL GRID SEARCH (On Surviving Kernels)
# -------------------------------------------------------
print("\n🚀 PHASE 2: Comprehensive Grid Search...")

# Define the Grid
pca_grid = [0.80, 0.90, 0.95]
C_grid = [0.1, 1, 10, 100]
gamma_grid = ['scale', 0.01, 0.1, 1.0]

grid_results = []
best_combo = None
best_score = -1

start_time = time.time()

# 1. Loop over PCA first (to save computation time)
for p_var in pca_grid:
    print(f"\n  Processing PCA Variance = {p_var}...")
    
    # Pre-calculate PCA for all folds for THIS variance
    fold_data_cache = []
    for idx_learn, idx_val in folds_indices:
        X_fold_train, y_fold_train = X_train[idx_learn], y_train[idx_learn]
        X_fold_val, y_fold_val = X_train[idx_val], y_train[idx_val]
        
        scaler = StandardScaler()
        X_fold_train_sc = scaler.fit_transform(X_fold_train)
        X_fold_val_sc   = scaler.transform(X_fold_val)
        
        pca = PCA(n_components=p_var, random_state=1)
        X_fold_train_pca = pca.fit_transform(X_fold_train_sc)
        X_fold_val_pca   = pca.transform(X_fold_val_sc)
        
        fold_data_cache.append((X_fold_train_pca, y_fold_train, X_fold_val_pca, y_fold_val))

    # 2. Loop over the rest of the Grid 
    for kernel in surviving_kernels:
        for C in C_grid:
            # Linear kernel ignores gamma, so we only run it once to save time
            g_list = ['scale'] if kernel == 'linear' else gamma_grid
            
            for gamma in g_list:
                t_accs, v_accs = [], []
                
                # Test combination on all folds
                for X_train_p, y_train_p, X_val_p, y_val_p in fold_data_cache:
                    t_acc, v_acc = evaluate_model(X_train_p, y_train_p, X_val_p, y_val_p, C, kernel, gamma)
                    t_accs.append(t_acc)
                    v_accs.append(v_acc)
                    
                mean_t = np.mean(t_accs)
                mean_v = np.mean(v_accs)
                combo_score = calculate_score(mean_t, mean_v)
                
                grid_results.append({
                    'pca': p_var, 'kernel': kernel, 'C': C, 'gamma': gamma, 
                    'train_acc': mean_t, 'val_acc': mean_v, 'score': combo_score
                })
                
                if combo_score > best_score:
                    best_score = combo_score
                    best_combo = grid_results[-1]

print(f"\n⏱️ Grid Search finished in {(time.time() - start_time):.1f} seconds!")
print("\n🏆 BEST HYPERPARAMETER COMBINATION:")
print(f"   PCA Variance: {best_combo['pca']}")
print(f"   Kernel:       {best_combo['kernel']}")
print(f"   C:            {best_combo['C']}")
print(f"   Gamma:        {best_combo['gamma']}")
print(f"   ↳ Validation Accuracy: {best_combo['val_acc']:.4f} (Score: {best_combo['score']:.4f})")

# -------------------------------------------------------
# PART 3: VISUALIZATION (Grid Search Analysis)
# -------------------------------------------------------
print("\n🎨 Plotting Grid Search Results...")
# Extracting top 15 results to visualize
sorted_results = sorted(grid_results, key=lambda x: x['score'], reverse=True)[:15]

labels = [f"{r['kernel'][:3]}|C={r['C']}|pca={r['pca']}" for r in sorted_results]
val_scores = [r['val_acc'] for r in sorted_results]
train_scores = [r['train_acc'] for r in sorted_results]

plt.figure(figsize=(12, 6))
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, train_scores, width, label='Train Acc', color='lightblue')
plt.bar(x + width/2, val_scores, width, label='Val Acc', color='royalblue')

plt.ylabel('Accuracy')
plt.title('Top 15 Hyperparameter Combinations')
plt.xticks(x, labels, rotation=45, ha="right")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# PART 4: DECISION BOUNDARIES & FINAL EVALUATION
# -------------------------------------------------------
# Retrieve the best values
best_pca_var = best_combo['pca']
best_C = best_combo['C']
best_gamma = best_combo['gamma']
best_kernel = best_combo['kernel']

print("\n🎨 Generating Decision Boundaries (PCA 2D)...")
# Scaler + PCA to 2D for visualization
scaler_vis = StandardScaler()
X_vis_norm = scaler_vis.fit_transform(X_train)
pca_vis = PCA(n_components=2, whiten=True)
X_vis_2d = pca_vis.fit_transform(X_vis_norm)

# Convert labels to integers
y_vis_num = np.zeros(y_train.shape, dtype=int)
for i, classname in enumerate(classnames):
    y_vis_num[y_train == classname] = i

model_vis = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma, random_state=0)
model_vis.fit(X_vis_2d, y_vis_num)

plt.figure(figsize=(6, 6))
plot_decision_boundaries(
    X_vis_2d, y_vis_num, model=model_vis, legend=classnames,
    title=f"SVM (C={best_C}, G={best_gamma}, PCA={best_pca_var})"
)
plt.show()

print("\n🏋️‍♂️ Final Training & Test on unseen data...")

# Final Pipeline applied to the entire Training set and evaluated on the Test set
scaler_final = StandardScaler()
X_train_final_sc = scaler_final.fit_transform(X_train)
X_test_final_sc  = scaler_final.transform(X_test)

pca_final = PCA(n_components=best_pca_var, random_state=1)
X_train_final_pca = pca_final.fit_transform(X_train_final_sc)
X_test_final_pca  = pca_final.transform(X_test_final_sc)

final_model = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma, probability=True, random_state=0)
final_model.fit(X_train_final_pca, y_train)

y_pred_train = final_model.predict(X_train_final_pca)
y_pred_test  = final_model.predict(X_test_final_pca)

print("-" * 40)
print(f"🎯 Final Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"📈 Final Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print("-" * 40)

print("\n📊 Confusion Matrix: TRAINING SET")
show_confusion_matrix(y_pred_train, y_train, classnames)

print("\n📊 Confusion Matrix: TEST SET")
show_confusion_matrix(y_pred_test, y_test, classnames)