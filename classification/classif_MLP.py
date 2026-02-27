import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

# Custom imports from your project structure
from src.classification.datasets import Dataset
from src.classification.utils.audio_student import AudioUtil, Feature_vector_DS
from src.classification.utils.plots import plot_decision_boundaries, show_confusion_matrix

# -------------------------------------------------------
# PART 0: SETUP & DATA GENERATION (STRICT SEPARATION)
# -------------------------------------------------------
dataset = Dataset()
# Filter classes
dataset.remove_class("background")
dataset.remove_class("fireworks")
dataset.remove_class("gunshot")
classnames = dataset.list_classes()

myds = Feature_vector_DS(dataset, Nft=512, nmel=20, duration=950, step=np.inf)
TARGET_SHAPE = (20, 20)

def get_fixed_feature(dataset, classname, idx):
    feat2d = dataset[classname, idx]
    if feat2d.shape != TARGET_SHAPE:
        feat2d = resize(feat2d, TARGET_SHAPE, mode='reflect', anti_aliasing=True)
    return feat2d.reshape(-1)

# Augmentation Config
aug_list_extras = ["time_shift", "add_noise", "add_echo"]
aug_list_base   = ["original"]

X_train_orig, y_train_orig = [], []   
X_train_aug, y_train_aug   = [], []   
aug_map = []                          
X_test, y_test = [], []

print("🔄 Generating Data (Separating Original vs Augmented)...")

for classname in classnames:
    n_audio = dataset.naudio[classname]
    indices = np.arange(n_audio)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # 1. Training Originals
    myds.mod_data_aug(aug_list_base)
    class_start_index = len(X_train_orig)
    for idx in train_idx:
        feat = get_fixed_feature(myds, classname, idx)
        X_train_orig.append(feat)
        y_train_orig.append(classname)
        
    # 2. Training Augmentations (Mapped)
    myds.mod_data_aug(aug_list_extras)
    for s in range(myds.data_aug_factor):
        for i, idx in enumerate(train_idx):
            feat = get_fixed_feature(myds, classname, idx)
            X_train_aug.append(feat)
            y_train_aug.append(classname)
            aug_map.append(class_start_index + i)

    # 3. Test Set (Pure)
    myds.mod_data_aug(aug_list_base)
    for idx in test_idx:
        feat = get_fixed_feature(myds, classname, idx)
        X_test.append(feat)
        y_test.append(classname)

X_train_orig = np.array(X_train_orig)
y_train_orig = np.array(y_train_orig)
X_train_aug  = np.array(X_train_aug)
y_train_aug  = np.array(y_train_aug)
aug_map      = np.array(aug_map)
X_test       = np.array(X_test)
y_test       = np.array(y_test)

print(f"✔ Data Ready: Originals={X_train_orig.shape}, Augmented={X_train_aug.shape}")

# -------------------------------------------------------
# HELPER: STRICT K-FOLD LOOP FOR MLP
# -------------------------------------------------------
def accuracy(preds, true):
    return accuracy_score(true, preds)

def run_strict_kfold_mlp(param_name, param_values, best_hidden=None, best_alpha=None):
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    train_scores = np.zeros((len(param_values), n_splits))
    val_scores   = np.zeros((len(param_values), n_splits))
    
    for i, val in enumerate(param_values):
        for k, (idx_learn, idx_val) in enumerate(kf.split(X_train_orig, y_train_orig)):
            
            # Build Folds
            X_fold_val = X_train_orig[idx_val]
            y_fold_val = y_train_orig[idx_val]
            
            X_learn_orig = X_train_orig[idx_learn]
            y_learn_orig = y_train_orig[idx_learn]
            
            mask_aug = np.isin(aug_map, idx_learn)
            X_learn_aug = X_train_aug[mask_aug]
            y_learn_aug = y_train_aug[mask_aug]
            
            X_fold_train = np.concatenate([X_learn_orig, X_learn_aug])
            y_fold_train = np.concatenate([y_learn_orig, y_learn_aug])
            X_fold_train, y_fold_train = shuffle(X_fold_train, y_fold_train, random_state=0)
            
            # Pre-processing
            scaler = StandardScaler()
            X_fold_train_sc = scaler.fit_transform(X_fold_train)
            X_fold_val_sc   = scaler.transform(X_fold_val)
            
            pca = PCA(n_components=0.8, random_state=1)
            X_fold_train_pca = pca.fit_transform(X_fold_train_sc)
            X_fold_val_pca   = pca.transform(X_fold_val_sc)
            
            # MLP Configuration (max_iter=500 to ensure convergence without taking forever)
            if param_name == 'hidden_layer_sizes':
                model = MLPClassifier(hidden_layer_sizes=val, alpha=0.0001, activation='relu', max_iter=500, random_state=1)
            elif param_name == 'alpha':
                model = MLPClassifier(hidden_layer_sizes=best_hidden, alpha=val, activation='relu', max_iter=500, random_state=1)
            elif param_name == 'activation':
                model = MLPClassifier(hidden_layer_sizes=best_hidden, alpha=best_alpha, activation=val, max_iter=500, random_state=1)
                
            model.fit(X_fold_train_pca, y_fold_train)
            
            val_scores[i, k]   = accuracy(model.predict(X_fold_val_pca), y_fold_val)
            train_scores[i, k] = accuracy(model.predict(X_fold_train_pca), y_fold_train)
            
    return train_scores, val_scores

# -------------------------------------------------------
# PART 1: OPTIMIZE HIDDEN LAYERS (Network Capacity)
# -------------------------------------------------------
print(f"\n🚀 STEP 1: Optimizing Hidden Layer Sizes...")
# Testing different architectures: 1 layer vs 2 layers, small vs large
hidden_layers = [(50,), (100,), (200,), (50, 50), (100, 50)]
layer_names = [str(hl) for hl in hidden_layers] # for plotting

train_acc, val_acc = run_strict_kfold_mlp('hidden_layer_sizes', hidden_layers)

means_train = train_acc.mean(axis=1)
means_val   = np.maximum(val_acc.mean(axis=1), 1e-10)
ratios = means_train / means_val
scores = means_val / ratios
best_idx = np.argmax(scores)
best_hidden = hidden_layers[best_idx]

# Plot Hidden Layers
plt.figure(figsize=(10, 4))
plt.plot(layer_names, means_train, ".-g", label="Train")
plt.plot(layer_names, means_val,   ".-r", label="Validation")
plt.plot(layer_names, ratios, ".--k", label="Overfitting Ratio")
plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
plt.ylabel("Score")
plt.title(f"Step 1: Best Architecture = {best_hidden}")
plt.legend()
plt.show()

print(f"🏆 Best Hidden Layers: {best_hidden}")

# -------------------------------------------------------
# PART 2: OPTIMIZE ALPHA (Regularization / Penalty)
# -------------------------------------------------------
print(f"\n🚀 STEP 2: Optimizing Alpha (Regularization)...")
# High alpha = heavy regularization (less overfitting, but might underfit)
Alphas = np.logspace(-4, 2, 20)
train_acc_a, val_acc_a = run_strict_kfold_mlp('alpha', Alphas, best_hidden=best_hidden)

means_train_a = train_acc_a.mean(axis=1)
means_val_a   = np.maximum(val_acc_a.mean(axis=1), 1e-10)
ratios_a = means_train_a / means_val_a
scores_a = means_val_a / ratios_a
best_idx_a = np.argmax(scores_a)
best_alpha = Alphas[best_idx_a]

# Plot Alpha
plt.figure(figsize=(8, 4))
plt.semilogx(Alphas, means_train_a, ".-g", label="Train")
plt.semilogx(Alphas, means_val_a,   ".-r", label="Validation")
plt.semilogx(Alphas, ratios_a, ".--k", label="Overfitting Ratio")
plt.axvline(x=best_alpha, color='blue', linestyle=':', label=f"Best Alpha: {best_alpha:.4f}")
plt.xlabel("Alpha (Log Scale)")
plt.ylabel("Score")
plt.title(f"Step 2: Best Alpha = {best_alpha:.4f}")
plt.legend()
plt.show()

print(f"🏆 Best Alpha: {best_alpha:.4f}")

# -------------------------------------------------------
# PART 3: OPTIMIZE ACTIVATION FUNCTION
# -------------------------------------------------------
print(f"\n🚀 STEP 3: Optimizing Activation Function...")
activations = ['relu', 'tanh', 'logistic']
train_acc_act, val_acc_act = run_strict_kfold_mlp('activation', activations, best_hidden=best_hidden, best_alpha=best_alpha)

means_train_act = train_acc_act.mean(axis=1)
means_val_act   = np.maximum(val_acc_act.mean(axis=1), 1e-10)
ratios_act = means_train_act / means_val_act
scores_act = means_val_act / ratios_act
best_idx_act = np.argmax(scores_act)
best_activation = activations[best_idx_act]

# Plot Activation
plt.figure(figsize=(8, 4))
plt.plot(activations, means_val_act, ".-r", label="Validation")
plt.plot(activations, ratios_act, ".--k", label="Overfitting Ratio")
plt.ylabel("Score / Ratio")
plt.title(f"Step 3: Best Activation = {best_activation}")
plt.legend()
plt.show()

print(f"🏆 Best Activation selected: {best_activation}")

# -------------------------------------------------------
# PART 4: VISUALIZATION (Decision Boundaries)
# -------------------------------------------------------
print("\n🎨 Generating Decision Boundary Plot (PCA 2D)...")
X_total_train = np.concatenate([X_train_orig, X_train_aug])
y_total_train = np.concatenate([y_train_orig, y_train_aug])

# Scaler + PCA to 2D
scaler_vis = StandardScaler()
X_vis_norm = scaler_vis.fit_transform(X_total_train)
pca_vis = PCA(n_components=2, whiten=True)
X_vis_2d = pca_vis.fit_transform(X_vis_norm)

# Convert labels to int
y_vis_num = np.zeros(y_total_train.shape, dtype=int)
for i, classname in enumerate(classnames):
    y_vis_num[y_total_train == classname] = i

model_vis = MLPClassifier(hidden_layer_sizes=best_hidden, alpha=best_alpha, activation=best_activation, max_iter=500, random_state=1)
model_vis.fit(X_vis_2d, y_vis_num)

plt.figure(figsize=(6, 6))
plot_decision_boundaries(
    X_vis_2d, y_vis_num, model=model_vis, legend=classnames,
    title=f"MLP Boundaries (Arch={best_hidden}, α={best_alpha:.4f})"
)
plt.show()

# -------------------------------------------------------
# PART 5: FINAL EVALUATION & MATRICES
# -------------------------------------------------------
print("\n🏋️‍♂️ Final Training & Test...")

# 1. Combine ALL Data
X_final_train = np.concatenate([X_train_orig, X_train_aug])
y_final_train = np.concatenate([y_train_orig, y_train_aug])
X_final_train, y_final_train = shuffle(X_final_train, y_final_train, random_state=0)

# 2. Final Pipeline
scaler_final = StandardScaler()
X_train_final_sc = scaler_final.fit_transform(X_final_train)
X_test_final_sc  = scaler_final.transform(X_test)

pca_final = PCA(n_components=0.8, random_state=1)
X_train_final_pca = pca_final.fit_transform(X_train_final_sc)
X_test_final_pca  = pca_final.transform(X_test_final_sc)

# 3. Train Final Model
final_model = MLPClassifier(hidden_layer_sizes=best_hidden, alpha=best_alpha, activation=best_activation, max_iter=1000, random_state=1)
final_model.fit(X_train_final_pca, y_final_train)

# 4. Predictions
y_pred_train = final_model.predict(X_train_final_pca)
y_pred_test  = final_model.predict(X_test_final_pca)

# 5. Final Report
print("-" * 40)
print(f"🎯 Final Test Accuracy: {accuracy(y_pred_test, y_test):.4f}")
print(f"📈 Final Train Accuracy: {accuracy(y_pred_train, y_final_train):.4f}")
print("-" * 40)

# 6. Confusion Matrices
print("\n📊 Confusion Matrix: TRAINING SET (Check fit)")
show_confusion_matrix(y_pred_train, y_final_train, classnames)

print("\n📊 Confusion Matrix: TEST SET (Check generalization)")
show_confusion_matrix(y_pred_test, y_test, classnames)