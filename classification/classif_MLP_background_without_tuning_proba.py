import os
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle # <-- NEW: Imported to save ensemble parameters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader

# Custom imports
from src.classification.utils.plots import show_confusion_matrix

# --- LOCK RANDOM SEEDS FOR REPRODUCIBILITY ---
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ---------------------------------------------

# -------------------------------------------------------
# PART 1: CONFIGURATION & HYPERPARAMETERS
# -------------------------------------------------------
INPUT_VECTORS_DIR = "classification\\feature_vector_sud11" 
FM_DIR = "classification\\data\\feature_matrices"
MODEL_DIR = "classification\\data\\models"

# --- NEW: File path for exporting the complete ensemble configuration ---
ENSEMBLE_PARAMS_PATH = os.path.join(MODEL_DIR, "ensemble_production_params_test.pkl")
# ----------------------------------------------------------------------

TARGET_SHAPE = (20, 20)

KEEP_ORIGINAL_DATA = True 

HIDDEN_UNITS = [500, 400, 300, 200]  
LEARNING_RATE = 0.00669
DROPOUT_RATE = 0.3863
WEIGHT_DECAY = 0.00004
BATCH_SIZE = 512
OPTIMIZER_NAME = 'AdamW'  

ALLOWED_PREFIXES = []

# --- ENSEMBLE MASKING CONFIGURATION ---
# TOP_BANDS_TO_HIDE = [0, 2, 3, 5, 6, 7,14] 
# BOTTOM_BANDS_TO_HIDE = [1, 2, 3, 5] 
TOP_BANDS_TO_HIDE = [0]
BOTTOM_BANDS_TO_HIDE = []

MAX_EPOCHS = 500
PATIENCE = 60  

os.makedirs(FM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# PART 2: NEURAL NETWORK ARCHITECTURE
# -------------------------------------------------------
class AudioMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_units_list, dropout_rate):
        super(AudioMLP, self).__init__()
        
        layers = []
        in_features = input_size
        n_layers = len(hidden_units_list)
        
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_units_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_units_list[i] 
            
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
        
    def predict_proba(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=1)

def hide_frequency_bands(matrix, num_bands, strategy='top'):
    if num_bands <= 0:
        return matrix
    masked_matrix = matrix.copy()
    max_bands = matrix.shape[0] 
    
    if strategy == 'top':
        bands_to_mask = list(range(max_bands - num_bands, max_bands))
    elif strategy == 'bottom':
        bands_to_mask = list(range(num_bands))
    else:
        bands_to_mask = []
        
    masked_matrix[bands_to_mask, :] = 0.0
    return masked_matrix

# -------------------------------------------------------
# PART 3: PRE-LOAD RAW DATASET INTO MEMORY
# -------------------------------------------------------
print(f"📂 Scanning directory: {INPUT_VECTORS_DIR}")
all_files = [f for f in os.listdir(INPUT_VECTORS_DIR) if f.endswith('.npy')]

background_matrices = []
for filename in all_files:
    if filename.startswith('background'):
        filepath = os.path.join(INPUT_VECTORS_DIR, filename)
        bg_matrix = np.load(filepath)
        if bg_matrix.shape == TARGET_SHAPE:
            background_matrices.append(bg_matrix)

master_raw_dataset = [] 
global_labels = []

for filename in all_files:
    if filename.startswith('background'):
        continue

    prefix = filename.split('_')[0]
    if len(ALLOWED_PREFIXES) > 0 and prefix not in ALLOWED_PREFIXES:
        continue

    classname = re.sub(r'\d+', '', prefix).lower()
    filepath = os.path.join(INPUT_VECTORS_DIR, filename)
    spec_matrix_orig = np.load(filepath)
    
    spec_matrix_orig = np.log(spec_matrix_orig + 1e-8)
    
    if spec_matrix_orig.shape == TARGET_SHAPE:
        if KEEP_ORIGINAL_DATA:
            master_raw_dataset.append({'matrix': spec_matrix_orig, 'label': classname})
            global_labels.append(classname)
            
        if len(background_matrices) > 0:
            random_bg = random.choice(background_matrices)
            attenuated_bg = random_bg * 0.1 
            spec_matrix_aug = spec_matrix_orig + attenuated_bg 
            
            master_raw_dataset.append({'matrix': spec_matrix_aug, 'label': classname})
            global_labels.append(classname)
            
        elif not KEEP_ORIGINAL_DATA: 
            master_raw_dataset.append({'matrix': spec_matrix_orig, 'label': classname})
            global_labels.append(classname)

label_encoder = LabelEncoder()
label_encoder.fit(global_labels)
classnames = sorted(list(set(global_labels)))
num_classes = len(classnames)

print(f"✔ Pre-loaded {len(master_raw_dataset)} raw matrices into memory.")
print(f"✔ Classes: {', '.join(classnames)}\n")

# -------------------------------------------------------
# PART 4: TRAIN MULTIPLE MODELS (ENSEMBLE LOOP)
# -------------------------------------------------------
configurations = []
for n in TOP_BANDS_TO_HIDE:
    configurations.append(('top', n))
for n in BOTTOM_BANDS_TO_HIDE:
    configurations.append(('bottom', n))

all_model_test_probabilities = []
all_model_precisions = []
ensemble_y_test_true = None 

# --- NEW: List to accumulate the specific parameters of each model ---
ensemble_export_data = []

for strategy, num_bands in configurations:
    config_name = f"mask_{strategy}_{num_bands}"
    print(f"==================================================")
    print(f"⚙️ TRAINING MODEL: {config_name.upper()}")
    print(f"==================================================")
    
    X_current, y_current = [], []
    for data in master_raw_dataset:
        masked_matrix = hide_frequency_bands(data['matrix'], num_bands, strategy)
        X_current.append(masked_matrix.flatten())
        y_current.append(data['label'])
        
    X_current = np.array(X_current)
    y_current_enc = label_encoder.transform(y_current)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_current, y_current_enc, test_size=0.3, random_state=42, stratify=y_current_enc)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    if ensemble_y_test_true is None:
        ensemble_y_test_true = y_test 
        
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    pca = PCA(n_components=0.8, random_state=1)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val_sc)
    X_test_pca  = pca.transform(X_test_sc)

    input_dim = X_train_pca.shape[1]

    train_dataset = TensorDataset(torch.FloatTensor(X_train_pca), torch.LongTensor(y_train))
    val_dataset   = TensorDataset(torch.FloatTensor(X_val_pca), torch.LongTensor(y_val))
    test_dataset  = TensorDataset(torch.FloatTensor(X_test_pca), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = AudioMLP(input_dim, num_classes, HIDDEN_UNITS, DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_val_loss = float('inf')
    patience_counter = 0
    current_best_model_path = os.path.join(MODEL_DIR, f"model_mlp_{config_name}.pth")
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), current_best_model_path) 
            patience_counter = 0 
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"⏹ Early stopping for {config_name} at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
            break 
            
    # Load best state for evaluation
    model.load_state_dict(torch.load(current_best_model_path))
    model.eval()
    
    # --- STEP 5.1: Calculate Per-Class Precision on VALIDATION SET ---
    val_y_pred, val_y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            probs = model.predict_proba(X_batch)
            _, predicted = torch.max(probs, 1)
            val_y_pred.extend(predicted.cpu().numpy())
            val_y_true.extend(y_batch.numpy())

    precisions, _, _, _ = precision_recall_fscore_support(
        val_y_true, val_y_pred, labels=range(num_classes), average=None, zero_division=0
    )
    all_model_precisions.append(precisions)
    
    formatted_precisions = [f"{p:.2f}" for p in precisions]
    print(f"🎯 Validation Precisions (Used for Ensemble Weights): {formatted_precisions}")
    
    # --- NEW: Save this specific model's preprocessing parameters into our export list ---
    ensemble_export_data.append({
        'pth_file': f"model_mlp_{config_name}.pth",
        'mask_strategy': strategy,
        'num_bands': num_bands,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'pca_components': pca.components_,
        'pca_mean': pca.mean_,
        'class_precisions': precisions
    })
    # -----------------------------------------------------------------------------------
    
    # --- STEP 5.2: Extract Probabilities on TEST SET ---
    model_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            probs = model.predict_proba(X_batch)
            model_probs.extend(probs.cpu().numpy())
            
    model_probs_np = np.array(model_probs)
    all_model_test_probabilities.append(model_probs_np)
    
    # --- STEP 5.3: Print Standalone Test Metrics for this Model ---
    test_y_pred_single = np.argmax(model_probs_np, axis=1)
    acc = accuracy_score(ensemble_y_test_true, test_y_pred_single)
    prec, rec, f1, _ = precision_recall_fscore_support(ensemble_y_test_true, test_y_pred_single, average='weighted', zero_division=0)
    
    print(f"📊 {config_name.upper()} Test Metrics -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}\n")

# -------------------------------------------------------
# PART 5: PRECISION-WEIGHTED SOFT VOTING ENSEMBLE
# -------------------------------------------------------
all_probs_3d = np.array(all_model_test_probabilities) 
all_precisions_2d = np.array(all_model_precisions)    

print(f"==================================================")
print(f"🔍 MODEL AGREEMENT ANALYSIS")
print(f"==================================================")

individual_predictions = np.argmax(all_probs_3d, axis=2)
num_models, num_samples = individual_predictions.shape

unanimous_mask = np.all(individual_predictions == individual_predictions[0, :], axis=0)
unanimous_count = np.sum(unanimous_mask)
split_count = num_samples - unanimous_count

print(f"Total Test Samples: {num_samples}")
print(f"Unanimous Agreement (All {num_models} models guessed the exact same class): {unanimous_count} ({unanimous_count/num_samples*100:.1f}%)")
print(f"Split Decisions (The precision weights were needed to break ties): {split_count} ({split_count/num_samples*100:.1f}%)\n")

print(f"==================================================")
print(f"🤝 EVALUATING PRECISION-WEIGHTED ENSEMBLE")
print(f"==================================================")

precisions_broadcast = all_precisions_2d[:, np.newaxis, :]
weighted_probs_3d = all_probs_3d * precisions_broadcast

final_scores = np.sum(weighted_probs_3d, axis=0)
ensemble_predictions = np.argmax(final_scores, axis=1)

y_pred_names = label_encoder.inverse_transform(ensemble_predictions)
y_true_names = label_encoder.inverse_transform(ensemble_y_test_true)

accuracy = accuracy_score(y_true_names, y_pred_names)
precision, recall, f1, _ = precision_recall_fscore_support(y_true_names, y_pred_names, average='weighted', zero_division=0)

print("\n📊 --- ENSEMBLE EVALUATION METRICS ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("--------------------------------------\n")

print("Detailed Ensemble Classification Report:\n")
print(classification_report(y_true_names, y_pred_names, zero_division=0))

show_confusion_matrix(y_pred_names, y_true_names, classnames)

# -------------------------------------------------------
# PART 6: EXPORT ENSEMBLE PARAMETERS TO PICKLE
# -------------------------------------------------------
print(f"==================================================")
print(f"💾 SAVING PRODUCTION PARAMETERS")
print(f"==================================================")

production_params = {
    'classes': classnames,       # Global array mapping predicted indices back to strings (e.g. ['chainsaw', 'fire'])
    'models': ensemble_export_data # List containing dicts of weights/PCA for every individual model
}

with open(ENSEMBLE_PARAMS_PATH, 'wb') as f:
    pickle.dump(production_params, f)

print(f"✅ Successfully exported all scalers, PCA projections, and precision weights to:")
print(f"   -> {ENSEMBLE_PARAMS_PATH}")
print(f"This file perfectly matches the expected input format for the new ensemble inference script.")