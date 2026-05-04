import os
import random
import re # <-- ADDED: Needed for regex string manipulation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle

# PyTorch & Optuna Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna 
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

# Custom imports
from src.classification.utils.plots import show_confusion_matrix

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
INPUT_VECTORS_DIR = "classification\\feature_vector_sud11" 
FM_DIR = "classification\\data\\feature_matrices"
MODEL_DIR = "classification\\data\\models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "model_mlp_V2.pth")
TARGET_SHAPE = (20, 20)

CLASSES_TO_REMOVE = ["background", "handsaw", "birds", "helicopter", "firorks"]

os.makedirs(FM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
# PART 0 & 1: Data Loading and Preprocessing
# -------------------------------------------------------
X_all, y_all = [], []
print(f"📂 Scanning directory: {INPUT_VECTORS_DIR}")

if not os.path.exists(INPUT_VECTORS_DIR):
    raise FileNotFoundError(f"❌ Directory not found: {INPUT_VECTORS_DIR}")

for filename in [f for f in os.listdir(INPUT_VECTORS_DIR) if f.endswith('.npy')]:
    
    # --- MODIFIED LOGIC HERE ---
    # 1. Grab everything before the first underscore (e.g., 'fire8' from 'fire8_01.npy')
    prefix = filename.split('_')[0]
    
    # 2. Strip all numbers out of the prefix to leave just the class name (e.g., 'fire')
    classname = re.sub(r'\d+', '', prefix).lower()
    # ---------------------------
    
    if classname in CLASSES_TO_REMOVE: continue
        
    filepath = os.path.join(INPUT_VECTORS_DIR, filename)
    spec_matrix = np.load(filepath)

    # --- ADD THE LOG TRANSFORMATION HERE ---
    # We add 1e-8 (a tiny number) to prevent log(0) errors if your audio has true silence
    # spec_matrix = np.log(spec_matrix + 1e-8)
    # ---------------------------------------
    
    if spec_matrix.shape == TARGET_SHAPE:
        X_all.append(spec_matrix.flatten())
        y_all.append(classname)

X_all, y_all = np.array(X_all), np.array(y_all)
classnames = sorted(list(set(y_all)))
print(f"✔ Classes kept: {', '.join(classnames)}")

X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc   = label_encoder.transform(y_val)
y_test_enc  = label_encoder.transform(y_test)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

pca = PCA(n_components=0.8, random_state=1)
X_train_pca = pca.fit_transform(X_train_sc)
X_val_pca   = pca.transform(X_val_sc)
X_test_pca  = pca.transform(X_test_sc)

input_dim = X_train_pca.shape[1]
num_classes = len(classnames)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(torch.FloatTensor(X_train_pca), torch.LongTensor(y_train_enc))
val_dataset   = TensorDataset(torch.FloatTensor(X_val_pca), torch.LongTensor(y_val_enc))

# -------------------------------------------------------
# PART 2: FULLY DYNAMIC NEURAL NETWORK
# -------------------------------------------------------
class AudioMLP(nn.Module):
    def __init__(self, input_size, num_classes, n_layers, hidden_units_list, dropout_rate):
        super(AudioMLP, self).__init__()
        
        layers = []
        in_features = input_size
        
        # Dynamically stack the exact number of layers Optuna asks for
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_units_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_units_list[i] # Output of this layer is input to the next
            
        # Add the final classification layer
        layers.append(nn.Linear(in_features, num_classes))
        
        # Unpack the list of layers into a Sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# -------------------------------------------------------
# PART 3: TRAINING FUNCTION
# -------------------------------------------------------
def train_and_evaluate(params, train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    # Instantiate the dynamic model
    model = AudioMLP(input_dim, num_classes, params['n_layers'], params['hidden_units_list'], params['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Dynamically select the optimizer
    if params['optimizer_name'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    else: # SGD
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'], momentum=0.9)
    
    epochs = 150 
    patience = 30
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    train_loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0 
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break 
            
    return best_val_loss, best_model_state, train_loss_history, val_loss_history

# -------------------------------------------------------
# PART 4: OPTUNA HYPERPARAMETER TUNING
# -------------------------------------------------------
print("\n🔍 Starting Optuna Hyperparameter Optimization (Ultimate Edition)...")

best_overall_loss = float('inf')
best_train_history = []
best_val_history = []

def objective(trial):
    global best_overall_loss, best_train_history, best_val_history
    
    # Let Optuna decide how many layers the network should have (1 to 4)
    # n_layers = trial.suggest_int('n_layers', 1, 4)
    n_layers = 4
    # Create a list to hold the number of neurons for each layer
    hidden_units_list = []
    for i in range(n_layers):
        # Optuna independently decides the width of layer 'i'
        hidden_units_list.append(trial.suggest_int(f'n_units_l{i}', 32, 256, step=32))
    
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.6),
        'batch_size': 128,
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'optimizer_name': 'AdamW',
        'n_layers': n_layers,
        'hidden_units_list': hidden_units_list
    }
    
    val_loss, model_state, t_hist, v_hist = train_and_evaluate(params, train_dataset, val_dataset)
    
    if val_loss < best_overall_loss:
        best_overall_loss = val_loss
        best_train_history = t_hist
        best_val_history = v_hist
        torch.save(model_state, BEST_MODEL_PATH)
        
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=140)

best_hyperparameters = study.best_params

print(f"\n==================================================")
print(f"👑 OPTUNA FINISHED! BEST HYPERPARAMETERS:")
print(f"{best_hyperparameters}")
print(f"🌟 Best Validation Loss: {study.best_value:.4f}")
print(f"==================================================")

# --- OPTUNA VISUAL DASHBOARDS ---
plot_optimization_history(study)
plt.title("Optuna: Optimization History (All Trials)")
plt.tight_layout()
plt.show()

plot_param_importances(study)
plt.title("Optuna: Hyperparameter Importance")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# PART 5: VISUALIZE TRAINING PROGRESS (Best Model)
# -------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(best_train_history, label='Training Loss', color='blue')
plt.plot(best_val_history, label='Validation Loss', color='red')
plt.title(f'Learning Curve for Best Model')
plt.xlabel('Epochs')
plt.ylabel('Loss (CrossEntropy)')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------
# PART 6: EVALUATION ON TEST SET 
# -------------------------------------------------------
print(f"\n📥 Evaluating Best Model on Test Set...")

# Extract the winning layer architecture from the dictionary
# n_layers_best = best_hyperparameters['n_layers']
n_layers_best = 4
hidden_units_best = [best_hyperparameters[f'n_units_l{i}'] for i in range(n_layers_best)]

best_model = AudioMLP(
    input_size=input_dim, 
    num_classes=num_classes, 
    n_layers=n_layers_best, 
    hidden_units_list=hidden_units_best, 
    dropout_rate=best_hyperparameters['dropout']
).to(device)

best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
best_model.eval()

test_dataset = TensorDataset(torch.FloatTensor(X_test_pca), torch.LongTensor(y_test_enc))
# test_loader  = DataLoader(test_dataset, batch_size=best_hyperparameters['batch_size'], shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)
y_pred_list, y_true_list = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = best_model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(y_batch.numpy())

y_pred_names = label_encoder.inverse_transform(y_pred_list)
y_true_names = label_encoder.inverse_transform(y_true_list)

accuracy = accuracy_score(y_true_names, y_pred_names)
precision, recall, f1, _ = precision_recall_fscore_support(y_true_names, y_pred_names, average='weighted', zero_division=0)

print("\n📊 --- FINAL EVALUATION METRICS ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("----------------------------------\n")

print("Detailed Classification Report:\n")
print(classification_report(y_true_names, y_pred_names, zero_division=0))

show_confusion_matrix(y_pred_names, y_true_names, classnames)

# -------------------------------------------------------
# PART 7: FINAL TRAINING ON ENTIRE DATASET
# -------------------------------------------------------
print("\n🚀 Starting FINAL training on 100% of the data...")

# 1. Combine all PCA-transformed features and labels
X_final_full = np.vstack((X_train_pca, X_val_pca, X_test_pca))
y_final_full = np.concatenate((y_train_enc, y_val_enc, y_test_enc))

full_dataset = TensorDataset(torch.FloatTensor(X_final_full), torch.LongTensor(y_final_full))
full_loader  = DataLoader(full_dataset, batch_size=128, shuffle=True)

# 2. Re-initialize the model with the winning architecture
# Note: We use the best_hyperparameters discovered by Optuna
n_layers_final = 4
hidden_units_final = [best_hyperparameters[f'n_units_l{i}'] for i in range(n_layers_final)]

final_model = AudioMLP(
    input_size=input_dim, 
    num_classes=num_classes, 
    n_layers=n_layers_final, 
    hidden_units_list=hidden_units_final, 
    dropout_rate=best_hyperparameters['dropout']
).to(device)

# 3. Setup optimizer (Using the best settings found)
optimizer = optim.AdamW(
    final_model.parameters(), 
    lr=best_hyperparameters['lr'], 
    weight_decay=best_hyperparameters['weight_decay']
)
criterion = nn.CrossEntropyLoss()

# 4. Final Training Loop
# We'll train for a fixed number of epochs (e.g., 100) 
# because there's no validation set left for early stopping!
epochs_final = 100 
final_model.train()

for epoch in range(epochs_final):
    running_loss = 0.0
    for X_batch, y_batch in full_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Final Training: Epoch [{epoch+1}/{epochs_final}], Loss: {running_loss/len(full_loader):.4f}")

# 5. Save the ULTIMATE version of the model
FINAL_PRODUCTION_MODEL_PATH = os.path.join(MODEL_DIR, "model_mlp_PRODUCTION.pth")
torch.save(final_model.state_dict(), FINAL_PRODUCTION_MODEL_PATH)

print(f"\n✅ DONE! The production-ready model is saved at: {FINAL_PRODUCTION_MODEL_PATH}")