import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Custom imports
from src.classification.utils.plots import show_confusion_matrix

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
INPUT_VECTORS_DIR = "classification\\feature_vector" 
FM_DIR = "classification\\data\\feature_matrices"
MODEL_DIR = "classification\\data\\models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_audio_mlp.pth")
TARGET_SHAPE = (20, 20)

# Classes to exclude
CLASSES_TO_REMOVE = ["background", "handsaw", "birds", "helicopter", "firorks"]

os.makedirs(FM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
# PART 0: Loading .npy files from directory
# -------------------------------------------------------
X_all = []
y_all = []

print(f"📂 Scanning directory: {INPUT_VECTORS_DIR}")

if not os.path.exists(INPUT_VECTORS_DIR):
    raise FileNotFoundError(f"❌ Directory not found: {INPUT_VECTORS_DIR}")

all_files = [f for f in os.listdir(INPUT_VECTORS_DIR) if f.endswith('.npy')]

for filename in all_files:
    classname = filename.split('_')[0]
    
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

classnames = sorted(list(set(y_all)))
print(f"✔ Classes kept: {', '.join(classnames)}")

# -------------------------------------------------------
# PART 1: Train / Val / Test Split & Preprocessing
# -------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

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

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Custom imports
from src.classification.utils.plots import show_confusion_matrix

# -------------------------------------------------------
# CONFIGURATION & PART 0-1 (Keep your existing code here)
# -------------------------------------------------------
# (Assuming X_train_pca, X_val_pca, X_test_pca, y_train_enc, y_val_enc, y_test_enc, 
# input_dim, and num_classes are all loaded and processed as per your snippet)

# Ensure device is set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "best_tuned_audio_mlp.pth"

# -------------------------------------------------------
# PART 2: DYNAMIC NEURAL NETWORK (For Hyperparameter Tuning)
# -------------------------------------------------------
class AudioMLP(nn.Module):
    # We pass hidden_units and dropout_rate dynamically now
    def __init__(self, input_size, num_classes, hidden_units, dropout_rate):
        super(AudioMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Technique to mitigate overfitting
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units[1], num_classes) 
        )

    def forward(self, x):
        return self.network(x)

# -------------------------------------------------------
# PART 3: TRAINING FUNCTION (Monitoring Progress & Early Stopping)
# -------------------------------------------------------
def train_and_evaluate(params, train_dataset, val_dataset):
    """Trains a model with specific hyperparameters and returns the best validation loss."""
    
    # 1. Setup DataLoaders with the chosen batch size
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    # 2. Initialize Model, Loss, and Optimizer
    model = AudioMLP(input_dim, num_classes, params['hidden_units'], params['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-3)
    
    # 3. Early Stopping Setup
    epochs = 50 
    patience = 7
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)

        # --- EARLY STOPPING LOGIC ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0 # Reset patience
        else:
            patience_counter += 1

        if patience_counter >= patience:
            # Stop early if the model is learning to memorize instead of generalise
            break 
            
    return best_val_loss, best_model_state

# -------------------------------------------------------
# PART 4: HYPERPARAMETER TUNING (Random Search)
# -------------------------------------------------------
print("\n🔍 Starting Hyperparameter Tuning (Random Search)...")

# Define the grid of hyperparameters to explore
param_grid = {
    'lr': [0.001, 0.0005, 0.0001],
    'batch_size': [64, 128, 256],
    'dropout': [0.3, 0.4, 0.5],
    'hidden_units': [[128, 64], [100, 50], [64, 32]]
}

# Create base PyTorch datasets
train_dataset = TensorDataset(torch.FloatTensor(X_train_pca), torch.LongTensor(y_train_enc))
val_dataset   = TensorDataset(torch.FloatTensor(X_val_pca), torch.LongTensor(y_val_enc))

NUM_SEARCHES = 10 # Number of random combinations to try
best_overall_loss = float('inf')
best_hyperparameters = None

for i in range(NUM_SEARCHES):
    # Randomly select a combination of hyperparameters
    current_params = {
        'lr': random.choice(param_grid['lr']),
        'batch_size': random.choice(param_grid['batch_size']),
        'dropout': random.choice(param_grid['dropout']),
        'hidden_units': random.choice(param_grid['hidden_units'])
    }
    
    print(f"[{i+1}/{NUM_SEARCHES}] Testing Params: {current_params}")
    
    # Train model with these parameters
    val_loss, model_state = train_and_evaluate(current_params, train_dataset, val_dataset)
    
    print(f"    ↳ Validation Loss: {val_loss:.4f}")
    
    # Save the best model across all searches
    if val_loss < best_overall_loss:
        best_overall_loss = val_loss
        best_hyperparameters = current_params
        torch.save(model_state, BEST_MODEL_PATH)

print(f"\n🏆 Best Hyperparameters Found: {best_hyperparameters}")
print(f"🌟 Best Validation Loss: {best_overall_loss:.4f}")

# -------------------------------------------------------
# PART 5: EVALUATION ON TEST SET (Accuracy, Precision, Recall, F1)
# -------------------------------------------------------
print(f"\n📥 Evaluating Best Model on Test Set...")

# Initialize the model using the BEST hyperparameters found
best_model = AudioMLP(
    input_size=input_dim, 
    num_classes=num_classes, 
    hidden_units=best_hyperparameters['hidden_units'], 
    dropout_rate=best_hyperparameters['dropout']
).to(device)

# Load the saved weights
best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
best_model.eval()

# Create Test DataLoader using the best batch size
test_dataset = TensorDataset(torch.FloatTensor(X_test_pca), torch.LongTensor(y_test_enc))
test_loader  = DataLoader(test_dataset, batch_size=best_hyperparameters['batch_size'], shuffle=False)

y_pred_list, y_true_list = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = best_model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(y_batch.numpy())

# Convert labels back to strings
y_pred_names = label_encoder.inverse_transform(y_pred_list)
y_true_names = label_encoder.inverse_transform(y_true_list)

# Calculate metrics from the image
accuracy = accuracy_score(y_true_names, y_pred_names)
precision, recall, f1, _ = precision_recall_fscore_support(y_true_names, y_pred_names, average='weighted')

print("\n📊 --- FINAL EVALUATION METRICS ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("----------------------------------\n")

# Detailed class-by-class report
print("Detailed Classification Report:\n")
print(classification_report(y_true_names, y_pred_names))

show_confusion_matrix(y_pred_names, y_true_names, classnames)