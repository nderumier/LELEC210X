import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Custom imports
from src.classification.utils.plots import show_confusion_matrix

# -------------------------------------------------------
# CONFIGURATION & HYPERPARAMETERS
# -------------------------------------------------------
INPUT_VECTORS_DIR = "classification\\feature_vector" 
FM_DIR = "classification\\data\\feature_matrices"
MODEL_DIR = "classification\\data\\models"
BEST_EVAL_MODEL_PATH = os.path.join(MODEL_DIR, "model_mlp_eval.pth")
FINAL_PRODUCTION_MODEL_PATH = os.path.join(MODEL_DIR, "model_mlp_normal.pth")
TARGET_SHAPE = (20, 20)

CLASSES_TO_REMOVE = ["background", "handsaw", "birds", "helicopter", "firorks"]

os.makedirs(FM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ⬇️ SET YOUR WINNING OPTUNA HYPERPARAMETERS HERE ⬇️
# normal feature vector
# HYPERPARAMS = {
#     'lr': 0.006524803678756985,                  # Learning rate
#     'dropout': 0.3337918868738315,               # Dropout rate
#     'batch_size': 128,            # Batch size
#     'weight_decay': 2.346137319834059e-05,         # Weight decay (L2 penalty)
#     'optimizer_name': 'AdamW',    # 'AdamW', 'Adam', or 'SGD'
#     'n_layers': 4,                # Number of hidden layers
#     'hidden_units_list': [32, 64, 32, 160] # Must have exactly 'n_layers' elements
# }

# log feature vector
HYPERPARAMS = {
    'lr': 0.00426406198135054,                  # Learning rate
    'dropout': 0.5643027358745762,               # Dropout rate
    'batch_size': 128,            # Batch size
    'weight_decay': 4.624159846965634e-05,         # Weight decay (L2 penalty)
    'optimizer_name': 'AdamW',    # 'AdamW', 'Adam', or 'SGD'
    'n_layers': 4,                # Number of hidden layers
    'hidden_units_list': [192, 256, 128, 256] # Must have exactly 'n_layers' elements
}

# -------------------------------------------------------
# PART 1: Data Loading and Preprocessing
# -------------------------------------------------------
X_all, y_all = [], []
print(f"📂 Scanning directory: {INPUT_VECTORS_DIR}")

if not os.path.exists(INPUT_VECTORS_DIR):
    raise FileNotFoundError(f"❌ Directory not found: {INPUT_VECTORS_DIR}")

for filename in [f for f in os.listdir(INPUT_VECTORS_DIR) if f.endswith('.npy')]:
    classname = filename.split('_')[0]
    if classname in CLASSES_TO_REMOVE: continue
        
    filepath = os.path.join(INPUT_VECTORS_DIR, filename)
    spec_matrix = np.load(filepath)
    # --- ADD THE LOG TRANSFORMATION HERE ---
    # We add 1e-8 (a tiny number) to prevent log(0) errors if your audio has true silence
    spec_matrix = np.log(spec_matrix + 1e-8)
    # ---------------------------------------

    if spec_matrix.shape == TARGET_SHAPE:
        X_all.append(spec_matrix.flatten())
        y_all.append(classname)

X_all, y_all = np.array(X_all), np.array(y_all)
classnames = sorted(list(set(y_all)))
print(f"✔ Classes kept: {', '.join(classnames)}")

# Split for evaluation purposes
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

pca = PCA(n_components=0.85, random_state=1)
X_train_pca = pca.fit_transform(X_train_sc)
X_val_pca   = pca.transform(X_val_sc)
X_test_pca  = pca.transform(X_test_sc)

# input_dim = X_train_sc.shape[1]
input_dim = X_train_pca.shape[1]
num_classes = len(classnames)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_dataset = TensorDataset(torch.FloatTensor(X_train_sc), torch.LongTensor(y_train_enc))
# val_dataset   = TensorDataset(torch.FloatTensor(X_val_sc), torch.LongTensor(y_val_enc))
train_dataset = TensorDataset(torch.FloatTensor(X_train_pca), torch.LongTensor(y_train_enc))
val_dataset   = TensorDataset(torch.FloatTensor(X_val_pca), torch.LongTensor(y_val_enc))

# -------------------------------------------------------
# PART 2: NEURAL NETWORK ARCHITECTURE
# -------------------------------------------------------
class AudioMLP(nn.Module):
    def __init__(self, input_size, num_classes, n_layers, hidden_units_list, dropout_rate):
        super(AudioMLP, self).__init__()
        
        layers = []
        in_features = input_size
        
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_units_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_units_list[i] 
            
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# -------------------------------------------------------
# PART 3: TRAINING FUNCTION (For Evaluation Model)
# -------------------------------------------------------
def train_and_evaluate(params, train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    model = AudioMLP(input_dim, num_classes, params['n_layers'], params['hidden_units_list'], params['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if params['optimizer_name'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    else: 
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'], momentum=0.9)
    
    epochs = 150 
    patience = 30
    patience_counter = 0
    best_val_loss = float('inf')
    
    train_loss_history, val_loss_history = [], []

    print("\n⏳ Training evaluation model...")
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
            torch.save(model.state_dict(), BEST_EVAL_MODEL_PATH)
            patience_counter = 0 
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏹ Early stopping triggered at epoch {epoch+1}")
            break 
            
    return train_loss_history, val_loss_history

train_history, val_history = train_and_evaluate(HYPERPARAMS, train_dataset, val_dataset)

# -------------------------------------------------------
# PART 4: VISUALIZE TRAINING PROGRESS
# -------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_history, label='Training Loss', color='blue')
plt.plot(val_history, label='Validation Loss', color='red')
plt.title('Learning Curve (Evaluation Model)')
plt.xlabel('Epochs')
plt.ylabel('Loss (CrossEntropy)')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------
# PART 5: EVALUATION ON TEST SET 
# -------------------------------------------------------
print(f"\n📥 Evaluating Model on Test Set...")

eval_model = AudioMLP(
    input_size=input_dim, 
    num_classes=num_classes, 
    n_layers=HYPERPARAMS['n_layers'], 
    hidden_units_list=HYPERPARAMS['hidden_units_list'], 
    dropout_rate=HYPERPARAMS['dropout']
).to(device)

eval_model.load_state_dict(torch.load(BEST_EVAL_MODEL_PATH))
eval_model.eval()

# test_dataset = TensorDataset(torch.FloatTensor(X_test_sc), torch.LongTensor(y_test_enc))
test_dataset = TensorDataset(torch.FloatTensor(X_test_pca), torch.LongTensor(y_test_enc))
test_loader  = DataLoader(test_dataset, batch_size=HYPERPARAMS['batch_size'], shuffle=False)
y_pred_list, y_true_list = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = eval_model(X_batch)
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
# PART 6: FINAL TRAINING ON ENTIRE DATASET
# -------------------------------------------------------
print("\n🚀 Starting FINAL training on 100% of the data...")

# X_final_full = np.vstack((X_train_sc, X_val_sc, X_test_sc))
X_final_full = np.vstack((X_train_pca, X_val_pca, X_test_pca))
y_final_full = np.concatenate((y_train_enc, y_val_enc, y_test_enc))

full_dataset = TensorDataset(torch.FloatTensor(X_final_full), torch.LongTensor(y_final_full))
full_loader  = DataLoader(full_dataset, batch_size=HYPERPARAMS['batch_size'], shuffle=True)

final_model = AudioMLP(
    input_size=input_dim, 
    num_classes=num_classes, 
    n_layers=HYPERPARAMS['n_layers'], 
    hidden_units_list=HYPERPARAMS['hidden_units_list'], 
    dropout_rate=HYPERPARAMS['dropout']
).to(device)

if HYPERPARAMS['optimizer_name'] == 'AdamW':
    optimizer = optim.AdamW(final_model.parameters(), lr=HYPERPARAMS['lr'], weight_decay=HYPERPARAMS['weight_decay'])
elif HYPERPARAMS['optimizer_name'] == 'Adam':
    optimizer = optim.Adam(final_model.parameters(), lr=HYPERPARAMS['lr'], weight_decay=HYPERPARAMS['weight_decay'])
else:
    optimizer = optim.SGD(final_model.parameters(), lr=HYPERPARAMS['lr'], weight_decay=HYPERPARAMS['weight_decay'], momentum=0.9)

criterion = nn.CrossEntropyLoss()

# Train for a fixed number of epochs since we have no validation set
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

torch.save(final_model.state_dict(), FINAL_PRODUCTION_MODEL_PATH)
import pickle

# After training is complete, save the scaler and encoder
preprocessing_data = {
    "scaler": scaler,
    "pca": pca,
    "label_encoder": label_encoder
}

with open(os.path.join(MODEL_DIR, "scaler_and_encoder.pickle"), "wb") as f:
    pickle.dump(preprocessing_data, f)
print(f"\n✅ DONE! The production-ready model is saved at: {FINAL_PRODUCTION_MODEL_PATH}")