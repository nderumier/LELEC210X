import os
import random
import re
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
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader

# Custom imports
from src.classification.utils.plots import show_confusion_matrix

# -------------------------------------------------------
# PART 1: CONFIGURATION & HYPERPARAMETERS
# -------------------------------------------------------
INPUT_VECTORS_DIR = "classification\\feature_vector_sud11" 
FM_DIR = "classification\\data\\feature_matrices"
MODEL_DIR = "classification\\data\\models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "model_mlp_V2.pth")
FINAL_PRODUCTION_MODEL_PATH = os.path.join(MODEL_DIR, "model_mlp_bottom_0hidden_proba.pth")
TARGET_SHAPE = (20, 20)

# Dataset configuration
KEEP_ORIGINAL_DATA = True 

# --- MANUAL HYPERPARAMETER CONFIGURATION ---
HIDDEN_UNITS = [500, 400, 300, 200]  
LEARNING_RATE = 0.00669
DROPOUT_RATE = 0.3863
WEIGHT_DECAY = 0.00004
BATCH_SIZE = 512
OPTIMIZER_NAME = 'AdamW'  

ALLOWED_PREFIXES = []

# --- NEW: FREQUENCY BAND HIDING CONFIGURATION ---
NUM_BANDS_TO_HIDE = 0 
MASKING_STRATEGY = 'top' 
# ------------------------------------------------

# Training configuration
MAX_EPOCHS = 500
PATIENCE = 30  

os.makedirs(FM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
# PART 2: Data Loading, Augmentation, and Preprocessing
# -------------------------------------------------------
print(f"📂 Scanning directory: {INPUT_VECTORS_DIR}")

if not os.path.exists(INPUT_VECTORS_DIR):
    raise FileNotFoundError(f"❌ Directory not found: {INPUT_VECTORS_DIR}")

all_files = [f for f in os.listdir(INPUT_VECTORS_DIR) if f.endswith('.npy')]

def hide_frequency_bands(matrix, num_bands, strategy='random'):
    if num_bands <= 0:
        return matrix
        
    masked_matrix = matrix.copy()
    max_bands = matrix.shape[0] 
    
    if strategy == 'random':
        bands_to_mask = random.sample(range(max_bands), num_bands)
    elif strategy == 'top':
        bands_to_mask = list(range(max_bands - num_bands, max_bands))
    elif strategy == 'bottom':
        bands_to_mask = list(range(num_bands))
    else:
        bands_to_mask = []
        
    masked_matrix[bands_to_mask, :] = 0.0
    return masked_matrix

# --- STEP A: LOAD ALL BACKGROUND FILES FIRST ---
background_matrices = []
for filename in all_files:
    if filename.startswith('background'):
        filepath = os.path.join(INPUT_VECTORS_DIR, filename)
        bg_matrix = np.load(filepath)
        if bg_matrix.shape == TARGET_SHAPE:
            background_matrices.append(bg_matrix)

print(f"🔊 Found {len(background_matrices)} valid background files for augmentation.")

# --- STEP B: LOAD TARGET FILES AND APPLY BACKGROUND ---
X_all, y_all = [], []

for filename in all_files:
    if filename.startswith('background'):
        continue

    prefix = filename.split('_')[0]

    if len(ALLOWED_PREFIXES) > 0 and prefix not in ALLOWED_PREFIXES:
        continue

    classname = re.sub(r'\d+', '', prefix).lower()
        
    filepath = os.path.join(INPUT_VECTORS_DIR, filename)
    spec_matrix_orig = np.load(filepath)
    
    # Log transformation
    spec_matrix_orig = np.log(spec_matrix_orig + 1e-8)
    
    if spec_matrix_orig.shape == TARGET_SHAPE:
        
        # 1. OPTIONALLY ADD ORIGINAL CLEAN DATA
        if KEEP_ORIGINAL_DATA:
            clean_masked = hide_frequency_bands(spec_matrix_orig, NUM_BANDS_TO_HIDE, MASKING_STRATEGY)
            X_all.append(clean_masked.flatten())
            y_all.append(classname)
            
        # 2. DATA AUGMENTATION: ADD 20dB ATTENUATED BACKGROUND
        if len(background_matrices) > 0:
            random_bg = random.choice(background_matrices)
            attenuated_bg = random_bg * 0.1 
            spec_matrix_aug = spec_matrix_orig + attenuated_bg 
            
            aug_masked = hide_frequency_bands(spec_matrix_aug, NUM_BANDS_TO_HIDE, MASKING_STRATEGY)
            X_all.append(aug_masked.flatten())
            y_all.append(classname)
            
        # Fallback if no backgrounds are found but KEEP_ORIGINAL_DATA is False
        elif not KEEP_ORIGINAL_DATA: 
            clean_masked = hide_frequency_bands(spec_matrix_orig, NUM_BANDS_TO_HIDE, MASKING_STRATEGY)
            X_all.append(clean_masked.flatten())
            y_all.append(classname)

X_all, y_all = np.array(X_all), np.array(y_all)
classnames = sorted(list(set(y_all)))
print(f"✔ Classes kept: {', '.join(classnames)}")
print(f"✔ Total samples in dataset: {len(X_all)}")

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------------------------------
# PART 3: NEURAL NETWORK ARCHITECTURE
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

# -------------------------------------------------------
# PART 4: TRAINING & VALIDATION (With Early Stopping)
# -------------------------------------------------------
print("\n⚙️ Starting Training Phase...")

model = AudioMLP(input_dim, num_classes, HIDDEN_UNITS, DROPOUT_RATE).to(device)
criterion = nn.CrossEntropyLoss()

if OPTIMIZER_NAME == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER_NAME == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else: 
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)

patience_counter = 0
best_val_loss = float('inf')
train_loss_history, val_loss_history = [], []

for epoch in range(MAX_EPOCHS):
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
        torch.save(model.state_dict(), BEST_MODEL_PATH) 
        patience_counter = 0 
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0 or patience_counter >= PATIENCE:
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
    if patience_counter >= PATIENCE:
        print(f"\n⏹ Early stopping triggered at epoch {epoch+1}.")
        break 

# Plot Learning Curve
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label='Training Loss', color='blue')
plt.plot(val_loss_history, label='Validation Loss', color='red')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss (CrossEntropy)')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------
# PART 5: EVALUATION ON TEST SET 
# -------------------------------------------------------
print(f"\n📥 Evaluating Best Model on Test Set...")

best_model = AudioMLP(input_dim, num_classes, HIDDEN_UNITS, DROPOUT_RATE).to(device)
best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
best_model.eval()

test_dataset = TensorDataset(torch.FloatTensor(X_test_pca), torch.LongTensor(y_test_enc))
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
y_pred_list, y_true_list = [], []

print(f"\n❌ --- Analysing Incorrect Predictions ---")

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        
        # Get 0.0 to 1.0 confidence scores
        probabilities = best_model.predict_proba(X_batch)
        max_probs, predicted = torch.max(probabilities, 1)
        
        # --- NEW: Loop through the batch to find mistakes ---
        for i in range(len(y_batch)):
            true_label = y_batch[i].item()
            pred_label = predicted[i].item()
            
            # If the prediction was wrong, print the details
            if true_label != pred_label:
                probs_formatted = [f"{p:.4f}" for p in probabilities[i].cpu().numpy()]
                pred_class_name = label_encoder.inverse_transform([pred_label])[0]
                true_class_name = label_encoder.inverse_transform([true_label])[0]
                
                print(f"Mismatch:")
                print(f"  Class Probs:  {probs_formatted}")
                print(f"  Confidence:   {max_probs[i].item() * 100:.2f}% (Guessed: {pred_class_name})")
                print(f"  Actually was: {true_class_name}\n")
        # --------------------------------------------------

        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(y_batch.numpy())

print("------------------------------------------\n")

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
print("\n🚀 Starting FINAL training on 100% of the data for Production...")

X_final_full = np.vstack((X_train_pca, X_val_pca, X_test_pca))
y_final_full = np.concatenate((y_train_enc, y_val_enc, y_test_enc))

full_dataset = TensorDataset(torch.FloatTensor(X_final_full), torch.LongTensor(y_final_full))
full_loader  = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

final_model = AudioMLP(input_dim, num_classes, HIDDEN_UNITS, DROPOUT_RATE).to(device)

if OPTIMIZER_NAME == 'AdamW':
    optimizer_final = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER_NAME == 'Adam':
    optimizer_final = optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    optimizer_final = optim.SGD(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)

criterion_final = nn.CrossEntropyLoss()

epochs_final = 100 
final_model.train()

for epoch in range(epochs_final):
    running_loss = 0.0
    for X_batch, y_batch in full_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer_final.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion_final(outputs, y_batch)
        loss.backward()
        optimizer_final.step()
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Production Training: Epoch [{epoch+1}/{epochs_final}], Loss: {running_loss/len(full_loader):.4f}")

torch.save(final_model.state_dict(), FINAL_PRODUCTION_MODEL_PATH)

print(f"\n✅ DONE! The production-ready model is saved at: {FINAL_PRODUCTION_MODEL_PATH}")