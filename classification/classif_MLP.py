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

# -------------------------------------------------------
# PART 2: PYTORCH DATASET & DATALOADERS
# -------------------------------------------------------
train_dataset = TensorDataset(torch.FloatTensor(X_train_pca), torch.LongTensor(y_train_enc))
val_dataset   = TensorDataset(torch.FloatTensor(X_val_pca), torch.LongTensor(y_val_enc))
test_dataset  = TensorDataset(torch.FloatTensor(X_test_pca), torch.LongTensor(y_test_enc))

BATCH_SIZE = 512 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------------------------------
# PART 3: DEFINING THE NEURAL NETWORK (Increased Dropout)
# -------------------------------------------------------
class AudioMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased to 0.5 to prevent memorization
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased to 0.5
            nn.Linear(50, num_classes) 
        )

    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioMLP(input_size=input_dim, num_classes=num_classes).to(device)

# -------------------------------------------------------
# PART 4: LOSS, OPTIMIZER & SCHEDULER (Tweaked params)
# -------------------------------------------------------
criterion = nn.CrossEntropyLoss()
# Lowered learning rate to 0.001 and increased weight_decay to 1e-3
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# -------------------------------------------------------
# PART 5: THE TRAINING LOOP (With Early Stopping)
# -------------------------------------------------------
EPOCHS = 100
train_losses, val_losses = [], []
best_val_loss = float('inf') # Set to infinity initially

print(f"\n🚀 Starting Training Loop on {device}...")
for epoch in range(EPOCHS):
    
    # --- TRAINING PHASE ---
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
    train_losses.append(epoch_train_loss)
    
    # --- VALIDATION PHASE ---
    model.eval() 
    running_val_loss = 0.0
    
    with torch.no_grad(): 
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_val_loss += loss.item() * X_batch.size(0)
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    # --- EARLY STOPPING & SAVING ---
    saved_flag = ""
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH) # Save the exact weights
        saved_flag = " 🌟 (New Best Model Saved!)"
    
    # --- SCHEDULER STEP ---
    scheduler.step(epoch_val_loss) 
    
    if (epoch+1) % 5 == 0 or saved_flag:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f} | LR: {current_lr:.6f}{saved_flag}")

# -------------------------------------------------------
# PART 6: PLOTTING LEARNING CURVES
# -------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss (CrossEntropy)')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------
# PART 7: FINAL EVALUATION ON TEST SET (Using the BEST model)
# -------------------------------------------------------
print(f"\n📥 Loading best model from {BEST_MODEL_PATH} for final test...")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

y_pred_list = []
y_true_list = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        
        _, predicted = torch.max(outputs, 1) 
        
        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(y_batch.numpy())

# Convert integer predictions back to string class names
y_pred_names = label_encoder.inverse_transform(y_pred_list)
y_true_names = label_encoder.inverse_transform(y_true_list)

print("\n🎯 Final Test Accuracy:", accuracy_score(y_true_names, y_pred_names))
print("\n📊 Classification Report:\n", classification_report(y_true_names, y_pred_names))

show_confusion_matrix(y_pred_names, y_true_names, classnames)