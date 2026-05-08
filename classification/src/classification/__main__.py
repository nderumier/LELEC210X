

# # ####### MLP #######

# import json
# import pickle
# import os
# from pathlib import Path
# from datetime import datetime  # <-- Added for saving

# import click
# import requests
# import numpy as np
# import cv2

# # PyTorch Imports
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import common
# from auth import PRINT_PREFIX
# from common.env import load_dotenv
# from common.logging import logger
# from leaderboard.utils import get_url

# from .utils import payload_to_melvecs

# print("Starting ensemble classification script...")
# load_dotenv()
# print("Environment variables loaded.")

# TARGET_SHAPE = (20, 20)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =======================================================
# # 1. DEFINE YOUR PYTORCH ARCHITECTURE HERE
# # (Must exactly match the model you trained)
# # =======================================================
# class AudioMLP(nn.Module):
#     def __init__(self, input_size, num_classes, hidden_units_list, dropout_rate):
#         super(AudioMLP, self).__init__()
#         layers = []
#         in_features = input_size
#         n_layers = len(hidden_units_list)
        
#         for i in range(n_layers):
#             layers.append(nn.Linear(in_features, hidden_units_list[i]))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout_rate))
#             in_features = hidden_units_list[i] 
            
#         layers.append(nn.Linear(in_features, num_classes))
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)
        
#     def predict_proba(self, x):
#         logits = self.network(x)
#         return F.softmax(logits, dim=1)

# # -------------------------------------------------------
# # HELPER: Frequency Masking
# # -------------------------------------------------------
# def hide_frequency_bands(matrix, num_bands, strategy='top'):
#     if num_bands <= 0:
#         return matrix
#     masked_matrix = matrix.copy()
#     max_bands = matrix.shape[0] 
    
#     if strategy == 'top':
#         bands_to_mask = list(range(max_bands - num_bands, max_bands))
#     elif strategy == 'bottom':
#         bands_to_mask = list(range(num_bands))
#     else:
#         bands_to_mask = []
        
#     masked_matrix[bands_to_mask, :] = 0.0
#     return masked_matrix

# # -------------------------------------------------------
# # HELPER: Feature Extraction with Forced Resize
# # -------------------------------------------------------
# def get_fixed_feature(melvec):
#     feat2d = melvec

#     if feat2d.shape != TARGET_SHAPE:
#         feat2d = cv2.resize(
#             feat2d,
#             (TARGET_SHAPE[1], TARGET_SHAPE[0]),
#             interpolation=cv2.INTER_AREA,
#         )

#     return feat2d.reshape(-1)


# @click.command()
# @click.option("-i", "--input", "_input", default="-", type=click.File("r"))
# @click.option("-m", "--model_dir", default="classification/data/models", type=click.Path(exists=True, file_okay=False, path_type=Path))
# @common.click.melvec_length
# @common.click.n_melvecs
# @click.option("--submit/--no-submit", default=True)
# @click.option("-u", "--url", default=None, envvar="LEADERBOARD_URL")
# @click.option("-k", "--key", default=None, envvar="LEADERBOARD_KEY")
# @common.click.verbosity
# def main(
#     _input: click.File | None,
#     model_dir: Path | None,
#     melvec_length: int,
#     n_melvecs: int,
#     submit: bool,
#     url: str | None,
#     key: str | None,
# ) -> None:

#     if submit:
#         if key is None:
#             raise click.UsageError("You must provide a key to submit guesses.")
#         url = url or get_url()

#     # =====================================================
#     # 2. LOAD ENSEMBLE PARAMETERS
#     # =====================================================
#     print("Loading Ensemble Parameters and Models...")
#     params_path = model_dir / "ensemble_production_params.pkl"
    
#     if not params_path.exists():
#         raise FileNotFoundError(f"Could not find {params_path}. Please run the training script first.")
        
#     with open(params_path, "rb") as f:
#         production_params = pickle.load(f)
        
#     global_classes = production_params['classes']
#     loaded_ensemble = []
    
#     # Must match the hyperparameters used during Optuna training
#     HIDDEN_UNITS = [500, 400, 300, 200]
#     DROPOUT_RATE = 0.3863

#     # =====================================================
#     # 3. INITIALIZE AND LOAD ALL PYTORCH MODELS
#     # =====================================================
#     for model_data in production_params['models']:
#         pth_path = model_dir / model_data['pth_file']
        
#         if not pth_path.exists():
#             print(f"⚠️ Warning: Model weights {pth_path} not found. Skipping this model.")
#             continue
            
#         model = AudioMLP(
#             input_size=model_data['pca_components'].shape[0],
#             num_classes=len(global_classes), 
#             hidden_units_list=HIDDEN_UNITS, 
#             dropout_rate=DROPOUT_RATE
#         ).to(device)

#         # Load weights
#         model.load_state_dict(torch.load(pth_path, map_location=device))
#         model.eval() 
        
#         # Store the instantiated model inside the dictionary
#         model_data['model_obj'] = model
#         loaded_ensemble.append(model_data)
        
#         print(f"✔️ Successfully loaded {model_data['pth_file']}")

#     if not loaded_ensemble:
#         raise RuntimeError("No models were successfully loaded. Cannot perform inference.")
        
#     print(f"\nEnsemble completely loaded! Operating with {len(loaded_ensemble)} models.\n")

#     # ----------------------------
#     # Stream payloads
#     # ----------------------------
#     for payload in _input:
#         print(f"Received payload: {payload.strip()}")
#         if PRINT_PREFIX in payload:
#             payload = payload[len(PRINT_PREFIX) :]

#             melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
#             logger.info(f"Parsed payload into Mel vectors: {melvecs.shape}")

#             # --- NEW: AUTOMATIC SAVING MECHANISM ---
#             os.makedirs("captured_live_audio", exist_ok=True) 
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#             save_path = os.path.join("captured_live_audio", f"live_melvec_{timestamp}.npy")
#             np.save(save_path, melvecs)
#             print(f"💾 Automatically saved raw feature vector to {save_path}")
#             # ---------------------------------------

#             # =====================================================
#             # 4. PRECISION-WEIGHTED ENSEMBLE PIPELINE
#             # =====================================================
            
#             # --- IMPORTANT: Apply the Global Log transformation! ---
#             melvecs = np.log(melvecs + 1e-8)
            
#             # Accumulator for final soft votes
#             final_ensemble_scores = np.zeros(len(global_classes))
            
#             for member in loaded_ensemble:
#                 # 4.a Apply specific frequency mask
#                 masked_melvecs = hide_frequency_bands(melvecs, member['num_bands'], member['mask_strategy'])
                
#                 # 4.b Resize and flatten
#                 feature_vector = get_fixed_feature(masked_melvecs)
                
#                 # 4.c Manual Standard Scaler implementation
#                 feat_scaled = (feature_vector - member['scaler_mean']) /  member['scaler_scale']
                
#                 # 4.d Manual PCA projection implementation
#                 feat_pca = np.dot(feat_scaled - member['pca_mean'], member['pca_components'].T)
                
#                 # 4.e Tensorize
#                 input_tensor = torch.FloatTensor(feat_pca).unsqueeze(0).to(device)
                
#                 # 4.f Get probabilities
#                 with torch.no_grad():
#                     probs = member['model_obj'].predict_proba(input_tensor).cpu().numpy()[0]
                    
#                 # 4.g Apply precision weights
#                 weighted_probs = probs * member['class_precisions']
#                 final_ensemble_scores += weighted_probs

#             # =====================================================
#             # 5. FINAL DECISION
#             # =====================================================
#             final_predicted_idx = np.argmax(final_ensemble_scores)
#             guess = global_classes[final_predicted_idx]
            
#             print(f"Predicted class: {guess}")
#             logger.info(f"Prediction: {guess}")

#             url_endpoint = "http://lelec210x.sipr.ucl.ac.be"
#             if submit:
#                 response = requests.post(
#                     f"{url_endpoint}/lelec210x/leaderboard/submit/{key}/{guess}"
#                 )

#                 response_as_dict = json.loads(response.text)

#                 if response.status_code == 200:
#                     logger.info(response_as_dict)
#                 else:
#                     logger.error(response_as_dict)

# if __name__ == "__main__":
#     main()

######### LOCAL TESTING PIPELINE #########


import json
import pickle
import os
from pathlib import Path
import sys

# --- THE SURGICAL PATH FIX ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../..")) # The LELEC210X root

# 1. Add the classification root
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../.."))) 

# 2. Add the specific 'src' folders for your other modules so Python finds the actual code!
sys.path.insert(0, os.path.join(project_root, "auth", "src"))
sys.path.insert(0, os.path.join(project_root, "common", "src"))
sys.path.insert(0, os.path.join(project_root, "leaderboard", "src"))
# --------------------------------

import click
import requests
import numpy as np
import cv2

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
from leaderboard.utils import get_url

# Fixed: Removed the relative dot because we are running this file directly!
from utils import payload_to_melvecs

print("Starting ensemble classification script...")
load_dotenv()
print("Environment variables loaded.")

TARGET_SHAPE = (20, 20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================================================
# 1. DEFINE YOUR PYTORCH ARCHITECTURE HERE
# =======================================================
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
# HELPER: Frequency Masking
# -------------------------------------------------------
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


@click.command()
@click.option("-i", "--input", "_input", default="-", type=click.File("r"))
@click.option("-m", "--model_dir", default="classification/data/models", type=click.Path(exists=True, file_okay=False, path_type=Path))
@common.click.melvec_length
@common.click.n_melvecs
@click.option("--submit/--no-submit", default=True)
@click.option("-u", "--url", default=None, envvar="LEADERBOARD_URL")
@click.option("-k", "--key", default=None, envvar="LEADERBOARD_KEY")
@common.click.verbosity
@click.option("--test-npy", default=None, type=click.Path(exists=True), help="Path to a .npy file to test directly")
def main(
    _input: click.File | None,
    model_dir: Path | None,
    melvec_length: int,
    n_melvecs: int,
    submit: bool,
    url: str | None,
    key: str | None,
    test_npy: str | None, 
) -> None:

    if submit and not test_npy:
        if key is None:
            raise click.UsageError("You must provide a key to submit guesses.")
        url = url or get_url()

    # =====================================================
    # 2. LOAD ENSEMBLE PARAMETERS
    # =====================================================
    print("Loading Ensemble Parameters and Models...")
    params_path = model_dir / "ensemble_production_params_test.pkl"
    
    if not params_path.exists():
        raise FileNotFoundError(f"Could not find {params_path}. Please run the training script first.")
        
    with open(params_path, "rb") as f:
        production_params = pickle.load(f)
        
    global_classes = production_params['classes']
    loaded_ensemble = []
    
    # Must match the hyperparameters used during Optuna training
    HIDDEN_UNITS = [500, 400, 300, 200]
    DROPOUT_RATE = 0.3863

    # =====================================================
    # 3. INITIALIZE AND LOAD ALL PYTORCH MODELS
    # =====================================================
    for model_data in production_params['models']:
        pth_path = model_dir / model_data['pth_file']
        
        if not pth_path.exists():
            print(f"⚠️ Warning: Model weights {pth_path} not found. Skipping this model.")
            continue
            
        model = AudioMLP(
            input_size=model_data['pca_components'].shape[0],
            num_classes=len(global_classes), 
            hidden_units_list=HIDDEN_UNITS, 
            dropout_rate=DROPOUT_RATE
        ).to(device)

        # Load weights
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.eval() 
        
        # Store the instantiated model inside the dictionary
        model_data['model_obj'] = model
        loaded_ensemble.append(model_data)
        
        print(f"✔️ Successfully loaded {model_data['pth_file']}")

    if not loaded_ensemble:
        raise RuntimeError("No models were successfully loaded. Cannot perform inference.")
        
    print(f"\nEnsemble completely loaded! Operating with {len(loaded_ensemble)} models.\n")

    # ----------------------------
    # STREAM OR TEST PAYLOADS
    # ----------------------------
    if test_npy:
        print(f"🧪 TESTING MODE: Loading {test_npy} directly...")
        melvecs = np.load(test_npy)
        payload_iterator = [melvecs] 
        is_live_stream = False
    else:
        payload_iterator = _input
        is_live_stream = True

    # ----------------------------
    # PIPELINE LOOP
    # ----------------------------
    for item in payload_iterator:
        
        # 1. Parse data depending on where it came from
        if is_live_stream:
            if isinstance(item, str):
                payload = item
            else:
                payload = item.read() # Read from buffer if it's a file stream
                
            print(f"Received payload: {payload.strip()}")
            if PRINT_PREFIX in payload:
                payload = payload[len(PRINT_PREFIX) :]
                melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
                logger.info(f"Parsed payload into Mel vectors: {melvecs.shape}")
            else:
                continue # Skip invalid serial prints
        else:
            melvecs = item 

        # =====================================================
        # 4. PRECISION-WEIGHTED ENSEMBLE PIPELINE
        # =====================================================
        
        # 4.a FIRST: Ensure the matrix is exactly TARGET_SHAPE (20x20)
        if melvecs.shape != TARGET_SHAPE:
            melvecs = cv2.resize(
                melvecs,
                (TARGET_SHAPE[1], TARGET_SHAPE[0]),
                interpolation=cv2.INTER_AREA,
            )
            
        # 4.b SECOND: Apply the Global Log transformation
        melvecs = np.log(melvecs + 1e-8)
        
        # Accumulator for final soft votes
        final_ensemble_scores = np.zeros(len(global_classes))
        
        for member in loaded_ensemble:
            # 4.c Apply specific frequency mask
            masked_melvecs = hide_frequency_bands(melvecs, member['num_bands'], member['mask_strategy'])
            
            # 4.d Flatten the matrix into a 400-length vector
            feature_vector = masked_melvecs.reshape(-1)
            
            # 4.e Manual Standard Scaler implementation
            feat_scaled = (feature_vector - member['scaler_mean']) / member['scaler_scale']
            
            # 4.f Manual PCA projection implementation
            feat_pca = np.dot(feat_scaled - member['pca_mean'], member['pca_components'].T)
            
            # 4.g Tensorize
            input_tensor = torch.FloatTensor(feat_pca).unsqueeze(0).to(device)
            
            # 4.h Get probabilities
            with torch.no_grad():
                probs = member['model_obj'].predict_proba(input_tensor).cpu().numpy()[0]
                
            # 4.i Apply precision weights
            weighted_probs = probs * member['class_precisions']
            final_ensemble_scores += weighted_probs

        # =====================================================
        # 5. FINAL DECISION
        # =====================================================
        final_predicted_idx = np.argmax(final_ensemble_scores)
        guess = global_classes[final_predicted_idx]
        
        print(f"Predicted class: {guess}")
        logger.info(f"Prediction: {guess}")

        # Only submit if we are NOT in test mode and submit is True
        if submit and not test_npy:
            url_endpoint = "http://lelec210x.sipr.ucl.ac.be"
            response = requests.post(
                f"{url_endpoint}/lelec210x/leaderboard/submit/{key}/{guess}"
            )

            response_as_dict = json.loads(response.text)

            if response.status_code == 200:
                logger.info(response_as_dict)
            else:
                logger.error(response_as_dict)

if __name__ == "__main__":
    main()