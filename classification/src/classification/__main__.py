#####  SVM  #####


# print("Importing libraries...")


# import json
# import pickle
# from pathlib import Path

# import click
# import requests
# import numpy as np
# import cv2

# import common
# from auth import PRINT_PREFIX
# from common.env import load_dotenv
# from common.logging import logger
# from leaderboard.utils import get_url

# from .utils import payload_to_melvecs
# print("Starting classification script...")
# load_dotenv()
# print("Environment variables loaded.")

# TARGET_SHAPE = (20, 20)


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
# @click.option(
#     "-i",
#     "--input",
#     "_input",
#     default="-",
#     type=click.File("r"),
# )
# @click.option(
#     "-m",
#     "--model",
#     default=None,
#     type=click.Path(exists=True, dir_okay=False, path_type=Path),
# )
# @common.click.melvec_length
# @common.click.n_melvecs
# @click.option("--submit/--no-submit", default=True)
# @click.option(
#     "-u",
#     "--url",
#     default=None,
#     envvar="LEADERBOARD_URL",
# )
# @click.option(
#     "-k",
#     "--key",
#     default=None,
#     envvar="LEADERBOARD_KEY",
# )
# @common.click.verbosity
# def main(
#     _input: click.File | None,
#     model: Path | None,
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

#     # ----------------------------
#     # Load model
#     # ----------------------------
#     with open("classification/data/models/model_audio_svm.pickle", "rb") as f:
#         clf = pickle.load(f)
#     print(f"Loaded model: {clf}")
#     # Same logic as your UART script
#     if isinstance(clf, dict):
#         scaler = clf.get("scaler")
#         pca = clf.get("pca")
#         model = clf["model"]
#     else:
#         model = clf
#         scaler = None
#         pca = None

#     # ----------------------------
#     # Stream payloads
#     # ----------------------------
#     for payload in _input:
#         print(f"Received payload: {payload.strip()}")
#         if PRINT_PREFIX in payload:
#             payload = payload[len(PRINT_PREFIX) :]

#             melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
#             print(f"Parsed payload into Mel vectors with shape: {melvecs.shape}")
#             logger.info(f"Parsed payload into Mel vectors: {melvecs.shape}")

#             if model:

#                 # =====================================================
#                 # YOUR EXACT CLASSIFICATION PIPELINE
#                 # =====================================================

#                 feature_vector = get_fixed_feature(melvecs)

#                 # ---- Normalisation ----
#                 norm_val = np.linalg.norm(feature_vector)
#                 if norm_val == 0:
#                     norm_val = 1e-9
#                 feature_norm = feature_vector / norm_val

#                 # ---- Optional scaler ----
#                 if scaler is not None:
#                     feature_norm = scaler.transform([feature_norm])[0]

#                 # ---- Optional PCA ----
#                 if pca is not None:
#                     feature_norm = pca.transform([feature_norm])[0]

#                 # ---- Prediction ----
#                 guess = model.predict([feature_norm])[0]
#                 print(f"Predicted class: {guess}")
#                 logger.info(f"Prediction: {guess}")

#                 # =====================================================
#                 url = "http://lelec210x.sipr.ucl.ac.be"
#                 if submit:
#                     response = requests.post(
#                         f"{url}/lelec210x/leaderboard/submit/{key}/{guess}"
#                     )

#                     response_as_dict = json.loads(response.text)

#                     if response.status_code == 200:
#                         logger.info(response_as_dict)
#                     else:
#                         logger.error(response_as_dict)


####### MLP #######


import json
import pickle
import os
from pathlib import Path

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

from .utils import payload_to_melvecs

print("Starting ensemble classification script...")
load_dotenv()
print("Environment variables loaded.")

TARGET_SHAPE = (20, 20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================================================
# 1. DEFINE YOUR PYTORCH ARCHITECTURE HERE
# (Must exactly match the model you trained)
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

# -------------------------------------------------------
# HELPER: Feature Extraction with Forced Resize
# -------------------------------------------------------
def get_fixed_feature(melvec):
    feat2d = melvec

    if feat2d.shape != TARGET_SHAPE:
        feat2d = cv2.resize(
            feat2d,
            (TARGET_SHAPE[1], TARGET_SHAPE[0]),
            interpolation=cv2.INTER_AREA,
        )

    return feat2d.reshape(-1)


@click.command()
@click.option("-i", "--input", "_input", default="-", type=click.File("r"))
@click.option("-m", "--model_dir", default="classification/data/models", type=click.Path(exists=True, file_okay=False, path_type=Path))
@common.click.melvec_length
@common.click.n_melvecs
@click.option("--submit/--no-submit", default=True)
@click.option("-u", "--url", default=None, envvar="LEADERBOARD_URL")
@click.option("-k", "--key", default=None, envvar="LEADERBOARD_KEY")
@common.click.verbosity
def main(
    _input: click.File | None,
    model_dir: Path | None,
    melvec_length: int,
    n_melvecs: int,
    submit: bool,
    url: str | None,
    key: str | None,
) -> None:

    if submit:
        if key is None:
            raise click.UsageError("You must provide a key to submit guesses.")
        url = url or get_url()

    # =====================================================
    # 2. LOAD ENSEMBLE PARAMETERS
    # =====================================================
    print("Loading Ensemble Parameters and Models...")
    params_path = model_dir / "ensemble_production_params.pkl"
    
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
    # Stream payloads
    # ----------------------------
    for payload in _input:
        print(f"Received payload: {payload.strip()}")
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX) :]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
            logger.info(f"Parsed payload into Mel vectors: {melvecs.shape}")

            # =====================================================
            # 4. PRECISION-WEIGHTED ENSEMBLE PIPELINE
            # =====================================================
            
            # --- IMPORTANT: Apply the Global Log transformation! ---
            melvecs = np.log(melvecs + 1e-8)
            
            # Accumulator for final soft votes
            final_ensemble_scores = np.zeros(len(global_classes))
            
            for member in loaded_ensemble:
                # 4.a Apply specific frequency mask
                masked_melvecs = hide_frequency_bands(melvecs, member['num_bands'], member['mask_strategy'])
                
                # 4.b Resize and flatten
                feature_vector = get_fixed_feature(masked_melvecs)
                
                # 4.c Manual Standard Scaler implementation
                feat_scaled = (feature_vector - member['scaler_mean']) / np.sqrt(member['scaler_var'] + 1e-8)
                
                # 4.d Manual PCA projection implementation
                feat_pca = np.dot(feat_scaled - member['pca_mean'], member['pca_components'].T)
                
                # 4.e Tensorize
                input_tensor = torch.FloatTensor(feat_pca).unsqueeze(0).to(device)
                
                # 4.f Get probabilities
                with torch.no_grad():
                    probs = member['model_obj'].predict_proba(input_tensor).cpu().numpy()[0]
                    
                # 4.g Apply precision weights
                weighted_probs = probs * member['class_precisions']
                final_ensemble_scores += weighted_probs

            # =====================================================
            # 5. FINAL DECISION
            # =====================================================
            final_predicted_idx = np.argmax(final_ensemble_scores)
            guess = global_classes[final_predicted_idx]
            
            print(f"Predicted class: {guess}")
            logger.info(f"Prediction: {guess}")

            url_endpoint = "http://lelec210x.sipr.ucl.ac.be"
            if submit:
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