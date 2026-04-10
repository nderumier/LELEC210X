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
from pathlib import Path

import click
import requests
import numpy as np
import cv2

# PyTorch Imports
import torch
import torch.nn as nn

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
from leaderboard.utils import get_url

from .utils import payload_to_melvecs

print("Starting classification script...")
load_dotenv()
print("Environment variables loaded.")

TARGET_SHAPE = (20, 20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================================================
# 1. DEFINE YOUR PYTORCH ARCHITECTURE HERE
# (Must exactly match the model you trained)
# =======================================================
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
@click.option("-m", "--model", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@common.click.melvec_length
@common.click.n_melvecs
@click.option("--submit/--no-submit", default=True)
@click.option("-u", "--url", default=None, envvar="LEADERBOARD_URL")
@click.option("-k", "--key", default=None, envvar="LEADERBOARD_KEY")
@common.click.verbosity
def main(
    _input: click.File | None,
    model_path: Path | None,
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
    # 2. LOAD SCALER AND LABEL ENCODER 
    # (You must save these in your training script!)
    # =====================================================
    print("Loading Scaler and Label Encoder...")
    with open("classification/data/models/scaler_and_encoder.pickle", "rb") as f:
        preprocessing_data = pickle.load(f)
        scaler = preprocessing_data["scaler"]
        label_encoder = preprocessing_data["label_encoder"]
        pca = preprocessing_data["pca"]  # ⬅️ ADD THIS LINE

    # =====================================================
    # 3. INITIALIZE AND LOAD PYTORCH MODEL
    # =====================================================
    # Insert the exact hyperparameters of your final production model here
    model = AudioMLP(
        input_size=pca.n_components_,
        num_classes=len(label_encoder.classes_), 
        n_layers=4, 
        hidden_units_list=[192, 256, 128, 256], 
        dropout_rate=0.5643
    ).to(device)

    print("Loading PyTorch model weights...")
    # Load weights (map_location ensures it loads correctly even if trained on GPU but deployed on CPU)
    model.load_state_dict(torch.load("classification/data/models/model_mlp_PRODUCTION.pth", map_location=device))
    model.eval() # Set model to evaluation mode (turns off dropout)
    print("Model ready.")

    # ----------------------------
    # Stream payloads
    # ----------------------------
    for payload in _input:
        print(f"Received payload: {payload.strip()}")
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX) :]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
            print(f"Parsed payload into Mel vectors with shape: {melvecs.shape}")
            logger.info(f"Parsed payload into Mel vectors: {melvecs.shape}")

            # =====================================================
            # 4. YOUR EXACT CLASSIFICATION PIPELINE
            # =====================================================
            
            # --- IMPORTANT: Apply the Log transformation you used in training! ---
            melvecs = np.log(melvecs + 1e-8)
            
            feature_vector = get_fixed_feature(melvecs)

            # ---- Scale feature vector ----
            feature_norm = scaler.transform([feature_vector])[0]
            # ---- Apply PCA ----
            feature_norm = pca.transform([feature_norm])[0]
            # ---- Convert to PyTorch Tensor ----
            # We unsqueeze(0) to add a batch dimension of 1 -> shape becomes [1, 400]
            input_tensor = torch.FloatTensor(feature_norm).unsqueeze(0).to(device)

            # ---- Prediction ----
            with torch.no_grad():
                outputs = model(input_tensor)
                # Get the index of the highest probability
                _, predicted_idx = torch.max(outputs, 1) 
            
            # Convert PyTorch index back to actual class name string (e.g. "chainsaw")
            guess = label_encoder.inverse_transform([predicted_idx.cpu().numpy()[0]])[0]
            
            print(f"Predicted class: {guess}")
            logger.info(f"Prediction: {guess}")

            # =====================================================
            url = "http://lelec210x.sipr.ucl.ac.be"
            if submit:
                response = requests.post(
                    f"{url}/lelec210x/leaderboard/submit/{key}/{guess}"
                )

                response_as_dict = json.loads(response.text)

                if response.status_code == 200:
                    logger.info(response_as_dict)
                else:
                    logger.error(response_as_dict)

if __name__ == "__main__":
    main()

