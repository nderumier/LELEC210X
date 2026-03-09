print("Importing libraries...")


import json
import pickle
from pathlib import Path

import click
import requests
import numpy as np
import cv2

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
@click.option(
    "-i",
    "--input",
    "_input",
    default="-",
    type=click.File("r"),
)
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@common.click.melvec_length
@common.click.n_melvecs
@click.option("--submit/--no-submit", default=True)
@click.option(
    "-u",
    "--url",
    default=None,
    envvar="LEADERBOARD_URL",
)
@click.option(
    "-k",
    "--key",
    default=None,
    envvar="LEADERBOARD_KEY",
)
@common.click.verbosity
def main(
    _input: click.File | None,
    model: Path | None,
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

    # ----------------------------
    # Load model
    # ----------------------------
    with open("classification/data/models/model_audio_svm.pickle", "rb") as f:
        clf = pickle.load(f)
    print(f"Loaded model: {clf}")
    # Same logic as your UART script
    if isinstance(clf, dict):
        scaler = clf.get("scaler")
        pca = clf.get("pca")
        model = clf["model"]
    else:
        model = clf
        scaler = None
        pca = None

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

            if model:

                # =====================================================
                # YOUR EXACT CLASSIFICATION PIPELINE
                # =====================================================

                feature_vector = get_fixed_feature(melvecs)

                # ---- Normalisation ----
                norm_val = np.linalg.norm(feature_vector)
                if norm_val == 0:
                    norm_val = 1e-9
                feature_norm = feature_vector / norm_val

                # ---- Optional scaler ----
                if scaler is not None:
                    feature_norm = scaler.transform([feature_norm])[0]

                # ---- Optional PCA ----
                if pca is not None:
                    feature_norm = pca.transform([feature_norm])[0]

                # ---- Prediction ----
                guess = model.predict([feature_norm])[0]
                print(f"Predicted class: {guess}")
                logger.info(f"Prediction: {guess}")

                # =====================================================
                url = "http://localhost:5000"
                if submit:
                    response = requests.post(
                        f"{url}/lelec210x/leaderboard/submit/{key}/{guess}"
                    )

                    response_as_dict = json.loads(response.text)

                    if response.status_code == 200:
                        logger.info(response_as_dict)
                    else:
                        logger.error(response_as_dict)
