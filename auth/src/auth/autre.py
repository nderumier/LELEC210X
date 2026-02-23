import os
from collections.abc import Iterator
import json
import pickle
from pathlib import Path

import click
import serial
import zmq
import numpy as np
import cv2
import requests

import common
from common.env import load_dotenv
from common.logging import logger
from leaderboard.utils import get_url

from . import PRINT_PREFIX, packet
from classification.utils import payload_to_melvecs

load_dotenv()

# ----------------------------
# Feature extraction helper
# ----------------------------
TARGET_SHAPE = (20, 20)

def get_fixed_feature(melvec):
    feat2d = melvec
    if feat2d.shape != TARGET_SHAPE:
        feat2d = cv2.resize(
            feat2d,
            (TARGET_SHAPE[1], TARGET_SHAPE[0]),
            interpolation=cv2.INTER_AREA,
        )
    return feat2d.reshape(-1)

# ----------------------------
# Main CLI
# ----------------------------
@click.command()
@click.option(
    "-i",
    "--input",
    "_input",
    default=None,
    type=click.File("r"),
    help="Where to read the input stream. If not specified, read from TCP address. You can pass '-' to read from stdin.",
)
@click.option(
    "-o",
    "--output",
    default="-",
    type=click.File("w"),
    help="Where to read the input stream. Default to '-', a.k.a. stdout.",
)
@click.option(
    "--serial-port",
    default=None,
    envvar="SERIAL_PORT",
    show_envvar=True,
    help="If specified, read the packet from the given serial port. E.g., '/dev/tty0'. This takes precedence of `--input` and `--tcp-address` options.",
)
@click.option(
    "--tcp-address",
    default="tcp://127.0.0.1:10000",
    envvar="TCP_ADDRESS",
    show_default=True,
    show_envvar=True,
    help="TCP address to be used to read the input stream.",
)
@click.option(
    "-k",
    "--auth-key",
    default=16 * "00",
    envvar="AUTH_KEY",
    callback=lambda ctx, param, value: bytes.fromhex(value),
    show_default=True,
    show_envvar=True,
    help="Authentification key (hex string).",
)
@click.option(
    "--authenticate/--no-authenticate",
    default=True,
    is_flag=True,
    help="Enable / disable authentication, useful for skipping authentication step.",
)
@common.click.melvec_length
@common.click.n_melvecs
@common.click.verbosity
def main(
    _input: click.File | None,
    output: click.File,
    serial_port: str | None,
    tcp_address: str,
    auth_key: bytes,
    authenticate: bool,
    melvec_length: int,
    n_melvecs: int,
    submit: bool,
    url: str | None,
    key: str | None,
    model: Path | None,
) -> None:
    """
    Parse packets from the MCU, perform authentication, and classify audio payloads.
    """
    logger.debug(f"Unwrapping packets with auth. key: {auth_key.hex()}")

    how_to_kill = (
        "Use Ctrl-C (or Ctrl-D) to terminate.\nIf that does not work, execute `"
        f"kill {os.getpid()}` in a separate terminal."
    )

    # ----------------------------
    # Load classification model
    # ----------------------------
    with open("classification/data/models/model_audio_svm.pickle", "rb") as f:
        clf = pickle.load(f)

    if isinstance(clf, dict):
        scaler = clf.get("scaler")
        pca = clf.get("pca")
        model = clf["model"]
    else:
        model = clf
        scaler = None
        pca = None

    print("Model loaded successfully.")

    # ----------------------------
    # Prepare packet unwrapper
    # ----------------------------
    unwrapper = packet.PacketUnwrapper(
        key=auth_key,
        allowed_senders=[0],
        authenticate=authenticate,
    )

    # ----------------------------
    # Prepare input stream
    # ----------------------------
    if serial_port:
        def reader() -> Iterator[bytes]:
            ser = serial.Serial(port=serial_port, baudrate=115200)
            ser.reset_input_buffer()
            ser.read_until(b"\n")
            logger.debug(f"Reading packets from serial port: {serial_port}")
            logger.info(how_to_kill)
            while True:
                line = ser.read_until(b"\n").decode("ascii").strip()
                pkt = parse_packet(line)
                if pkt is not None:
                    yield pkt

    elif _input:
        def reader() -> Iterator[bytes]:
            logger.debug(f"Reading packets from input: {_input!s}")
            logger.info(how_to_kill)
            for line in _input:
                pkt = parse_packet(line)
                if pkt is not None:
                    yield pkt
    else:
        def reader() -> Iterator[bytes]:
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.connect(tcp_address)
            logger.debug(f"Reading packets from TCP address: {tcp_address}")
            logger.info(how_to_kill)
            while True:
                msg = socket.recv(2 * melvec_length * n_melvecs)
                yield msg

    # ----------------------------
    # Process packets
    # ----------------------------
    print("Starting to read packets...")
    input_stream = reader()
    print("Done! Now unwrapping packets...")

    for msg in input_stream:
        try:
            print(f"Received raw message: {msg.hex()}")
            sender, payload = unwrapper.unwrap_packet(msg)
            print(f"Unwrapped packet from sender {sender}: {payload.hex()}")

            # ===============================
            # CLASSIFICATION PIPELINE
            # ===============================
            melvecs = payload_to_melvecs(payload.hex(), melvec_length, n_melvecs)
            feature_vector = get_fixed_feature(melvecs)

            # Normalisation
            norm_val = np.linalg.norm(feature_vector)
            if norm_val == 0:
                norm_val = 1e-9
            feature_norm = feature_vector / norm_val

            # Optional scaler
            if scaler is not None:
                feature_norm = scaler.transform([feature_norm])[0]

            # Optional PCA
            if pca is not None:
                feature_norm = pca.transform([feature_norm])[0]

            # Prediction
            guess = model.predict([feature_norm])[0]
            print(f"Predicted class: {guess}")
            logger.info(f"Prediction: {guess}")

            # Write unwrapped packet
            #output.write(PRINT_PREFIX + payload.hex() + "\n")
            #output.flush()
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
        except packet.InvalidPacket as e:
            print(f"Received invalid packet: {e.args[0]}")
            logger.error(f"Invalid packet error: {e.args[0]}")

    print("Finished processing packets.")

# ----------------------------
# Helper function
# ----------------------------
def parse_packet(line: str) -> bytes | None:
    """Parse a line into a packet."""
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX):])
    else:
        return None
