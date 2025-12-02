import os
from collections.abc import Iterator

import click
import serial
import zmq

import common
from common.env import load_dotenv
from common.logging import logger

from . import PRINT_PREFIX, packet

load_dotenv()


def parse_packet(line: str) -> bytes:
    """Parse a line into a packet."""
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        return None


def hex_to_bytes(ctx: click.Context, param: click.Parameter, value: str) -> bytes:
    """Convert a hex string into bytes."""
    return bytes.fromhex(value)


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
    callback=hex_to_bytes,
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
) -> None:
    """
    Parse packets from the MCU and perform authentication.
    """
    logger.debug(f"Unwrapping packets with auth. key: {auth_key.hex()}")

    how_to_kill = (
        "Use Ctrl-C (or Ctrl-D) to terminate.\nIf that does not work, execute `"
        f"kill {os.getpid()}` in a separate terminal."
    )

    unwrapper = packet.PacketUnwrapper(
        key=auth_key,
        allowed_senders=[
            0,
        ],
        authenticate=authenticate,
    )

    if serial_port:  # Read from serial port

        def reader() -> Iterator[str]:
            ser = serial.Serial(port=serial_port, baudrate=115200)
            ser.reset_input_buffer()
            ser.read_until(b"\n")

            logger.debug(f"Reading packets from serial port: {serial_port}")
            logger.info(how_to_kill)

            while True:
                line = ser.read_until(b"\n").decode("ascii").strip()
                packet = parse_packet(line)
                if packet is not None:
                    yield packet

    elif _input:  # Read from file-like

        def reader() -> Iterator[str]:
            logger.debug(f"Reading packets from input: {_input!s}")
            logger.info(how_to_kill)

            for line in _input:
                packet = parse_packet(line)
                if packet is not None:
                    yield packet

    else:  # Read from zmq GNU Radio interface

        def reader() -> Iterator[str]:
            context = zmq.Context()
            socket = context.socket(zmq.SUB)

            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.setsockopt(zmq.CONFLATE, 1)  # last msg only.

            socket.connect(tcp_address)

            logger.debug(f"Reading packets from TCP address: {tcp_address}")
            logger.info(how_to_kill)

            while True:
                msg = socket.recv(2 * melvec_length * n_melvecs)
                yield msg

    input_stream = reader()
    for msg in input_stream:
        try:
            sender, payload = unwrapper.unwrap_packet(msg)
            logger.debug(f"From {sender}, received packet: {payload.hex()}")
            output.write(PRINT_PREFIX + payload.hex() + "\n")
            output.flush()

        except packet.InvalidPacket as e:
            logger.error(
                f"Invalid packet error: {e.args[0]}",
            )

# chat version to classify audio packets using a pre-trained ML model.


# import os
# import time
# import pickle
# import numpy as np
# import csv
# import pandas as pd
# from collections.abc import Iterator
# from datetime import datetime

# import click
# import serial
# import zmq
# from skimage.transform import resize

# # Import your existing common modules
# import common
# from common.env import load_dotenv
# from common.logging import logger
# from . import PRINT_PREFIX, packet

# load_dotenv()

# # --- CONFIGURATION ---
# MODEL_PATH = "data/models/model_audio_svm.pickle"
# PREDICTION_LOG_FILE = "predictions.csv"
# GROUND_TRUTH_FILE = "ground_truth.csv" # To read what is currently playing
# # need to create this file with the player script
# WINDOW_DURATION = 5.0  # Seconds
# TARGET_SHAPE = (20, 20) # Must match training shape

# def load_ml_model(path):
#     """Load the trained scaler, pca, and model."""
#     if not os.path.exists(path):
#         logger.error(f"❌ Model file not found at {path}")
#         return None
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def get_current_ground_truth(sequence_id):
#     """
#     Reads the ground_truth.csv to find what sound corresponds to this sequence_id.
#     This allows live comparison in the terminal.
#     """
#     if not os.path.exists(GROUND_TRUTH_FILE):
#         return "Unknown (No GT file)"
    
#     try:
#         # Read the csv safely
#         df = pd.read_csv(GROUND_TRUTH_FILE)
#         # Find the row with the matching sequence_id
#         row = df[df['sequence_id'] == sequence_id]
#         if not row.empty:
#             return row.iloc[0]['class_name']
#         else:
#             return "Waiting for Player..."
#     except Exception:
#         return "Read Error"

# def parse_packet(line: str) -> bytes:
#     line = line.strip()
#     if line.startswith(PRINT_PREFIX):
#         return bytes.fromhex(line[len(PRINT_PREFIX) :])
#     else:
#         return None

# def hex_to_bytes(ctx: click.Context, param: click.Parameter, value: str) -> bytes:
#     return bytes.fromhex(value)

# @click.command()
# @click.option("-i", "--input", "_input", default=None, type=click.File("r"), help="Input stream")
# @click.option("-o", "--output", default="-", type=click.File("w"), help="Output stream")
# @click.option("--serial-port", default=None, envvar="SERIAL_PORT", show_envvar=True, help="Serial port")
# @click.option("--tcp-address", default="tcp://127.0.0.1:10000", envvar="TCP_ADDRESS", show_default=True, show_envvar=True, help="TCP address")
# @click.option("-k", "--auth-key", default=16 * "00", envvar="AUTH_KEY", callback=hex_to_bytes, show_default=True, show_envvar=True, help="Auth key")
# @click.option("--authenticate/--no-authenticate", default=True, is_flag=True, help="Auth enable")
# @common.click.melvec_length
# @common.click.n_melvecs
# @common.click.verbosity
# def main(_input, output, serial_port, tcp_address, auth_key, authenticate, melvec_length, n_melvecs):
#     """
#     Receive packets -> Accumulate 5s -> Predict -> Compare with GT
#     """
#     # 1. Load ML Model
#     ml_artifacts = load_ml_model(MODEL_PATH)
#     if not ml_artifacts:
#         return
    
#     scaler = ml_artifacts["scaler"]
#     pca = ml_artifacts["pca"]
#     model = ml_artifacts["model"]
#     logger.info(f"✅ Loaded ML model from {MODEL_PATH}")

#     # 2. Setup Prediction Log
#     with open(PREDICTION_LOG_FILE, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["timestamp", "sequence_id", "predicted_class"])
    
#     # 3. Setup Communication
#     logger.debug(f"Unwrapping packets with auth. key: {auth_key.hex()}")
#     unwrapper = packet.PacketUnwrapper(key=auth_key, allowed_senders=[0], authenticate=authenticate)

#     # Reader factory (Standard)
#     if serial_port:
#         def reader() -> Iterator[str]:
#             ser = serial.Serial(port=serial_port, baudrate=115200)
#             ser.reset_input_buffer()
#             ser.read_until(b"\n")
#             logger.info(f"Listening on {serial_port}...")
#             while True:
#                 yield parse_packet(ser.read_until(b"\n").decode("ascii").strip())
#     elif _input:
#         def reader() -> Iterator[str]:
#             for line in _input:
#                 yield parse_packet(line)
#     else:
#         def reader() -> Iterator[str]:
#             context = zmq.Context()
#             socket = context.socket(zmq.SUB)
#             socket.setsockopt(zmq.SUBSCRIBE, b"")
#             socket.connect(tcp_address)
#             logger.info(f"Listening on {tcp_address}...")
#             while True:
#                 yield socket.recv(2 * melvec_length * n_melvecs)

#     input_stream = reader()
    
#     # --- 4. CLASSIFICATION LOOP ---
    
#     feature_buffer = []
#     start_time = time.time()
#     prediction_count = 0 # Corresponds to sequence_id
    
#     print("\n🚀 System Active. Waiting for MCU packets...")

#     for msg in input_stream:
#         if msg is None: continue
        
#         try:
#             sender, payload = unwrapper.unwrap_packet(msg)
            
#             # --- A. Decode Payload ---
#             # IMPORTANT: Ensure dtype matches your MCU code (float32 or uint16?)
#             # If your MCU sends raw Mel values as floats, use float32.
#             vector = np.frombuffer(payload, dtype=np.float32) 
            
#             # Store vector
#             feature_buffer.append(vector)
            
#             current_time = time.time()
            
#             # --- B. Check Time Window (5 seconds) ---
#             if current_time - start_time >= WINDOW_DURATION:
                
#                 # If we have data, make a prediction
#                 if len(feature_buffer) > 0:
#                     prediction_count += 1 # Increment sequence ID
                    
#                     # 1. Stack and Transpose
#                     # Result shape: (Time, Mel) -> Transpose to (Mel, Time) for resizing
#                     spectrogram = np.stack(feature_buffer).T 
                    
#                     # 2. Resize to Training Shape (20, 20)
#                     resized_spec = resize(spectrogram, TARGET_SHAPE, mode='reflect', anti_aliasing=True)
                    
#                     # 3. Flatten (1, 400)
#                     feat_vector = resized_spec.reshape(1, -1)
                    
#                     # 4. Predict
#                     feat_scaled = scaler.transform(feat_vector)
#                     feat_pca = pca.transform(feat_scaled)
#                     pred_class = model.predict(feat_pca)[0]
                    
#                     # 5. Live Comparison
#                     actual_class = get_current_ground_truth(prediction_count)
                    
#                     # 6. Console Output
#                     timestamp_str = datetime.now().strftime('%H:%M:%S')
#                     status_icon = "✅" if pred_class == actual_class else "❌"
                    
#                     print(f"[{timestamp_str}] SEQ #{prediction_count} | Pred: {pred_class.ljust(10)} | Real: {actual_class.ljust(10)} | {status_icon}")
                    
#                     # 7. Log to CSV
#                     with open(PREDICTION_LOG_FILE, 'a', newline='') as f:
#                         writer = csv.writer(f)
#                         writer.writerow([current_time, prediction_count, pred_class])
                    
#                     # Reset
#                     feature_buffer = []
#                     start_time = time.time()
#                 else:
#                     # No data received in window (Silence?) - Reset timer
#                     start_time = current_time

#         except packet.InvalidPacket as e:
#             pass # Ignore packet errors

# if __name__ == "__main__":
#     main()