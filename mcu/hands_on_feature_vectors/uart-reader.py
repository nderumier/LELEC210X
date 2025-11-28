"""
uart-reader.py
ELEC PROJECT - 210x
"""

import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports

from classification.utils.plots import plot_specgram

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

dt = np.dtype(np.uint16).newbyteorder("<")


def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=2 * N_MELVECS * MELVEC_LENGTH).decode(
                "ascii"
            )
            print(line)
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    args = argParser.parse_args()
    print("uart-reader launched...\n")

    if args.port is None:
        print(
            "No port specified, here is a list of serial communication port available"
        )
        print("================")
        port = list(list_ports.comports())
        for p in port:
            print(p.device)
        print("================")
        print("Launch this script with [-p PORT_REF] to access the communication port")

    else:
        input_stream = reader(port=args.port)
        msg_counter = 0

        # Load SVM model
        print("Loading audio classification model...")
        with open(r"C:\Users\natha\Documents\Master1\Q1\LELEC2102_-_Project_in_Electrical_Engineering_Integratio_of_wireles_embedded_sensing_systems\LELEC210X\classification\data\models\model_audio_svm.pickle", "rb") as f:
            clf = pickle.load(f)
        scaler=clf["scaler"]
        pca = clf["pca"]
        model = clf["model"]

        dataset_dir = r"C:\Users\natha\Documents\Master1\Q1\LELEC2102_-_Project_in_Electrical_Engineering_Integratio_of_wireles_embedded_sensing_systems\New_dataset"
        os.makedirs(dataset_dir, exist_ok=True)

        X_save = []   # feature vectors
        y_save = []   # predicted classes
        print("Model loaded.\n")

        for melvec in input_stream:
            msg_counter += 1

            print(f"MEL Spectrogram #{msg_counter}")

            feat = melvec.reshape(-1).astype(float)

            # Flatten into 1D vector
            feature_vector = melvec.reshape(-1).astype(float)

            # Normalisation identique
            feature_norm = feature_vector / np.linalg.norm(feature_vector)

            # PCA transform
            feature_pca = pca.transform([feature_norm])

            # Prediction
            prediction = model.predict(feature_pca)[0]
            print(f"Predicted class: {prediction}")

            # ----- SAVE VECTORS -----
            X_save.append(feat)
            y_save.append(prediction)

            # Save periodically
            if msg_counter % 10 == 0:
                np.save(dataset_dir + "feature_vectors.npy", np.array(X_save))
                np.save(dataset_dir + "labels.npy", np.array(y_save))
                print(f"💾 Saved {msg_counter} samples so far...")


            # ----- PLOTTING -----

            plt.figure()
            plot_specgram(
                melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T,
                ax=plt.gca(),
                is_mel=True,
                title=f"MEL Spectrogram #{msg_counter}",
                xlabel="Mel vector",
            )
            plt.draw()
            plt.pause(0.001)
            plt.clf()
