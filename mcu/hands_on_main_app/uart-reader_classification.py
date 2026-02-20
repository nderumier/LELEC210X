"""
uart-reader.py
ELEC PROJECT - 210x
put ground truth.csv in same folder as uart-reader.py
run command : python uart-reader.py  
"""

import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import serial
import pandas as pd
from serial.tools import list_ports
from sklearn.metrics import accuracy_score, confusion_matrix

import cv2
import numpy as np



# Ensure you have the plot utils in your path, or comment this out if testing without it
# from classification.utils.plots import plot_specgram

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
    argParser.add_argument("-g", "--ground-truth", help="Path to CSV file containing real classification labels")
    argParser.add_argument("-o", "--output-csv", help="Path to create the Prediction vs Real CSV", default="results_comparison.csv")
    
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

        # --- LOAD GROUND TRUTH ---
        ground_truth_labels = []
        if args.ground_truth:
            try:
                print(f"Loading ground truth from: {args.ground_truth}")
                
                # UPDATED LOGIC: Read CSV with headers and select "Prefix (Class)" column
                df_gt = pd.read_csv(args.ground_truth)
                
                # Check if the column exists (robustness)
                target_col = "Prefix (Class)"
                if target_col not in df_gt.columns:
                    # Fallback if column names differ slightly or csv has no header
                    print(f"⚠️ Column '{target_col}' not found. Using the 3rd column (index 2) by default.")
                    ground_truth_labels = df_gt.iloc[:, 2].astype(str).tolist()
                else:
                    ground_truth_labels = df_gt[target_col].astype(str).tolist()

                print(f"✅ Loaded {len(ground_truth_labels)} labels. First 3: {ground_truth_labels[:3]}")
                
                # Initialize Output CSV
                with open(args.output_csv, "w") as f:
                    f.write("Prediction,Real\n")
                print(f"Saving comparison results to: {args.output_csv}")
                
            except Exception as e:
                print(f"❌ Error reading ground truth CSV: {e}")
                exit(1)

        y_true_acc = []
        y_pred_acc = []
        # Load SVM model
        print("Loading audio classification model...")
        # NOTE: Verify this path exists on your machine
        model_path = r"C:\Users\natha\Documents\Master1\Q1\LELEC2102_-_Project_in_Electrical_Engineering_Integratio_of_wireles_embedded_sensing_systems\LELEC210X\classification\data\models\model_audio_svm.pickle"
        
        if not os.path.exists(model_path):
             print(f"WARNING: Model not found at {model_path}. Please update path in code.")

        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        
        # Check if clf is a dict (standard for the project) or the model itself
        if isinstance(clf, dict):
            scaler = clf.get("scaler") # Might be None based on your code snippet logic
            pca = clf["pca"]
            model = clf["model"]
        else:
            model = clf
            pca = None

        X_save = []   # feature vectors
        y_save = []   # predicted classes
        print("Model loaded.\n")

        for melvec in input_stream:
            msg_counter += 1

            print(f"\n--- Processing Message #{msg_counter} ---")

            feat = melvec.reshape(-1).astype(float)

            # Flatten into 1D vector
            #feature_vector = melvec.reshape(-1).astype(float)


            TARGET_SHAPE = (20, 20)

            # -------------------------------------------------------
            # HELPER: Feature Extraction with Forced Resize
            # -------------------------------------------------------
            def get_fixed_feature(melvec):
                # Retrieve the spectrogram (2D array)   
                feat2d = melvec
                
                # If the shape is not 20x20 (e.g. it is 20x107), force resize it
                if feat2d.shape != TARGET_SHAPE:
                    # mode='reflect' handles borders smoothly, anti_aliasing prevents artifacts
                    feat2d = cv2.resize(feat2d,(TARGET_SHAPE[1], TARGET_SHAPE[0]),interpolation=cv2.INTER_AREA)
                    
                # Flatten to 1D vector (length 400)
                return feat2d.reshape(-1)

            feature_vector = get_fixed_feature(melvec)

            # Normalisation (Using user's logic)
            norm_val = np.linalg.norm(feature_vector)
            if norm_val == 0: 
                norm_val = 1e-9 # Avoid div by zero
            feature_norm = feature_vector / norm_val


            
            # PCA transform
            # if pca:
            #     feature_pca = pca.transform([feature_norm])
            # else:
            #     feature_pca = [feature_norm]

            # Prediction
            prediction = model.predict(feature_norm)[0]
            print(f"Predicted class: {prediction}")

            # --- COMPARISON LOGIC ---
            if args.ground_truth:
                # Check if we still have labels in the ground truth list
                if msg_counter <= len(ground_truth_labels):
                    real_label = ground_truth_labels[msg_counter - 1]
                    print(f"Real class:      {real_label}")
                    
                    # Accumulate for metrics
                    y_pred_acc.append(prediction)
                    y_true_acc.append(real_label)
                    
                    # Append to comparison CSV
                    with open(args.output_csv, "a") as f:
                        f.write(f"{prediction},{real_label}\n")
                    
                    # Compute intermediate accuracy
                    curr_acc = accuracy_score(y_true_acc, y_pred_acc)
                    print(f"Running Accuracy: {curr_acc:.2%}")

                    # If we just processed the LAST label in the ground truth
                    if msg_counter == len(ground_truth_labels):
                        print("\n" + "="*40)
                        print("FINAL RESULTS")
                        print("="*40)
                        
                        final_acc = accuracy_score(y_true_acc, y_pred_acc)
                        cm = confusion_matrix(y_true_acc, y_pred_acc)
                        
                        print(f"Final Accuracy: {final_acc:.2%}")
                        print("\nConfusion Matrix:")
                        print(cm)
                        
                        # Get unique labels to make matrix readable
                        labels = sorted(list(set(y_true_acc + y_pred_acc)))
                        print(f"\nLabels order: {labels}")
                        print("="*40 + "\n")
                else:
                    print("Warning: Received more audio packets than labels provided in ground truth CSV.")

            # ----- SAVE VECTORS (Original Logic) -----
            X_save.append(feat)
            y_save.append(prediction)

            # if msg_counter % 10 == 0:
            #     np.save(os.path.join(dataset_dir, "feature_vectors.npy"), np.array(X_save))
            #     np.save(os.path.join(dataset_dir, "labels.npy"), np.array(y_save))
            #     print(f"💾 Saved {msg_counter} samples so far...")


            # ----- PLOTTING -----
            # plt.figure(1)
            # plot_specgram(
            #     melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T,
            #     ax=plt.gca(),
            #     is_mel=True,
            #     title=f"MEL Spectrogram #{msg_counter} (Pred: {prediction})",
            #     xlabel="Mel vector",
            # )
            # plt.draw()
            # plt.pause(0.001)
            # plt.clf()