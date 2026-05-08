import numpy as np
import matplotlib.pyplot as plt

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../classification/src"))
sys.path.append(BASE_DIR)

# 1. Load the file
file_path = "classification\\feature_vector_sud11\\fire9_10.npy"
data = np.load(file_path)

filename = file_path.split("\\")[-1]          # récupère "gunshot1_53.npy"
title = filename.split(".")[0]                # enlève ".npy"



# 2. Plotting
plt.figure(figsize=(6, 5))

# We use .T (transpose) because your MCU sends Time on Axis 0, 
# but spectrograms usually show Frequency on the Y-axis.
#plt.imshow(data.T, origin='lower', aspect='auto', cmap='viridis')
plt.imshow(data.T, origin='lower', aspect='auto', cmap='viridis')

# 3. Formatting
plt.colorbar(label='Intensity')
#plt.title(f"Melspectrogram: {file_path.split('/')[-1]}")
plt.title(f"Melspectrogram: {title}")
plt.xlabel("Time (Vector Index)")
plt.ylabel("Frequency (Mel Band)")

plt.show()