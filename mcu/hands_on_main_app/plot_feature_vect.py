import numpy as np
import matplotlib.pyplot as plt

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../classification/src"))
sys.path.append(BASE_DIR)

from classification.datasets import Dataset
from classification.utils.audio_student import AudioUtil, Feature_vector_DS


# 1. Load the file
file_path = "classification\\feature_vector_sud11\\fireworks9_78.npy"
data = np.load(file_path)

filename = file_path.split("\\")[-1]          # récupère "gunshot1_53.npy"
title = filename.split(".")[0]                # enlève ".npy"



# Data augmentation: apply data augmentation fonction to feature vector
data_aug = data.copy()
data_aug = AudioUtil.add_noise_to_mel(data, noise_db=-20) 
data_aug = AudioUtil.echo_to_mel(data)
data_aug = AudioUtil.hide_band_mel_bandwidth(data, band_width=1)
data_aug = AudioUtil.hide_random_bands_mel(data, n_bands=5)


# 2. Plotting
plt.figure(figsize=(6, 5))

# We use .T (transpose) because your MCU sends Time on Axis 0, 
# but spectrograms usually show Frequency on the Y-axis.
#plt.imshow(data.T, origin='lower', aspect='auto', cmap='viridis')
plt.imshow(data_aug.T, origin='lower', aspect='auto', cmap='viridis')

# 3. Formatting
plt.colorbar(label='Intensity')
#plt.title(f"Melspectrogram: {file_path.split('/')[-1]}")
plt.title(f"Melspectrogram: {title}")
plt.xlabel("Time (Vector Index)")
plt.ylabel("Frequency (Mel Band)")

plt.show()