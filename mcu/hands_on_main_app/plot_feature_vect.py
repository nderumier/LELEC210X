import numpy as np
import matplotlib.pyplot as plt

# 1. Load the file
file_path = "feature_vector\\gunshot1_10.npy"
data = np.load(file_path)

# 2. Plotting
plt.figure(figsize=(6, 5))

# We use .T (transpose) because your MCU sends Time on Axis 0, 
# but spectrograms usually show Frequency on the Y-axis.
plt.imshow(data.T, origin='lower', aspect='auto', cmap='viridis')

# 3. Formatting
plt.colorbar(label='Intensity')
plt.title(f"Melspectrogram: {file_path.split('/')[-1]}")
plt.xlabel("Time (Vector Index)")
plt.ylabel("Frequency (Mel Band)")

plt.show()