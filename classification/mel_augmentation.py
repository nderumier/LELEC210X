import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

from src.classification.datasets import Dataset
from src.classification.utils.audio_student import Feature_vector_DS

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
TARGET_SHAPE = (20,20)

# Load dataset
dataset = Dataset()

# Remove unwanted classes
dataset.remove_class("background")
dataset.remove_class("fireworks")
dataset.remove_class("gunshot")

classnames = dataset.list_classes()

# Feature extractor
myds = Feature_vector_DS(dataset, Nft=512, nmel=20, duration=950, step=np.inf)

# -------------------------------------------------------
# Function to resize mel spectrogram
# -------------------------------------------------------
def get_mel(dataset, classname, idx):
    
    mel = dataset[classname, idx]
    
    if mel.shape != TARGET_SHAPE:
        mel = resize(mel, TARGET_SHAPE, mode='reflect', anti_aliasing=True)
    
    return mel

# -------------------------------------------------------
# Choose one example
# -------------------------------------------------------
classname = classnames[0]   # first class
idx = 0                     # first audio sample

print(f"Class used: {classname}")

# -------------------------------------------------------
# Original
# -------------------------------------------------------
myds.mod_data_aug(["original"])
mel_original = get_mel(myds, classname, idx)

# -------------------------------------------------------
# Augmentations
# -------------------------------------------------------
augmentations = ["noise", "echo", "bg_fixed"]

mel_augmented = []

for aug in augmentations:
    
    myds.mod_data_aug([aug])
    
    mel = get_mel(myds, classname, idx)
    
    mel_augmented.append(mel)

# -------------------------------------------------------
# Plot
# -------------------------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(mel_original, aspect='auto', origin='lower')
plt.title("Original")
plt.colorbar()

for i, aug in enumerate(augmentations):
    
    plt.subplot(1,4,i+2)
    plt.imshow(mel_augmented[i], aspect='auto', origin='lower')
    plt.title(aug)
    plt.colorbar()

plt.suptitle(f"Mel Spectrogram Comparison - Class: {classname}")

plt.tight_layout()
plt.show()