import numpy as np

audio = np.load(r"C:\Users\tvang\LELEC210X\contrib\src\contrib\data\audio\audio_20251008-181802-452895.npy")

print(audio.shape, audio.dtype)
import sounddevice as sd

# Sampling rate — set it to what you used for recording, e.g. 16000 Hz
fs = 10200
sd.play(audio, fs)
sd.wait()  # Wait until playback finishes
