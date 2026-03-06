import os
import time
import soundfile as sf
import sounddevice as sd

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
DATASET_DIR = "classification\\src\\classification\\datasets\\soundfiles"
EXTENSIONS = ('.wav')
FILTER_PREFIXES = ["chainsaw"]  # Only plays files starting with these strings

def play_filtered_files():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Directory not found: {DATASET_DIR}")
        return

    # 1. Collect all valid .wav files from the directory (and subdirectories)
    playable_files = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(EXTENSIONS):
                # Check if the filename starts with any of our allowed prefixes
                if any(f.lower().startswith(p.lower()) for p in FILTER_PREFIXES):
                    playable_files.append(os.path.join(root, f))

    if not playable_files:
        print(f"⚠️  No files found matching prefixes: {FILTER_PREFIXES}")
        return

    playable_files.sort()
    print(f"🔊 Found {len(playable_files)} matching files. Starting playback...")
    time.sleep(1)

    # 2. Loop and Play
    try:
        for i, file_path in enumerate(playable_files, 1):
            filename = os.path.basename(file_path)
            print(f"[{i}/{len(playable_files)}] ▶️  Playing: {filename}")
            
            try:
                data, fs = sf.read(file_path)
                sd.play(data, fs)
                sd.wait() # Ensures the script waits for the audio to finish
            except Exception as e:
                print(f"❌ Error playing {filename}: {e}")

        print("\n✅ All files played.")

    except KeyboardInterrupt:
        print("\n\n🛑 Playback stopped by user.")
        sd.stop()

if __name__ == "__main__":
    play_filtered_files()