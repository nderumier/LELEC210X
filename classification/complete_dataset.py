import os
import time
import soundfile as sf
import sounddevice as sd
import csv
# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
DATASET_DIR = "classification\\src\\classification\\datasets\\soundfiles"        # Path to your dataset root folder
DELAY_BETWEEN_SOUNDS = 0    # Seconds to wait after playing (time to push button)
EXTENSIONS = ('.wav')       # Audio extensions to look for

# Define which files to play. 
# Examples: ["fire"] or ["bird", "chain"] or ["chainsaw_01"]
# Leave empty [] to play ALL files.
FILTER_PREFIXES = [ "chainsaw"] 
CSV_FILENAME = "classification\\played_files_log.csv"
def play_dataset_sequentially():
    # Convert to absolute path for clarity in debug
    abs_dataset_dir = os.path.abspath(DATASET_DIR)

    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Dataset directory not found.")
        print(f"   Looking at: {abs_dataset_dir}")
        return

    print(f"🔍 Dataset Directory: {abs_dataset_dir}")
    if FILTER_PREFIXES:
        print(f"🔎 Filter active: Only playing files starting with {FILTER_PREFIXES}")

    # Check content
    try:
        all_items = os.listdir(DATASET_DIR)
        # Check for directories (Class folders)
        classes = [d for d in all_items if os.path.isdir(os.path.join(DATASET_DIR, d))]
        classes.sort()
        
        # Check for files directly (Flat structure)
        flat_files = [f for f in all_items if f.lower().endswith(EXTENSIONS)]
        
        # Apply filter to flat files
        if FILTER_PREFIXES:
            flat_files = [f for f in flat_files if any(f.lower().startswith(p.lower()) for p in FILTER_PREFIXES)]
            
        flat_files.sort()
        
    except Exception as e:
        print(f"❌ Error reading directory: {e}")
        return

    # Determine mode
    mode = "empty"
    if classes:
        mode = "nested"
        print(f"📂 Found {len(classes)} class folders. Mode: Nested.")
    elif flat_files:
        mode = "flat"
        print(f"📂 Found {len(flat_files)} matching audio files in root. Mode: Flat.")
    else:
        print(f"⚠️  No class folders or matching .wav files found.")
        if FILTER_PREFIXES:
            print("   (Check if your filter matches existing file names)")
        return

    print(f"⏱️  Inter-sound delay set to {DELAY_BETWEEN_SOUNDS} seconds.")
    print("🚀 Starting sequence in 3 seconds...")
    time.sleep(3)

    total_count = 0
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # Write Header
            writer.writerow(["Index", "Filename", "Prefix (Class)", "Full Path"])

            if mode == "nested":
                for class_name in classes:
                    class_path = os.path.join(DATASET_DIR, class_name)
                    files = [f for f in os.listdir(class_path) if f.lower().endswith(EXTENSIONS)]
                    if FILTER_PREFIXES:
                        files = [f for f in files if any(f.lower().startswith(p.lower()) for p in FILTER_PREFIXES)]
                    files.sort()

                    if not files: continue

                    print(f"\n--- 📂 Entering Class: {class_name} ---")
                    
                    for filename in files:
                        total_count += 1
                        full_path = os.path.join(class_path, filename)
                        
                        # 1. PLAY
                        play_audio_file(full_path, filename, total_count)
                        
                        # 2. LOG TO CSV (Prefix = Folder Name)
                        prefix = class_name
                        writer.writerow([total_count, filename, prefix, full_path])
                        csv_file.flush() # Save immediately

            elif mode == "flat":
                print(f"\n--- 📂 Playing files in root directory ---")
                for filename in flat_files:
                    total_count += 1
                    full_path = os.path.join(DATASET_DIR, filename)

                    # 1. PLAY
                    play_audio_file(full_path, filename, total_count)
                    
                    # 2. LOG TO CSV (Prefix = First part of filename before '_')
                    # Example: "chainsaw_01.wav" -> "chainsaw"
                    prefix = filename.split('_')[0] 
                    
                    # NOTE: If you want to MANUALLY type the prefix while listening, 
                    # uncomment the line below and comment out the line above:
                    # prefix = input(f"   ⌨️  Enter prefix for {filename}: ")

                    writer.writerow([total_count, filename, prefix, full_path])
                    csv_file.flush() # Save immediately

            print(f"\n✅ Sequence finished. Played {total_count} files.")
            print(f"📝 Log saved to {CSV_FILENAME}")

def play_audio_file(file_path, filename, count):
    print(f"[{count}] ▶️  Playing: {filename} ... ", end="", flush=True)
    
    try:
        # Load audio
        data, fs = sf.read(file_path)
        
        # Play audio
        sd.play(data, fs)
        sd.wait()  # Wait until the file finishes playing
        print("Done.")
        
    except Exception as e:
        print(f"\n❌ Error playing {filename}: {e}")
        return

    # Pause to allow for acquisition button press
    if DELAY_BETWEEN_SOUNDS > 0:
        print(f"   ⏳ Waiting {DELAY_BETWEEN_SOUNDS}s for acquisition...", end="\r")
        time.sleep(DELAY_BETWEEN_SOUNDS)
        print(" " * 40, end="\r") # Clear line

if __name__ == "__main__":
    try:
        play_dataset_sequentially()
    except KeyboardInterrupt:
        print("\n\n🛑 Sequence stopped by user.")
        sd.stop()