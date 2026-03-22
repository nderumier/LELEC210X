"""
uart-reader.py
ELEC PROJECT - 210x - HEX Melspectrogram Saver
run: python uart-reader_feature_vect_acquisition.py -p COM3 -n test -d feature_vector
"""

import argparse
import os
import numpy as np
import serial
from serial.tools import list_ports

PRINT_PREFIX = "DF:HEX:"
MELVEC_LENGTH = 20  # Number of bins in one vector
N_MELVECS = 20      # Number of vectors per full spectrogram

# The MCU sends 16-bit integers in Little Endian or Big Endian. 
# Usually, %04x in C sends Big Endian (MSB first).
dt = np.dtype(np.uint16).newbyteorder(">") 

def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        try:
            # Extract the hex string after the prefix
            hex_string = line[len(PRINT_PREFIX):].strip()
            # Convert hex string to raw bytes
            raw_bytes = bytes.fromhex(hex_string)
            # Convert bytes to numpy array of uint16
            return np.frombuffer(raw_bytes, dtype=dt)
        except Exception as e:
            print(f"Parsing Error: {e}")
            return None
    else:
        # If it's a regular log, print it
        if line: print(f"MCU Log: {line}")
        return None

def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200, timeout=1)
    while True:
        line = ser.readline().decode("ascii", errors="ignore")
        if not line:
            continue
        
        buffer_array = parse_buffer(line)
        if buffer_array is not None:
            yield buffer_array

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Serial port (e.g. COM3)")
    argParser.add_argument("-n", "--name", help="Base filename", default="test")
    argParser.add_argument("-d", "--dir", help="Save directory", default="feature_vector")
    
    args = argParser.parse_args()

    if args.port is None:
        print("Please specify a port with -p.")
        for p in list_ports.comports(): print(p.device)
    else:
        os.makedirs(args.dir, exist_ok=True)
        input_stream = reader(port=args.port)
        
        spectrogram_bucket = []
        spec_counter = 0
        
        print(f"Listening on {args.port}... Waiting for {N_MELVECS} vectors per file.")

        for melvec in input_stream:
            # melvec is a 1D array of 20 elements
            if len(melvec) != MELVEC_LENGTH:
                # Sometimes UART drops bytes; we ignore incomplete vectors
                continue
                
            spectrogram_bucket.append(melvec.astype(float))
            
            # Show progress on one line
            print(f"Vector {len(spectrogram_bucket)}/{N_MELVECS} received", end="\r")

            if len(spectrogram_bucket) == N_MELVECS:
                spec_counter += 1
                
                # Create the (20, 20) matrix
                spectrogram_matrix = np.array(spectrogram_bucket)
                
                filename = f"{args.name}_{spec_counter:02d}.npy"
                filepath = os.path.join(args.dir, filename)
                np.save(filepath, spectrogram_matrix)
                
                print(f"\n✅ Saved Spectrogram #{spec_counter}: {filename}")
                
                # Reset for next acquisition
                spectrogram_bucket = []