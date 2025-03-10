"""
Processes and saves radar frames from locally stored mmWave radar data.

This script reads raw IQ data from binary files, reshapes it into structured frames,
and saves each frame as a NumPy file for further analysis.

Usage:
    - Ensure the `radar_utility.py` module is available in the same directory.
    - Modify the `cases` list to specify which capture IDs to process.
    - Run the script in a Python environment.

Dependencies:
    - numpy
    - tqdm
    - radar_utility.py (contains helper functions)

Attributes:
    cases (range): Specifies the range of radar capture IDs to process.
    remote_files (list): List of binary files required for processing.
"""

import os
import time

import numpy as np
from tqdm import tqdm

from radar_utility import get_frames_count, config, get_iq_data_per_chip

# Define capture IDs to process
cases = range(440, 450)
# cases = [399]

# List of required binary files
remote_files = [
    'master_0000_idx.bin', 'master_0000_data.bin',
    'slave1_0000_data.bin', 'slave2_0000_data.bin',
    'slave3_0000_data.bin'
]

for capture_id in cases:
    """
    Processes radar frames for a given capture case.

    Steps:
        1. Reads the number of frames from the index file.
        2. Computes the expected number of samples per frame.
        3. Loads binary IQ data and reshapes it into structured frames.
        4. Saves each frame as a `.npy` file.

    Raises:
        AssertionError: If the number of loaded samples does not match expectations.
    """
    start_time = time.time()
    print(f"Processing capture ID: {capture_id}, Start time: {start_time}")

    # Define local directories for storage
    local_dir = f'/home/ttoha12/crowd/dataset/crowd_dataset-capture_{capture_id:05d}-cascaded/'
    frames_dir = f'{local_dir}frames/'

    # Get total number of frames from the index file
    num_of_frames = get_frames_count(local_dir + remote_files[0])
    print(f"Number of frames = {num_of_frames}")

    # Compute the expected number of samples per frame
    samples_per_frame = (
            config['adc_per_chirp'] * config['chirp_per_loop'] *
            config['num_of_loops'] * config['rx_per_device'] *
            config['num_of_waves']
    )
    print(f"Number of samples per frame = {samples_per_frame}")

    # Load binary IQ data
    data_per_chip = np.fromfile(local_dir + remote_files[1], dtype=np.uint16)

    # Validate the data size
    assert data_per_chip.shape[0] == (num_of_frames * samples_per_frame), (
        f"Capture {capture_id} => Size mismatch"
    )

    # Create the frames directory if it does not exist
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Initialize an empty frame buffer
    frame = np.zeros(
        (config['chirp_per_loop'], config['rx_per_device'] * 4,
         config['num_of_loops'], config['adc_per_chirp'], 2), dtype=np.int16
    )

    # Process each frame and store the reshaped data
    for frame_id in tqdm(range(num_of_frames), desc="Processing frames"):
        r = 0
        for dp in remote_files[1:]:
            data_per_chip = get_iq_data_per_chip(
                local_dir + dp, frame_id, samples_per_frame, config
            )
            frame[:, r:r + 4] = data_per_chip
            r += 4

        # Save the processed frame as a NumPy file
        out_file = f'{frames_dir}frame_{frame_id}.npy'
        np.save(out_file, frame)

    print(f"Capture {capture_id} processed in {time.time() - start_time:.2f} seconds.")
