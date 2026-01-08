"""
Script to download radar data from an mmWave radar's DSP board via SCP.

This script connects to the DSP board over SSH and transfers the specified radar
data files to a local directory using `download_files` from `radar_utility.py`.

Usage:
    - Modify `cases` to specify the capture IDs to download.
    - Ensure SCP is configured with the necessary SSH access.

Dependencies:
    - `radar_utility.py` (must contain the `download_files` function).
    - `time` (standard Python library).

Attributes:
    cases (list): List of radar capture IDs to download.
    remote_files (list): List of radar data files to transfer.

"""

import time

from radar_utility import download_files

# List of capture case IDs to download
# cases = [221]
cases = range(250, 290)  # Example: Download a range of captures
dataset_dir = 'wall_dataset'

# List of radar data files to transfer
remote_files = [
    "master_0000_idx.bin",
    "master_0000_data.bin",
    "slave1_0000_data.bin",
    "slave2_0000_data.bin",
    "slave3_0000_data.bin"
]

for capture_id in cases:
    """
    Downloads radar data for a specific capture case.

    Steps:
    
    1. Constructs the remote and local file paths based on `capture_id`.
    2. Calls `download_files()` to transfer data via SCP.
    3. Prints timing information for performance tracking.

    Parameters:
        capture_id (int): Unique identifier for the radar data capture.

    Example:
        >>> capture_id = 400
        >>> # This will download the data files for the given capture ID.

    Raises:
        RuntimeError: If the SCP transfer fails.
    """

    start_time = time.time()
    print(f"Downloading data for capture {capture_id}...")

    # Define remote DSP board directory and local save directory
    remote_dir = f'root@192.168.33.180:/mnt/ssd/{dataset_dir}-capture_{capture_id:05d}-cascaded/'
    local_dir = f'/home/ttoha12/{dataset_dir}/{dataset_dir}-capture_{capture_id:05d}-cascaded/'

    # Initiate the file transfer
    download_files(remote_dir, local_dir, remote_files, start_time)

    # Print elapsed time for debugging and tracking
    print(f"Capture {capture_id} downloaded in {time.time() - start_time:.2f} seconds.")
