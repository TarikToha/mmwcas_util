"""
Processes and visualizes radar data from locally stored mmWave radar files.

This script reads, transforms, and saves radar frames by computing Range FFT, Doppler FFT, and Azimuth FFT

If processed data already exists, it loads and visualizes it.

Usage:
    - Ensure the `radar_utility.py` module is available.
    - Modify `cases` to specify the capture IDs to process.
    - Run the script in a Python environment.

Dependencies:
    - numpy
    - matplotlib
    - tqdm
    - radar_utility.py (contains helper functions)

Attributes:
    cases (range): Specifies the range of radar capture IDs to process.
    base_dir (str): Directory where radar data is stored.
    show (bool): If True, displays the processed azimuth FFT visualization.

"""

from matplotlib import pyplot as plt

from param import *
from radar_utility import *

# Define capture IDs to process
cases = range(440, 450)
# cases = [398]

# Base directory for dataset storage
base_dir = '/home/ttoha12/crowd/dataset/'
show = False  # Set to True to visualize azimuth FFT output

for capture_id in cases:
    """
    Processes and visualizes radar frames for a given capture case.

    Steps:
        1. Reads stored frames if available, otherwise processes from raw data.
        2. Computes:
            - Range FFT (frequency-domain transformation of raw radar data)
            - Doppler FFT (velocity analysis)
            - Azimuth FFT (spatial representation)
        3. Saves the processed data for future use.
        4. Optionally visualizes the results.

    Parameters:
        capture_id (int): Unique identifier for the radar data capture.

    Example:
        >>> capture_id = 440
        >>> # This will process and save the azimuth FFT for the given capture ID.

    Raises:
        FileNotFoundError: If required radar files are missing.
    """

    start_time = time.time()
    print(f"Processing capture {capture_id}...")

    # Define local file paths
    data_dir = f"{base_dir}crowd_dataset-capture_{capture_id:05d}-cascaded/"
    resume_path = f"{data_dir}frames/azi_fft_{capture_id}.npy"

    if not os.path.exists(resume_path):
        # Load raw frames from binary files
        frames = read_frames(data_dir, start_frame, end_frame, calibration=True)
        print("frames.shape =", frames.shape, time.time() - start_time, frames.dtype, frames.nbytes / 2 ** 20)

        # Compute Range FFT
        range_fft = get_range_fft(frames)[:, :, :, :max_range_bins]
        print("range_fft.shape =", range_fft.shape, time.time() - start_time, range_fft.dtype,
              range_fft.nbytes / 2 ** 20)
        del frames

        # Compute Doppler FFT
        dop_fft = get_doppler_fft(range_fft, n=max_dop_bins, dc_offset=False)
        print("dop_fft.shape =", dop_fft.shape, time.time() - start_time, dop_fft.dtype, dop_fft.nbytes / 2 ** 20)
        del range_fft

        # Compute Azimuth FFT
        azi_data = get_unique_azimuth_data(dop_fft)
        azi_fft = get_azimuth_fft(azi_data, n=max_azi_bins)
        print("azi_fft.shape =", azi_fft.shape, time.time() - start_time, azi_fft.dtype, azi_fft.nbytes / 2 ** 20)
        del dop_fft
        del azi_data

        # Save processed data
        np.save(resume_path, azi_fft)

    else:
        print(f"{resume_path} loading ...")
        azi_fft = np.load(resume_path)
        print("azi_fft.shape =", azi_fft.shape, time.time() - start_time, azi_fft.dtype, azi_fft.nbytes / 2 ** 20)

    # Post-process and clean the azimuth FFT
    azi_fft = 10 * np.log10((np.abs(azi_fft) ** 2).sum(axis=-2))
    azi_fft = clean_heatmap(azi_fft, 50)
    print("azi_fft.shape =", azi_fft.shape, time.time() - start_time, azi_fft.dtype)

    # Visualization (if enabled)
    if show:
        plt.title(f"Case-{capture_id}")
        plt.imshow(azi_fft[0])
        plt.show()

    print(f"Capture {capture_id} processed in {time.time() - start_time:.2f} seconds.\n")
