import os
import subprocess
import time
from threading import Thread

import numpy as np
import scipy

np.random.seed(0)
np.set_printoptions(suppress=True)

# Radar Configuration Parameters
config = {
    'adc_per_chirp': 256,
    'chirp_per_loop': 12,
    'num_of_loops': 128,
    # 'num_of_loops': 64,
    'rx_per_device': 4,
    'num_of_waves': 2
}

# Virtual Antenna Array Configuration
TI_CASCADED_TX_POSITION_AZI: np.ndarray = np.array([11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0])
TI_CASCADED_TX_POSITION_ELEV: np.ndarray = np.array([6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
TI_CASCADED_RX_POSITION_AZI: np.ndarray = np.array([11, 12, 13, 14, 50, 51, 52, 53, 46, 47, 48, 49, 0, 1, 2, 3])
TI_CASCADED_RX_POSITION_ELEV: np.ndarray = np.zeros(TI_CASCADED_RX_POSITION_AZI.shape, dtype=np.int64)

TI_CASCADED_TX_ID: list = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
TI_CASCADED_RX_ID: list = [12, 13, 14, 15, 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7]


def get_frames_count(idx_path: str) -> int:
    """
    Reads the number of frames from an index file.

    Args:
        idx_path (str): Path to the index binary file.

    Returns:
        int: Number of frames in the dataset.
    """
    dt = np.dtype([
        ("tag", np.uint32),
        ("version", np.uint32),
        ("flags", np.uint32),
        ("numIdx", np.uint32),  # Number of frames
        ("dataFileSize", np.uint64)  # Total data size written into file
    ])

    header = np.fromfile(idx_path, dtype=dt, count=1)[0]
    return header[3]


def get_iq_data_per_chip(data_path: str, frame_id: int, samples_per_frame: int, config: dict) -> np.ndarray:
    """
    Extracts in-phase (I) and quadrature (Q) data for a specific radar frame.

    Args:
        data_path (str): Path to the binary data file.
        frame_id (int): Index of the frame to extract.
        samples_per_frame (int): Total number of samples per frame.
        config (dict): Radar configuration dictionary.

    Returns:
        np.ndarray: Extracted IQ data with shape (chirp_per_loop, rx_per_device, num_of_loops, adc_per_chirp, 2).
    """
    data_per_chip = np.fromfile(data_path, dtype=np.int16,
                                count=samples_per_frame, offset=frame_id * samples_per_frame * 2)
    in_phase, quadrature = data_per_chip[::2], data_per_chip[1::2]
    data_per_chip = np.column_stack((in_phase, quadrature))

    data_per_chip = data_per_chip.reshape((config['rx_per_device'], config['adc_per_chirp'],
                                           config['chirp_per_loop'], config['num_of_loops'], 2), order='F')
    data_per_chip = data_per_chip.transpose((2, 0, 3, 1, 4))
    return data_per_chip


def run_scp(remote_path: str, local_path: str, start_time: float) -> None:
    """
    Transfers a file from the mmWave radar's DSP board to a local machine using SCP.

    Args:
        remote_path (str): Full path to the file on the DSP board.
        local_path (str): Destination path on the local machine.
        start_time (float): Timestamp for tracking transfer duration.
    """
    subprocess.run(['scp', remote_path, local_path])
    print(remote_path, time.time() - start_time)


def download_files(remote_dir: str, local_dir: str, files: list, start_time: float) -> None:
    """
    Downloads multiple radar data files from an mmWave radar's DSP board using SCP in parallel threads.

    Args:
        remote_dir (str): The directory path on the mmWave radar's DSP board containing the data files.
        local_dir (str): The local directory where the files will be saved.
        files (list): A list of file names to be downloaded.
        start_time (float): The timestamp marking the start of the download process.
    """
    os.makedirs(local_dir, exist_ok=True)

    threads = []
    for file in files:
        local_path = f'{local_dir}{file}'
        if os.path.exists(local_path):
            print(f'{local_path} exists')
            continue

        t = Thread(target=run_scp, args=(f'{remote_dir}{file}', local_dir, start_time))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def get_range_fft(frames: np.ndarray, n: int = None) -> np.ndarray:
    """
    Computes the Range FFT of radar frames.

    Args:
        frames (np.ndarray): Input radar frames.
        n (int, optional): FFT length. Defaults to None.

    Returns:
        np.ndarray: Range FFT output.
    """
    range_fft = frames - frames.mean(axis=-1, keepdims=True)
    range_fft *= np.hanning(range_fft.shape[-1])
    range_fft = scipy.fft.fft(range_fft, n=n, axis=-1)
    return range_fft


def get_doppler_fft(range_fft: np.ndarray, n: int = None, dc_offset: bool = None) -> np.ndarray:
    """
    Computes the Doppler FFT of range-transformed data.

    Args:
        range_fft (np.ndarray): FFT-transformed range data.
        n (int, optional): FFT length. Defaults to None.
        dc_offset (bool, optional): If True, compensates for DC offset across all frames.
                                    If False, compensates within each Doppler bin. Defaults to None.

    Returns:
        np.ndarray: Doppler FFT output.
    """
    if dc_offset is None:
        dop_fft = range_fft.copy()
    elif dc_offset:
        dop_fft = range_fft - range_fft.mean(axis=0)
    else:
        dop_fft = range_fft - range_fft.mean(axis=-2, keepdims=True)

    dop_fft *= np.hanning(dop_fft.shape[-2]).reshape(-1, 1)
    dop_fft = scipy.fft.fft(dop_fft, n=n, axis=-2)
    dop_fft = scipy.fft.fftshift(dop_fft, axes=-2)
    return dop_fft


def clean_heatmap(fft_in: np.ndarray, q: float) -> np.ndarray:
    """
    Applies percentile-based thresholding to clean a heatmap.

    Args:
        fft_in (np.ndarray): Input heatmap data (e.g., FFT output).
        q (float): Percentile threshold (0-100). Values below this percentile are set to the threshold.

    Returns:
        np.ndarray: Processed heatmap with low values clipped.
    """
    data = fft_in.copy()
    vmin = np.percentile(data, q)
    data[data < vmin] = vmin
    return data
