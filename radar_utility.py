import os
import subprocess
import time
from threading import Thread

import numpy as np
import scipy
from scipy import constants
from tqdm import tqdm

np.random.seed(0)
np.set_printoptions(suppress=True)

"""
Radar configuration parameters.

Defines key settings for radar signal processing, including ADC sampling,
chirp structure, receiver channels, and waveform details.
"""
config = {
    'adc_per_chirp': 256,
    'chirp_per_loop': 12,
    'num_of_loops': 128,
    # 'num_of_loops': 64,
    'rx_per_device': 4,
    'num_of_waves': 2
}

"""
Defines the virtual antenna array configuration for a cascaded mmWave radar system.

Based on section 2.6.3 "Virtual Antenna Array" from TI document SWRU553A.
These arrays define the azimuth and elevation positions of the transmit (TX) 
and receive (RX) antennas in a cascaded radar setup.
"""
# See: 2.6.3 Virtual Antenna Array, SWRU553A
TI_CASCADED_TX_POSITION_AZI = np.array([11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0])
TI_CASCADED_TX_POSITION_ELEV = np.array([6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
TI_CASCADED_RX_POSITION_AZI = np.array([11, 12, 13, 14, 50, 51, 52, 53, 46, 47, 48, 49, 0, 1, 2, 3])
TI_CASCADED_RX_POSITION_ELEV = np.zeros(TI_CASCADED_RX_POSITION_AZI.shape, dtype=np.int64)

TI_CASCADED_TX_ID = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
TI_CASCADED_RX_ID = [12, 13, 14, 15, 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7]


def get_frames_count(idx_path):
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
        ("numIdx", np.uint32),  # number of frames
        ("dataFileSize", np.uint64)  # total data size written into file
    ])

    header = np.fromfile(idx_path, dtype=dt, count=1)[0]
    return header[3]


def get_iq_data_per_chip(data_path, frame_id, samples_per_frame, config):
    """
    Extracts in-phase (I) and quadrature (Q) data for a specific radar frame.

    Args:
        data_path (str): Path to the binary data file.
        frame_id (int): Index of the frame to extract.
        samples_per_frame (int): Total number of samples per frame.
        config (dict): Radar configuration dictionary.

    Returns:
        np.ndarray: Extracted IQ data with shape
                    (chirp_per_loop, rx_per_device, num_of_loops, adc_per_chirp, 2).
    """
    data_per_chip = np.fromfile(data_path, dtype=np.int16,
                                count=samples_per_frame, offset=frame_id * samples_per_frame * 2)
    in_phase, quadrature = data_per_chip[::2], data_per_chip[1::2]
    data_per_chip = np.column_stack((in_phase, quadrature))

    data_per_chip = data_per_chip.reshape((config['rx_per_device'], config['adc_per_chirp'],
                                           config['chirp_per_loop'], config['num_of_loops'], 2), order='F')
    # chirp_per_loop/rx_per_device/num_of_loops/adc_per_chirp/IQ
    data_per_chip = data_per_chip.transpose((2, 0, 3, 1, 4))
    return data_per_chip


def run_scp(remote_path, local_path, start_time):
    """
    Transfers a file from the mmWave radar's DSP board to a local machine using SCP.

    Args:
        remote_path (str): Full path to the file on the DSP board.
        local_path (str): Destination path on the local machine.
        start_time (float): Timestamp for tracking transfer duration.
    """
    subprocess.run(['scp', remote_path, local_path])
    print(remote_path, time.time() - start_time)


def download_files(remote_dir, local_dir, files, start_time):
    """
    Downloads multiple radar data files from an mmWave radar's DSP board using SCP in parallel threads.

    Args:
        remote_dir (str): The directory path on the mmWave radar's DSP board containing the data files.
        local_dir (str): The local directory where the files will be saved.
        files (list): A list of file names to be downloaded.
        start_time (float): The timestamp marking the start of the download process.
    """
    os.makedirs(local_dir, exist_ok=True)

    thread = []
    for file in files:
        local_path = f'{local_dir}{file}'
        if os.path.exists(local_path):
            print(f'{local_path} exists')
            continue

        t = Thread(target=run_scp, args=(f'{remote_dir}{file}', local_dir, start_time))
        t.start()
        thread.append(t)

    for t in thread:
        t.join()


def read_frames(data_dir, start_frame, end_frame, skip_frame=0, calibration=True):
    """
    Loads and processes radar frames from stored NumPy files.

    Args:
        data_dir (str): Directory containing the radar frames.
        start_frame (int): Index of the first frame to read.
        end_frame (int): Index of the last frame to read.
        skip_frame (int, optional): Number of frames to skip between reads. Defaults to 0.
        calibration (bool, optional): If True, applies calibration correction. Defaults to True.

    Returns:
        np.ndarray: Complex radar data with shape (nF, nTx*nRx, nC, nS).
    """
    total_frames = end_frame - start_frame + 1

    # nF/nTx/nRx/nC/nS
    frames = np.zeros((total_frames, config['chirp_per_loop'], config['rx_per_device'] * 4,
                       config['num_of_loops'], config['adc_per_chirp']), dtype=np.complex64)
    for frame_id in tqdm(range(start_frame, end_frame + 1, skip_frame + 1)):
        frame_path = f'{data_dir}frames/frame_{frame_id}.npy'
        frame = np.load(frame_path)
        frame = frame[:, :, :, :, 0] + 1j * frame[:, :, :, :, 1]
        frames[frame_id - start_frame] = frame
        # print(frame_id, frame.shape)

    if calibration:
        frames *= get_calibration_constant()
    # nF/nTx/nRx/nC/nS
    frames = frames.reshape((frames.shape[0], -1, frames.shape[3], frames.shape[4]))
    return frames


def get_calibration_constant():
    """
    Computes the calibration constant for range correction.

    Returns:
       np.ndarray: Frequency bias factor for calibration.
    """
    slope = 7.8e13
    rate = 5.6e6
    adc = 256

    f = np.arange(adc) * slope / rate
    f = f.reshape((1, 1, 1, 1, 1, adc))

    range_offset_mm = 100
    delay_offset = range_offset_mm / 1e3 * 2 / constants.c
    delay_offsets = np.full((192, 1), delay_offset)
    freq_bias_factor = np.exp(-1j * 2 * np.pi * delay_offsets * f)
    freq_bias_factor = freq_bias_factor.reshape((1, 12, 16, 1, adc))

    return freq_bias_factor


def get_range_fft(frames, n=None):
    """
    Computes the Range FFT of radar frames.

    Args:
        frames (np.ndarray): Input radar frames.
        n (int, optional): FFT length. Defaults to None.

    Returns:
        np.ndarray: Range FFT output.
    """
    # DC offset compensation
    range_fft = frames - frames.mean(axis=-1, keepdims=True)
    # Range-domain windowing
    range_fft *= np.hanning(range_fft.shape[-1])
    # Range FFT
    range_fft = scipy.fft.fft(range_fft, n=n, axis=-1)
    return range_fft


def get_doppler_fft(range_fft, n=None, dc_offset=None):
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
    # Zero doppler compensation
    if dc_offset is None:
        dop_fft = range_fft.copy()
    elif dc_offset:
        dop_fft = range_fft - range_fft.mean(axis=0)
    else:
        dop_fft = range_fft - range_fft.mean(axis=-2, keepdims=True)

    # Doppler-domain windowing
    dop_fft *= np.hanning(dop_fft.shape[-2]).reshape(-1, 1)
    # Doppler FFT
    dop_fft = scipy.fft.fft(dop_fft, n=n, axis=-2)
    dop_fft = scipy.fft.fftshift(dop_fft, axes=-2)
    return dop_fft


def get_azimuth_fft(elev_azi_data, n=None):
    """
    Computes the Azimuth FFT of radar elevation-azimuth data.

    Args:
        elev_azi_data (np.ndarray): Input data with elevation and azimuth information.
        n (int, optional): FFT length. Defaults to None.

    Returns:
        np.ndarray: Azimuth FFT output.
    """
    azi_fft = elev_azi_data.copy()
    azi_fft *= np.hanning(azi_fft.shape[-3]).reshape(-1, 1, 1)
    azi_fft = scipy.fft.fft(azi_fft, n=n, axis=-3)
    azi_fft = scipy.fft.fftshift(azi_fft, axes=-3)
    return azi_fft


def get_unique_antenna():
    """
    Identifies unique virtual antenna positions for azimuth processing.

    Returns:
        np.ndarray: A mapping of unique elevation and azimuth antenna positions.
    """
    # Stacking TX
    tx_pos_azi = TI_CASCADED_TX_POSITION_AZI[TI_CASCADED_TX_ID]
    tx_pos_elev = TI_CASCADED_TX_POSITION_ELEV[TI_CASCADED_TX_ID]
    tx_pos = np.column_stack((tx_pos_azi, tx_pos_elev))

    # Stacking RX
    rx_pos_azi = TI_CASCADED_RX_POSITION_AZI[TI_CASCADED_RX_ID]
    rx_pos_elev = TI_CASCADED_RX_POSITION_ELEV[TI_CASCADED_RX_ID]
    rx_pos = np.column_stack((rx_pos_azi, rx_pos_elev))

    # Stacking virtual channel
    tx_pos = tx_pos.reshape((config['chirp_per_loop'], 1, 2))
    rx_pos = rx_pos.reshape((1, config['rx_per_device'] * 4, 2))
    ant_pos = tx_pos + rx_pos
    ant_pos = ant_pos.reshape((-1, 2))
    ant_idx = np.arange(ant_pos.shape[0])

    elev_azi_pos = np.full(ant_pos.max(axis=0) + 1, -1).T
    for elev in range(elev_azi_pos.shape[0]):
        azi_idx = np.where(ant_pos[:, 1] == elev)[0]
        unique_pos, unique_idx = np.unique(ant_pos[azi_idx][:, 0], axis=0, return_index=True)
        elev_azi_pos[elev, unique_pos] = ant_idx[azi_idx][unique_idx]

    return elev_azi_pos


def get_unique_azimuth_data(dop_fft):
    """
    Extracts unique azimuth data from Doppler FFT output.

    Args:
        dop_fft (np.ndarray): Doppler FFT data.

    Returns:
        np.ndarray: Processed azimuth data with unique antenna positions.
    """
    elev_azi_pos = get_unique_antenna()
    azi_pos = elev_azi_pos[0]
    azi_idx = np.argwhere(azi_pos != -1)
    azi_data = np.zeros((dop_fft.shape[0], 86, dop_fft.shape[-2], dop_fft.shape[-1]), dtype=np.complex64)
    azi_data[:, azi_idx] = dop_fft[:, azi_pos[azi_idx]]
    return azi_data


def clean_heatmap(fft_in, q):
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
