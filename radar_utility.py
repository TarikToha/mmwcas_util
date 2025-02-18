import os
import subprocess
import time
from threading import Thread

import numpy as np
import scipy
from mmwave import dsp
from scipy import constants
from tqdm import tqdm

np.random.seed(0)
np.set_printoptions(suppress=True)

config = {
    'adc_per_chirp': 256,
    'chirp_per_loop': 12,
    'num_of_loops': 128,
    # 'num_of_loops': 64,
    'rx_per_device': 4,
    'num_of_waves': 2
}

# See: 2.6.3 Virtual Antenna Array, SWRU553A
TI_CASCADED_TX_POSITION_AZI = np.array([11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0])
TI_CASCADED_TX_POSITION_ELEV = np.array([6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
TI_CASCADED_RX_POSITION_AZI = np.array([11, 12, 13, 14, 50, 51, 52, 53, 46, 47, 48, 49, 0, 1, 2, 3])
TI_CASCADED_RX_POSITION_ELEV = np.zeros(TI_CASCADED_RX_POSITION_AZI.shape, dtype=np.int64)

TI_CASCADED_TX_ID = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
TI_CASCADED_RX_ID = [12, 13, 14, 15, 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7]

ground_truth = np.zeros(316, dtype=np.int16)
# One Person: 115, 240-251
ground_truth[115], ground_truth[240:252] = 1, 1
# Two Persons: 41-91, 93-113, 193-216
ground_truth[41:92], ground_truth[93:114], ground_truth[193:217] = 2, 2, 2
# Three Persons: 120-127, 149-153, 155-185, 186-191, 217-239
(ground_truth[120:128], ground_truth[149:154], ground_truth[155:186],
 ground_truth[186:192], ground_truth[217:240]) = 3, 3, 3, 3, 3
# Four Persons: 129-146, 147-148
ground_truth[129:147], ground_truth[147:149] = 4, 4

# Two Persons: 253-258, 265-267, 312-315
ground_truth[253:259], ground_truth[265:268], ground_truth[312:316] = 2, 2, 2
# Three Persons: 259-264, 304-311
ground_truth[259:265], ground_truth[304:312] = 3, 3
# Four Persons: 268-269, 297-303
ground_truth[268:270], ground_truth[297:304] = 4, 4
# Five Persons: 270, 285-296
ground_truth[270:271], ground_truth[285:297] = 5, 5
# Seven Persons: 271-284
ground_truth[271:285] = 7


def get_frames_count(idx_path):
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
    subprocess.run(['scp', remote_path, local_path])
    print(remote_path, time.time() - start_time)


def download_files(remote_dir, local_dir, files, start_time):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    else:
        print('data file exists')
        return

    thread = []
    for file in files:
        t = Thread(target=run_scp, args=(f'{remote_dir}{file}', local_dir, start_time))
        t.start()
        thread.append(t)

    for t in thread:
        t.join()


def read_frames(data_dir, start_frame, end_frame, skip_frame=0, calibration=True):
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
    # DC offset compensation
    range_fft = frames - frames.mean(axis=-1, keepdims=True)
    # Range-domain windowing
    range_fft *= np.hanning(range_fft.shape[-1])
    # Range FFT
    range_fft = scipy.fft.fft(range_fft, n=n, axis=-1)
    return range_fft


def get_doppler_fft(range_fft, n=None, dc_offset=None):
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
    azi_fft = elev_azi_data.copy()
    azi_fft *= np.hanning(azi_fft.shape[-3]).reshape(-1, 1, 1)
    azi_fft = scipy.fft.fft(azi_fft, n=n, axis=-3)
    azi_fft = scipy.fft.fftshift(azi_fft, axes=-3)
    return azi_fft


def get_elevation_fft(azi_fft, n=None):
    elev_fft = azi_fft.copy()
    elev_fft *= np.hanning(elev_fft.shape[-4]).reshape(-1, 1, 1, 1)
    elev_fft = scipy.fft.fft(elev_fft, n=n, axis=-4)
    elev_fft = scipy.fft.fftshift(elev_fft, axes=-4)
    return elev_fft


def get_frame_fft(frame_cube, n=None):
    # frame_fft = frame_cube.copy()
    frame_fft = frame_cube - frame_cube.mean(axis=-1, keepdims=True)
    frame_fft *= np.hanning(frame_fft.shape[-1])
    frame_fft = scipy.fft.fft(frame_fft, n=n, axis=-1)
    return frame_fft


def get_unique_antenna():
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
    elev_azi_pos = get_unique_antenna()
    azi_pos = elev_azi_pos[0]
    azi_idx = np.argwhere(azi_pos != -1)
    azi_data = np.zeros((dop_fft.shape[0], 86, dop_fft.shape[-2], dop_fft.shape[-1]), dtype=np.complex64)
    azi_data[:, azi_idx] = dop_fft[:, azi_pos[azi_idx]]
    return azi_data


def get_peaks(data):
    x_det = np.apply_along_axis(func1d=dsp.ca, axis=0, arr=data.T,
                                l_bound=5, guard_len=4, noise_len=16)
    y_thresh, y_noise = np.apply_along_axis(func1d=dsp.ca_, axis=0, arr=data,
                                            l_bound=5, guard_len=4, noise_len=16)
    y_det = data > y_thresh
    mask = y_det & x_det.T
    peaks = np.argwhere(mask)
    vals = data[peaks[:, 0], peaks[:, 1]]
    snr = vals / y_noise[peaks[:, 0], peaks[:, 1]]
    peaks = np.column_stack((peaks, vals, snr))
    return peaks


def bin2polar(range_idx, azi_idx, range_bin_size, max_azi_bins):
    r = range_idx * range_bin_size
    phases = np.linspace(-np.pi, np.pi, max_azi_bins)
    azi_idx = azi_idx.astype(np.int16)
    theta = np.arcsin(phases[azi_idx] / (2 * np.pi * 0.5129))
    return r, theta


def polar2cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def polar2cartesian3d(r, theta, phi):
    x = r * np.sin(-theta) * np.cos(phi)
    y = r * np.cos(-theta) * np.cos(phi)
    z = r * np.sin(-phi)
    return x, y, z


def remove_ghost_points(pts, range_bin_size, max_azi_bins, min_range_dist, max_azi_dist):
    pts = pts[pts[:, 2] > 1]
    azi_idx, range_idx = pts[:, 0], pts[:, 1]
    r, theta = bin2polar(range_idx, azi_idx, range_bin_size, max_azi_bins)
    xs, ys = polar2cartesian(r, theta)
    pts, ys = pts[xs > min_range_dist], ys[xs > min_range_dist]
    pts = pts[abs(ys) < max_azi_dist]
    return pts


def get_vital_sources(ic_data, freq_list, freq_type, max_bf, min_bs, debug):
    # freq_sources = ic_data.copy()
    freq_sources = ic_data * np.hanning(ic_data.shape[-1])
    freq_sources = scipy.fft.fft(freq_sources, axis=-1)
    freq_sources = np.abs(freq_sources) ** 2

    freq_sources = freq_sources[:, :freq_sources.shape[-1] // 2]
    # print(freq_sources.shape)

    freq_weight = freq_sources.max(axis=-1) / freq_sources.sum(axis=-1)
    if debug:
        print(freq_weight)

    # freq_weight = np.zeros(freq_sources.shape[0])
    # for idx, ic in enumerate(freq_sources):
    #     freq_peaks, _ = scipy.signal.find_peaks(ic)
    #     if freq_peaks.size == 0:
    #         continue
    #
    #     freq_peaks = ic[freq_peaks]
    #     freq_weight[idx] = freq_peaks.max() / freq_peaks.sum()
    #
    # print(freq_weight)

    if freq_type == 'peak':
        freq_peaks = np.argmax(freq_sources, axis=-1)
        freq_peaks = freq_list[freq_peaks]
    else:
        freq_peaks = np.sum(freq_list * freq_sources, axis=-1) / np.sum(freq_sources, axis=-1)

    if debug:
        print(freq_peaks)

    vital_sources = (0.1 <= freq_peaks) & (freq_peaks <= max_bf) & (freq_weight >= min_bs)

    if debug:
        print(freq_type, max_bf, min_bs)

    vital_peaks, vital_weight = freq_peaks[vital_sources], freq_weight[vital_sources]
    return freq_sources, vital_sources, vital_weight


def clean_heatmap(fft_in, q):
    data = fft_in.copy()
    vmin = np.percentile(data, q)
    data[data < vmin] = vmin
    return data


def get_steering_vec(ang_est_range, ang_est_resolution, num_ant):
    num_vec = 2 * ang_est_range / ang_est_resolution
    num_vec = int(round(num_vec))
    steering_vectors = np.zeros((num_vec, num_ant), dtype=np.complex64)
    for kk in range(num_vec):
        for jj in range(num_ant):
            theta = -1 * np.pi * jj * np.sin((-ang_est_range + kk * ang_est_resolution) * np.pi / 180)
            real = np.cos(theta)
            imag = np.sin(theta)
            steering_vectors[kk, jj] = real + 1j * imag

    return num_vec, steering_vectors


def cov_matrix(x):
    Rxx = x @ np.conjugate(x.T)
    Rxx = np.divide(Rxx, x.shape[1])
    return Rxx


def get_noise_subspace(covariance):
    _, ev = np.linalg.eigh(covariance)
    ev = ev[:, :-16]
    return ev


def apply_music_algorithm(vx_chirps, steering_vec):
    cov = cov_matrix(vx_chirps)
    noise_subspace = get_noise_subspace(cov)
    v = noise_subspace.T.conj() @ steering_vec.T
    spectrum = np.reciprocal(np.sum(v * v.conj(), axis=0).real)
    return spectrum


def pad_data_2d(data_in, bound, axis):
    data = data_in.copy()
    gap = bound - data.shape[axis]
    if axis == 0:
        data = np.pad(data, ((0, gap), (0, 0)), 'constant')
    else:
        data = np.pad(data, ((0, 0), (0, gap)), 'constant')
    return data


def pad_data_3d(data_in, bound, axis):
    data = data_in.copy()
    gap = bound - data.shape[axis]
    if axis == 0:
        data = np.pad(data, ((0, gap), (0, 0), (0, 0)), 'constant')
    elif axis == 1:
        data = np.pad(data, ((0, 0), (0, gap), (0, 0)), 'constant')
    else:
        data = np.pad(data, ((0, 0), (0, 0), (0, gap)), 'constant')
    return data
