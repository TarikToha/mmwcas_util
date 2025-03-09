"""
Processes and saves radar frames from locally stored mmWave radar data.

Reads raw IQ data from binary files, reshapes it into structured frames,
and stores each frame as a NumPy file for further analysis.
"""
import os
import time

import numpy as np
from tqdm import tqdm

from radar_utility import get_frames_count, config, get_iq_data_per_chip

cases = range(440, 450)
# cases = [399]
remote_files = ['master_0000_idx.bin', 'master_0000_data.bin', 'slave1_0000_data.bin',
                'slave2_0000_data.bin', 'slave3_0000_data.bin']

for capture_id in cases:
    start_time = time.time()
    print(capture_id, start_time)

    local_dir = f'/home/ttoha12/crowd/dataset/crowd_dataset-capture_{capture_id:05d}-cascaded/'
    frames_dir = f'{local_dir}frames/'

    num_of_frames = get_frames_count(local_dir + remote_files[0])
    print(f'number of frames = {num_of_frames}')

    samples_per_frame = (config['adc_per_chirp'] * config['chirp_per_loop'] * config['num_of_loops'] *
                         config['rx_per_device'] * config['num_of_waves'])
    print(f'number of samples per frame = {samples_per_frame}')

    data_per_chip = np.fromfile(local_dir + remote_files[1], dtype=np.uint16)

    assert data_per_chip.shape[0] == (num_of_frames * samples_per_frame), \
        f'case {capture_id} => size mismatch'

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # num_of_frames/chirp_per_loop/total_rx/num_of_loops/adc_per_chirp/IQ
    frame = np.zeros((config['chirp_per_loop'], config['rx_per_device'] * 4,
                      config['num_of_loops'], config['adc_per_chirp'], 2), dtype=np.int16)

    for frame_id in tqdm(range(num_of_frames)):
        r = 0
        for dp in remote_files[1:]:
            data_per_chip = get_iq_data_per_chip(local_dir + dp, frame_id, samples_per_frame, config)
            # print(data_per_chip.shape)  # chirp_per_loop/rx_per_device/num_of_loops/adc_per_chirp/IQ
            frame[:, r:r + 4] = data_per_chip
            r += 4

        out_file = f'{frames_dir}frame_{frame_id}.npy'
        np.save(out_file, frame)
        frame = np.load(out_file)
        # print(out_file, frame.shape, frame.dtype)

    print(capture_id, time.time() - start_time)
