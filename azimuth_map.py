"""
Processes and visualizes radar data from locally stored mmWave radar files.

This script reads, transforms, and saves radar frames by computing range, Doppler,
and azimuth FFTs. If processed data already exists, it loads and visualizes it.
"""
from matplotlib import pyplot as plt

from param import *
from radar_utility import *

cases = range(440, 450)
# cases = [398]
# matplotlib.use('Qt5Agg')
base_dir = '/home/ttoha12/crowd/dataset/'
show = False

for capture_id in cases:
    start_time = time.time()

    data_dir = f'{base_dir}crowd_dataset-capture_{capture_id:05d}-cascaded/'
    resume_path = f'{data_dir}frames/azi_fft_{capture_id}.npy'

    if not os.path.exists(resume_path):
        # nF/nVx/nC/nS
        frames = read_frames(data_dir, start_frame, end_frame, calibration=True)
        print('frames.shape =', frames.shape, time.time() - start_time, frames.dtype, frames.nbytes / 2 ** 20)

        range_fft = get_range_fft(frames)[:, :, :, :max_range_bins]
        print('range_fft.shape =', range_fft.shape, time.time() - start_time, range_fft.dtype,
              range_fft.nbytes / 2 ** 20)
        del frames

        dop_fft = get_doppler_fft(range_fft, n=max_dop_bins, dc_offset=False)
        print('dop_fft.shape =', dop_fft.shape, time.time() - start_time, dop_fft.dtype, dop_fft.nbytes / 2 ** 20)
        del range_fft

        azi_data = get_unique_azimuth_data(dop_fft)
        azi_fft = get_azimuth_fft(azi_data, n=max_azi_bins)
        print('azi_fft.shape =', azi_fft.shape, time.time() - start_time, azi_fft.dtype, azi_fft.nbytes / 2 ** 20)
        del dop_fft
        del azi_data

        np.save(resume_path, azi_fft)

    else:
        print(f'{resume_path} loading ... ')
        azi_fft = np.load(resume_path)
        print('azi_fft.shape =', azi_fft.shape, time.time() - start_time, azi_fft.dtype, azi_fft.nbytes / 2 ** 20)

    azi_fft = 10 * np.log10((np.abs(azi_fft) ** 2).sum(axis=-2))
    azi_fft = clean_heatmap(azi_fft, 50)
    print('azi_fft.shape =', azi_fft.shape, time.time() - start_time, azi_fft.dtype)

    if show:
        plt.title(f'Case-{capture_id}')
        plt.imshow(azi_fft[0])
        plt.show()

    print(capture_id, time.time() - start_time, '\n')
