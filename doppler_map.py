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
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from param import *
from radar_utility import *

matplotlib.use('macosx')

capture_id = 224
# cases = range(150, 160)
dataset_dir = 'wall_dataset'
# base_dir = f'/home/ttoha12/{dataset_dir}/'
base_dir = f'/Users/tarik/odrive/unc/thesis/data/{dataset_dir}/'

start_time = time.time()
print(f"Processing capture {capture_id}...")

# Define local file paths
data_dir = f"{base_dir}{dataset_dir}-capture_{capture_id:05d}-cascaded/"
resume_path = f"{data_dir}frames/azi_fft_{capture_id}.npy"

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

dop_fft = 10 * np.log10((np.abs(dop_fft) ** 2).sum(axis=1))
dop_fft = clean_heatmap(dop_fft, 50)
print("dop_fft.shape =", dop_fft.shape, time.time() - start_time, dop_fft.dtype, dop_fft.nbytes / 2 ** 20)

fig, ax = plt.subplots(figsize=(10, 10))
dop_plot = ax.imshow(dop_fft[start_frame])

fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
frame_slider = Slider(ax=ax_slider, label='Frames ',
                      valmin=start_frame, valmax=end_frame,
                      valinit=start_frame, valstep=1)


def update(fid):
    dop_plot.set_data(dop_fft[fid - start_frame])
    fig.canvas.draw_idle()


frame_slider.on_changed(update)
plt.show()
