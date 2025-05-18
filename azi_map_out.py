import sys

from matplotlib import pyplot as plt

from param import *
from radar_utility import *

# cases = [256]
# cases = range(125, 140)
cases = range(int(sys.argv[1]), int(sys.argv[2]))
dataset_dir = 'wall_dataset'
base_dir = f'/home/ttoha12/{dataset_dir}/'
save = True

for capture_id in cases:
    start_time = time.time()
    print(f"Processing capture {capture_id}...")

    data_dir = f"{base_dir}{dataset_dir}-capture_{capture_id:05d}-cascaded/"
    resume_path = f"{data_dir}frames/azi_fft_{capture_id}.npy"
    azi_frame_path = f'{data_dir}frames/azi_frame_{capture_id}'

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

    azi_fft = 10 * np.log10((np.abs(azi_fft) ** 2).sum(axis=-2))
    azi_fft = clean_heatmap(azi_fft, 50)
    print("azi_fft.shape =", azi_fft.shape, time.time() - start_time, azi_fft.dtype)

    for frame_id in tqdm(range(azi_fft.shape[0])):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()

        cb = ax.imshow(azi_fft[frame_id], cmap='jet', aspect='auto')

        fig.tight_layout()
        if save:
            plt.axis('off')
            plt.savefig(f'{azi_frame_path}_{frame_id}.jpg',
                        dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            fig.colorbar(cb, label='Reflection Intensity')
            plt.show()
            break

    print(f"Capture {capture_id} processed in {time.time() - start_time:.2f} seconds")
