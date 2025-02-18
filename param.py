range_bin_size = 0.0421 * 100 / 2.54 / 12  # feet
dop_bin_size = 0.0212 * 1000  # mm/s
frame_time = 0.09
sampling_rate = 0.5
max_range_bins = 140
max_dop_bins = 128
max_azi_bins = 128
min_range_dist = 1
dop_center = max_dop_bins / 2
max_azi_dist = 6

start_frame = 0
end_frame = 59  # including
max_frames = end_frame - start_frame + 1
