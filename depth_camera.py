import glob
import json
import os
from datetime import datetime

save = False
base_dir = '/home/ttoha12/crowd/dataset/'
depth_files = glob.glob(f'{base_dir}*.npy')

timestamps = []
for depth_path in depth_files:
    depth_file = depth_path.split(base_dir)[-1]
    timestamp = depth_file.split('_depth.npy')[0]
    timestamps.append(int(timestamp))

timestamps.sort()

start_id = 371
for idx, timestamp in enumerate(timestamps):
    capture_id = start_id + idx

    color_image_in = f'{base_dir}{timestamp}_color.jpg'
    depth_image_in = f'{base_dir}{timestamp}_depth.jpg'
    depth_frame_in = f'{base_dir}{timestamp}_depth.npy'
    capture_log = f'{base_dir}capture_{capture_id:05d}/capture.log'

    out_dir = f'{base_dir}capture_{capture_id:05d}/'
    color_image_out = f'{out_dir}capture_{capture_id:05d}_color.jpg'
    depth_image_out = f'{out_dir}capture_{capture_id:05d}_depth.jpg'
    depth_frame_out = f'{out_dir}capture_{capture_id:05d}_depth.npy'

    timestamp = datetime.fromtimestamp(timestamp)

    with open(capture_log) as file:
        capture_log = file.readlines()
        capture_log = json.loads(capture_log[-1])['text']
        capture_log = capture_log.split('|')[0]
        capture_log = datetime.strptime(capture_log, '%Y-%m-%d %H:%M:%S.%f ')
        capture_log = capture_log.replace(microsecond=0)
        delay = timestamp - capture_log
        delay = int(delay.total_seconds())
        print(f'{capture_id}: {timestamp} - {capture_log} = {delay}; {0 < delay < 6}')

    if save and 0 < delay < 6:
        os.rename(color_image_in, color_image_out)
        os.rename(depth_image_in, depth_image_out)
        os.rename(depth_frame_in, depth_frame_out)
