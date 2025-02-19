from radar_utility import *

cases = range(430, 440)
# cases = [399]
remote_files = ['master_0000_idx.bin', 'master_0000_data.bin', 'slave1_0000_data.bin',
                'slave2_0000_data.bin', 'slave3_0000_data.bin']

for capture_id in cases:
    start_time = time.time()
    print(capture_id, start_time)

    remote_dir = f'root@192.168.33.180:/mnt/ssd/crowd_dataset-capture_{capture_id:05d}-cascaded/'
    local_dir = f'/home/ttoha12/crowd/dataset/crowd_dataset-capture_{capture_id:05d}-cascaded/'

    download_files(remote_dir, local_dir, remote_files, start_time)
    print(capture_id, time.time() - start_time)
