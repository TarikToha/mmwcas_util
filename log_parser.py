import json
import os.path

cases = range(300, 400)
# cases = [316]
base_dir = '/home/ttoha12/crowd/dataset/'

for capture_id in cases:
    capture_log = f'{base_dir}capture_{capture_id:05d}/capture.log'

    if not os.path.exists(capture_log):
        print(f'{capture_log} does not exist')
        break

    with open(capture_log) as file:
        capture_log = file.readlines()
        capture_log = json.loads(capture_log[-1])['text']
        capture_log = capture_log.split('|')[0]
        print(capture_id, capture_log)
