import os.path
import shutil

base_dir = '/home/ttoha12/crowd/dataset/'
cases = range(500)
save = False

for capture_id in cases:
    in_path = f'{base_dir}capture_{capture_id:05d}/capture_{capture_id:05d}_color.jpg'
    if not os.path.exists(in_path):
        continue

    out_dir = f'{base_dir}ground_truths/'
    if save:
        shutil.copy2(in_path, out_dir)

    print(capture_id)
