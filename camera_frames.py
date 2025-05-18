import os.path
import shutil

base_dir = '/home/ttoha12/wall_dataset/'
out_dir = f'{base_dir}ground_truths/'
cases = range(500)
save = True
if save:
    os.makedirs(out_dir, exist_ok=True)

for capture_id in cases:
    in_path = f'{base_dir}capture_{capture_id:05d}/capture_{capture_id:05d}_color.jpg'
    if not os.path.exists(in_path):
        continue

    if save:
        shutil.copy2(in_path, out_dir)

    print(capture_id, save)
