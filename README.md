# mmWave Cascade Radar Utility

![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)

Utility scripts for processing mmWave radar recordings, including downloading data, extracting frames, and generating
Range-Azimuth spectrograms.

## Requirements

- Python 3.9+
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```

## Radar Capture

Record radar data using the [mmWave Capture Toolkit](https://github.com/mmwave-capture-std/mmwave-capture-std).

## Download Recordings

1. Configure `download_frames.py`:
    - Set capture IDs and local save directory.
2. Run:
   ```sh
   python download_frames.py
   ```

## Extract Frames

1. Ensure `radar_utility.py` settings match your radar.
2. Configure `process_frames.py`:
    - Set capture IDs and save directory.
3. Run:
   ```sh
   python process_frames.py
   ```

## Generate Spectrogram

1. Configure FFT parameters in `param.py` if needed.
2. Set up `azimuth_map.py`:
    - Define capture IDs and base directory.
3. Run:
   ```sh
   python azimuth_map.py
   ```

## License

Licensed under the [MIT License](https://mit-license.org/).
