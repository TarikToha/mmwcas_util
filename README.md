# mmWave Cascade Radar Utility

![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)

This repository provides utility scripts for processing mmWave radar recordings, including downloading data, extracting
frames, and generating Range-Azimuth spectrograms.

## Table of Contents

- [Requirements](#requirements)
- [Radar Capture](#radar-capture)
- [Download Recordings](#download-recordings)
- [Extract Frames](#extract-frames)
- [Generate Spectrogram](#generate-spectrogram)
- [License](#license)

## Requirements

- Python 3.9+
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```

## Radar Capture

To record radar data, use the Linux-based mmWave capture toolkit:

[mmWave Capture Toolkit](https://github.com/mmwave-capture-std/mmwave-capture-std)

## Download Recordings

To transfer recorded radar data from the DSP board to a local machine:

1. Open `download_frames.py` and configure:
    - Capture IDs: Specify the recordings to download.
    - Local directory: Set where recordings will be saved.

2. Run the script:
   ```sh
   python download_frames.py
   ```

This script fetches `.bin` files from the DSP board and stores them locally.

## Extract Frames

To extract frames from the recorded radar data:

1. Ensure the radar configuration in `radar_utility.py` matches your setup.
2. Open `process_frames.py` and configure:
    - Capture IDs: Specify the recordings to process.
    - Local directory: Set where extracted frames will be stored.

3. Run the script:
   ```sh
   python process_frames.py
   ```

This script merges data from multiple radar chips, extracts frames, and saves them for further processing.

## Generate Spectrogram

To generate a Range-Azimuth spectrogram from radar frames:

1. Configure FFT parameters (if needed):
    - Modify `param.py` to adjust FFT settings.

2. Set up processing parameters:
    - Open `azimuth_map.py` and configure:
        - Capture IDs: Specify the recordings to process.
        - Base directory: Set where recordings are stored.

3. Run the script:
   ```sh
   python azimuth_map.py
   ```

This script calibrates radar frames, applies FFT transformations, and visualizes the spectrogram.

## License

This project is licensed under the [MIT License](https://mit-license.org/).
