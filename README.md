# mmWave Cascade Radar Utility

This repository provides utility scripts for working with mmWave radar recordings, including downloading files,
extracting frames, and generating Range-Azimuth spectrograms.

## Requirements

- Python 3.9+
- Install dependencies with:
  ```sh
  pip install -r requirements.txt
  ```

## Radar Capture

To record radar data, use the Linux-based mmWave capture toolkit available at:

[mmWave Capture Toolkit](https://github.com/mmwave-capture-std/mmwave-capture-std)

## Download Recordings

To download recorded radar data:

1. Open `download_frames.py` and configure the following:
    - Capture IDs: Specify the IDs of the recordings to download.
    - Local folder path: Set the directory where the recordings will be saved.
2. Run the script:
   ```sh
   python download_frames.py
   ```

This script transfers `.bin` files from the DSP board to the local machine.

## Extract Frames

To extract frames from recorded radar data:

1. Ensure the radar configuration in `radar_utility.py` matches your radar settings.
2. Open `process_frames.py` and configure the following:
    - Capture IDs: Specify the recordings to process.
    - Local folder path: Set the directory where the extracted frames will be stored.
3. Run the script:
   ```sh
   python process_frames.py
   ```

This script merges data from four radar chips, extracts frames, and saves them to the specified folder.

## Generate Spectrogram

To generate a Range-Azimuth spectrogram from radar frames:

1. Configure FFT parameters (if needed):
    - Open `param.py` and update the FFT parameters as required.

2. Set up processing parameters:
    - Open `azimuth_map.py` and configure the following:
        - Capture IDs: Specify the IDs of the recordings to process.
        - Base folder path: Set the directory where the recordings are stored.

3. Run the script:
   ```sh
   python azimuth_map.py
   ```

This script calibrates the radar frames, applies FFT transformations, and visualizes the spectrogram.
