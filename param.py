"""
Radar Processing Constants

Defines key parameters for radar data processing, including:
- Maximum bins for range, Doppler, and azimuth
- Frame indexing for processing
"""

max_range_bins: int = 200  # Maximum number of range bins
max_dop_bins: int = 64  # Maximum number of Doppler bins
max_azi_bins: int = 128  # Maximum number of azimuth bins

start_frame: int = 0  # First frame index
end_frame: int = 59  # Last frame index (inclusive)

discard_cases = [221]
