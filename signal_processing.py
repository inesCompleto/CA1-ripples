"""
Basic processing procedures for time series (e.g., performing a z-score of a
signal, or filtering a signal).

.. autosummary::
    :toctree: _toctree/signal_processing

    zscore
    cross_correlation_function
    butter
    wavelet_transform
    hilbert
    rauc
    derivative

"""


import numpy as np
from scipy.signal import butter, buttord, sosfiltfilt
from scipy.ndimage import uniform_filter1d  # Used for RMS smoothing
from scipy.interpolate import interp1d

def norm_time(firing_rate, number_bins=10):
    """
    Normalize a time series array using a fixed number of bins

    1) Check if interpolation is needed.
    2) Interpolate if the array is too short.
    3) Apply temporal binning afterward.
    """
    original_length = len(firing_rate)

    # Step 1: Interpolate if too few points
    if original_length < number_bins:
        # Interpolate to number_bins points
        original_time = np.linspace(0, 1, original_length)
        target_time = np.linspace(0, 1, number_bins)
        interp_func = interp1d(original_time, firing_rate, kind='linear', fill_value='extrapolate')
        time_interp = interp_func(target_time)
        return time_interp

    # Step 2: Temporal binning
    bins = np.linspace(0, original_length, number_bins + 1, dtype=int)
    binned = [np.mean(zscore[bins[i]:bins[i+1]]) for i in range(number_bins)]
    return np.array(binned)


def bandpass_filter(data, f_low, f_high, fs, fallback_order=3):
    """
    Robust bandpass Butterworth filter with automatic gpass/gstop selection and fallback.

    Parameters:
        data (np.ndarray): 1D signal.
        f_low (float): Lower cutoff (Hz).
        f_high (float): Upper cutoff (Hz).
        fs (float): Sampling frequency (Hz).
        fallback_order (int): Order used if estimation fails (default = 3).

    Returns:
        np.ndarray: Filtered signal.
    """
    nyq = fs / 2.0
    wp = [f_low / nyq, f_high / nyq]

    # Auto-tune gpass/gstop based on center frequency
    center_freq = (f_high + f_low) / 2

    if center_freq < 20:
        gpass = 3
        gstop = 20
    elif center_freq < 100:
        gpass = 2
        gstop = 30
    else:
        gpass = 1
        gstop = 50

    # Add stopband margins
    margin = min(0.5 * wp[0], 0.5 * (1 - wp[1]), 0.05)
    ws = [max(wp[0] - margin, 0.001), min(wp[1] + margin, 0.999)]

    try:
        N, Wn = buttord(wp, ws, gpass, gstop)
        if N > 10 or N < 1:
            raise ValueError(f"Unstable order estimated: N = {N}")
        sos = butter(N, Wn, btype='band', output='sos')
    except Exception as e:
        print(f"[bandpass_filter] Warning: Falling back to fixed order={fallback_order} ({e})")
        sos = butter(fallback_order, wp, btype='band', output='sos')
        
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data


def compute_fr(spikes, t_range, bin_size):

    tbin_edge = np.arange(t_range[0], t_range[1]+bin_size/2, bin_size)

    spk = spikes[(spikes > t_range[0]) & (spikes < t_range[1])]
    cnt, _ = np.histogram(spk, bins=tbin_edge)

    fr = cnt / bin_size

    t_bin = (tbin_edge[1:] + tbin_edge[:-1]) / 2

    return fr, t_bin


def compute_rms_envelope(signal, bin_size):
    """Computes the RMS power envelope using a moving window.
    Instantaneous power of filtered LFP estimated by calculating the root mean square over a bin_size interval.
    """
    squared_signal = signal ** 2  # Compute power
    rms_envelope = np.sqrt(np.clip(uniform_filter1d(squared_signal, size=bin_size), 0, None))  # Smooth using uniform filter
                                                                                               # clip minimum values to zero
                                                                                               # sometimes minimum values is -1e-20
                                                                                               # which is very close to zero but not quite
                                                                                               # and can't compute sqrt
    return rms_envelope



