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
from scipy.signal import butter, buttord, filtfilt
from scipy.ndimage import uniform_filter1d  # Used for RMS smoothing


def bandpass_filter(data, f_low, f_high, fs, gpass=3, gstop=20):
    """
    Bandpass Butterworth filter with automatic order selection and zero-phase filtering.

    Parameters:
        data (np.ndarray): 1D input signal.
        fs (float): Sampling frequency in Hz.
        f_low (float): Lower cutoff frequency (Hz).
        f_high (float): Upper cutoff frequency (Hz).
        gpass (float): Maximum ripple in the passband (dB). Default is 3.
        gstop (float): Minimum attenuation in the stopband (dB). Default is 20.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyq = fs / 2.0

    # Normalize cutoff frequencies
    wp = [f_low / nyq, f_high / nyq]

    # Add stopband margin, and clamp
    margin = 0.05
    ws = [wp[0] - margin, wp[1] + margin]
    ws = [max(ws[0], 0.001), min(ws[1], 0.999)]

    # Determine filter order and cutoff
    N, Wn = buttord(wp, ws, gpass, gstop)
    b, a = butter(N, Wn, btype='band')

    filtered_data = filtfilt(b, a, data)

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



