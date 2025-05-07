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
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d  # Used for RMS smoothing

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
     Design a Butterworth bandpass filter. (maybe use a FIR filter; see chat gpt comments on this)

     The butter() function (from scipy.signal) expects the cutoff frequencies to be specified 
     in terms of a fraction of the Nyquist frequency, i.e. the highest frequency that can be accurately represented 
     (half of sampling rate)
    """

    # Compute Nyquist frequency
    nyquist = 0.5 * fs

    # Normalize cutoff frequencies by the Nyquist
    low = lowcut / nyquist   
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to the data
    filtered_data = filtfilt(b, a, data) # The filtfilt() function applies the filter to the data.
                                         # in both directions—forward and backward—which eliminates any phase shift 
                                         # (i.e., it does not distort the time alignment of the data).
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



