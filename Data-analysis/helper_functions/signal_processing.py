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
from scipy.ndimage import uniform_filter1d  # Used for RMS smoothing
from scipy.interpolate import interp1d
from scipy.signal import welch, hilbert,  firwin, filtfilt



def rayleigh_test(phases):
    """
    Calculate the normalized Rayleigh vector for a set of phase angles (in rad)
    Formula: R = sqrt((1/n) * (sum(sin(phases))**2 + sum(cos(phases))**2))

    Params:
    phases (numpy array): phase angles in rad

    Returns:
    R (float): normalized Rayleigh vector magnitude
    mean_angle (float): mean phase angle
    p_value (float): p-value for Rayleigh test
    """
    
    # Remove NaN values
    phases = phases[~np.isnan(phases)]
    n = len(phases)
    
    # Handle edge case: no valid phases
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Handle edge case: only one phase value
    if n == 1:
        return 1.0, phases[0], 0.0  # Perfect phase locking with 1 spike
    
    sum_sin = np.sum(np.sin(phases))  # No need for nansum since we removed NaNs
    sum_cos = np.sum(np.cos(phases))

    R = np.sqrt(sum_sin**2 + sum_cos**2) / n
    mean_angle = (np.arctan2(sum_sin, sum_cos))
    Z = n * R**2  # Rayleigh's Z statistics
    p_value = np.exp(-Z)  # approximation for large n

    return R, mean_angle, p_value


def shuffle_test(spk_times, trial_duration, phase_interpolator, sampling_rate, n_shuffles=500):
    
        theta_R_shuffled = []
        for _ in range(n_shuffles):
            shift = np.random.uniform(0, trial_duration)
            shifted_times = (spk_times + shift) % trial_duration 
            shuffled_phases = get_spike_phase(phase_interpolator, shifted_times, sampling_rate)
            shuffled_R, *_ = rayleigh_test(shuffled_phases)         
            theta_R_shuffled.append(shuffled_R)
        theta_R_shuffled = np.array(theta_R_shuffled)
        
        return theta_R_shuffled




def get_signal_phase(
    lfp: np.ndarray,
    sampling_rate: float,
    clip_value: float = 0,
    power_thresh: float = 5,
    cycle_start: str = "peak",
    bandpass_range: tuple = (4.0, 12.0),
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """
    Compute theta phase, cycle numbers, and cycle boundaries.

    Parameters
    ----------
    lfp : np.ndarray
        Input LFP signal (1D).
    sampling_rate : float
        Sampling rate in Hz.
    clip_value : float, optional
        If >0, cycles with clipped values beyond this threshold are discarded.
    power_thresh : float, optional
        Percentile threshold for rejecting low-power cycles.
    cycle_start : str, optional
        Whether to define cycle boundaries at "peak" or "trough".
    bandpass_range : tuple, optional
        Bandpass filter frequency range (low, high).

    Returns
    -------
    signal_phase_out : np.ndarray
        Instantaneous theta phase, NaN for bad cycles.
    cycle_numbers_out : np.ndarray
        Cycle ID per sample, NaN for bad cycles.
    cycle_boundaries : list of tuples
        List of (cycle_number, start_index, end_index) for valid cycles.
    """

    if cycle_start not in ["peak", "trough"]:
        raise ValueError("cycle_start must be either 'peak' or 'trough'")

    n_samples = lfp.shape[0]
    low_freq, high_freq = bandpass_range

    # Pre-allocate outputs
    signal_phase_out = np.zeros(n_samples)
    cycle_numbers_out = np.zeros(n_samples, dtype=float)

    # Filter LFP signal in theta range
    filtered_lfp = bandpass_filter(
        lfp,
        low_freq,
        high_freq,
        sampling_rate
    )

    # Hilbert transform
    analytic_signal = hilbert(filtered_lfp)

    # Extract phase
    signal_phase = np.angle(analytic_signal)

    # Identify phase transitions
    phase_diff = np.diff(signal_phase)
    phase_transitions = np.zeros(n_samples, dtype=bool)

    if cycle_start == "peak":
        phase_transitions[1:] = phase_diff < -np.pi
    else:  # "trough"
        phase_transitions[1:] = np.abs(phase_diff) > np.pi

    # Assign cycle numbers
    cycle_numbers = np.cumsum(phase_transitions)

    # Stats
    max_cycle = cycle_numbers[-1] + 1
    bincount_cycles = np.bincount(cycle_numbers, minlength=max_cycle)

    # Power per cycle
    power = filtered_lfp**2
    power_per_cycle = np.bincount(
        cycle_numbers, weights=power, minlength=max_cycle
    )
    mask = bincount_cycles > 0
    power_per_cycle[mask] /= bincount_cycles[mask]

    # Identify bad cycles
    bad_cycles_set = set()

    # 1. Clipped cycles
    if clip_value > 0:
        clipped_indices = np.where(np.abs(lfp) > clip_value)[0]
        if len(clipped_indices) > 0:
            clipped_cycles = np.unique(cycle_numbers[clipped_indices])
            bad_cycles_set.update(clipped_cycles.astype(int))

    # 2. Low-power cycles
    power_threshold = np.nanpercentile(power_per_cycle, power_thresh)
    bad_power_cycles = np.where(power_per_cycle < power_threshold)[0]
    bad_cycles_set.update(bad_power_cycles.astype(int))

    # 3. Bad length cycles
    cycle_length_lim = np.ceil(
        1 / np.array([high_freq, low_freq]) * sampling_rate
    ).astype(int)
    bad_length_mask = (bincount_cycles < cycle_length_lim[0]) | (
        bincount_cycles > cycle_length_lim[1]
    )
    bad_length_cycles = np.where(bad_length_mask)[0]
    bad_cycles_set.update(bad_length_cycles.astype(int))

    # Create mask of bad cycles
    is_bad_cycle = np.zeros(max_cycle, dtype=bool)
    if bad_cycles_set:
        is_bad_cycle[np.array(list(bad_cycles_set), dtype=int)] = True
    bad_sample_mask = is_bad_cycle[cycle_numbers]

    # Apply mask to outputs
    channel_phase = signal_phase.copy()
    channel_phase[bad_sample_mask] = np.nan
    signal_phase_out[:] = channel_phase

    channel_cycles = cycle_numbers.astype(float)
    channel_cycles[bad_sample_mask] = np.nan
    cycle_numbers_out[:] = channel_cycles

    # Compute cycle boundaries for valid cycles
    cycle_boundaries = []
    valid_cycles = np.unique(channel_cycles[~np.isnan(channel_cycles)]).astype(int)
    for cycle in valid_cycles:
        indices = np.where(channel_cycles == cycle)[0]
        if len(indices) > 0:
            start_idx = indices[0]
            end_idx = indices[-1]
            cycle_boundaries.append((cycle, start_idx, end_idx))

    return signal_phase_out, cycle_numbers_out, cycle_boundaries


def get_spike_phase(
    phase: np.ndarray,
    spike_times: np.ndarray,
    sampling_rate: float
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate the signal phase of spikes based on the LFP.
    """
    
    # Convert spike times to indices
    spike_indices = np.floor(spike_times * sampling_rate).astype(np.int64)

    # Create output arrays with NaNs
    spike_phases = np.full_like(spike_times, np.nan, dtype=float)

    # Find valid indices
    valid_mask = (spike_indices >= 0) & (spike_indices < len(phase))

    # Extract phases and frequencies for valid spikes
    if np.any(valid_mask):
        valid_indices = spike_indices[valid_mask]
        spike_phases[valid_mask] = phase[valid_indices]

    return spike_phases




def choose_lfp_ch(lfp, eeg_sr, freq_range):
    """
    Chooses best lfp channel for analysis of given rhythm.

    Returns
    -------
    best_ch_idx : TYPE
        DESCRIPTION.

    """
    
    
    n_channels = len(lfp) # number of LFP channels

    SNR = np.zeros(n_channels) # to store signal-to-noise ratio for each channel

    for ch in range(n_channels):
        channel = lfp[ch]
            
        # Compute power spectral density (PSD)
        freqs, psd = welch(channel, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')

        # Identify the peak frequency in the theta range (5-11 Hz)
        band_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if not np.any(band_mask):
            SNR[ch] = 0.0
            continue
            
        rhythm_freqs = freqs[band_mask]
        rhythm_psd = psd[band_mask]
        peak_idx = np.argmax(rhythm_psd)
        peak_freq = rhythm_freqs[peak_idx]

        # Compute SNR as the mean power in peak band ±0.5 Hz / total power
        peak_band_mask = (freqs >= (peak_freq - 0.5)) & (freqs <= (peak_freq + 0.5))
        peak_band_power = np.mean(psd[peak_band_mask])
        total_power = np.sum(psd)
     
        SNR[ch] = peak_band_power / total_power if total_power > 0 else np.nan

    # Select best channel
    best_ch_idx = np.argmax(SNR)
    
    return best_ch_idx



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
    binned = [np.mean(firing_rate[bins[i]:bins[i+1]]) for i in range(number_bins)]
    return np.array(binned)





def bandpass_filter(data, f_low, f_high, fs):

    # Optimize filter design
    filter_order = min(int(fs / 2), 1001)
    if filter_order % 2 == 0:
        filter_order += 1  # Ensure odd order for zero phase filtering

    # Design bandpass filter for this channel
    filter_taps = firwin(
            filter_order,
            [f_low, f_high],
            pass_zero=False,
            window="blackman",
            fs=fs,
        )

    n_samples = data.shape[0]
    # Calculate appropriate padding for filtfilt
    pad_length = min(3 * (len(filter_taps) - 1), n_samples - 1, 1000)

    # Filter LFP for this channel
    filtered_lfp = filtfilt(filter_taps, 1, data, padlen=pad_length)
    
    return filtered_lfp






def bandpass_filter_mine(data, f_low, f_high, fs, fallback_order=3):
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



