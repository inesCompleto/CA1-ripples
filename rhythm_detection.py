"""Find start and end times of sharp wave ripple events (100-250 Hz)
    based on [1].

    Parameters
    ----------

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
      [1] Laurenz

    """

import numpy as np
from scipy.signal import welch, hilbert, find_peaks
from scipy.signal import butter, buttord, filtfilt
import signal_processing as sp

# Suppress MNE and joblib (verbose=False) logs
import mne
mne.set_log_level("ERROR")


def detect_theta(filtered_lfp, eeg_sr, theta_band = (5,11)):
    """
    1) Identifies discrete theta cycles instead of treating all time as one continuous theta rhythm
    2) Quality control - only keeps cycles with high envelope power (strong theta)
    3) Natural cycle boundaries - uses actual peaks/troughs instead of arbitrary time windows  

    Returns:
    valid_theta_cycles: index of beginning and end of theta cycles
    """

    # Measure phase
    hilb = hilbert(filtered_lfp)
    mag = np.abs(hilb)

    peaks,_ = find_peaks(-filtered_lfp, distance=eeg_sr / theta_band[1]) # Find peaks or troughs in the filtered_lfp
    theta_cycles = list(zip(peaks[:-1], peaks[1:]))   # Each pair of consecutive troughs defines a theta cycle
                                                  # (start, end) of each cycle

    # keep cycles where the envelope power is high during that segment
    threshold = 0.5 * np.mean(mag)
    valid_theta_cycles = []
    for start, end in theta_cycles:
        if np.mean(mag[start:end]) > threshold:
            valid_theta_cycles.append((start, end))

    return valid_theta_cycles



def detect_ripples(power_env, min_dur, max_dur, eeg_sr):
    """
    psd: power envelope
    min_dur: minimum ripple peak duration
    eeg_sr: eeg sampling rate, in Hz
    """
  
    # Detect peak (interval) of ripple events
    thr_ripple_peak = np.percentile(power_env, 99) 
    is_above_threshold = power_env >= thr_ripple_peak

    int_signal = np.concatenate(([0], is_above_threshold.astype(int), [0])) # turn boolean into arrays of 0 and 1 and pad it in the beginning and end to detect edges correctly 
    diff_signal = np.diff(int_signal) # difference between consecutive entries to find start and end of ripple (0->1: start, 1->0: end)
    ripple_starts = np.where(diff_signal == 1)[0] 
    ripple_ends = np.where(diff_signal == -1)[0]

    # Filter out short ripple events
    min_samples = int((min_dur / 1000) * eeg_sr) # convert min_ripples_dur from ms to sample size
    valid_ripples = (ripple_ends - ripple_starts) >= min_samples    

    ripple_starts = ripple_starts[valid_ripples] # index of start of ripples
    ripple_ends = ripple_ends[valid_ripples] # index of end of ripples
    ripp_indices = np.column_stack((ripple_starts, ripple_ends))

    # Extend beggining and end of ripple events
    mean = np.mean(power_env)
    std = np.std(power_env)
    thr_ripple_begg_end = mean+std

    # Extend the start and end of ripple event to when power goes below mean
    for j,(left_idx,right_idx) in enumerate(ripp_indices):
        new_left_idx = next((i for i in range(left_idx, 0, -1) if power_env[i] < thr_ripple_begg_end), left_idx)
        new_right_idx = next((i for i in range(right_idx, len(power_env),1) if power_env[i] < thr_ripple_begg_end), right_idx)

        ripp_indices[j] = [new_left_idx, new_right_idx] 

    # Remove duplicate ripple events
    mask = np.ones(len(ripp_indices), dtype=bool)
    mask[1:] = ~(np.all(ripp_indices[1:] == ripp_indices[:-1], axis=1))
    ripp_indices = ripp_indices[mask]

    # Remove ripple events longer than max_dur
    max_samples = int((max_dur / 1000) * eeg_sr)
    mask_long = (ripp_indices[:, 1] - ripp_indices[:, 0]) < max_samples
    ripp_indices = ripp_indices[mask_long]

    return ripp_indices

    
def ripp_modulation(norm_firing_rate, ripp_events, bin_fr=0.005, bins_notRipp=10, bins_ripp=10, lag_default=0.05):

    total_ripples = len(ripp_events)
    
    norm_fr_ripp = [] # store firing rate during ripples
    norm_fr_nonripp = [] # store firing rate during non-ripple periods
    pre_ripp, ripp, post_ripp = [], [], [] # store normalized zscored f.r. before, during and after ripp
        
    for j, (start,end) in enumerate(ripp_events): 
        # 1. Compute lag to consider for each ripple
        current_ripple = ripp_events[j]
        
        if j==0:     
            next_ripple = ripp_events[j+1]
            distance_to_next= next_ripple[0] - current_ripple[1]
            lag = min(lag_default, distance_to_next)
    
        elif j==total_ripples-1:
            previous_ripple = ripp_events[j-1]
            distance_to_previous = current_ripple[0] - previous_ripple[1]
            lag = min(lag_default, distance_to_previous)

        else:
            next_ripple = ripp_events[j+1]
            distance_to_next = next_ripple[0] - current_ripple[1]
            previous_ripple = ripp_events[j-1]
            distance_to_previous = current_ripple[0] - previous_ripple[1]
            lag = min(lag_default, distance_to_previous, distance_to_next)

        if lag<0.015 or start-lag<0:
            continue 
            
        borders = np.array([start-lag, start, end, end+lag])/bin_fr
        
        
        pre_ripp_fr = norm_firing_rate[int(borders[0]):int(borders[1])]
        norm_fr_nonripp.extend(pre_ripp_fr) # to compute mean non ripp activity
        norm_pre = sp.norm_time(pre_ripp_fr, number_bins=bins_notRipp) # Normalized time
        pre_ripp.append(norm_pre) # to compute av firing pattern before/during and after ripp
            
        ripp_fr = norm_firing_rate[int(borders[1]):int(borders[2])]
        norm_fr_ripp.extend(ripp_fr)
        norm_ripp = sp.norm_time(ripp_fr, number_bins=bins_ripp)
        ripp.append(norm_ripp)
            
        post_ripp_fr = norm_firing_rate[int(borders[2]):int(borders[3])]
        norm_fr_nonripp.extend(post_ripp_fr)
        norm_post = sp.norm_time(post_ripp_fr, number_bins=bins_notRipp)
        post_ripp.append(norm_post)
        
    pre_ripp_mean = np.mean(pre_ripp, axis=0)
    ripp_mean = np.mean(ripp, axis=0)   
    post_ripp_mean = np.mean(post_ripp, axis=0)
    all_fr = np.concatenate([pre_ripp_mean, ripp_mean, post_ripp_mean]) 
        
    mean_ripp_act = np.mean(norm_fr_ripp)
    mean_nonripp_act = np.mean(norm_fr_nonripp)
    ripp_modulation = mean_ripp_act - mean_nonripp_act
        
    return all_fr, ripp_modulation


def detect_brain_states(
        speed, pos_sr,
        lfp, eeg_sr, theta_band=(5, 11), delta_band=(1.5, 4),
        bin_size=1.6, step_size=0.8, 
        speed_thr = 2.5,TDRatio_thr = 2
        ):
    """
    speed: instantaneous speed of rat during trial in cm/s
    pos_sr (int): sampling rate of rat's position in Hz 
    bin_speed (int): bin size used to compute mean speed from instantaneous speed in seconds
    step_speed (int): step size used to compute mean speed from instantaneous speed in seconds
    """
 
    #---- 2.1 Get speed in cm/s ----#
    # Define window size in samples
    bin_speed_samples = int(bin_size * pos_sr)
    step_speed_samples = int(step_size * pos_sr)

    # Compute number of windows to compute mean speed
    nWin_speed = int(np.floor((len(speed) - bin_speed_samples) / step_speed_samples)) # number of bins for each we will compute the PSD

    mean_speed = np.zeros(nWin_speed)
    for i in range(nWin_speed):
        start = i * step_speed_samples
        end = start + bin_speed_samples
        if end > len(speed):
            break

        mean_speed[i] = np.nanmean(speed[start:end])

#    slow_mov_idx = np.where(mean_speed < speed_thr)[0]   # Speed indices where speed is below threshold
#    slow_mov_times = slow_mov_idx/pos_sr             # times where mean speed is below threshold

    #---- 2.2 Compute theta/delta ratio ----#   
    # Run/REM: theta-dominant; SWS: delta-dominant

    #-- 2.2.1 Choose best channel based on power in delta range--#
    n_channels = len(lfp)
    SNR = np.empty(n_channels) # to store signal-to-noise ratio for each channel

    for ch in range(n_channels):

        channel = lfp[ch]
    
        # Filter lfp data to get <2.5 cm/s movement epochs
    #    slow_mov_lfp_idx = (slow_mov_times*eeg_sr).astype(int) # lfp indices where speed is below threshold 

    #    lfp_slow_epochs = channel[slow_mov_lfp_idx]  # Extract LFP data only for slow speed epochs
    
        # Compute power spectral density (PSD)
        freqs, psd = welch(channel, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')

        # Identify the peak frequency in the delta range (2-3.5 Hz). WHY USE THIS RANGE AND THEN WHEN COMPUTING THE THETA/DELTA RATIO USING DELTA=[1.5,4]
        delta_band_mask = (freqs >= 2) & (freqs <= 3.5)
        delta_freqs = freqs[delta_band_mask]
        delta_psd = psd[delta_band_mask]

        peak_idx = np.argmax(delta_psd)
        peak_freq = delta_freqs[peak_idx]

        # Compute SNR as the mean power in peak band ±0.5 Hz / total power. WHY THE ±0.5 Hz??
        peak_band_mask = (freqs >= (peak_freq - 0.5)) & (freqs <= (peak_freq + 0.5))
        peak_band_power = np.mean(psd[peak_band_mask])
        total_power = np.sum(psd)

        SNR[ch] = peak_band_power / total_power if total_power > 0 else np.nan

    best_delta_ch_idx = np.argmax(SNR)
    selected_lfp = lfp[best_delta_ch_idx]

    #-- 2.2.2 Compute power spectral density (PSD) in selected channel --#
    # How much power is in different frequency bands

    # Compute number of windows
    trial_dur = len(lfp[0]) # trial duration in lfp samples

    bin_eeg_samples = int(bin_size * eeg_sr) # Define window size in samples
    step_eeg_samples = int(step_size * eeg_sr)

    nWin_psd = int(np.floor((trial_dur - bin_eeg_samples) / step_eeg_samples)) # number of bins for each we will compute the PSD

    # Compute multitaper PSD in sliding windows (for a total of nWin windows). 
    # See notes from 05/02/2025 for visuals   
    TDRatio = np.zeros(nWin_psd)
    for i in range(nWin_psd):
        start = i * step_eeg_samples
        end = start + bin_eeg_samples 
        if end > len(selected_lfp):
            break

        eeg_segment = selected_lfp[start:end]
    
        # Compute frequency power for window i
        freqs, psd = welch(eeg_segment, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')
         
        # Compute mean power in theta and delta bands
        theta_power = np.mean(psd[(freqs >= theta_band[0]) & (freqs <= theta_band[1])])
        delta_power = np.mean(psd[(freqs >= delta_band[0]) & (freqs <= delta_band[1])])

        TDRatio[i] = theta_power / delta_power

    # Reshape TDRatio. Only needed for r631_P18 for which TDRatio has one more bin than mean_speed.
    if mean_speed.shape != TDRatio.shape: 
        TDRatio = TDRatio[:len(mean_speed)]
         
    #---- 2.3 Get brain states ----#
    state_run = (mean_speed >= speed_thr) & (TDRatio >= TDRatio_thr)
    state_sws = (mean_speed < speed_thr) & (TDRatio < TDRatio_thr)
    state_rem = (mean_speed < speed_thr) & (TDRatio > TDRatio_thr)

    return state_run, state_sws, state_rem 

def exclude_movement(ripp_times, state_sws, eeg_sr):
    """
    Find valid ripple events, i.e. ripples that happen during sws
    """

    int_sws = np.concatenate(([0], state_sws.astype(int), [0]))
    diff_sws = np.diff(int_sws) 

    sws_starts = np.where(diff_sws == 1)[0]
    sws_ends = np.where(diff_sws == -1)[0]
    sws_indices = np.column_stack((sws_starts, sws_ends))

    # Convert indices to time
    sws_times = sws_indices*0.8 # each bin corresponds to 0.8 sec
    sws_in_lfp_idx = sws_times*eeg_sr

    # Check if each event starts after the start of a state epoch (compare 1 ripple event with all sws states)
    ripp_start_ok = ripp_times[:, [0]] > sws_times[:, 0] 

    # Check if each event ends before the end of a state epoch (compare 1 ripple event with all sws states)
    ripp_end_ok = ripp_times[:, [1]] < sws_times[:, 1]  

    # Find events that are fully contained within a state epoch
    ripp_event_all = ripp_start_ok & ripp_end_ok # (ripp_events, sws_events). all-by-all boolean matrix

    # Identify valid windows (if an event is within ANY state epoch, keep it)
    ripp_event_ok = np.any(ripp_event_all, axis=1) 

    # Keep only valid events
    valid_ripp_times = ripp_times[ripp_event_ok, :]

    return valid_ripp_times


