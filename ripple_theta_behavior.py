# -*- coding: utf-8 -*-
"""
Classify interneurons

Classifies tpes of interneurons based on their behavior during ripple events   

"""
import numpy as np
import os
import h5py
from scipy.signal import welch, hilbert
import signal_processing as sp
import rhythm_detection as rd
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0, 1)) # for fr normalization


def ripp_diff(spk_times, duration_trial, ripp_events):
    
    bins_notRipp = 10
    bins_ripp = 10
    lag_default = 0.05
    bin_fr = 0.005 # sec, for firing rate 
    
    full_firing_rate, time_bins = sp.compute_fr(spk_times[i], t_range=[0, duration_trial], bin_size=bin_fr)
    norm_firing_rate = scaler.fit_transform(full_firing_rate.reshape(-1, 1)).flatten() # normalized firing rate

    norm_fr_ripp = [] # store firing rate during ripples
    norm_fr_nonripp = [] # store firing rate during non-ripple periods
    pre_ripp, ripp, post_ripp = [], [], [] # store normalized zscored f.r. before, during and after ripp
        
    for j, (start,end) in enumerate(ripp_events): 
        # Compute lag to consider for each ripple
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
        norm_fr_nonripp.extend(pre_ripp_fr) # too compute mean non ripp activity
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
    normalized_fr.append(all_fr)
        
    mean_ripp_act = np.mean(norm_fr_ripp)
    mean_nonripp_act = np.mean(norm_fr_nonripp)
        
    return (mean_ripp_act - mean_nonripp_act)

 
 
pathway_win = r"D:\data_CA1\hdf5_NoHighGain"       # directory where data files are stored
trial_files = [f for f in os.listdir(pathway_win)] # Load name of all rat files
                  
eeg_sr = 4800
theta_range = (5,11)
theta_bandwidth = 3 # Hz around peak
n_shuffles = 500

for trial in trial_files:
    
    print(trial)

    file_path = os.path.join(pathway_win, trial)

    # ---- Load data --- # 
    with h5py.File(file_path,'r') as f:
        ripp_events = f['ripp_times']['postRun_sleep'][:]
        lfp_run = f['lfp']['run'][:]
        spk_times_run = f['spike_times']['run'][:]
        spk_times_sleep = f['spike_times']['postRun_sleep'][:]
        lfp_sleep = f['lfp']['postRun_sleep'][:]
        
        duration_run = len(lfp_run[0])/eeg_sr
        duration_sleep = len(lfp_sleep[0])/eeg_sr

    # ---- Selected best channel for theta detection ---- #
#    n_channels = len(lfp_run) # number of LFP channels

#    SNR_theta = np.zeros(n_channels) # to store signal-to-noise ratio for each channel

#    for ch in range(n_channels):
#        channel = lfp_run[ch]
        
        # Compute power spectral density (PSD)
#        freqs, psd = welch(channel, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')

        # Identify the peak frequency in the theta range (5-11 Hz)
#        theta_band_mask = (freqs >= theta_range[0]) & (freqs <= theta_range[1])
#        if not np.any(theta_band_mask):
#            SNR_theta[ch] = 0.0
#            continue
        
#        theta_freqs = freqs[theta_band_mask]
#        theta_psd = psd[theta_band_mask]
#        peak_idx = np.argmax(theta_psd)
#        peak_freq = theta_freqs[peak_idx]

        # Compute SNR as the mean power in peak band ±0.5 Hz / total power
#        peak_band_mask = (freqs >= (peak_freq - 0.5)) & (freqs <= (peak_freq + 0.5))
#        peak_band_power = np.mean(psd[peak_band_mask])
#        total_power = np.sum(psd)
 
#        SNR_theta[ch] = peak_band_power / total_power if total_power > 0 else np.nan

    # Select best channel
#    best_theta_ch_idx = np.argmax(SNR_theta)
#    selected_lfp = lfp_run[best_theta_ch_idx]
    
    # Redefine theta range around peak
#    freqs, psd = welch(selected_lfp, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')
#    theta_peak = np.argmax(psd)
#    theta_low = theta_peak - theta_bandwidth
#    theta_high = theta_peak + theta_bandwidth  
   
    # Filter and extract phase
#    filtered_lfp =sp.bandpass_filter(selected_lfp, theta_low, theta_high, eeg_sr)  
#    hilb = hilbert(filtered_lfp)
#    phase = np.angle(hilb) # phase in rad [-pi,pi]
    
    # Interpolator for phase values
#    eeg_time = np.arange(len(phase)) / eeg_sr
#    phase_interpolator = interp1d(eeg_time, phase, kind='linear', bounds_error=False, fill_value="extrapolate")

# ---- Get theta phase-locking info during RUN 
#      and firing rate info during SLEEP for each neuron ---- #

    total_neurons = len(spk_times_sleep)
    observed_ripp_diff = np.zeros(total_neurons)
    mag_R = np.zeros(total_neurons)           # for theta phase-locking analysis
    mean_phase = np.zeros(total_neurons)      # for theta phase-locking analysis
    phase_lock_sign = np.zeros(total_neurons) # is theta phase-lock significant?
    ripp_act_significant = np.zeros(total_neurons) # is ripp activity analysis significant?
    normalized_fr = []                    # to store normalize f.r. before,during and after ripp

    # Define phase bins for theta cycle analysis
#    n_phase_bins = 18  # 20-degree bins (360/18 = 20 degrees per bin)
#    phase_bin_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1) 
#    phase_bin_centers = (phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2

   # Storage for phase-binned firing rates
#    phase_binned_fr = np.zeros((total_neurons, n_phase_bins))  # mean firing rate for each neuron at each phase bin
    
    # Calculate time spent in each phase bin during RUN period (do this once for all neurons)
#    run_phases = phase_interpolator(np.arange(0, duration_run, 1/eeg_sr))
#    run_phases = run_phases[~np.isnan(run_phases)]

#    if len(run_phases) > 0:
#        phase_time_counts, _ = np.histogram(run_phases, bins=phase_bin_edges)
        # Convert counts to time (in seconds)
#        phase_time_spent = phase_time_counts / eeg_sr
        
        # Avoid division by zero
#        phase_time_spent[phase_time_spent == 0] = np.finfo(float).eps  # Use small number instead of inf
#    else:
#        phase_time_spent = np.ones(n_phase_bins) * np.finfo(float).eps
  

    for i in range(total_neurons):

        # ---- Ripple analysis during SLEEP ---- #
        total_ripples = len(ripp_events)
        
        # Compute firing rate of neuron i for the entire trial
        observed_ripp_diff[i] = ripp_diff(spk_times_sleep, duration_sleep, ripp_events)
         
        shuffled_diff = []
        for _ in range(n_shuffles):
            shift = np.random.uniform(0, duration_sleep)
            shifted_times = (spk_times_sleep[i] + shift) % duration_sleep
            shuffled_diff.append(ripp_diff(spk_times_sleep, duration_sleep, ripp_events))
            
        shuffled_diff = np.array(shuffled_diff)

        # Significance threshold
        threshold_ripp = np.percentile(shuffled_diff, 95)  # 95% confidence
        if observed_ripp_diff[i] > threshold_ripp:
            ripp_act_significant[i] = 1

        # ---- Phase-locking analysis during RUN ---- #
        # Get interpolated phase values at spike times
#        spike_phases = phase_interpolator(spk_times_run[i])  
 
        # ---- Compute phase-binned firing rates ---- #
        # Bin the spike phases
#        phase_counts, _ = np.histogram(spike_phases, bins=phase_bin_edges)
            
        # Calculate firing rate for each phase bin (spikes/second)
#        phase_binned_fr[i, :] = phase_counts / phase_time_spent         
           
        # Perform Rayleigh test
#        R, mean_angle, _ = rd.rayleigh_test(spike_phases)

#        mag_R[i] = R
#        mean_phase[i] = mean_angle
        
        # Theta shuffle test
#        mag_R_shuffled = []
#        for _ in range(n_shuffles):
#            shift = np.random.uniform(0, duration_run)
#            shifted_times = (spk_times_run[i] + shift) % duration_run
#            shuffled_phases = phase_interpolator(shifted_times)
#            shuffled_R, *_ = rd.rayleigh_test(shuffled_phases)         
#            mag_R_shuffled.append(shuffled_R)

#        mag_R_shuffled = np.array(mag_R_shuffled)

        # Significance threshold
#        threshold = np.percentile(mag_R_shuffled, 95)  # 95% confidence
         
#        if R>threshold:
#            phase_lock_sign[i] = 1
           
    # Save results
    with h5py.File(file_path, 'a') as f:
        
        # List of datasets to update
        datasets_to_update = {
#            'mag_R': mag_R,
#            'theta_phase': mean_phase,
#            'theta_frequency': theta_peak,
#            'theta_lock_sign': phase_lock_sign,
#            'theta_fr': phase_binned_fr,
#            'norm_fr_ripp': normalized_fr,
            'diff_ripp_act': observed_ripp_diff,
            'ripp_act_sign': ripp_act_significant
            }
         
        # Remove existing datasets and create ones
        for dataset_name, data in datasets_to_update.items():
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(name=dataset_name, data=data)
   
    print(f"Saved: {file_path}")
        
