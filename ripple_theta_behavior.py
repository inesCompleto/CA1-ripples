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
  

pathway_win = r"D:\data_CA1\hdf5_NoHighGain"       # directory where data files are stored
trial_files = [f for f in os.listdir(pathway_win)] # Load name of all rat files

bin_fr = 0.005 # sec, for firing rate                 
eeg_sr = 4800
theta_range = (5,11)
theta_bandwidth = 3 # Hz around peak

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
        

# ---- Selected best channel for theta detection ---- #
    n_channels = len(lfp_run) # number of LFP channels

    SNR_theta = np.zeros(n_channels) # to store signal-to-noise ratio for each channel

    for ch in range(n_channels):
        channel = lfp_run[ch]
        
        # Compute power spectral density (PSD)
        freqs, psd = welch(channel, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')

        # Identify the peak frequency in the theta range (5-11 Hz)
        theta_band_mask = (freqs >= theta_range[0]) & (freqs <= theta_range[1])
        if not np.any(theta_band_mask):
            SNR_theta[ch] = 0.0
            continue
        
        theta_freqs = freqs[theta_band_mask]
        theta_psd = psd[theta_band_mask]
        peak_idx = np.argmax(theta_psd)
        peak_freq = theta_freqs[peak_idx]

        # Compute SNR as the mean power in peak band ±0.5 Hz / total power
        peak_band_mask = (freqs >= (peak_freq - 0.5)) & (freqs <= (peak_freq + 0.5))
        peak_band_power = np.mean(psd[peak_band_mask])
        total_power = np.sum(psd)
 
        SNR_theta[ch] = peak_band_power / total_power if total_power > 0 else np.nan

    # Select best channel
    best_theta_ch_idx = np.argmax(SNR_theta)
    selected_lfp = lfp_run[best_theta_ch_idx]
    
    # Redefine theta range around peak
    freqs, psd = welch(selected_lfp, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')
    theta_peak = np.argmax(psd)
    theta_low = theta_peak - theta_bandwidth
    theta_high = theta_peak + theta_bandwidth
   
    # Filter and extract phase
    filtered_lfp =sp.bandpass_filter(selected_lfp, theta_low, theta_high, eeg_sr)  
    hilb = hilbert(filtered_lfp)
    phase = np.angle(hilb, deg=True)

    # Phase conversion
    phase = np.where(phase<0, phase+360, phase)

# ---- Get theta phase-locking info during RUN 
#      and firing rate info during SLEEP for each neuron ---- #

    total_neurons = len(spk_times_sleep)
    
    mean_rate_ripp = np.zeros(total_neurons)    # store mean neuron rate during ripple period
    mean_rate_nonripp = np.zeros(total_neurons) # store mean neuron rate during non-ripple period
    mag_R = np.zeros(total_neurons)             # for theta phase-locking analysis
    mean_phase = np.empty(total_neurons)        # for theta phase-locking analysis
    
    for i in range(total_neurons):
        
        # ---- Ripple analysis during SLEEP ---- #
        trial_duration = len(lfp_sleep[0])/eeg_sr
        spk_times = spk_times_sleep[i]
        total_ripples = len(ripp_events)
        
        # Compute firing rate of neuron i for the entire trial
        full_firing_rate, time_bins = sp.compute_fr(spk_times, t_range=[0, trial_duration], bin_size=bin_fr)
        
        fr_ripp = [] # store firing rate during ripples
        fr_nonripp = [] # store firing rate during non-ripple periods
        
        for j, (start,end) in enumerate(ripp_events): 
            
            # Convert time intervals to bin indices
            start_bin = int(start / bin_fr)
            end_bin = int(end / bin_fr)

            # Firing rate during ripples
            fr_ripp.extend(full_firing_rate[start_bin:end_bin])

            # firing rate during non-ripples 
            if j==0:
                # Before first ripple
                fr_nonripp.extend(full_firing_rate[0:start_bin])
            else:
                # Between ripples
                previous_end = ripp_events[j-1][1]
                previous_end_bin = int(previous_end / bin_fr)
                fr_nonripp.extend(full_firing_rate[previous_end_bin:start_bin])

            if j==(total_ripples-1):
                # After last ripple
                fr_nonripp.extend(full_firing_rate[end_bin:])
        
        mean_rate_ripp[i] = np.mean(fr_ripp)
        mean_rate_nonripp[i] = np.mean(fr_nonripp)
 
        # ---- Phase-locking analysis during RUN ---- #
        # Convert spike times to indices
        spk_idx = np.array((spk_times_run[i] * eeg_sr).astype(int))
        
        # Extract phases at spike times
        spike_phases = phase[spk_idx] 
        spike_phases_rad = np.deg2rad(spike_phases)
        
        # Perform Rayleigh test
        R, mean_angle, p_value = rd.rayleigh_test(spike_phases_rad)

        mag_R[i] = R
        mean_phase[i] = mean_angle

    # Save results
    with h5py.File(file_path, 'a') as f:
        
        # List of datasets to update
        datasets_to_update = {
            'mag_R': mag_R,
            'theta_phase': mean_phase,
            'theta_frequency': theta_peak,
            'neuron_rate_ripp': mean_rate_ripp,
            'neuron_rate_nonripp': mean_rate_nonripp
            }
        
        # Remove existing datasets and create ones
        for dataset_name, data in datasets_to_update.items():
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(name=dataset_name, data=data)
   
    print(f"Saved: {file_path}")
        
