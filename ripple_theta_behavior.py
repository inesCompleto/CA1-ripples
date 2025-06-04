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

# Load times of ripple events
pathway_win = r"D:\data_CA1\hdf5_NoHighGain"                   # directory where data files are stored
trial_files = [f for f in os.listdir(pathway_win)] # Load name of all rat files

# Define saving path
save_path = r"D:\data_CA1\interneurons"
os.makedirs(save_path, exist_ok=True)  # Create folder if it doesn't exist


bin_fr = 0.005 # sec, for firing rate                 
lag = 0.2

eeg_sr = 4800

for trial in trial_files:
    
    print(trial)
        
    # Initialize empty lists for each variable
    interneuron_id, mag_R, mean_phase, frequency = [], [], [], []

    file_path = os.path.join(pathway_win, trial)

# ---- Load data --- #
    with h5py.File(file_path,'r') as f:
        
        cell_type = f['cell_type'][:]
        mask_inhib = cell_type==b'inhib'
        
        ripp_events = f['ripp_times']['postRun_sleep'][:]
        
        lfp_run = f['lfp']['run'][:]
        spk_times_run = f['spike_times']['run'][:]

        
# ---- Behavior during ripples, during sleep ---- #        
        # mean ripple duration
        ripp_dur = np.squeeze(np.diff(ripp_events, axis=1))
        mean_dur = round(np.mean(ripp_dur),2)
        number_ripp = len(ripp_dur)


        spk_times_sleep = f['spike_times']['postRun_sleep']
        spk_sleep_inhib = spk_times_sleep[mask_inhib]
        
        n_inhib = len(spk_sleep_inhib)
        
    if n_inhib==0:
        continue

    fr = []

    # Number of bins used for normalization
    bins_notRipp = 10
    bins_ripp = 10
    
    mean_rate_ripp = np.empty(n_inhib) # store mean neuron rate during ripple period
    mean_rate_nonripp = np.empty(n_inhib) # store mean neuron rate during non-ripple period
    

    for i in range(n_inhib):

        spk_times = spk_sleep_inhib[i]
        pre_ripp, ripp, post_ripp = [], [], [] # to store firing rates for all the ripple events
        spks_scatter = [] # store spike times (non-normalized) for scatter plot

        total_ripples = len(ripp_events)
        
        fr_ripp = [] # store firing rate during ripples
        fr_nonripp = [] # store firing rate during non-ripple periods
        
        for j, (start,end) in enumerate(ripp_events): 
            
            # -- Mean neuron firing rate during ripple and non-ripple period
            # firing rate during ripples
            firing_rate_ripp,_ = sp.compute_fr(spk_times, t_range=[start,end],bin_size=bin_fr)

            fr_ripp.extend(firing_rate_ripp)

            # firing rate during non-ripples 
            if j==0:
                firing_rate_nonripp,_ = sp.compute_fr(spk_times, t_range=[0,start],bin_size=bin_fr)
            else:
                previous_end = ripp_events[j][1]
                firing_rate_nonripp,_ = sp.compute_fr(spk_times, t_range=[previous_end,start],bin_size=bin_fr)

            fr_nonripp.extend(firing_rate_nonripp)

            if j==(total_ripples-1):
                firing_rate_nonripp,_ = sp.compute_fr(spk_times, t_range=[end,-1],bin_size=bin_fr)
                fr_nonripp.extend(firing_rate_nonripp)
                    

        mean_rate_ripp[i] = np.mean(fr_ripp)
        mean_rate_nonripp[i] = np.mean(fr_nonripp) 

    
# ---- Behavior during theta, during run trial ---- #
    # 1) Selected best channel for theta detection
    # Choose highest SNR in the theta band for the RUN trial
    n_channels = len(lfp_run) # number of LFP channels

    SNR_theta = np.empty(n_channels) # to store signal-to-noise ratio for each channel

    for ch in range(n_channels):

        channel = lfp_run[ch]
        
        # Compute power spectral density (PSD)
        freqs, psd = welch(channel, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')

        # Identify the peak frequency in the theta range (5-11 Hz)
        theta_band_mask = (freqs >= 5) & (freqs <= 11)
        theta_freqs = freqs[theta_band_mask]
        theta_psd = psd[theta_band_mask]

        peak_idx = np.argmax(theta_psd)
        peak_freq = theta_freqs[peak_idx]

        # Compute SNR as the mean power in peak band ±0.5 Hz / total power
        peak_band_mask = (freqs >= (peak_freq - 0.5)) & (freqs <= (peak_freq + 0.5))
        peak_band_power = np.mean(psd[peak_band_mask])
        total_power = np.sum(psd)
 
        SNR_theta[ch] = peak_band_power / total_power if total_power > 0 else np.nan

    best_theta_ch_idx = np.argmax(SNR_theta)
    selected_lfp = lfp_run[best_theta_ch_idx]

    # Redefine theta range. best practice, given that theta might chang with age
    freqs, psd = welch(selected_lfp, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')
    theta_peak = np.argmax(psd)
    theta_band = (theta_peak-3,theta_peak+3)    
    filtered_lfp =sp.bandpass_filter(selected_lfp, theta_band[0], theta_band[1], eeg_sr)  
    
    # Hilbert transform. Measure phase
    hilb = hilbert(filtered_lfp)
    phase = np.angle(hilb, deg=True)
    phase[phase < 0] = phase[phase < 0] + 360
    
    spk_run_inhib = spk_times_run[mask_inhib]
    
    # Check at which phase inhibitory neuron is spiking
    for j in range(n_inhib):
        spk_idx = np.array((spk_run_inhib[j] * eeg_sr).astype(int))
        
        # 3. Get phase of the LFP at the spike times
        spike_phases = phase[spk_idx] 
        
        # Convert to radians for circular plot
        spike_phases_rad = np.deg2rad(spike_phases)
#        mean_angle = stats.circmean(spike_phases_rad)
        
        # Perform Rayleigh test
        R, mean_angle, p_value = rd.rayleigh_test(spike_phases_rad)

        
        # 5. Save info about inhibitory neurons in file
        interneuron_id.append(j)
        mag_R.append(R)
        mean_phase.append(mean_angle)
        frequency.append(theta_peak)
        
    # Define trial name and HDF5 save path
    trial_name = trial[:-3] # Removes the last 3 characters ('.h5')
    hdf5_file_path = os.path.join(save_path, f"{trial_name}_interneurons.h5")

    # Save Data in HDF5 Format
    with h5py.File(hdf5_file_path, 'a') as f:  # 'a' mode allows appending new variables
        if "interneuron_id" in f:
            del f['interneuron_id']
        f.create_dataset(name='interneuron_id', data=interneuron_id)
        
        if "mag_R" in f:
            del f['mag_R']
        f.create_dataset(name='mag_R', data=mag_R)
        
        if "mean_phase" in f:
            del f['mean_phase']
        f.create_dataset(name='mean_phase', data=mean_phase)
        
        if "frequency" in f:
            del f['frequency']
        f.create_dataset(name='frequency', data=frequency)
        
        if "neuron_rate_ripp" in f:
            del f['neuron_rate_ripp']
        f.create_dataset(name='neuron_rate_ripp', data=mean_rate_ripp)
        
        if "neuron_rate_nonripp" in f:
            del f['neuron_rate_nonripp']
        f.create_dataset(name='neuron_rate_nonripp', data=mean_rate_nonripp)
   
    print(f"Saved: {hdf5_file_path}")
        
