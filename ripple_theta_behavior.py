# -*- coding: utf-8 -*-
"""
How do firing patterns change during development?
- Analyze firing patterns of pyr cells and interneuron during theta cycles and SWRs

Main steps:
    (Theta phase analysis)
    1. Select lfp channel for theta phase analysis
    2. Extract theta phase during RUN trial
    3. Get interpolated phase values at spike times
    4. Compute preferred spiking phase
     
    
    (SWR modulation analysis)
    5. Normalize fr over entire SLEEP trial
    6. Load idx of SWR events
    7. Compute ripple modulation as difference between ripple and non-ripple 
       mean normalized activity
    
    8. Save ....

"""

import os
import h5py 
from scipy.signal import welch
import numpy as np
import signal_processing as sp
import rhythm_detection as rd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1)) # for fr normalization

# Parameters
eeg_sr = 4800
theta_range = (5,11)
SWR_range = (100,250)
filt_half_bandwidth = 3 # Hz around peak

bin_fr=0.005
  
pathway_win = r"D:\data_CA1\hdf5_NoHighGain_old"       # directory where data files are stored
pathway_win2 = r"D:\data_CA1\hdf5_NoHighGain_new"       # directory where data files are stored
trial_files = [f for f in os.listdir(pathway_win)] # Load name of all rat files
     
# (Optional) To store theta spiking phase for all neurons in a age group
phase_group1 = []
phase_group2 = []
phase_group3 = []
phase_group4 = []

for trial in trial_files[15:]:  
    print(trial)
    file_path = os.path.join(pathway_win, trial)
    file_path2 = os.path.join(pathway_win2, trial)

    with h5py.File(file_path2,'r') as f:
        selected_tetrode = f['theta_lfp_tetrode'][()]    

    with h5py.File(file_path,'r') as f:
        cell_tetrode = f['cell_id']
        
        spk_times_run = f['spike_times']['run'][:]
        spk_times_sleep = f['spike_times']['postRun_sleep'][:]
        total_neurons = len(spk_times_run)
        
        # Initialize variables
        # Theta phase analysis
        theta_R = np.zeros(total_neurons)
        mean_theta_phase = np.zeros(total_neurons)
        theta_lock_sign = np.zeros(total_neurons)
        
        # SWR modulation analysis
        ripp_modulation = np.zeros(total_neurons)
        
        # Theta phase analysis
        # 1. Select lfp ch for theta phase analysis
        all_tetrodes = f['tetrode_id'][:]  # index of tetrode in lfp ch
#        selected_tetrode = f['theta_lfp_tetrode'][()]        
        lfp_idx = np.where(all_tetrodes == selected_tetrode)[0][0] # which lfp ch to use? 1st one from this tetrode
        selected_lfp = f['lfp']['run'][lfp_idx]
        
        # 2. Extract theta phase during RUN trial
        # 2.1 Redefine theta range around peak
        freqs, psd = welch(selected_lfp, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')
        theta_band_mask = (freqs >= theta_range[0]) & (freqs <= theta_range[1]) # Identify the peak frequency in the theta range
        theta_psd = psd[theta_band_mask]
        peak_idx = np.argmax(theta_psd)
        theta_freqs = freqs[theta_band_mask]
        theta_peak_freq = theta_freqs[peak_idx]
        
        theta_low = theta_peak_freq - filt_half_bandwidth
        theta_high = theta_peak_freq + filt_half_bandwidth
             
        # 2.2 Create function that  estimates phase
        phase_1d, _ = sp.get_signal_phase(
                lfp=selected_lfp,
                sampling_rate=eeg_sr,
                power_thresh=5,
                cycle_start="peak",
                bandpass_range = (theta_low, theta_high),
                )             
        
        for i in range(total_neurons): 
            
            # 3. Get phase values at spike times (in rad)
            spike_theta = sp.get_spike_phase(
                phase = phase_1d,        
                spike_times=spk_times_run[i],
                sampling_rate=eeg_sr,
                ) # 0-2pi
            
            # 4. Compute preferred spiking phase
            # 4.1 Rayleigh test
            theta_R[i], mean_theta_phase[i], _ = sp.rayleigh_test(spike_theta)
             
            # 4.2 Theta shuffle test
            duration_run = len(selected_lfp)/eeg_sr
            theta_R_shuffled = sp.shuffle_test(spk_times_run[i], duration_run, phase_1d, eeg_sr)
            threshold = np.percentile(theta_R_shuffled, 95)  # significance threshold
            theta_lock_sign[i] = 1 if theta_R[i]>threshold else theta_lock_sign[i] # set lock sign to one if R significant
         
            # SWR modulation analysis
            # 5. Normalize fr over entire SLEEP trial
            # 5.1 Get duration of sleep trial based on duration of lfp
            example_lfp_sleep = f['lfp']['postRun_sleep'][0]
            duration_sleep = len(example_lfp_sleep)/eeg_sr
            
            # 5.2 Compute firing rate from spk times
            full_firing_rate, time_bins = sp.compute_fr(spk_times_sleep[i], 
                                                        t_range=[0, duration_sleep], 
                                                        bin_size=bin_fr)
            
            # 5.3 Normalize firing rate for entire trial
            norm_firing_rate = scaler.fit_transform(full_firing_rate.reshape(-1, 1)).flatten()
             
            # 6. Load idx of SWR events
            ripp_events = f['ripp_times']['postRun_sleep'][:]
            
            # 7. Compute ripple modulation as difference between ripple and non-ripple mean activity
            _, ripp_modulation[i] = rd.ripp_modulation(norm_firing_rate, ripp_events)
 
  
    # Save results
    with h5py.File(file_path, 'a') as f:
        
        # List of datasets to update
        datasets_to_update = {
            'mag_R': theta_R,
            'theta_phase': mean_theta_phase,
            'theta_lock_phase': theta_lock_sign,
            'ripp_modulation': ripp_modulation          
            }
         
        # Remove existing datasets and create ones
        for dataset_name, data in datasets_to_update.items():
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(name=dataset_name, data=data)
   
    print(f"Saved: {file_path}")
        
