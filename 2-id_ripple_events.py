"""
During sleep trials
1) Compute power envelope of LFP
2) Choose eeg channel with highest std to detect ripples
3) Detect ripples:
    3.1 Find ripple events in power envelope of select eeg channel
    3.2 Find slow wave sleep states
    3.3 Keep ripple events that happen during slow wave sleep epochs
"""

# Load packages
import numpy as np
import os
import h5py
import signal_processing as sp
import rhythm_detection as rd


pathway_win = r"D:\data_CA1\hdf5_NoHighGain"                   # directory where data files are stored
trial_files = [f for f in os.listdir(pathway_win)] # Load name of all rat files


for trial in trial_files:
    
    print(trial)
    file_path = os.path.join(pathway_win, trial)       # File path to that rat data

    # Load data
    with h5py.File(file_path, 'a') as f:
    

        # 1) Compute power envelope of all eeg channels
        lfp_group = f['lfp']
        lfp_sleep = lfp_group['postRun_sleep']
        
        lfp_channels = len(lfp_sleep) # number of LFP channels
        print(lfp_channels)

        # Filter EEG to ripple frequency range (100-250 Hz)
        eeg_sr = 4800   # lfp sampling rate, Hz
        lowcut = 100   
        highcut = 250

        filtered_data = []
        for n in range(lfp_channels): # Apply the bandpass filter to all LFP channels
            filtered_data.append(sp.bandpass_filter(lfp_sleep[n], lowcut, highcut, eeg_sr))
    
        # Compute power envelope of filtered lfp signal and respective standard deviation
        window_size_ms = 15 # compute power of root mean square (see rms_envelope function) over 15 ms
        bin_size = int((window_size_ms / 1000) * eeg_sr)  # Convert ms to samples
        if bin_size % 2 == 0:
            bin_size += 1  # Ensure window size is odd for symmetric smoothing
    
        rms_power_envelopes = []
        std_rms_envelopes = []

        for n in range(lfp_channels):
            rms_envelope = sp.compute_rms_envelope(filtered_data[n], bin_size)
            rms_power_envelopes.append(rms_envelope)
            std_rms_envelopes.append(np.std(rms_envelope))

        # 2) Select channels with rms envelope standard deviation
        best_channel_idx = np.argmax(std_rms_envelopes)
        best_channel_data = filtered_data[best_channel_idx]
        best_channel_rms = rms_power_envelopes[best_channel_idx]
        
        if 'SWR_lfp_ch' in f:
            del f['SWR_lfp_ch']
        
        f.create_dataset('SWR_lfp_ch', data=best_channel_idx)

        # 3) Detect ripple events on selected channel
        # 3.1 Find ripple events
        min_ripple_dur = 5 # Must cross threshold for at least this long, in ms
        max_ripple_dur = 200 # ms
     
        # Convert indices to time
        ripp_indices = rd.detect_ripples(best_channel_rms, min_ripple_dur, max_ripple_dur, eeg_sr)
        ripp_times = ripp_indices / eeg_sr

        # 3.2 Detect brain states
        speed_sleep = f['speed']['postRun_sleep']
        
        #---- 2.1 Get speed in cm/s ----#
        pos_sr = 50 # Hz
 
        state_run, state_sws, state_rem = rd.detect_brain_states(speed_sleep, pos_sr, lfp_sleep, eeg_sr)

        # 3.3 Detect coincident ripple events and slow-wave sleep events
        valid_ripp_times = rd.exclude_movement(ripp_times, state_sws, eeg_sr)

        if 'ripp_times' not in f:
            group = f.create_group('ripp_times')
        else:
            group = f['ripp_times']
    
        if 'postRun_sleep' in group:
            del group['postRun_sleep']
    
        group.create_dataset('postRun_sleep', data=valid_ripp_times)
