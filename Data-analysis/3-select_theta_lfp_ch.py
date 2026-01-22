import os
import h5py
from collections import Counter
import numpy as np
from scipy.signal import welch

pathway_win = r"D:\data_CA1\hdf5_NoHighGain_new"       # directory where data files are stored
trial_files = [f for f in os.listdir(pathway_win)] # Load name of all rat files

eeg_sr = 4800
theta_range = (5,11)

for trial in trial_files:  
    print(trial)
    file_path = os.path.join(pathway_win, trial)

    with h5py.File(file_path,'r') as f:
        cell_type = f['cell_type'][:]
        cell_id = f['cell_id'][:]
        all_tetrodes = f['tetrode_id'][:]  # index of tetrode in lfp ch
        
        exc_cell_tetrodes = []
        
        for i in range(len(cell_type)):
            if cell_type[i] == b'exc':
                exc_cell_tetrodes.append(cell_id[i])
    
        counter = Counter(exc_cell_tetrodes)
        most_common = counter.most_common(1)[0]
        selected_theta_tetrode,_ = most_common
     
        lfp_ch_idx = np.where(all_tetrodes == selected_theta_tetrode)[0] # lfp ch from selected tetrode    
        lfp_chosen_tetrode = f['lfp']['run'][lfp_ch_idx]

        # Use lfp channel with highest SNR
        n_channels = len(lfp_ch_idx)
        SNR_theta = np.zeros(n_channels)
    
        for ch in range(n_channels):
            lfp = lfp_chosen_tetrode[ch]

            # Compute power spectral density (PSD)
            freqs, psd = welch(lfp, fs=eeg_sr, nperseg=int(eeg_sr), scaling='density')

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
    best_theta_ch_idx = lfp_ch_idx[np.argmax(SNR_theta)]
    
    with h5py.File(file_path, 'a') as f:
        if 'theta_lfp_ch' in f:
            del f['theta_lfp_ch']
        
        f.create_dataset('theta_lfp_ch', data=best_theta_ch_idx)
