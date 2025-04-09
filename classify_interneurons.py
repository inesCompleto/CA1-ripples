# -*- coding: utf-8 -*-
"""
Classify interneurons
"""

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

plt.rcParams['axes.titley'] = 0.99

# Load times of ripple events
pathway_win = r"D:\data_CA1\hdf5"                   # directory where data files are stored
trial_files = [f for f in os.listdir(pathway_win)] # Load name of all rat files

bin_size = 0.005 # sec
lag = 0.2

for trial in trial_files:
    
    print(trial)

    file_path = os.path.join(pathway_win, trial)

    with h5py.File(file_path,'r') as f:

        ripp_group = f['ripp_times']
        ripp_events = ripp_group['postRun_sleep'][:]

        # find shortest ripple. use this to clip other ripples
        ripp_dur = np.squeeze(np.diff(ripp_events, axis=1))

        cell_type = f['cell_type'][:]
        mask_inhib = cell_type==b'inhib'

        spk_times_sleep = f['spike_times'][2]
        spk_times_inhib = spk_times_sleep[mask_inhib]
        
        n_inhib = len(spk_times_inhib)
        
    if n_inhib==0:
        continue

 
    bin_size = 0.05 # sec
    lag = 0.04

    # Bins for histogram
    bins_pre = np.arange(-1, 0, bin_size)  # Pre-ripple range from -1 to 0
    bins_post = np.arange(1, 2 + bin_size, bin_size)  # Post-ripple range from 1 to 2
    bins_ripp = np.linspace(0,1, len(bins_pre)+1)
    bins_all = np.concatenate([bins_pre, bins_ripp[1:], bins_post[1:]])  # avoid duplicate edges

    # Create a figure with n rows and 2 columns for subplots
    fig = plt.figure(figsize=(12, 5 * n_inhib))
    outer_grid = GridSpec(n_inhib, 2, figure=fig, wspace=0.3)

    for n in range(n_inhib):
        print('Inhib neuron #', n)
        spk_times = spk_times_inhib[n]

        pre_ripp_spks, ripp_spks, post_ripp_spks = [], [], []  # Store spike times before, during and after ripples
        spks_scatter = [] # store spike times (non-normalized) for scatter plot
    
        for j, (start, end) in enumerate(ripp_events): # Loop through all ripple events and align spike times

            duration = ripp_dur[j] 

            # Pre-ripple spikes (normalize to [-1, 0])
            pre_ripp_spks.extend(((spk_times[(spk_times >= start - lag) & (spk_times <= start)] - (start - lag)) / lag) - 1)

            # During ripple (normalized to [0, 1])
            ripp_spks.extend((spk_times[(spk_times >= start) & (spk_times <= end)] - start) / duration)

            # Post-ripple spikes (normalize to [1, 2])
            post_ripp_spks.extend(((spk_times[(spk_times >= end) & (spk_times <= end + lag)] - end) / lag) + 1)

            spks_to_plot = spk_times[(spk_times>= start-lag) & (spk_times<=end+lag)] - start
            spks_scatter.append((spks_to_plot, j)) # track ripple event index for y-axis

        all_spikes = np.concatenate([pre_ripp_spks, ripp_spks, post_ripp_spks])

        psth, bins = np.histogram(all_spikes, bins=bins_all)

        # Convert spike counts into mean firing rate (Hz)
        mean_fr = psth / (len(ripp_events) * bin_size)

        # Compute standard error of the mean (SEM)
        sem = np.sqrt(psth) / (len(ripp_events) * bin_size)  # Assuming Poisson variance: sqrt(N)

        # Plot results
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # --- Left subplot: Mean firing rate with SEM (PSTH) ---
        ax_psth = fig.add_subplot(outer_grid[n, 0])
       
        ax_psth.plot(bin_centers, mean_fr, color='b')
        ax_psth.fill_between(bin_centers, mean_fr - sem, mean_fr + sem, color='b', alpha=0.3)
        ax_psth.axvline(0, linestyle='--', color='gray', label='Ripple Start')
        ax_psth.axvline(1, linestyle='--', color='gray', label='Ripple End')
        ax_psth.set_title(f'Neuron {n} - PSTH')
        ax_psth.set_ylabel('Firing Rate (Hz)')
        ax_psth.set_xlabel('Time (normalized)')


        # --- Right subplot: Raster plot ---
        inner_grid = GridSpecFromSubplotSpec(len(spks_scatter), 1, subplot_spec=outer_grid[n, 1], hspace=0.2)
        for spk_times_event, j in spks_scatter:
            ax_raster = fig.add_subplot(inner_grid[j])
            y = np.ones_like(spk_times_event) * j
            ax_raster.scatter(spk_times_event, y, marker='.', color='b', s=10)
            ax_raster.axvspan(0, ripp_dur[j], color='gray', alpha=0.3)
            ax_raster.set_xlim([-lag, lag+ripp_dur[j]])
            ax_raster.set_yticks([])
            if j==len(spks_scatter)-1:
                ax_raster.set_xticks([-lag, lag+ripp_dur[j]], ['-lag','lag'])
                ax_raster.spines[['right', 'top', 'left']].set_visible(False)
            else:
                ax_raster.set_xticks([])
                ax_raster.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    # Show the final figure with all subplots
    fig.suptitle(f'Mean firing rate, {trial}')
    plt.savefig(f'{trial}.png')
    plt.show()