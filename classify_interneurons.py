# -*- coding: utf-8 -*-
"""
Classify interneurons
"""

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
        ripp_dur = np.diff(ripp_events, axis=1)
        min_dur = min(ripp_dur)

        cell_type = f['cell_type'][:]
        mask_inhib = cell_type==b'inhib'

        spk_times_sleep = f['spike_times'][2]
        spk_times_inhib = spk_times_sleep[mask_inhib]
        
        n_inhib = len(spk_times_inhib)
        
    if n_inhib==0:
        continue

    # Align spikes relative to start of ripple
    time_window_start = (-lag, min_dur/2)  # count spikes 100ms before beggining of ripple until min_dur/2
    bins_start = np.arange(time_window_start[0], time_window_start[1] + bin_size, bin_size)  # Histogram bins

    # Align spikes relative to end of ripple
    time_window_end = (-min_dur/2, lag)  # count spikes 500ms before to 500ms after start (or end) of ripple
    bins_end = np.arange(time_window_end[0], time_window_end[1] + bin_size, bin_size)  # Histogram bins
 
    bin_centers_start = bins_start[:-1] + bin_size / 2  # Compute bin centers for plotting
    bin_centers_end = bins_end[:-1] + bin_size / 2  # Compute bin centers for plotting

    # Create a figure with n rows and 2 columns for subplots
    fig = plt.figure(figsize=(5, 2 * n_inhib))  # Adjust height based on n_inhib
    gs = gridspec.GridSpec(n_inhib, 2, figure=fig, wspace=0.03, hspace=0.5)

    for n in range(n_inhib):
        print('Inhib neuron #', n)
        spk_times = spk_times_inhib[n]

        aligned_spikes_start = []  # Store spike times relative to ripple start
        aligned_spikes_end = []    # Store spike times relative to ripple end
    
        for start, end in ripp_events: # Loop through all ripple events and align spike times
    
            aligned_spikes_start.extend(spk_times[(spk_times >= start + time_window_start[0]) & (spk_times <= start + time_window_start[1])] - start) # time of spikes that happens shortly before or after beginning of ripple
            aligned_spikes_end.extend(spk_times[(spk_times >= end + time_window_end[0]) & (spk_times <= end + time_window_end[1])] - end) # time of spikes that happens shortly before or after end of ripple

        psth_start, _ = np.histogram(aligned_spikes_start, bins=bins_start)
        psth_end, _ = np.histogram(aligned_spikes_end, bins=bins_end)

        # Convert spike counts into mean firing rate (Hz)
        mean_firing_rate_start = psth_start / (len(ripp_events) * bin_size)
        mean_firing_rate_end = psth_end / (len(ripp_events) * bin_size)

        # Compute standard error of the mean (SEM)
        sem_start = np.sqrt(psth_start) / (len(ripp_events) * bin_size)  # Assuming Poisson variance: sqrt(N)
        sem_end = np.sqrt(psth_end) / (len(ripp_events) * bin_size)

        # Plot results
        # Create ax1 and ax2 for each row
        ax1 = fig.add_subplot(gs[n, 0])  # Left subplot
        ax2 = fig.add_subplot(gs[n, 1])  # Right subplot, sharing y-axis with ax1

        # Plot mean firing rate with error bars as a line plot
        ax1.plot(bin_centers_start, mean_firing_rate_start, color='b', linestyle='-', label="Aligned to Start")
        ax1.axvline(x=0, color='black', linestyle='--')
        ax1.fill_between(bin_centers_start, 
                     [m - s for m, s in zip(mean_firing_rate_start, sem_start)], 
                     [m + s for m, s in zip(mean_firing_rate_start, sem_start)], 
                     color='b', alpha=0.3)
        ax1.set_ylabel("Mean Firing Rate (Hz)")
      #  ax1.set_title(f"Aligned to Ripple Start (n={n+1})")

        ax2.plot(bin_centers_end, mean_firing_rate_end, color='b', linestyle='-', label="Aligned to End")
        ax2.axvline(x=0, color='black', linestyle='--')
        ax2.fill_between(bin_centers_end, 
                     [m - s for m, s in zip(mean_firing_rate_end, sem_end)], 
                     [m + s for m, s in zip(mean_firing_rate_end, sem_end)], 
                     color='b', alpha=0.3)
      #  ax2.set_title(f"Aligned to Ripple End (n={n+1})")
        ax2.set_xlabel("Time (s)")

        # Remove unnecessary axes for the broken effect
        ax1.spines.right.set_visible(False)
        ax2.spines.left.set_visible(False)
        ax2.tick_params(left=False)  # Remove left ticks on ax2
        ax2.set_yticks([]) 

        # Add broken axis effect (diagonal lines)
        d = 0.02  # Size of diagonal break
        kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False)

        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        kwargs = dict(transform=ax2.transAxes, color='black', clip_on=False)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # Show the final figure with all subplots
    fig.suptitle(f'{trial}')
    plt.savefig(f'{trial}.png')
    plt.show()