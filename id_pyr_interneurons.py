import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

"""
Pyramidal cells: bursting index < 25 ms and spike width > 300 ms
Inhibitory cells: bursting index > 25 ms and spike width < 300 ms

Bursting index (aka weighted period mean):
Use autocorrelation to compute bursting index as the weighted mean of the autocorrelation. 
Autocorrelation function shows  the correlation between a signal at two points in time 
separated by a fixed lag, evaluated over different values of lag.

Consider a max lag of 50 ms. theta cycle has a time scale of 100ms, so considering
50 ms ensures we don't include the theta cycle.

Note: bigger bursting index, less bursting.

"""

pathway_win = r"D:\data_CA1\pkl/"
trial_files = [f for f in os.listdir(pathway_win)]


for file_name in trial_files:
    
    print(file_name)
    file_path = os.path.join(pathway_win, file_name)

#---- Load relevant data ----#
    data = joblib.load(file_path)

    # Spike width, and spike times and mean rates during run trials
    mean_rate = data['mean_rate'][1]
    spk_times = data['spike_times'][1]
    spk_width = data['spike_width'][1] 
    spk_width = np.nan_to_num(spk_width, nan=0)# Replace nan entries with 0
    
    n_neurons = len(spk_times)
    print('Number of neurons:', n_neurons)

#---- Autocorrelation ----#
    bin_size = 0.001 # seconds
    max_lag = 0.05 # maximum time diff. (lag) in seconds
    bins = np.arange(0, max_lag + bin_size, bin_size) # Create bins for histogram
    invalid_neurons = []
    weighted_mean = np.zeros(n_neurons)
        
    
    for n in range(n_neurons):                          # Count occurrences within each bin
        # Only consider neurons that are spiking enough
        if spk_times[n].ndim == 0 or len(spk_times[n])<75:
            invalid_neurons.append(n)
            continue # skip to next neuron 'n'
            
        # Get spike train from spike times
        duration = spk_times[n][-1] # Total duration in s
        num_total_bins = int(duration / bin_size) + 1
        spike_train = np.zeros(num_total_bins)

        spike_indices = (spk_times[n] / bin_size).astype(int)
        spike_indices = np.clip(spike_indices, 0, num_total_bins - 1)  # Prevent out-of-bounds errors
        spike_train[spike_indices] = 1
            
        # Compute autocorrelation 
        len_spk_train = len(spike_train)
        max_slid = int(max_lag/bin_size)
        max_slid = min(max_slid, len_spk_train - 1)  # Ensure max_lag doesn't exceed the signal length

        # Initialize autocorrelation array
        autocorr = np.zeros(max_slid + 1)

        # Compute autocorrelation only for lags up to max_lag
        for lag in range(max_slid + 1):
            overlap = spike_train[:len_spk_train-lag] * spike_train[lag:]
            autocorr[lag] = np.sum(overlap)

        lags = np.arange(0, max_slid + 1)*bin_size

#---- Bursting index ----#
        # Compute the first moment (mean lag weighted by a autocorrelation, excluding t = 0)
        weighted_mean[n] = 0 if np.sum(autocorr[1:])==0 else np.average(lags[1:], weights=autocorr[1:])
        
        data['BI'] = weighted_mean
        
#---- ID pyr and inhib cells ----#
    # Clustering pyramidal cells and interneurons
    total_neurons = range(n_neurons)
    valid_neurons = np.setdiff1d(total_neurons, invalid_neurons) # Get valid neuron indices (complement)
    neuron_info = np.column_stack((total_neurons, weighted_mean*1000, spk_width*1000)) # bursting index and spike width in ms
    valid_neuron_info = neuron_info[valid_neurons]
    
    # Extract bursting index and spike width as separate arrays
    burst_index = valid_neuron_info[:, 1]
    spike_width = valid_neuron_info[:, 2]

    # Get masks of pyramidal cell, inhibitory cells, and non-identified cells
    pyr_mask = (burst_index <= 25) & (spike_width >= 300)
    inhib_mask = (burst_index > 25) & (spike_width < 300)
    remaining_mask = ~pyr_mask & ~inhib_mask  # Everything else

    # Apply masks to get filtered neuron data (index, bursting index, and spike width)
    pyr_cells = valid_neuron_info[pyr_mask]
    inhib_cells = valid_neuron_info[inhib_mask]
    remaining_cells = valid_neuron_info[remaining_mask]
    
#---- Save results in joblib file ----#
    # Initialize cell_type array with default value 'none'
    cell_type = np.full(n_neurons, 'none', dtype='<U5')  # '<U5' specifies a string array of length 5

    # Assign labels based on the boolean masks
    cell_type[valid_neurons[pyr_mask]] = 'pyr'      # Mark pyramidal cells
    cell_type[valid_neurons[inhib_mask]] = 'inhib'  # Mark inhibitory cells

    # Add the cell_type array to the data dictionary
    data['cell_type'] = cell_type

    # Save the updated data dictionary back to the joblib file
    joblib.dump(data, file_path)
