import os
import h5py
from collections import Counter

pathway_win = r"D:\data_CA1\hdf5_NoHighGain_new"       # directory where data files are stored
trial_files = [f for f in os.listdir(pathway_win)] # Load name of all rat files

for trial in trial_files:  
    print(trial)
    file_path = os.path.join(pathway_win, trial)

    with h5py.File(file_path,'r') as f:
        cell_type = f['cell_type'][:]
        cell_id = f['cell_id'][:]
        
        exc_cell_tetrodes = []
    for i in range(len(cell_type)):
        if cell_type[i] == b'exc':
            exc_cell_tetrodes.append(cell_id[i])
    
    counter = Counter(exc_cell_tetrodes)
    most_common = counter.most_common(1)[0]
    best_tetrode,_ = most_common
    print(best_tetrode)
    
    with h5py.File(file_path, 'a') as f:
        if 'theta_lfp_tetrode' in f:
            del f['theta_lfp_tetrode']
        
        f.create_dataset('theta_lfp_tetrode', data=best_tetrode)
