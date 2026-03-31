'''
Parameters neural dynamics:
    AA and BC from hippocampome
    pyr cells from Skinner paper. 
    
    
From Frankland (Ramsaran): weak lateral inhibition

    
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import cauchy

# Simulation parameters
tmin = 0.0
tmax = 100.0  # ms
dt = 0.01  # ms
steps = int((tmax - tmin) / dt)
t = np.linspace(tmin, tmax, steps)

# Network size
N_e = 12000  # Excitatory neurons
N_i = 200    # Inhibitory neurons


# Neuron parameters
parameters_set = {
    # PV+BC
    'BC': {'k': 1.19, 'Vr': -57.63, 'Vth': -35.53, 'alpha': 0.005, 'beta': 0.22, 'C': 114, 'ujump': 2, 'Vreset': -48.7, 'Vpeak':21.72 },
     
    # AA
    'AA': {'k': 9.83, 'Vr': -66.76, 'Vth': -65.78, 'alpha': 0.002, 'beta': 14.47, 'C': 234, 'ujump': 26, 'Vreset': -62.22, 'Vpeak':3.18 },
    
    # Pyramidal cells, Ferguson (2014)
   'pyr': {'k': 3.3, 'Vr': -61.8, 'Vth': -57, 'alpha': 0.0012, 'beta': 3, 'C': 115, 'ujump': 10, 'Vreset': -65.8, 'Vpeak':22.6 }

    # Pyramidal cells, hippocampome
#    'pyr': {'k': 1.56, 'Vr': -69.36, 'Vth': -53.22, 'alpha': 0, 'beta': -17.25, 'C': 334, 'ujump': 16, 'Vreset': -60.22, 'Vpeak':25.46 }
}

I = parameters_set['BC']
E = parameters_set['pyr']

# Gaussian distributuion for background current
Delta=1
etabar=0
eta_i = cauchy.rvs(loc=etabar, scale=Delta, size=N_i)
eta_e = cauchy.rvs(loc=etabar, scale=Delta, size=N_e)

# Synaptic parameters
# Excitatory
E_e = 0.0         # mV (excitatory reversal)
E_i = -68.0       # mV (inhibitory reversal)

# Inhibitory
E_I_i = -75.0     # mV (inhibitory reversal for I cells)

# E to E (AMPA on pyramidal cells)
tau_d_ee = 1.8   # ms
g_peak_ee = 0.9  # nS

# I to E (GABA on pyramidal cells) 
tau_d_ei = 2.0   # ms. post-pre
g_peak_ei = 9.0  # nS

# E to I (AMPA on interneurons)
tau_d_ie = 1.2   # ms
g_peak_ie = 3.0  # nS

# I to I (GABA on interneurons)
tau_d_ii = 1.2   # ms
g_peak_ii = 5.0  # nS


# Sharp wave parameters
peak_time = 50.0   # ms
temporal_sigma = 10.0  # ms
sharp_wave_amplitude = 60.0  # nS
#n_excited = 400
n_excited = 800
amplitude_CV = 0.5

# Initialize state variables 
V_e = np.zeros((N_e,len(t)))
V_e[:,0] = np.random.normal(E['Vr'], 0.1, N_e)
u_e = np.zeros((N_e,len(t)))

V_i = np.zeros((N_i,len(t)))
V_i[:,0] = np.random.normal(I['Vr'], 0.1, N_i)
u_i = np.zeros((N_i,len(t)))

# Synaptic conductance variables (double exponential)
g_ee = np.zeros(N_e)  # AMPA on E cells
g_ei = np.zeros(N_e)  # GABA on E cells

g_ie = np.zeros(N_i)  # AMPA on I cells
g_ii = np.zeros(N_i)  # GABA on I cells

# External drive conductance
g_ext = np.zeros(N_e)

# Spike recording
spikes_e = np.zeros((N_e, steps), dtype=bool)
spikes_i = np.zeros((N_i, steps), dtype=bool)

# Connectivity matrices
np.random.seed(1)
p_ee = 0.01 # connectivity probability. post_pre
p_ii = 0.2 
p_ei = 0.17
p_ie = 0.17

conn_ee = (np.random.rand(N_e, N_e) < p_ee).astype(float)
conn_ii = (np.random.rand(N_i, N_i) < p_ii).astype(float)
conn_ei = (np.random.rand(N_e, N_i) < p_ei).astype(float)
conn_ie = (np.random.rand(N_i, N_e) < p_ie).astype(float)

# Generate sharp wave profile (Gaussian)
def gaussian_form(t, sigma, mu):
    return (1.0 / (np.sqrt(2.0 * np.pi * sigma**2))) * np.exp(-(t - mu)**2 / (2.0 * sigma**2))

# Select excited neurons
shuffled_E_cells = np.random.permutation(N_e)[:n_excited]
active_neurons = np.zeros(N_e)
active_neurons[shuffled_E_cells] = 1.0

# Generate amplitude distribution
amplitude = np.random.normal(sharp_wave_amplitude, amplitude_CV * sharp_wave_amplitude, N_e)
amplitude[amplitude <= 0] = 0
amplitude *= active_neurons

# Pre-compute sharp wave time course
normalization = 1.0 / np.sqrt(2.0 * np.pi * temporal_sigma**2)
g_chr_array = np.zeros((steps, N_e))
for i, time in enumerate(t):
    g_chr_array[i, :] = amplitude * (1.0 / normalization) * gaussian_form(time, temporal_sigma, peak_time)
 
# Synaptic delay (1 ms)
#delay_steps = int(1.0 / dt)
delay_steps = int(1.8 / dt)

print("Starting simulation...")

# Main simulation loop
i=0 
while i < len(t)-(1+delay_steps): 
    
    # --- Excitatory neurons ---
    # External drive
    g_ext = g_chr_array[i, :]

    # Update synaptic conductances (decay)
    input_ee = np.dot(conn_ee, spikes_e[:, i].astype(float))
    g_ee = g_ee + dt * (-g_ee / tau_d_ee + g_peak_ee * input_ee/dt)
    input_ei = np.dot(conn_ei, spikes_i[:, i].astype(float))  
    g_ei = g_ei + dt * (-g_ei / tau_d_ei + g_peak_ei * input_ei/dt)
      
    # Total synaptic current for each E neuron (vectorized)
    I_syn_e = (g_ee + g_ext) * (E_e - V_e[:,i]) + g_ei * (E_i - V_e[:,i])
    
    # Voltage update (Euler method)
    dV_e = (E['k']*(V_e[:,i] - E['Vr'])*(V_e[:,i] - E['Vth']) - u_e[:,i] + I_syn_e + eta_e)/E['C']
    du_e = E['alpha']*(E['beta']*(V_e[:,i] - E['Vr']) - u_e[:,i])    
    V_e[:,i+1] = V_e[:,i] + dt * dV_e
    u_e[:,i+1] = u_e[:,i] + dt* du_e
    
    # Check for spikes
    spike_mask_e = V_e[:,i+1] >= E['Vpeak']
    V_e[spike_mask_e,i+1] = E['Vreset']
    u_e[spike_mask_e,i+1]=u_e[spike_mask_e,i] + E['ujump']    
    spikes_e[:, i+delay_steps] = spike_mask_e
  
    
    # --- Inhibitory neurons ---  
    # Update synaptic conductances (decay)
    input_ie = np.dot(conn_ie, spikes_e[:, i].astype(float))
    g_ie = g_ie + dt * (-g_ie / tau_d_ie + g_peak_ie * input_ie/dt)
    input_ii = np.dot(conn_ii, spikes_i[:, i].astype(float))
    g_ii = g_ii + dt * (-g_ii / tau_d_ii + g_peak_ii * input_ii/dt)  
    
    # Total synaptic current for each i neuron
    I_syn_i = g_ie * (E_e - V_i[:,i]) + g_ii * (E_I_i - V_i[:,i])
    
    # Voltage update
    dV_i = (I['k']*(V_i[:,i] - I['Vr'])*(V_i[:,i] - I['Vth']) - u_i[:,i] + I_syn_i + eta_i)/I['C']
    du_i = I['alpha']*(I['beta']*(V_i[:,i] - I['Vr']) - u_i[:,i])    
    V_i[:,i+1] = V_i[:,i] + dt * dV_i
    u_i[:,i+1] = u_i[:,i] + dt* du_i    
     
    # Check for spikes
    spike_mask_i = V_i[:,i+1] >= I['Vpeak']
    V_i[spike_mask_i, i+1] = I['Vreset']
    u_i[spike_mask_i,i+1]=u_i[spike_mask_i,i] + I['ujump']
    spikes_i[:, i+delay_steps] = spike_mask_i

    i+=1
      
print("Simulation complete!")

# --- Analysis ---
# Calculate population rates with Gaussian smoothing
resolution = 0.5  # ms 
sigma_smooth = resolution / dt / 2.355  # Convert to std dev in time steps

def smooth_rate(spike_array, N_neurons):
    """Calculate smoothed population firing rate"""
    instantaneous_rate = np.sum(spike_array, axis=0) / dt  # spikes/ms
    rate = instantaneous_rate / N_neurons * 1000  # Convert to Hz
    # Gaussian smoothing
    window_size = int(6 * sigma_smooth)
    if window_size % 2 == 0:
        window_size += 1
    x = np.arange(window_size) - window_size // 2
    gaussian = np.exp(-x**2 / (2 * sigma_smooth**2))
    gaussian /= gaussian.sum()
    rate_smooth = np.convolve(rate, gaussian, mode='same')
    return rate_smooth

population_rate_e = smooth_rate(spikes_e, N_e)
population_rate_i = smooth_rate(spikes_i, N_i)

# Calculate frequency spectrum
t_min_idx = int(0 / dt)
t_max_idx = int(100 / dt)
sampling_freq = 1000.0 / dt

freq_e, spec_e = signal.welch(population_rate_e[t_min_idx:t_max_idx], 
                               sampling_freq, nperseg=len(t)//5, 
                               nfft=5*len(t), window='hann', scaling='density')
freq_max_e = freq_e[np.argmax(spec_e)]

freq_i, spec_i = signal.welch(population_rate_i[t_min_idx:t_max_idx], 
                               sampling_freq, nperseg=len(t)//5, 
                               nfft=5*len(t), window='hann', scaling='density')
freq_max_i = freq_i[np.argmax(spec_i)]

# Spike statistics
number_of_spikes_e = np.sum(spikes_e, axis=1)
number_of_spikes_i = np.sum(spikes_i, axis=1)

mean_spikes_e_nonsilent = np.mean(number_of_spikes_e[number_of_spikes_e > 0])
mean_spikes_i_nonsilent = np.mean(number_of_spikes_i[number_of_spikes_i > 0])

silent_percentage_e = np.sum(number_of_spikes_e == 0) / N_e
silent_percentage_i = np.sum(number_of_spikes_i == 0) / N_i

print(f"\nResults:")
print(f"Mean spikes (non-silent E/I): {mean_spikes_e_nonsilent:.1f} / {mean_spikes_i_nonsilent:.1f}")
print(f"Silent E neurons: {np.sum(number_of_spikes_e == 0)} ({silent_percentage_e:.2%})")
print(f"Silent I neurons: {np.sum(number_of_spikes_i == 0)} ({silent_percentage_i:.2%})")
print(f"Firing E neurons: {np.sum(number_of_spikes_e > 0)}, Excited: {len(shuffled_E_cells)}")
print(f"Peak frequencies (E/I): {freq_max_e:.1f} Hz / {freq_max_i:.1f} Hz")


# --- Plotting ---
fig = plt.figure(figsize=(7.5, 5.0), dpi=100)

# External drive
ax0 = fig.add_subplot(411)
for j in shuffled_E_cells[:100]:  # Plot subset for visibility
    ax0.plot(t, g_chr_array[:, j], '-', linewidth=0.5, alpha=0.8, color='k')
ax0.axvline(x=50, color='r', linestyle='--', linewidth=2.0)
ax0.axhline(y=sharp_wave_amplitude, color='r', linestyle='--', linewidth=1.0)
ax0.set_ylabel(r'$g_{ext}$ [nS]', fontsize=12)
ax0.set_xlim(0, 100)
ax0.set_xticks(np.arange(0, 101, 10))
ax0.set_xticklabels([])

# E cell raster
ax1 = fig.add_subplot(412)
spike_indices_e, spike_time_indices_e = np.where(spikes_e)
spike_times_e = t[spike_time_indices_e]

# Only plot spikes within valid time range
ax1.plot(spike_times_e, spike_indices_e, 'r.', markersize=1.0)
ax1.axvline(x=50, color='k', linestyle='--', linewidth=1.0)
ax1.set_ylabel('E cell', fontsize=12)
ax1.set_xlim(0, 100)
ax1.set_xticks(np.arange(0, 101, 10))
ax1.set_xticklabels([])
ax1.set_title(f"Avg spikes E/I: {mean_spikes_e_nonsilent:.1f}/{mean_spikes_i_nonsilent:.1f}, " +
              f"Active fraction E(I): {1-silent_percentage_e:.3f}({1-silent_percentage_i:.3f})", fontsize=10)

# I cell raster
ax2 = fig.add_subplot(413)
spike_indices_i, spike_time_indices_i = np.where(spikes_i)
spike_times_i = t[spike_time_indices_i]

# Only plot spikes within valid time range
ax2.plot(spike_times_i, spike_indices_i, 'b.', markersize=2.0)
ax2.axvline(x=50, color='k', linestyle='--', linewidth=1.0)
ax2.set_ylabel('I cell', fontsize=12)
ax2.set_xlim(0, 100)
ax2.set_xticks(np.arange(0, 101, 10))
ax2.set_xticklabels([])

# Population rates
ax3 = fig.add_subplot(414)
rate_e = N_e * population_rate_e / 1000.0  # Convert to kHz
rate_i = N_i * population_rate_i / 1000.0
ax3.plot(t, rate_e, '-r', label='E')
ax3.plot(t, rate_i, '-b', label='I')
ax3.axvline(x=50, color='k', linestyle='--', linewidth=1.0)
ax3.set_ylabel(r'$A_E, A_I$ [kHz]', fontsize=12)
ax3.set_xlabel('t (ms)', fontsize=12)
ax3.set_xlim(0, 100)
ax3.set_xticks(np.arange(0, 101, 10))
ax3.text(2, 0.7*np.max(rate_i), f'$f_E$={freq_max_e:.1f} Hz', color='r', fontsize=10)
ax3.text(2, 0.3*np.max(rate_i), f'$f_I$={freq_max_i:.1f} Hz', color='b', fontsize=10)

plt.tight_layout()
plt.show()

# Spike count histograms
fig2 = plt.figure(figsize=(10, 3), dpi=100)

ax6 = fig2.add_subplot(121)
bins_e = np.arange(0, np.max(number_of_spikes_e) + 2)
ax6.hist(number_of_spikes_e, bins=bins_e, histtype='step', color='r', linewidth=2.0, log=True)
ax6.axvline(x=mean_spikes_e_nonsilent, color='r', linewidth=2.0, linestyle='--')
ax6.set_ylabel('No. of E cells', fontsize=12)
ax6.set_xlabel('Spikes', fontsize=12)
ax6.set_xticks(np.arange(0, np.max(bins_e), step=2))

ax7 = fig2.add_subplot(122)
bins_i = np.arange(0, np.max(number_of_spikes_i) + 2)
ax7.hist(number_of_spikes_i, bins=bins_i, histtype='step', color='b', linewidth=2.0, log=True)
ax7.axvline(x=mean_spikes_i_nonsilent, color='b', linewidth=2.0, linestyle='--')
ax7.set_ylabel('No. of I cells', fontsize=12)
ax7.set_xlabel('Spikes', fontsize=12)
ax7.set_xticks(np.arange(0, np.max(bins_i), step=2))

plt.tight_layout()
plt.show()