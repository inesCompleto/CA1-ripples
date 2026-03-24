import math 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
from scipy.signal import butter,filtfilt

# ----------  PARAMETERS  ---------- #
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


#I_ext = 200  #external current
Delta=10
etabar=0

# Network size
N_e = 6000  # Excitatory neurons
N_i = 6000    # Inhibitory neurons

# Synaptic parameters
# Excitatory
E_e = 0.0         # mV (excitatory reversal)
E_i = -68.0       # mV (inhibitory reversal)

# Inhibitory
E_I_i = -75.0     # mV (inhibitory reversal for I cells)

# E to E (AMPA on pyramidal cells)
tau_d_ee = 1.8   # ms
g_peak_ee = 0.8  # nS

# I to E (GABA on pyramidal cells) 
tau_d_ei = 2.0   # ms. post-pre
g_peak_ei = 8.0  # nS

# E to I (AMPA on interneurons)
tau_d_ie = 1.2   # ms
g_peak_ie = 2.0  # nS

# I to I (GABA on interneurons)
tau_d_ii = 1.2   # ms
g_peak_ii = 4.0  # nS

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


# ----------  VARIABLES  ---------- #
tmin = 0.0
tmax = 100 # msec
dt=0.01
steps = int((tmax - tmin)/dt) 
t = np.linspace(tmin, tmax, steps)
I_ext = np.zeros(len(t))

# Fast-spiking interneuron
aI = I['k']
bI = -I['k']*(I['Vr'] + I['Vth'])
cI = I['k']*I['Vr']*I['Vth']
tauI=I['C']
rI = np.zeros(len(t))
v_mean_I = np.zeros(len(t))
v_mean_I[0]=-55
u_mean_I = np.zeros(len(t))
gII_mean=0
gIO_mean=0
gIE_mean=0


V_I = np.zeros((N_i,len(t)))
V_I[:,0]=-55
u_I=np.zeros((N_i,len(t)))
V_av_I = np.zeros(len(t))
count_I=np.zeros(len(t))


x_I=[]
y_I=[]

# Pyramidal Cell
V_E = np.zeros((N_e,len(t)))
V_E[:,0]=-64
u_E=np.zeros((N_e,len(t)))
V_av_E = np.zeros(len(t))

rE = np.zeros(len(t))
u_mean_E=np.zeros(len(t))
v_mean_E = np.zeros(len(t))
v_mean_E[0]=-64
gEE_mean=0
gEI_mean=0

count_E = np.zeros(len(t))

aE = E['k']
bE = -E['k']*(E['Vr'] + E['Vth'])
cE = E['k']*E['Vr']*E['Vth']
tauE=E['C']


# Synaptic conductance variables (double exponential)
g_ee = np.zeros(N_e)  # AMPA on E cells
g_ei = np.zeros(N_e)  # GABA on E cells

g_ie = np.zeros(N_i)  # AMPA on I cells
g_ii = np.zeros(N_i)  # GABA on I cells

delay_steps = int(1.8 / dt)

x_E=[]
y_E=[]


# Gaussian distributuion for background current
eta_i = cauchy.rvs(loc=etabar, scale=Delta, size=N_i)
eta_e = cauchy.rvs(loc=etabar, scale=Delta, size=N_e)

# Spike recording
spikes_e = np.zeros((N_e, steps), dtype=bool)
spikes_i = np.zeros((N_i, steps), dtype=bool)

t1=100
t2=200
t3=300
t4=400


i=0



while i < len(t)-(1+delay_steps):
    I_ext[i] = 170 + 120*np.heaviside(i*dt-t1,1)*np.heaviside(t2-i*dt,1) \
                   -  60*np.heaviside(i*dt-t3,1)*np.heaviside(t4-i*dt,1)

    # ---------------------------------------------------------------
    # MEAN FIELD EQUATIONS
    # ---------------------------------------------------------------

    # Fast-spiking interneurons
    # Mean field conductances (population mean, consistent with sparse network)
    gII_mean = gII_mean + dt*(-gII_mean/tau_d_ii + g_peak_ii * p_ii * rI[i-delay_steps] )
    gIE_mean = gIE_mean + dt*(-gIE_mean/tau_d_ie +  g_peak_ie * p_ie * rE[i-delay_steps])  
    
    rI[i+1] = rI[i] + dt*((bI - gII_mean - gIE_mean)*rI[i]
                           + 2*aI*rI[i]*v_mean_I[i]
                           + Delta*aI/(math.pi*tauI))/tauI

    v_mean_I[i+1] = v_mean_I[i] + dt*((-(math.pi*rI[i]*tauI)**2)/aI
                                       + aI*v_mean_I[i]**2
                                       + (bI - gII_mean - gIE_mean)*v_mean_I[i]
                                       + cI + gII_mean*E_I_i
                                       - u_mean_I[i] + I_ext[i] + etabar)/tauI

    u_mean_I[i+1] = u_mean_I[i] + dt*(I['alpha']*(I['beta']*(v_mean_I[i]-I['Vr'])
                                       - u_mean_I[i]) + I['ujump']*rI[i])


    # Pyramidal cells
    # Mean field conductances (population mean, consistent with sparse network)
    gEE_mean = gEE_mean + dt*(-gEE_mean/tau_d_ee + g_peak_ee * p_ee * rE[i-delay_steps])
    gEI_mean = gEI_mean + dt*(-gEI_mean/tau_d_ei + g_peak_ei * p_ei * rI[i-delay_steps])   
    
    rE[i+1] = rE[i] + dt*((bE - gEE_mean - gEI_mean)*rE[i]
                           + 2*aE*rE[i]*v_mean_E[i]
                           + Delta*aE/(math.pi*tauE))/tauE

    v_mean_E[i+1] = v_mean_E[i] + dt*((-(math.pi*rE[i]*tauE)**2)/aE
                                       + aE*v_mean_E[i]**2
                                       + (bE - gEE_mean - gEI_mean)*v_mean_E[i]
                                       + cE - u_mean_E[i]
                                       + gEI_mean*E_i + I_ext[i] + etabar)/tauE

    u_mean_E[i+1] = u_mean_E[i] + dt*(E['alpha']*(E['beta']*(v_mean_E[i]-E['Vr'])
                                       - u_mean_E[i]) + E['ujump']*rE[i])


    # ---------------------------------------------------------------
    # QIF / SPIKING NETWORK EQUATIONS
    # ---------------------------------------------------------------

    # --- Inhibitory neurons ---
    # Update inhibitory conductances from previous spikes
    input_ie = np.dot(conn_ie, spikes_e[:, i].astype(float))
    g_ie     = g_ie + dt*(-g_ie/tau_d_ie + g_peak_ie * input_ie /( N_e*dt) )
    input_ii = np.dot(conn_ii, spikes_i[:, i].astype(float))
    g_ii     = g_ii + dt*(-g_ii/tau_d_ii + g_peak_ii * input_ii /( N_i*dt)) 
    
    I_synI = g_ie * (E_e - V_I[:,i]) + g_ii * (E_I_i - V_I[:,i])

    V_I[:,i+1] = V_I[:,i] + dt*(aI*V_I[:,i]**2 + bI*V_I[:,i] + cI
                                 - u_I[:,i] + I_ext[i] + I_synI + eta_i)/tauI
    u_I[:,i+1] = u_I[:,i] + dt*(I['alpha']*(I['beta']*(V_I[:,i]-I['Vr'])
                                 - u_I[:,i]))
    
    # Spike detection — inhibitory
    spike_mask_i          = V_I[:,i+1] >= 1000
    V_I[spike_mask_i,i+1] = -1000
    u_I[spike_mask_i,i+1] = u_I[spike_mask_i,i] + I['ujump']
    count_I[i+1]           = np.sum(spike_mask_i) / (N_i * dt)
    spikes_i[:, i+delay_steps]         = spike_mask_i

    # --- Excitatory neurons ---
    # Update excitatory conductances from previous spikes
    input_ee = np.dot(conn_ee, spikes_e[:, i].astype(float))
    g_ee     = g_ee + dt*(-g_ee/tau_d_ee + g_peak_ee * input_ee / (N_e*dt))
    input_ei = np.dot(conn_ei, spikes_i[:, i].astype(float))
    g_ei     = g_ei + dt*(-g_ei/tau_d_ei + g_peak_ei * input_ei / (N_i*dt))
    
    I_synE = g_ee * (E_e - V_E[:,i]) + g_ei * (E_i - V_E[:,i])

    V_E[:,i+1] = V_E[:,i] + dt*(aE*V_E[:,i]**2 + bE*V_E[:,i] + cE
                                 - u_E[:,i] + I_ext[i] + I_synE + eta_e)/tauE
    u_E[:,i+1] = u_E[:,i] + dt*(E['alpha']*(E['beta']*(V_E[:,i]-E['Vr'])
                                 - u_E[:,i]))

    # Spike detection — excitatory
    spike_mask_e          = V_E[:,i+1] >= 1000
    V_E[spike_mask_e,i+1] = -1000
    u_E[spike_mask_e,i+1] = u_E[spike_mask_e,i] + E['ujump']
    count_E[i+1]           = np.sum(spike_mask_e) / (N_e * dt)
    spikes_e[:, i+delay_steps]         = spike_mask_e

    i += 1



b,a = butter(2, 0.09, 'low')
count_I = filtfilt(b,a,count_I)
count_E = filtfilt(b,a,count_E)

plt.figure()
plt.subplot(5,1,1)
plt.plot(t, I_ext, color='k')
plt.ylabel('$I_{ext}$',  fontsize=15)
plt.tick_params(labelbottom=False) 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False) 


plt.subplot(5,1,2)
plt.scatter(x_I,y_I, color='k', marker=".", s=0.7)
plt.scatter(x_E,y_E, color='k', marker=".", s=0.7)
plt.ylabel('Neuron #',  fontsize=17)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tick_params(labelbottom=False) 


plt.subplot(5,1,4)
plt.plot(t,count_I, color='#B0B8E8', label='Full net')
plt.plot(t,rI, color='#303F9F', label='Reduced')
plt.ylabel('$r_I$(t)',  fontsize=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tick_params(labelbottom=False) 
plt.legend(loc=1, frameon=False)

plt.subplot(5,1,5)
plt.plot(t,count_E, color='#ECA577', label='Full net')
plt.plot(t,rE, color='#9D2503', label='Reduced')
plt.ylabel('$r_E$(t)',  fontsize=15)
plt.xlabel('time (msec)',  fontsize=15, x=1, y=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.legend(loc=1, frameon=False)

plt.show()