'''
Parameters neural dynamics:
    AA and BC from hippocampome
    pyr cells from Skinner paper. 
    
'''

import math 
import numpy as np
import matplotlib.pyplot as plt


tmin = 0.0
tmax = 1000 # msec
dt = 0.01
steps = ((tmax - tmin)/dt) // 1 
t = np.linspace(tmin, tmax, int(steps))


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

BC = parameters_set['BC']
AA = parameters_set['AA']
E = parameters_set['pyr']


I_ext = 500


# PV+BC
N_BC=1
V_BC = np.zeros((N_BC,len(t)))
V_BC[:,0] = -55
u_BC = np.zeros((N_BC,len(t)))

# AA
N_AA=1
V_AA = np.zeros((N_AA,len(t)))
V_AA[:,0] = -55
u_AA = np.zeros((N_AA,len(t)))

# pyr cell
N_E=1
V_E = np.zeros((N_E,len(t)))
V_E[:,0] = -60
u_E = np.zeros((N_E,len(t)))



i=0

while i < len(t)-1: 
    
    # PV+BC
    V_BC[:,i+1] = V_BC[:,i] + dt*(BC['k']*(V_BC[:,i] - BC['Vr'])*(V_BC[:,i] - BC['Vth']) - u_BC[:,i] + I_ext)/BC['C']
    u_BC[:,i+1] = u_BC[:,i] + dt*(BC['alpha']*(BC['beta']*(V_BC[:,i] - BC['Vr']) - u_BC[:,i]) )
    
    for j in range(N_BC):
        if V_BC[j,i+1]>=BC['Vpeak']:
            V_BC[j,i+1]=BC['Vreset']
            u_BC[j,i+1] = u_BC[j,i] +BC['ujump']    
    
    # AA
    V_AA[:,i+1] = V_AA[:,i] + dt*(AA['k']*(V_AA[:,i] - AA['Vr'])*(V_AA[:,i] - AA['Vth']) - u_AA[:,i] + I_ext)/AA['C']
    u_AA[:,i+1] = u_AA[:,i] + dt*(AA['alpha']*(AA['beta']*(V_AA[:,i] - AA['Vr']) - u_AA[:,i]) )
    
    for j in range(N_AA):
        if V_AA[j,i+1]>=AA['Vpeak']:
            V_AA[j,i+1]=AA['Vreset']
            u_AA[j,i+1] = u_AA[j,i] +AA['ujump']
 
    
    # pyr cell
    V_E[:,i+1] = V_E[:,i] + dt*(E['k']*(V_E[:,i] - E['Vr'])*(V_E[:,i] - E['Vth']) - u_E[:,i] + I_ext)/E['C']
    u_E[:,i+1] = u_E[:,i] + dt*(E['alpha']*(E['beta']*(V_E[:,i] - E['Vr']) - u_E[:,i]) )

    # Check for spike (threshold crossing)
    for j in range(N_E):
        if V_E[j,i+1]>=E['Vpeak']:
            V_E[j,i+1]=E['Vreset']
            u_E[j,i+1] = u_E[j,i] +E['ujump']
 
    i+=1


# plot it

f, (a0, a1, a2) = plt.subplots(3, 1,sharex=True)
a0.plot(t,V_BC[0,:], color='k')
a0.spines['right'].set_visible(False)
a0.spines['top'].set_visible(False)
#a0.set_ylim([-400,400])
a0.tick_params(axis='both', labelsize=18)

a1.plot(t,V_AA[0,:], color='k')
a1.spines['right'].set_visible(False)
a1.spines['top'].set_visible(False)
#a1.set_ylim([-400,400])
#a0.set_yticks(np.arange(0, 0.3+0.01, 0.2))
a1.tick_params(axis='both', labelsize=18)

a2.plot(t, V_E[0,:], color='k')
a2.set_ylabel('Voltage (mV)', fontsize=25)
a2.spines['right'].set_visible(False)
a2.spines['top'].set_visible(False)
#a2.set_ylim([-400,400])
a2.tick_params(axis='both', labelsize=18)


f.tight_layout()

#f.savefig('Evolution_neurons.pdf')

plt.show()