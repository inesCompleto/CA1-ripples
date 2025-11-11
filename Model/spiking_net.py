'''

    
'''

import math 
import numpy as np
import matplotlib.pyplot as plt


tmin = 0.0
tmax = 1000 # msec
dt = 0.01
steps = ((tmax - tmin)/dt) // 1 
t = np.linspace(tmin, tmax, int(steps))

I_ext = 50

# Pyramidal cells
N_E=500
g_leak = 0.1
V_rest = -60
Vpeak = 400
Vreset = -400

V_E = np.zeros((N_E,len(t)))
V_E[:,0] = 20
count_E = np.zeros(len(t))

p_EE = 0.3 # connectivity probability
conn_EE = np.zeros(N_E)
num_ones = int(p_EE*N_E) 
conn_EE[:num_ones] = 1
g_EE=0
tau_s=1
s_EE=1 
Er_E=0


i=0

while i < len(t)-1:

    # QIF equations
    g_EE = g_EE + dt*(-g_EE/tau_s + conn_EE*s_EE*count_E[i]) 
    I_synE = g_EE*(Er_E - V_E[:,i])  
    V_E[:,i+1] = V_E[:,i] + dt*(g_leak*(V_rest - V_E[:,i]) + I_synE + I_ext)
    
    for j in range(N_E):
        if V_E[j,i]>Vpeak:
            V_E[j,i]=Vreset
            count_E[i+1]=count_E[i+1]+1/(N_E*dt) # count number of neurons spiking at this time    
            
    i+=1


# plot it

f, (a0, a1, a2, a3) = plt.subplots(4, 1,sharex=True)
a0.plot(t,V_E[0,:], color='k')
a0.spines['right'].set_visible(False)
a0.spines['top'].set_visible(False)
a0.set_ylim([-400,400])
a0.tick_params(axis='both', labelsize=18)

a1.plot(t,V_E[1,:], color='k')
a1.spines['right'].set_visible(False)
a1.spines['top'].set_visible(False)
a1.set_ylim([-400,400])
#a0.set_yticks(np.arange(0, 0.3+0.01, 0.2))
a1.tick_params(axis='both', labelsize=18)

a2.plot(t, V_E[2,:], color='k')
a2.set_ylabel('Voltage (mV)', fontsize=25)
a2.spines['right'].set_visible(False)
a2.spines['top'].set_visible(False)
a2.set_ylim([-400,400])
a2.tick_params(axis='both', labelsize=18)


a3.plot(t, V_E[3,:], color='k')
a3.spines['right'].set_visible(False)
a3.spines['top'].set_visible(False)
a3.tick_params(axis='both', labelsize=18)
a3.set_ylim([-400,400])
#a3.set_yticks(np.arange(-2, 3, 4))
a3.set_xlabel('time (msec)', fontsize=25)

f.tight_layout()

#f.savefig('Evolution_neurons.pdf')

plt.show()