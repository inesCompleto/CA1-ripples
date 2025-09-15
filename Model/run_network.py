import numpy as np
from neurons import Neuron

parameter_sets = {
    'class2': {'a': 0.04, 'b': 5, 'c': 155, 'alpha': 0.2, 'beta': 0.26, 'Vr': -65, 'Cm': 1, 'ujump': 0},
    'subthreshold': {'a': 0.0454, 'b': 5.02, 'c': 137.76, 'alpha': 0.05, 'beta': 1.1, 'Vr': -60, 'Cm': 2, 'ujump': 0},
    'tonic_spiking': {'a': 0.04, 'b': 4.93, 'c': 152, 'alpha': 0.02, 'beta': 0.2, 'Vr': -60, 'Cm': 1, 'ujump': 2},
    'rebound_burst': {'a': 0.04, 'b': 5.3, 'c': 174, 'Cm': 1, 'alpha': 0.01, 'beta': 0.9, 'Vr': -65, 'ujump': 0},
    'rebound_spike': {'a': 0.04, 'b': 4.99, 'c': 154, 'alpha': 0.03, 'beta': 0.25, 'Vr': -56, 'Cm': 1, 'ujump': 4},
    'class1': {'a': 0.04, 'b': 4.96, 'c': 154, 'alpha': 0.02, 'beta': 0.1, 'Vr': -65, 'Cm': 1, 'ujump': 4}
}


class1_neurons = Neuron(parameter_sets['class1'], N=200)


I_ext = 6
tmin = 0.0
tmax = 200
dt = 0.01
steps = ((tmax - tmin)/dt) // 1 
t = np.linspace(tmin, tmax, int(steps))


i = 0


while i < len(t)-1:
    I_ext[i] = 0 + 6*np.heaviside(i*dt-t1, 1)*np.heaviside(t2-i*dt, 1) if t[i] < 130 else 2*np.sin(0.1*t[i])

# Mean field eq
    g_mean = g_mean + dt*(-g_mean/tau_s + p*r[i])
    r[i+1] = r[i] + dt*drdt(params, v_mean[i], r[i], g_mean)
    v_mean[i+1] = v_mean[i] + dt*dvdt(params, v_mean[i], r[i], u_mean[i], g_mean, I_ext[i])
    u_mean[i+1] = u_mean[i] + dt*(dudt(params, v_mean[i], u_mean[i]) + params['ujump']*r[i])


# QIF equations
    V_av[i] = np.sum(V[:,i])/N

    for j in range(N):
        if V[j,i] > 200:
            V[j,i] = -200
            u[j,i] = u[j,i] + params['ujump']
            count[i+1] = count[i+1] + 1/(N*dt)

    g = g + dt*(-g/tau_s + p*count[i])
    I_syn = V[:,i]*g
    V[:,i+1] = V[:,i] + dt*(params['a']*V[:,i]**2 + params['b']*V[:,i] + params['c'] - u[:,i] + I_ext[i] - I_syn + eta)/params['Cm']
    u[:,i+1] = u[:,i] + dt*(params['alpha']*(params['beta']*(V[:,i]-params['Vr'])-u[:,i]))

    i += 1
    
    

print(r[-1], v_mean[-1], u_mean[-1])

b,a = butter(2, 0.05, 'low')
count = filtfilt(b,a,count)
#V_av = filtfilt(b,a,V_av)


from sklearn.metrics import mean_absolute_error as mae

#calculate MAE
av_error_v=mae(v_mean[:-2], V_av[:-2])
print(av_error_v)

av_error_r=mae(r[:-2], count[:-2])
print(av_error_r)

# plot it

#f, (a0, a1) = plt.subplots(2, 1,sharex=True, gridspec_kw={'height_ratios': [4, 1]})
f, (a0) = plt.subplots(1, 1, figsize=(6,4))
a0.plot(t,count, color='#a6bddb', label='Full network')
a0.plot(t,r, linewidth=2,  color='#163050', label='Mean field')
a0.set_ylabel('Firing rate', fontsize=22)
a0.spines['right'].set_visible(False)
a0.spines['top'].set_visible(False)
a0.spines['right'].set_visible(False)
a0.spines['top'].set_visible(False)
a0.tick_params(axis='both', labelsize=16)
a0.set_xlabel('time (msec)', fontsize=22)
a0.set_title('Class 1', fontsize=22)
a0.legend(fontsize=18)

#a0.plot(t, I_ext, linewidth=2, color='k')
#a0.set_ylabel('$I_{ext}$', fontsize=18)
#a0.set_xlabel('time (msec)', fontsize=16)
#a0.spines['right'].set_visible(False)
#a0.spines['top'].set_visible(False)
#a0.tick_params(axis='both', labelsize=14)
#f.suptitle('Class 1, $\\alpha = 0.06\ msec^{-1}$', fontsize=16)

f.tight_layout()
f.savefig('class1.pdf')

plt.show()
