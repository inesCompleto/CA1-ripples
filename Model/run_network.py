import numpy as np
from neurons import Neuron

parameter_sets = {
    'pyr': {'a': 0.5, 'b': 52.5, 'c': 1350, 'alpha': 0.02, 'beta': 0.5, 'Vr': -60, 'Cm': 50, 'ujump': 50, 'V_reset': -60, 'V_peak': 30},
    'subthreshold': {'a': 0.0454, 'b': 5.02, 'c': 137.76, 'alpha': 0.05, 'beta': 1.1, 'Vr': -60, 'Cm': 2, 'ujump': 0},
    'tonic_spiking': {'a': 0.04, 'b': 4.93, 'c': 152, 'alpha': 0.02, 'beta': 0.2, 'Vr': -60, 'Cm': 1, 'ujump': 2},
    'rebound_burst': {'a': 0.04, 'b': 5.3, 'c': 174, 'Cm': 1, 'alpha': 0.01, 'beta': 0.9, 'Vr': -65, 'ujump': 0},
    'rebound_spike': {'a': 0.04, 'b': 4.99, 'c': 154, 'alpha': 0.03, 'beta': 0.25, 'Vr': -56, 'Cm': 1, 'ujump': 4},
    'class1': {'a': 0.04, 'b': 4.96, 'c': 154, 'alpha': 0.02, 'beta': 0.1, 'Vr': -65, 'Cm': 1, 'ujump': 4}
}


class1_neurons = Neuron(parameter_sets['pyr'], N=200)


I_ext = 100
tmin = 0.0
tmax = 200
dt = 0.01
steps = ((tmax - tmin)/dt) // 1 
t = np.linspace(tmin, tmax, int(steps))

V,u = class1_neurons.simulate(dt, t, I_ext)

class1_neurons.plot_results(t, 'Class1')
