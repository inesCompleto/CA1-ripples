import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """ Defines a population of Izhikevich neurons"""
    
    def __init__(self, params, N):
        """
        Parameters
        ----------
        params : dict
            Dictionary containing neuron parameters (a, b, c, alpha, beta, Vr, Cm, ujump)
        N : int
            Number of neurons in populations
        """
        
        self.params = params
        self.N = N
        
        # Initialize state variables (V, u)
        self.V = None
        self.u = None
        
    def initialize_state(self, t_steps, V_init=-70, u_init=10):
        """ Initialize state variables for simulation """
        
        self.V = np.zeros((self.N, t_steps))
        self.V[:,0] = V_init
        self.u = np.zeros((self.N, t_steps))
        self.u[:,0] = u_init

    def neuron_dynamics(self, V, u, I_ext):
        """Time evolution of membrane potential of neurons"""
        dvdt = (self.params['a']*V**2 + self.params['b']*V + self.params['c'] - u + I_ext) / self.params['Cm']        
        dudt = self.params['alpha'] * (self.params['beta'] * (V - self.params['Vr']) - u)
        return dvdt, dudt
    
    
    def simulate(self, dt, t, I_ext, tau_s=1, p=0.1):
        """
        Run full simulation
        
        Parameters:
        -----------
        t : array
            Time array
        I_ext : array
            External current array
        tau_s : float
            Synaptic time constant
        p : float
            Connection probability
        
        Returns:
        --------
        dict : Simulation results
        """
        self.initialize_state(len(t))
        
        for i in range(len(t)-1):
            dvdt, dudt = self.neuron_dynamics(self.V[:,i], self.u[:,i], I_ext)
            # Update voltage and recovery variable for all neurons      
            self.V[:, i+1] = self.V[:, i] + dt * dvdt
            self.u[:, i+1] = self.u[:, i] + dt * dudt
        
            # Spike detection
            for j in range(self.N):
                if self.V[j,i+1] > self.params['V_peak']:
                    self.V[j,i+1] = self.params['V_reset']
                    self.u[j,i+1] = self.u[j,i+1] + self.params['ujump']   
    
        return self.V, self.u


    def plot_results(self, t, title='Neuron Population'):
        """Plot firing rate comparison"""
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(t, self.V[0,:], color='#a6bddb', label='Full network')
        ax.set_ylabel('Membrane potential', fontsize=22)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlabel('time (msec)', fontsize=22)
        ax.set_title(title, fontsize=22)
        ax.legend(fontsize=18)
        fig.tight_layout()
        return fig, ax
        