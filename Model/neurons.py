import numpy as np

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
            # Update voltage and recovery variable for all neurons
            self.V[:, i+1] = self.V[:, i] + dt * np.array([
    self.dvdt_individual(self.V[j, i], self.u[j, i], I_ext, I_syn[j], self.eta[j])
    for j in range(self.N)
])

self.u[:, i+1] = self.u[:, i] + dt * np.array([
    self.dudt(self.V[j, i], self.u[j, i]) for j in range(self.N)
])
        
        
        