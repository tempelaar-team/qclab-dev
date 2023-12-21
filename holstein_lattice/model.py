import numpy as np

def initialize(sim):
    # model specific parameter default values
    defaults = {
        "temp":1,# temperature
        "w":1, # classical oscillator frequency
        "j":1, # hopping integral
        "num_states":20, # number of states
        "g":1, # electron-phonon coupling
        "quantum_rotation":None, # rotation of quantum subspace
        "classical_rotation":None, # rotation of classical subspace
    }
    inputs = list(sim.input_params)  # inputs is list of keys in input_params
    for key in inputs:  # copy input values into defaults
        defaults[key] = sim.input_params[key]
    # load model specific parameters
    sim.g = defaults["g"]
    sim.temp = defaults["temp"]
    sim.j = defaults["j"]
    sim.num_states = defaults["num_states"]
    sim.w = defaults["w"]
    def init_classical():
        """
        Initialize classical coordiantes according to Boltzmann statistics
        :return: position (q) and momentum (p)
        """
        q = np.random.normal(loc = 0, scale = np.sqrt(sim.temp),size = sim.num_states)
        p = np.random.normal(loc = 0, scale = np.sqrt(sim.temp/(sim.w)), size=sim.num_states)
        return q, p
    def h_q():
        """
        Nearest-neighbor tight-binding Hamiltonian with periodic boundary conditions and dimension num_states.
        :return: h_q Hamiltonian
        """
        out = np.zeros((sim.num_states, sim.num_states), dtype=complex)
        for i in range(sim.num_states-1):
            out[i,i+1] = -sim.j
            out[i+1,i] = -sim.j
        out[0,-1] = -sim.j
        out[-1,0] = -sim.j
        return out

    def h_qc(q,p):
        """
        Holstein Hamiltonian on a lattice in real-space
        :param q: position coordinate q(t)
        :param p: momentum coordiante p(t)
        :return: h_qc(q,p) Hamiltonian
        """
        out = np.diag(sim.g*np.sqrt(2*(sim.w**3))*q)
        return out

    def h_c(q, p):
        """
        Harmonic osccilator Hamiltonian
        :param q: position coordinate q(t)
        :param p: momentum coordinate p(t)
        :return: h_c(q,p) Hamiltonian
        """
        return np.real(np.sum((1/2)*((p**2) + (sim.w**2)*(q**2))))
    """
    Initialize the rotation matrices of quantum and classical subsystems
    """
    if sim.quantum_rotation == 'fourier':
        def U_q():
            out = np.fft.fft(np.identity(sim.num_states))/np.sqrt(sim.num_states)
            return out
    else:
        def U_q():
            out = np.identity(sim.num_states)
            return out
    if sim.classical_rotation == 'fourier':
        def U_c():
            out = np.fft.fft(np.identity(sim.num_states))/np.sqrt(sim.num_states)
            return out
    else:
        def U_c():
            out = np.identity(sim.num_states)
            return out


    # initialize derivatives of h wrt q and p
    # tensors have dimension # classical osc \times # quantum states \times # quantum states
    dq_mat = np.zeros((sim.num_states, sim.num_states, sim.num_states),dtype=complex)
    dp_mat = np.zeros((sim.num_states, sim.num_states, sim.num_states),dtype=complex)
    for i in range(sim.num_states):
        dq_mat[i,i,i] = sim.g*np.sqrt(2*(sim.w**3))
    dq_shape = np.shape(dq_mat)
    dp_shape = np.shape(dp_mat)
    # position of nonzero matrix elements
    dq_ind = np.where(np.abs(dq_mat) > 1e-18)
    dp_ind = np.where(np.abs(dp_mat) > 1e-18)
    # nonzero matrix elements
    dq_mels = dq_mat[dq_ind]
    dp_mels = dp_mat[dp_ind]
    # necessary variables for computing expectation values
    dq_vars = (dq_shape, dq_ind, dq_mels, dp_shape, dp_ind, dp_mels)

    # equip simulation object with necessary functions
    sim.init_classical = init_classical
    sim.w_c = sim.w
    sim.h_q = h_q
    sim.h_qc = h_qc
    sim.h_c = h_c
    sim.U_c = U_c
    sim.U_q = U_q
    sim.dq_vars = dq_vars
    sim.calc_dir = 'holstein_lattice_g_'+str(sim.g)+'_j_'+str(sim.j)+'_w_'+str(sim.w)+\
                   '_temp_'+str(sim.temp)+'_nstates_'+str(sim.num_states)
    sim.psi_db_0 = 1/np.sqrt(sim.num_states) * np.ones(sim.num_states,dtype=complex)

    return sim




