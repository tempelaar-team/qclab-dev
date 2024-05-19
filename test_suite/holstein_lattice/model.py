import numpy as np
import auxilliary

def initialize(sim):
    # model specific parameter default values
    defaults = {
        "temp": 1,  # temperature
        "w": 1,  # classical oscillator frequency
        "j": 1,  # hopping integral
        "num_states": 20,  # number of states
        "g": 1,  # electron-phonon coupling
        "m": 1, # mass of the classical oscillators
        "quantum_rotation": None,  # rotation of quantum subspace
        "classical_rotation": None,  # rotation of classical subspace
        "calc_dir":"output"
    }
    inputs = list(sim.input_params)  # inputs is list of keys in input_params
    for key in inputs:  # copy input values into defaults
        defaults[key] = sim.input_params[key]
    # load model specific parameters
    sim.g = defaults["g"]
    sim.temp = defaults["temp"]
    sim.j = defaults["j"]
    sim.m = defaults["m"]
    sim.num_states = defaults["num_states"]
    sim.w = defaults["w"]
    sim.calc_dir = defaults["calc_dir"]


    def h_q(sim):
        """
        Nearest-neighbor tight-binding Hamiltonian with periodic boundary conditions and dimension num_states.
        :return: h_q Hamiltonian
        """
        out = np.zeros((sim.num_states, sim.num_states), dtype=complex)
        for n in range(sim.num_states - 1):
            out[n, n + 1] = -sim.j
            out[n + 1, n] = -sim.j
        out[0, -1] = -sim.j
        out[-1, 0] = -sim.j
        return out

    def h_qc(z, sim):
        """
        Holstein Hamiltonian on a lattice in real-space, z and zc are frequency weighted
        :param z: z coordinate
        :param zc: z^{*} conjugate z
        :return: h_qc(z,z^{*}) Hamiltonian
        """
        out = np.diag(sim.g * np.sqrt(sim.h) * (z + np.conj(z)))
        return out




    # initialize derivatives of h wrt z and zc
    # tensors have dimension # classical osc \times # quantum states \times # quantum states
    dz_mat = np.zeros((sim.num_states, sim.num_states, sim.num_states), dtype=complex)
    dzc_mat = np.zeros((sim.num_states, sim.num_states, sim.num_states), dtype=complex)
    for i in range(sim.num_states):
        dz_mat[i, i, i] = sim.g*np.sqrt(sim.w)
        dzc_mat[i, i, i] = sim.g*np.sqrt(sim.w)
    dz_shape = np.shape(dz_mat)
    dzc_shape = np.shape(dzc_mat)
    # position of nonzero matrix elements
    dz_ind = np.where(np.abs(dz_mat) > 1e-12)
    dzc_ind = np.where(np.abs(dzc_mat) > 1e-12)
    # nonzero matrix elements
    dz_mels = dz_mat[dz_ind]
    dzc_mels = dzc_mat[dzc_ind]
    # necessary variables for computing expectation values
    diff_vars = (dz_shape, dz_ind, dz_mels, dzc_shape, dzc_ind, dzc_mels)

    def dh_qc_dz(psi_a, psi_b, z, sim):
        """
        Computes <\psi_a| dH_qc/dz  |\psi_b>
        :param psi_a:
        :param psi_b:
        :return:
        """
        return auxilliary.matprod_sparse(dz_shape, dz_ind, dz_mels, psi_a, psi_b)
    def dh_qc_dzc(psi_a, psi_b, z, sim):
        """
        Computes <\psi_a| dH_qc/dz*  |\psi_b>
        :param psi_a:
        :param psi_b:
        :return:
        """
        return auxilliary.matprod_sparse(dzc_shape, dzc_ind, dzc_mels, psi_a, psi_b)

    def hop(z, delta_z, ev_diff, sim):
        hopped = False
        delta_zc = np.conj(delta_z)
        zc = np.conj(z)
        akj_z = np.real(np.sum(sim.h * delta_zc * delta_z))
        bkj_z = np.real(np.sum(1j * sim.h * (zc * delta_z - z * delta_zc)))
        ckj_z = ev_diff
        disc = bkj_z ** 2 - 4 * akj_z * ckj_z
        if disc >= 0:
            if bkj_z < 0:
                gamma = bkj_z + np.sqrt(disc)
            else:
                gamma = bkj_z - np.sqrt(disc)
            if akj_z == 0:
                gamma = 0
            else:
                gamma = gamma / (2 * akj_z)
            # adjust classical coordinates
            z = z - 1.0j * np.real(gamma) * delta_z
            zc = zc + 1.0j * np.real(gamma) * delta_zc
            hopped = True
        return z, hopped

    def harmonic_oscillator_boltzmann(sim):
        """
        Initialize classical coordiantes according to Boltzmann statistics
        :param sim: simulation object with temperature, harmonic oscillator mass and frequency
        :return: z = sqrt(w*h/2)*(q + i*(p/((w*h))), z* = sqrt(w*h/2)*(q - i*(p/((w*h)))
        """
        q = np.random.normal(loc=0, scale=np.sqrt(sim.temp / (sim.m * (sim.h ** 2))),
                             size=sim.num_states)
        p = np.random.normal(loc=0, scale=np.sqrt(sim.temp), size=sim.num_states)
        z = np.sqrt(sim.h * sim.m / 2) * (q + 1.0j * (p / (sim.h * sim.m)))
        return z

    def harmonic_oscillator(z, sim):
        """
        Harmonic oscillator Hamiltonian
        :param z: z(t)
        :param zc: conjugate z(t)
        :return: h_c(z,zc) Hamiltonian
        """
        return np.real(np.sum(sim.h * np.conj(z) * z))

    def harmonic_oscillator_dh_c_dz(z, sim):
        """
        Gradient of harmonic oscillator hamiltonian w.r.t z
        :param z: z coordinate
        :param zc: z* coordinate
        :param sim: simulation object
        :return:
        """
        return sim.h * np.conj(z)

    def harmonic_oscillator_dh_c_dzc(z, sim):
        """
        Gradient of harmonic oscillator hamiltonian wrt z*
        :param z: z coordinate
        :param zc: z* coordinate
        :param sim: simulation object
        :return:
        """
        return sim.h * z

    def observables(sim, rho_db_branch, z_branch):
        rho_db = np.sum(rho_db_branch,axis=0)
        pops_db = np.diag(rho_db)
        contributions = np.einsum('nii->n',rho_db_branch) # weights of each branch
        z = np.einsum('n,nj->j',contributions, z_branch)
        output_dictionary = {'rho_db':rho_db,'pops_db':pops_db,'ph_occ':np.abs(z)**2}
        return output_dictionary
    
    # equip simulation object with necessary functions
    sim.init_classical = harmonic_oscillator_boltzmann
    sim.hop = hop
    sim.h_q = h_q
    sim.h_qc = h_qc
    sim.h_c = harmonic_oscillator
    sim.dh_qc_dz = dh_qc_dz
    sim.dh_qc_dzc = dh_qc_dzc
    sim.dh_c_dz = harmonic_oscillator_dh_c_dz
    sim.dh_c_dzc = harmonic_oscillator_dh_c_dzc
    sim.h = sim.w*np.ones(sim.num_states)
    sim.mf_observables = observables
    sim.fssh_observables = observables
    sim.cfssh_observables = observables
    #sim.calc_dir = 'holstein_lattice_g_' + str(sim.g) + '_j_' + str(sim.j) + '_w_' + str(sim.w) + \
    #               '_temp_' + str(sim.temp) + '_nstates_' + str(sim.num_states)
    sim.psi_db_0 = 1 / np.sqrt(sim.num_states) * np.ones(sim.num_states, dtype=complex)

    return sim
