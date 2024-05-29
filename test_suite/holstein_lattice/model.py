import numpy as np
import auxilliary
from numba import njit

def initialize(sim):
    # model specific parameter default values
    defaults = {
        "temp": 1,  # temperature
        "w": 1,  # classical oscillator frequency
        "j": 1,  # hopping integral
        "num_states": 20,  # number of states
        "g": 1,  # electron-phonon coupling
        "m": 1, # mass of the classical oscillators
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



    # Define model parameters


    def h_q_branch(sim):
        """
        Nearest-neighbor tight-binding Hamiltonian with periodic boundary conditions and dimension num_states.
        :return: h_q Hamiltonian
        """
        out = np.zeros((sim.num_branches, sim.num_states, sim.num_states), dtype=complex)
        for n in range(sim.num_states - 1):
            out[:, n, n + 1] = -sim.j
            out[:, n + 1, n] = -sim.j
        out[:, 0, -1] = -sim.j
        out[:, -1, 0] = -sim.j
        return out

    def h_qc_branch(z_branch, sim):
        """
        Holstein Hamiltonian on a lattice in real-space, z and zc are frequency weighted
        :param z: z coordinate
        :param zc: z^{*} conjugate z
        :return: h_qc(z,z^{*}) Hamiltonian
        """
        h_qc_out = np.zeros((sim.num_branches, sim.num_states, sim.num_states), dtype=complex)
        h_qc_diag = sim.g * np.sqrt(sim.h) * (z_branch + np.conj(z_branch))
        np.einsum('...jj->...j', h_qc_out)[...] = h_qc_diag
        return h_qc_out




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
    dz_mels = dz_mat[dz_ind] + 0.0j
    dzc_mels = dzc_mat[dzc_ind] + 0.0j
    @njit
    def dh_qc_dz_branch(psi_a_branch, psi_b_branch, z_branch):
        """
        Computes <\psi_a| dH_qc/dz  |\psi_b> in each branch
        :param psi_a_branch: left vector in each branch
        :param psi_b_branch: right vector in each branch
        :param z_branch: z coordinate in each branch
        :return:
        """
        out = np.ascontiguousarray(np.zeros((len(psi_a_branch), dz_shape[0])))+0.0j
        for n in range(len(psi_a_branch)):
            out[n] = auxilliary.matprod_sparse(dz_shape, dz_ind, dz_mels, psi_a_branch[n], psi_b_branch[n])
        return out
    @njit
    def dh_qc_dzc_branch(psi_a_branch, psi_b_branch, z_branch):
        """
        Computes <\psi_a| dH_qc/dzc  |\psi_b> in each branch
        :param psi_a_branch: left vector in each branch
        :param psi_b_branch: right vector in each branch
        :param z_branch: z coordinate in each branch
        :return:
        """
        out = np.ascontiguousarray(np.zeros((len(psi_a_branch), dzc_shape[0])))+0.0j
        for n in range(len(psi_a_branch)):
            out[n] = auxilliary.matprod_sparse(dzc_shape, dzc_ind, dzc_mels, psi_a_branch[n], psi_b_branch[n]) # conjugation is done by matprod_sparse
        return out

    def hop(z, delta_z, ev_diff, sim):
        """
        Carries out the hopping procedure for a harmonic oscillator Hamiltonian, defined on a single branch only. 
        :param z: z coordinate
        :param delta_z: rescaling direction
        :param ev_diff: change in quantum energy following a hop: e_{final} - e_{initial}
        :param sim: simulation object
        :return z, hopped: updated z coordinate and boolean indicating if a hop has or has not occured
        """
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
            # adjust classical coordinate
            z = z - 1.0j * np.real(gamma) * delta_z
            hopped = True
        return z, hopped

    def harmonic_oscillator_boltzmann_branch(sim):
        """
        Initialize classical coordiantes according to Boltzmann statistics in each branch
        :param sim: simulation object with temperature, harmonic oscillator mass and frequency
        :return: z = sqrt(w*h/2)*(q + i*(p/((w*h))), z* = sqrt(w*h/2)*(q - i*(p/((w*h)))
        """
        if sim.dynamics_method != 'MF':
            q = np.random.normal(loc=0, scale=np.sqrt(sim.temp / (sim.m * (sim.h ** 2))), size=sim.num_states)
            p = np.random.normal(loc=0, scale=np.sqrt(sim.temp), size=sim.num_states)
            z = np.sqrt(sim.h * sim.m / 2) * (q + 1.0j * (p / (sim.h * sim.m)))

            z_branch = np.zeros((sim.num_branches, len(z)), dtype=complex)
            z_branch[:] = z
        if sim.dynamics_method == 'MF':
            q_branch = np.random.normal(loc=0, scale=np.sqrt(sim.temp / (sim.m * (sim.h ** 2))), size=(sim.num_branches, sim.num_states))
            p_branch = np.random.normal(loc=0, scale=np.sqrt(sim.temp), size=(sim.num_branches,sim.num_states))
            z_branch = np.sqrt(sim.h[np.newaxis,:] * sim.m[np.newaxis,:] / 2) * (q_branch + 1.0j * (p_branch / (sim.h[np.newaxis,:] * sim.m[np.newaxis,:])))
        return z_branch

    def harmonic_oscillator(z_branch, sim):
        """
        Harmonic oscillator Hamiltonian
        :param z: z(t)
        :param zc: conjugate z(t)
        :return: h_c(z,zc) Hamiltonian
        """
        return np.real(np.sum(sim.h[np.newaxis,:] * np.conj(z_branch) * z_branch, axis=1))

    def harmonic_oscillator_dh_c_dz_branch(z_branch, sim):
        """
        Gradient of harmonic oscillator hamiltonian wrt z_branch
        :param z_branch: z coordinate in each branch
        :param sim: simulation object
        :return:
        """
        return sim.h[np.newaxis, :] * np.conj(z_branch)

    def harmonic_oscillator_dh_c_dzc_branch(z_branch, sim):
        """
        Gradient of harmonic oscillator hamiltonian wrt zc_branch
        :param z_branch: z coordinate in each branch
        :param sim: simulation object
        :return:
        """
        return sim.h[np.newaxis, :] * z_branch
    state_vars_list = ['rho_db_fssh','rho_db_mf','rho_db_cfssh','z_branch','psi_db_branch', 'h_tot_branch', 'evals_branch','act_surf_ind_branch']
    def observables(sim, state_vars):
        output_dictionary = {}
        z_branch = state_vars['z_branch']
        output_dictionary['E_c'] = np.sum(sim.h_c_branch(z_branch,sim))
        if sim.dynamics_method=='MF':
            psi_db_branch = state_vars['psi_db_branch']
            h_tot_branch = state_vars['h_tot_branch']
            output_dictionary['E_q'] = np.real(np.einsum('ni,nij,nj', np.conjugate(psi_db_branch),h_tot_branch, psi_db_branch))
        if sim.dynamics_method=='FSSH' or sim.dynamics_method=='CFSSH':
            evals_branch = state_vars['evals_branch']
            act_surf_ind_branch = state_vars['act_surf_ind_branch']
            eq = 0
            for n in range(len(act_surf_ind_branch)):
                eq += evals_branch[n][act_surf_ind_branch[n]]
            output_dictionary['E_q'] = eq
        if 'rho_db_fssh' in state_vars.keys():
            output_dictionary['pops_db_fssh'] = np.real(np.diag(state_vars['rho_db_fssh']))
        if 'rho_db_cfssh' in state_vars.keys():
            output_dictionary['pops_db_cfssh'] = np.real(np.diag(state_vars['rho_db_cfssh']))
        if 'rho_db_mf' in state_vars.keys():
            output_dictionary['pops_db_mf'] = np.real(np.diag(state_vars['rho_db_mf']))
        return output_dictionary
    
    # equip simulation object with necessary functions
    sim.init_classical_branch = harmonic_oscillator_boltzmann_branch
    sim.hop = hop
    sim.h_q_branch = h_q_branch
    sim.h_qc_branch = h_qc_branch
    sim.h_c_branch = harmonic_oscillator
    sim.dh_qc_dz_branch = dh_qc_dz_branch
    sim.dh_qc_dzc_branch = dh_qc_dzc_branch
    sim.dh_c_dz_branch = harmonic_oscillator_dh_c_dz_branch
    sim.dh_c_dzc_branch = harmonic_oscillator_dh_c_dzc_branch
    sim.h = sim.w*np.ones(sim.num_states)
    sim.m = sim.m*np.ones(sim.num_states)
    sim.w = sim.w*np.ones(sim.num_states)
    sim.mf_observables = observables
    sim.fssh_observables = observables
    sim.cfssh_observables = observables
    sim.state_vars_list = state_vars_list
    sim.psi_db_0 = 1 / np.sqrt(sim.num_states) * np.ones(sim.num_states, dtype=complex)
    sim.psi_db_0 *= 0
    sim.psi_db_0[0] = 1

    return sim
