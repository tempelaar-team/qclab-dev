import numpy as np
import qclab.auxiliary as auxiliary
from numba import njit


class HolsteinLatticeModel:
    def __init__(self, input_params):
        # Here we can define some input parameters that the model accepts and use them to construct the relevant
        # aspects of the physical system
        self.num_states = input_params['num_states']  # number of states
        self.temp = input_params['temp']  # temperature
        self.j = input_params['j']  # hopping integral
        self.w = input_params['w']  # classical oscillator frequency
        self.g = input_params['g']  # electron-phonon coupling
        self.open = input_params['open']  # open or closed boundary conditions
        self.mass = input_params['m'] * np.ones(self.num_states)  # mass of the classical oscillators
        self.pq_weight = np.ones(self.num_states) * self.w
        self.h_c_params = self.pq_weight
        self.h_qc_params = (self.pq_weight, self.g)
        self.h_q_params = (self.j, self.open)
        self.num_classical_coordinates = self.num_states
        self.init_classical = auxiliary.harmonic_oscillator_boltzmann_init_classical
        self.dh_c_dzc = auxiliary.harmonic_oscillator_dh_c_dzc
        # initialize derivatives of pq_weight wrt z and zc
        # tensors have dimension # classical osc \times # quantum states \times # quantum states
        dz_mat = np.zeros((self.num_states, self.num_states, self.num_states), dtype=complex)
        for i in range(self.num_states):
            dz_mat[i, i, i] = self.g * self.w
        dz_shape = np.shape(dz_mat)
        # position of nonzero matrix elements
        dz_ind = np.where(np.abs(dz_mat) > 1e-12)
        # nonzero matrix elements
        dz_mels = dz_mat[dz_ind] + 0.0j

        @njit
        def dh_qc_dz(self, state, psi_a, psi_b):
            """
            Computes <psi_a| dH_qc/dz  |psi_b> in each branch
            :param psi_a: left vector in each branch
            :param psi_b: right vector in each branch
            :return:
            """
            out = np.ascontiguousarray(np.zeros((len(psi_a), dz_shape[0]))) + 0.0j
            for n in range(len(psi_a)):
                out[n] = auxiliary.matprod_sparse(dz_shape, dz_ind, dz_mels, psi_a[n], psi_b[n])
            return out

        @njit
        def dh_qc_dzc(self, state, psi_a, psi_b):
            """
            Computes <psi_a| dH_qc/dzc  |psi_b> in each branch
            :param psi_a: left vector in each branch
            :param psi_b: right vector in each branch
            :return:
            """
            return np.conj(dh_qc_dz(self, state, psi_a, psi_b))

        def h_q(self, state):
            """
            Nearest-neighbor tight-binding Hamiltonian with periodic boundary conditions and dimension num_states.
            :return: h_q Hamiltonian
            """
            out = np.zeros((self.num_states, self.num_states), dtype=complex)
            for n in range(self.num_states - 1):
                out[n, n + 1] = -self.j
                out[n + 1, n] = -self.j
            if not self.open_boundaries:
                out[0, -1] = -self.j
                out[-1, 0] = -self.j
            return out

        def h_qc(self, state):
            """
            Holstein Hamiltonian on a lattice in real-space, z and zc are frequency weighted
            :return: h_qc(z,z^{*}) Hamiltonian
            """
            #h, g = h_qc_params
            #z_coord = h_qc_vars
            h_qc_out = np.zeros((self.batch_size * self.num_branches, self.num_states, self.num_states), dtype=complex)
            h_qc_diag = self.g * self.h[np.newaxis, :] * (state.z_coord + np.conj(state.z_coord))
            np.einsum('...jj->...j', h_qc_out)[...] = h_qc_diag
            return h_qc_out

        self.dh_qc_dz = dh_qc_dz
        self.dh_qc_dzc = dh_qc_dzc
        self.h_qc = h_qc
        self.h_q = h_q
