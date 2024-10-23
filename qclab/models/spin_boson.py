import numpy as np
import qclab.auxiliary as auxiliary
from numba import njit


class SpinBosonModel:
    def __init__(self, input_params):
        # Here we can define some input parameters that the model accepts and use them to construct the relevant aspects of the physical system
        self.temp = input_params['temp']  # temperature
        self.V = input_params['V']  # offdiagonal coupling
        self.E = input_params['E']  # diagonal energy
        self.A = input_params['A']  # total number of classical oscillators
        self.W = input_params['W']  # characteristic frequency
        self.l = input_params['l']  # reorganization energy
        self.w = self.W * np.tan(
            ((np.arange(self.A) + 1) - (1 / 2)) * np.pi / (2 * self.A))  # classical oscillator frequency
        self.g = self.w * np.sqrt(2 * self.l / self.A)  # electron-phonon coupling
        self.pq_weight = self.w
        self.mass = np.ones_like(self.w)
        self.num_states = 2  # number of states
        self.num_classical_coordinates = self.A

        

        #@njit
        def dh_qc_dz(state, psi_a, psi_b):
            """
            Computes <psi_a| dH_qc/dz  |psi_b> in each branch
            :param psi_a: left vector in each branch
            :param psi_b: right vector in each branch
            :return:
            """
            # initialize derivatives of h wrt z and zc
            # tensors have dimension # classical osc \times # quantum states \times # quantum states
            dz_mat = np.zeros((state.model.A, state.model.num_states, state.model.num_states), dtype=complex)
            dz_mat[:, 0, 0] = state.model.g * np.sqrt(1 / (2 * state.model.mass * state.model.pq_weight))
            dz_mat[:, 1, 1] = -state.model.g * np.sqrt(1 / (2 * state.model.mass * state.model.pq_weight))
            dz_shape = np.shape(dz_mat)
            # position of nonzero matrix elements
            dz_ind = np.where(np.abs(dz_mat) > 1e-12)
            # nonzero matrix elements
            dz_mels = dz_mat[dz_ind] + 0.0j
            return auxiliary.matprod_sparse(dz_shape, dz_ind, dz_mels, psi_a, psi_b)

        #@njit
        def dh_qc_dzc(state, psi_a, psi_b):
            """
            Computes <psi_a| dH_qc/dzc  |psi_b> in each branch
            :param psi_a: left vector in each branch
            :param psi_b: right vector in each branch
            :return:
            """
            return np.conj(dh_qc_dz(state, psi_a, psi_b))

        def h_q(state):
            """
            Nearest-neighbor tight-binding Hamiltonian with periodic boundary conditions and dimension num_states.
            :return: h_q Hamiltonian
            """
            out = np.zeros((state.model.num_states, state.model.num_states), dtype=complex)
            out[0, 0] = state.model.E
            out[1, 1] = -state.model.E
            out[0, 1] = state.model.V
            out[1, 0] = state.model.V
            return out[np.newaxis, np.newaxis]

        def h_qc(state):
            """
            Holstein Hamiltonian on a lattice in real-space using frequency-weighted coordinates
            :return:
            """
            h_qc_out = np.zeros((state.model.batch_size, state.model.num_branches, state.model.num_states, state.model.num_states), dtype=complex)
            h_qc_out[:, :, 0, 0] = np.sum(state.model.g[np.newaxis, np.newaxis, :] * np.sqrt(1 / (2 * state.model.mass * state.model.pq_weight))[np.newaxis, np.newaxis, :] * (
                        state.z_coord + np.conj(state.z_coord)), axis=-1)
            h_qc_out[:, :, 1, 1] = np.sum(-state.model.g[np.newaxis, np.newaxis, :] * np.sqrt(1 / (2 * state.model.mass * state.model.pq_weight))[np.newaxis, np.newaxis, :] * (
                        state.z_coord + np.conj(state.z_coord)), axis=-1)
            return h_qc_out

        self.dh_qc_dz = dh_qc_dz
        self.dh_qc_dzc = dh_qc_dzc
        self.h_qc = h_qc
        self.h_q = h_q
