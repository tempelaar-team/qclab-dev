import numpy as np


class DonorBridgeAcceptorModel:
    def __init__(self, input_params):
        # Here we can define some input parameters that the model accepts and use them to construct the relevant
        # aspects of the physical system
        self.temp = input_params['temp']  # temperature
        self.V = input_params['V']  # donor-bridge bridge-acceptor coupling
        self.E_D = input_params['E_D']  # donor energy
        self.E_B = input_params['E_B']  # bridge energy
        self.E_A = input_params['E_A']  # acceptor energy
        self.A = input_params['A']  # total number of classical oscillators
        self.W = input_params['W']  # characteristic frequency
        self.l_reorg = input_params['l_reorg']  # reorganization energy
        self.w = self.W * np.tan(
            ((np.arange(self.A) + 1) - (1 / 2)) * np.pi / (2 * self.A))  # classical oscillator frequency
        self.g = self.w * np.sqrt(2 * self.l_reorg / self.A)  # electron-phonon coupling
        self.g = np.concatenate((self.g, self.g, self.g))
        self.w = np.concatenate((self.w, self.w, self.w))
        self.pq_weight = self.w
        self.mass = np.ones_like(self.w)
        self.num_states = 3  # number of states
        self.num_classical_coordinates = int(self.A * 3)
        self.h_q_params = (self.E_D, self.E_B, self.E_A, self.V)

        def dh_qc_dz(state, z_coord, psi_a, psi_b):
            """
            Computes <psi_a| dH_qc/dz  |psi_b> in each branch
            :param psi_a: left vector in each branch
            :param psi_b: right vector in each branch
            :return:
            """
            dz_mat = np.zeros((state.model.num_states * state.model.A,
                               state.model.num_states, state.model.num_states), dtype=complex)
            dz_mat[0:state.model.A, 0, 0] = \
                (state.model.g * np.sqrt(1 / (2 * state.model.mass * state.model.pq_weight)))[0:state.model.A]
            dz_mat[state.model.A:2 * self.A, 1, 1] = \
                (self.g * np.sqrt(1 / (2 * self.mass * state.model.pq_weight)))[state.model.A:2 * state.model.A]
            dz_mat[2 * state.model.A:3 * state.model.A, 2, 2] = \
                (state.model.g * np.sqrt(1 / (2 * state.model.mass * state.model.pq_weight)))[2 * state.model.A:3 * state.model.A]
            return np.einsum('...i,cij,...j->...c', np.conj(psi_a), dz_mat, psi_b, optimize='greedy')

        def dh_qc_dzc(state, z_coord, psi_a, psi_b):
            """
            Computes <psi_a| dH_qc/dzc  |psi_b> in each branch
            :param psi_a: left vector in each branch
            :param psi_b: right vector in each branch
            :return:
            """
            return np.conj(dh_qc_dz(state, z_coord, psi_a, psi_b))

        def h_q(state):
            """
            Nearest-neighbor tight-binding Hamiltonian with periodic boundary conditions and dimension num_states.
            :return: h_q Hamiltonian
            """
            out = np.zeros((state.model.num_states, state.model.num_states), dtype=complex)
            out[0, 0] = state.model.E_D
            out[1, 1] = state.model.E_B
            out[2, 2] = state.model.E_A
            out[0, 1] = state.model.V
            out[1, 0] = state.model.V
            out[1, 2] = state.model.V
            out[2, 1] = state.model.V
            return out

        def h_qc(state, z_coord):
            """
            Holstein Hamiltonian on a lattice in real-space with frequency-weighted coordinates
            :return:
            """
            h_qc_out = np.zeros((state.model.batch_size, state.model.num_branches,
                                 state.model.num_states, state.model.num_states), dtype=complex)
            mel = (state.model.g[..., :] * np.sqrt(1 / (2 * state.model.mass * state.model.pq_weight))[..., :]
                   * (z_coord + np.conj(z_coord)))
            h_qc_out[..., 0, 0] = np.sum(mel[..., 0:state.model.A], axis=-1)
            h_qc_out[..., 1, 1] = np.sum(mel[..., state.model.A:2 * state.model.A], axis=-1)
            h_qc_out[..., 2, 2] = np.sum(mel[..., 2 * state.model.A:3 * state.model.A], axis=-1)
            return h_qc_out

        self.dh_qc_dz = dh_qc_dz
        self.dh_qc_dzc = dh_qc_dzc
        self.h_qc = h_qc
        self.h_q = h_q
