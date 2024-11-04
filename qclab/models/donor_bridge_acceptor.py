import numpy as np


class DonorBridgeAcceptorModel:
    def __init__(self, input_params):
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

        def dh_qc_dz(state, model, params, z_coord, psi_a, psi_b):
            dz_mat = np.zeros((model.num_states * model.A,
                               model.num_states, model.num_states), dtype=complex)
            dz_mat[0:model.A, 0, 0] = \
                (model.g * np.sqrt(1 / (2 * model.mass * model.pq_weight)))[0:model.A]
            dz_mat[model.A:2 * self.A, 1, 1] = \
                (self.g * np.sqrt(1 / (2 * self.mass * model.pq_weight)))[model.A:2 * model.A]
            dz_mat[2 * model.A:3 * model.A, 2, 2] = \
                (model.g * np.sqrt(1 / (2 * model.mass * model.pq_weight)))[2 * model.A:3 * model.A]
            return np.einsum('...i,cij,...j->...c', np.conj(psi_a), dz_mat, psi_b, optimize='greedy')

        def dh_qc_dzc(state, model, params, z_coord, psi_a, psi_b):
            return np.conj(dh_qc_dz(state, model, params, z_coord, psi_a, psi_b))

        def h_q(state, model, params):
            out = np.zeros((model.num_states, model.num_states), dtype=complex)
            out[0, 0] = model.E_D
            out[1, 1] = model.E_B
            out[2, 2] = model.E_A
            out[0, 1] = model.V
            out[1, 0] = model.V
            out[1, 2] = model.V
            out[2, 1] = model.V
            return out

        def h_qc(state, model, params, z_coord):
            h_qc_out = np.zeros((model.batch_size, model.num_branches,
                                 model.num_states, model.num_states), dtype=complex)
            mel = (model.g[..., :] * np.sqrt(1 / (2 * model.mass * model.pq_weight))[..., :]
                   * (z_coord + np.conj(z_coord)))
            h_qc_out[..., 0, 0] = np.sum(mel[..., 0:model.A], axis=-1)
            h_qc_out[..., 1, 1] = np.sum(mel[..., model.A:2 * model.A], axis=-1)
            h_qc_out[..., 2, 2] = np.sum(mel[..., 2 * model.A:3 * model.A], axis=-1)
            return h_qc_out

        self.dh_qc_dz = dh_qc_dz
        self.dh_qc_dzc = dh_qc_dzc
        self.h_qc = h_qc
        self.h_q = h_q
