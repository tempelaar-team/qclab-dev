import numpy as np
import qclab.auxiliary as auxiliary


class HolsteinLatticeModel:
    def __init__(self, input_params):
        self.num_states = input_params['num_states']  # number of states
        self.temp = input_params['temp']  # temperature
        self.j = input_params['j']  # hopping integral
        self.w = input_params['w']  # classical oscillator frequency
        self.g = input_params['g']  # electron-phonon coupling
        self.open_boundaries = input_params['open_boundaries']  # open or closed boundary conditions
        self.mass = input_params['m'] * np.ones(self.num_states)  # mass of the classical oscillators
        self.pq_weight = np.ones(self.num_states) * self.w
        self.num_classical_coordinates = self.num_states
        self.init_classical = auxiliary.harmonic_oscillator_boltzmann_init_classical
        self.dh_c_dzc = auxiliary.harmonic_oscillator_dh_c_dzc

        def dh_qc_dz(state, z_coord, psi_a, psi_b):
            return np.conj(psi_a) * state.model.g * state.model.pq_weight[..., :] * psi_b

        def dh_qc_dzc(state, z_coord, psi_a, psi_b):
            return np.conj(dh_qc_dz(state, z_coord, psi_a, psi_b))

        def h_q(state):
            out = np.zeros((state.model.num_states, state.model.num_states), dtype=complex)
            for n in range(state.model.num_states - 1):
                out[n, n + 1] = -state.model.j
                out[n + 1, n] = -state.model.j
            if not state.model.open_boundaries:
                out[0, -1] = -state.model.j
                out[-1, 0] = -state.model.j
            return out

        def h_qc(state, z_coord):
            h_qc_out = np.zeros((state.model.batch_size, state.model.num_branches,
                                 state.model.num_states, state.model.num_states), dtype=complex)
            h_qc_diag = state.model.g * state.model.pq_weight[np.newaxis, :] * (z_coord + np.conj(z_coord))
            np.einsum('...jj->...j', h_qc_out)[...] = h_qc_diag
            return h_qc_out

        self.dh_qc_dz = dh_qc_dz
        self.dh_qc_dzc = dh_qc_dzc
        self.h_qc = h_qc
        self.h_q = h_q
