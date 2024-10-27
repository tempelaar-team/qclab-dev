
import numpy as np
import qclab.auxiliary as auxiliary
from numba import njit


class SpinHolsteinLatticeModel:
    def __init__(self, input_params):

        # Here we can define some input parameters that the model accepts and use them to construct the relevant aspects of the physical system 
        self.num_states=input_params['num_states']  # number of states
        self.temp=input_params['temp']  # temperature
        self.j=input_params['j']  # hopping integral
        self.w=input_params['w']  # classical oscillator frequency
        self.g=input_params['g']  # electron-phonon coupling
        self.open = input_params['open'] # open or closed boundary conditions
        self.mass = input_params['m']*np.ones(int(self.num_states/2)) # mass of the classical oscillators
        self.pq_weight=np.ones(int(self.num_states/2))*self.w
        self.num_classical_coordinates = int(self.num_states/2)
        self.init_classical = auxiliary.harmonic_oscillator_boltzmann_init_classical
        self.dh_c_dzc = auxiliary.harmonic_oscillator_dh_c_dzc
        
        def dh_qc_dz(state, z_coord, psi_a, psi_b):
            out = np.conj(psi_a[...,:int(state.model.num_states/2)]) * state.model.g * state.model.pq_weight[..., :] * psi_b[...,:int(state.model.num_states/2)] +\
                np.conj(psi_a[...,int(state.model.num_states/2):]) * state.model.g * state.model.pq_weight[..., :] * psi_b[...,int(state.model.num_states/2):]
            return out

        def dh_qc_dzc(state, z_coord, psi_a, psi_b):
            return np.conj(dh_qc_dz(state, z_coord, psi_a, psi_b))
        
        def h_q(state):
                out = np.zeros((state.model.num_states, state.model.num_states), dtype=complex)
                for n in range(int(state.model.num_states/2) - 1):
                    out[n, n + 1] = -state.model.j
                    out[n + 1, n] = -state.model.j
                    out[n+int(state.model.num_states/2), n+int(state.model.num_states/2) + 1] = -state.model.j
                    out[n+int(state.model.num_states/2) + 1, n+int(state.model.num_states/2)] = -state.model.j
                if not(open):
                    out[0, int(state.model.num_states/2)-1] = -state.model.j
                    out[int(state.model.num_states/2)-1, 0] = -state.model.j
                    out[int(state.model.num_states/2), int(state.model.num_states)-1] = -state.model.j
                    out[int(state.model.num_states)-1, int(state.model.num_states/2)] = -state.model.j
                return out #+ np.identity(self.num_states)*2*self.j + np.identity(self.num_states)*(0.5/0.0252488)

        def h_qc(state,z_coord):
            h_qc_out = np.zeros((state.model.batch_size, state.model.num_branches,
                                 state.model.num_states, state.model.num_states), dtype=complex)
            h_qc_diag = state.model.g * state.model.pq_weight[np.newaxis, :] * (z_coord + np.conj(z_coord))
            np.einsum('...jj->...j', h_qc_out)[...] =  np.concatenate((h_qc_diag, h_qc_diag), axis=-1)#h_qc_diag
            return h_qc_out
        
        self.dh_qc_dz = dh_qc_dz
        self.dh_qc_dzc = dh_qc_dzc
        self.h_qc = h_qc
        self.h_q = h_q