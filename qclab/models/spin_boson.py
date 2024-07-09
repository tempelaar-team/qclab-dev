import numpy as np
import qclab.auxilliary as auxilliary
from numba import njit


class SpinBosonModel:
    def __init__(self, input_params):
        # Here we can define some input parameters that the model accepts and use them to construct the relevant aspects of the physical system 
        self.temp=input_params['temp']  # temperature
        self.V=input_params['V']  # offdiagonal coupling
        self.E=input_params['E']  # diagonal energy
        self.A = input_params['A'] # total number of classical oscillators
        self.W = input_params['W'] # characteristic frequency
        self.l=input_params['l'] # reorganization energy
        self.w=self.W*np.tan(((np.arange(self.A)+1) - (1/2))*np.pi/(2*self.A))  # classical oscillator frequency
        self.g=self.w*np.sqrt(2*self.l/self.A)  # electron-phonon coupling
        self.h=self.w
        self.m=np.ones_like(self.w)
        self.num_states=2  # number of states
        self.h_q_params = (self.E, self.V)
        self.h_qc_params = None
        self.num_classical_coordinates = self.A

        # initialize derivatives of h wrt z and zc
        # tensors have dimension # classical osc \times # quantum states \times # quantum states
        dz_mat = np.zeros((self.A, self.num_states, self.num_states), dtype=complex)
        dzc_mat = np.zeros((self.A, self.num_states, self.num_states), dtype=complex)
        dz_mat[:, 0, 0] = self.g*np.sqrt(1/(2*self.m*self.h))
        dz_mat[:, 1, 1] = -self.g*np.sqrt(1/(2*self.m*self.h))
        dzc_mat[:, 0, 0] = self.g*np.sqrt(1/(2*self.m*self.h))
        dzc_mat[:, 1, 1] = -self.g*np.sqrt(1/(2*self.m*self.h))
        dz_shape = np.shape(dz_mat)
        dzc_shape = np.shape(dzc_mat)
        # position of nonzero matrix elements
        dz_ind = np.where(np.abs(dz_mat) > 1e-12)
        dzc_ind = np.where(np.abs(dzc_mat) > 1e-12)
        # nonzero matrix elements
        dz_mels = dz_mat[dz_ind] + 0.0j
        dzc_mels = dzc_mat[dzc_ind] + 0.0j
        @njit
        def dh_qc_dz_branch(h_qc_params, psi_a_branch, psi_b_branch, z_branch):
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
        def dh_qc_dzc_branch(h_qc_params, psi_a_branch, psi_b_branch, z_branch):
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
        def h_q(h_q_params):
                """
                Nearest-neighbor tight-binding Hamiltonian with periodic boundary conditions and dimension num_states.
                :return: h_q Hamiltonian
                """
                E,V = h_q_params
                out = np.zeros((self.num_states, self.num_states), dtype=complex)
                out[0,0] = E
                out[1,1] = -E
                out[0,1] = V
                out[1,0] = V
                return out
        def h_qc_branch(h_qc_params,z_branch):
            """
            Holstein Hamiltonian on a lattice in real-space, z and zc are frequency weighted
            :param z: z coordinate
            :param zc: z^{*} conjugate z
            :return: h_qc(z,z^{*}) Hamiltonian
            """
            h_qc_out = np.zeros((len(z_branch), self.num_states, self.num_states), dtype=complex)
            h_qc_out[:,0,0] = np.sum(self.g[np.newaxis,:] * np.sqrt(1/(2*self.m*self.h))[np.newaxis,:] * (z_branch + np.conj(z_branch)),axis=1)
            h_qc_out[:,1,1] = np.sum(-self.g[np.newaxis,:] * np.sqrt(1/(2*self.m*self.h))[np.newaxis,:] * (z_branch + np.conj(z_branch)),axis=1)
            return h_qc_out
        self.dh_qc_dz_branch = dh_qc_dz_branch
        self.dh_qc_dzc_branch = dh_qc_dzc_branch
        self.h_qc_branch = h_qc_branch
        self.h_q = h_q
