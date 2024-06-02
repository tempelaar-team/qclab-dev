import numpy as np
import auxilliary
from numba import njit


class HolsteinLatticeModel:
    def __init__(self, input_params):
        self.num_states=input_params['num_states']  # number of states
        self.temp=input_params['temp']  # temperature
        self.j=input_params['j']  # hopping integral
        self.w=input_params['w']  # classical oscillator frequency
        self.g=input_params['g']  # electron-phonon coupling
        self.m=input_params['m']*np.ones(self.num_states) # mass of the classical oscillators
        self.h=np.ones(self.num_states)*self.w

        self.dynamics_method=None
        self.tmax = None
        self.dt = None
        self.dt_bath = None
        self.calc_mf_obs = None
        self.calc_fssh_obs = None
        self.calc_cfssh_obs = None
        self.mf_observables = None
        self.fssh_observables = None
        self.cfssh_observables = None
        self.gauge_fix = None
        self.dmat_const = None
        self.cfssh_branch_pair_update = None

        # initialize derivatives of h wrt z and zc
        # tensors have dimension # classical osc \times # quantum states \times # quantum states
        dz_mat = np.zeros((self.num_states, self.num_states, self.num_states), dtype=complex)
        dzc_mat = np.zeros((self.num_states, self.num_states, self.num_states), dtype=complex)
        for i in range(self.num_states):
            dz_mat[i, i, i] = self.g*np.sqrt(self.w)
            dzc_mat[i, i, i] = self.g*np.sqrt(self.w)
        dz_shape = np.shape(dz_mat)
        dzc_shape = np.shape(dzc_mat)
        # position of nonzero matrix elements
        dz_ind = np.where(np.abs(dz_mat) > 1e-12)
        dzc_ind = np.where(np.abs(dzc_mat) > 1e-12)
        # nonzero matrix elements
        dz_mels = dz_mat[dz_ind] + 0.0j
        dzc_mels = dzc_mat[dzc_ind] + 0.0j
        @njit
        def dh_qc_dz(psi_a, psi_b, z):
            """
            Computes <\psi_a| dH_qc/dz  |\psi_b> in each branch
            :param psi_a_branch: left vector in each branch
            :param psi_b_branch: right vector in each branch
            :param z_branch: z coordinate in each branch
            :return:
            """
            return auxilliary.matprod_sparse(dz_shape, dz_ind, dz_mels, psi_a, psi_b)
        @njit
        def dh_qc_dzc(psi_a, psi_b, z):
            """
            Computes <\psi_a| dH_qc/dzc  |\psi_b> in each branch
            :param psi_a_branch: left vector in each branch
            :param psi_b_branch: right vector in each branch
            :param z_branch: z coordinate in each branch
            :return:
            """
            return auxilliary.matprod_sparse(dzc_shape, dzc_ind, dzc_mels, psi_a, psi_b) # conjugation is done by matprod_sparse
        self.dh_qc_dz = dh_qc_dz
        self.dh_qc_dzc = dh_qc_dzc

        

        

    def h_q(self):
            """
            Nearest-neighbor tight-binding Hamiltonian with periodic boundary conditions and dimension num_states.
            :return: h_q Hamiltonian
            """
            out = np.zeros((self.num_states, self.num_states), dtype=complex)
            for n in range(self.num_states - 1):
                out[n, n + 1] = -self.j
                out[n + 1, n] = -self.j
            out[0, -1] = -self.j
            out[-1, 0] = -self.j
            return out

    def h_qc(self, z):
            """
            Holstein Hamiltonian on a lattice in real-space, z and zc are frequency weighted
            :param z: z coordinate
            :return: h_qc(z) Hamiltonian
            """
            return np.diag(self.g * np.sqrt(self.h) * (z + np.conj(z)))
    def h_c(self, z):
        """
        Harmonic oscillator Hamiltonian
        :param z: z(t)
        :param zc: conjugate z(t)
        :return: h_c(z,zc) Hamiltonian
        """
        return np.real(np.sum(self.h* np.conj(z) * z))
    def dh_c_dz(self, z):
        """
        Gradient of harmonic oscillator hamiltonian wrt z_branch
        :param z_branch: z coordinate in each branch
        :param sim: simulation object
        :return:
        """
        return self.h*np.conj(z)

    def dh_c_dzc(self, z):
        """
        Gradient of harmonic oscillator hamiltonian wrt zc_branch
        :param z_branch: z coordinate in each branch
        :param sim: simulation object
        :return:
        """
        return self.h * z
    def init_classical(self, seed=None):
        """
        Initialize classical coordiantes according to Boltzmann statistics
        :param sim: simulation object with temperature, harmonic oscillator mass and frequency
        :return: z = sqrt(w*h/2)*(q + i*(p/((w*h))), z* = sqrt(w*h/2)*(q - i*(p/((w*h)))
        """
        np.random.seed(seed)
        q = np.random.normal(loc=0, scale=np.sqrt(self.temp / (self.m * (self.h ** 2))), size=self.num_states)
        p = np.random.normal(loc=0, scale=np.sqrt(self.temp), size=self.num_states)
        z = np.sqrt(self.h * self.m / 2) * (q + 1.0j * (p / (self.h * self.m)))
        return z
    
    def hop(self, z, delta_z, ev_diff):
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
        akj_z = np.real(np.sum(self.h * delta_zc * delta_z))
        bkj_z = np.real(np.sum(1j * self.h * (zc * delta_z - z * delta_zc)))
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