import numpy as np
import auxilliary
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

        self.state_vars_list = []
        self.mf_observables = auxilliary.no_observables
        self.fssh_observables = auxilliary.no_observables
        self.cfssh_observables = auxilliary.no_observables
        self.num_states=2  # number of states

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
        self.dh_qc_dz_branch = dh_qc_dz_branch
        self.dh_qc_dzc_branch = dh_qc_dzc_branch

        

        

    def h_q(self):
            """
            Nearest-neighbor tight-binding Hamiltonian with periodic boundary conditions and dimension num_states.
            :return: h_q Hamiltonian
            """
            out = np.zeros((self.num_states, self.num_states), dtype=complex)
            out[0,0] = self.E
            out[1,1] = -self.E
            out[0,1] = self.V
            out[1,0] = self.V
            return out
    def h_qc_branch(self,z_branch):
        """
        Holstein Hamiltonian on a lattice in real-space, z and zc are frequency weighted
        :param z: z coordinate
        :param zc: z^{*} conjugate z
        :return: h_qc(z,z^{*}) Hamiltonian
        """
        h_qc_out = np.zeros((self.num_branches, self.num_states, self.num_states), dtype=complex)
        h_qc_out[:,0,0] = np.sum(self.g[np.newaxis,:] * np.sqrt(1/(2*self.m*self.h))[np.newaxis,:] * (z_branch + np.conj(z_branch)),axis=1)
        h_qc_out[:,1,1] = np.sum(-self.g[np.newaxis,:] * np.sqrt(1/(2*self.m*self.h))[np.newaxis,:] * (z_branch + np.conj(z_branch)),axis=1)
        return h_qc_out
    def h_c(self, z):
        """
        Harmonic oscillator Hamiltonian
        :param z: z(t)
        :param zc: conjugate z(t)
        :return: h_c(z,zc) Hamiltonian
        """
        return np.real(np.sum(self.h* np.conj(z) * z))

    def h_c_branch(self, z_branch):
        return np.real(np.sum(self.h*np.conj(z_branch)*z_branch, axis=1))
    def dh_c_dz_branch(self, z_branch):
        """
        Gradient of harmonic oscillator hamiltonian wrt z_branch
        :param z_branch: z coordinate in each branch
        :param sim: simulation object
        :return:
        """
        return self.h[np.newaxis, :] * np.conj(z_branch)

    def dh_c_dzc_branch(self, z_branch):
        """
        Gradient of harmonic oscillator hamiltonian wrt zc_branch
        :param z_branch: z coordinate in each branch
        :param sim: simulation object
        :return:
        """
        return self.h[np.newaxis, :] * z_branch
    def init_classical(self, seed=None):
        """
        Initialize classical coordiantes according to Boltzmann statistics
        :param sim: simulation object with temperature, harmonic oscillator mass and frequency
        :return: z = sqrt(w*h/2)*(q + i*(p/((w*h))), z* = sqrt(w*h/2)*(q - i*(p/((w*h)))
        """
        np.random.seed(seed)
        q = np.random.normal(loc=0, scale=np.sqrt(self.temp / (self.m * (self.h ** 2))), size=self.A)
        p = np.random.normal(loc=0, scale=np.sqrt(self.temp), size=self.A)
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