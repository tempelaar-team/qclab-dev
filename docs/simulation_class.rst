Simulation Class Structure
===================================

A Simulation Class contains the following required attributes

Simulation Attributes
---------------------
 .. function:: num_trajs (int) -> 1

        An integer v*alue giving the numer of trajectories that the dynamics core should execute at a single time

* .. function:: num_branches (int) -> num_states

        An integer value giving the total number of branches (realizations of a single set of initial classical coordinates) to be executed. 
        if ``sh_deterministic=True`` then ``num_branches = num_states`` otherwise num_branches takes on an algorithm-dependent quantity. 

* .. function:: dt (float) -> 0.01

        A float giving the timestep for the propagation of the mixed quantum-classical dynamics

* .. function:: dt_output (float) -> 0.1

        A float giving the timestep for the calculation of observables, this will be the output resolution. 

* .. function:: tmax (float) -> 10

        A float giving the total simulation time

Physical System Attributes
--------------------------
* .. function:: num_states (int)

        An integer giving the number of quantum states (i.e. the dimension of the quantum Hamiltonian)

* .. function:: num_classical_coordinates (int)

        An integer value giving the total number of classical coordinates in a single realization of the physical system.

* .. function:: h (ndarray, ``shape=(num_classical_coordinates), dtype=float``)

        A numpy array with length ``num_classical_coordinates`` that serves as an arbitrary parameter to define the complex-valued classical coordinates. See reference [1] (Miyazaki 2024) for details on this. When under a harmonic potential, it is convenient to set ``h`` for each coordinate to be its frequency. 

* .. function:: h_q_params ( tuple)

        Any parameters needed to generate the Hamiltonian matrix of the quantum subsystem

* .. function:: h_qc_params (tuple)

        Any parameters needed to generate the Hamiltonian matrix describing the quantum-classical interaction. 

* .. function:: h_c_params (tuple)

        Any parameters needed to generate the Hamiltonian function of the classical subsystem. 

* .. function:: h_q(h_q_params: tuple) -> ndarray

    Calculate the Hamiltonian matrix of the quantum subsystem

    :Parameters:
        h_q_params : tuple

    :Returns:
        ndarray, ``shape=(num_states, num_states), dtype=complex``
            The quantum Hamiltonian


    :See Also:
        h_q_params

* .. function:: h_qc(h_qc_params: tuple, z_coord: ndarray) -> ndarray

    Calculate the Hamiltonian matrix of the quantum-classical interaction over the first dimension of 
    z_coord.

    :Parameters:
        h_qc_params : tuple
                The parameters needed to calculate the quantum-classical interaction
        z_coord : ndarray, ``shape=(num_branches * num_trajs, num_classical_coordinates), dtype=complex``
                The complex classical coordinates. 
    :Returns:
        ndarray, ``shape=(num_trajs*num_branches, num_states, num_states), dtype=complex``
            The quantum-classical interaction Hamiltonian for every term in z_coord.

    :See Also:
        h_qc_params
        z_coord

* .. function:: h_c(h_c_params: tuple, z_coord: ndarray) -> ndarray

    Calculate the Hamiltonian function of the classical subsystem over the first dimension of 
    z_coord.

    :Parameters:
        h_c_params : tuple
                Parameters needed to calculate the classical Hamiltonian function

        z_coord : ndarray, ``shape=(num_branches * num_trajs, num_classical_coordinates), dtype=complex``
                The complex classical coordinates

    :Returns:
        ndarray, ``shape=(num_trajs*num_branches, num_states, num_states), dtype=complex``
            The energy of the classical susbsytem for every term in z_coord.

    :See Also:
        h_c_params
        z_coord

* .. function:: dh_qc_dz(h_qc_params: tuple, wf_a: ndarray, wf_b: ndarray, z_coord: ndarray) -> ndarray

        Calculate the expectation value of the derivative of the quantum-classical interaction term with-respect-to the
        z coordinate over the first dimension of z_coord: <a| dH_{qc}/dz| b> 

        :Parameters:
                h_qc_params : tuple
                        Parameters needed to compute the quantum-classical interaction

                wf_a: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The left wavefunction in the expectation value, not conjugated or transposed: |a>
                
                wf_b: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The right wavefunction in the expectation value: |b>
                
                z_coord: ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
                        The complex classical coordinates. 
        
        :Returns:
                ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
        
        :See Also:
                dh_qc_dzc
                h_qc_params
                z_coord

* .. function:: dh_qc_dzc(h_qc_params: tuple, wf_a: ndarray, wf_b: ndarray, z_coord: ndarray) -> ndarray

        Calculate the expectation value of the derivative of the quantum-classical interaction term with-respect-to the
        conjugate z coordinate over the first dimension of z_coord: <a| dH_{qc}/dz*| b> 

        :Parameters:
                h_qc_params : tuple
                        Parameters needed to compute the quantum-classical interaction

                wf_a: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The left wavefunction in the expectation value, not conjugated or transposed: |a>
                
                wf_b: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The right wavefunction in the expectation value: |b>
                
                z_coord: ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
                        The complex classical coordinates. 
        
        :Returns:
                ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
        
        :See Also:
                dh_qc_dz
                h_qc_params
                z_coord

* .. function:: dh_c_dz(h_c_params: tuple, wf_a: ndarray, wf_b: ndarray, z_coord: ndarray) -> ndarray

        Calculate the expectation value of the derivative of the classical Hamiltonian with-respect-to the
        z coordinate over the first dimension of z_coord: <a| dH_{c}/dz| b> 

        :Parameters:
                h_c_params : tuple
                        Parameters needed to compute the classical Hamiltonian

                wf_a: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The left wavefunction in the expectation value, not conjugated or transposed: |a>
                
                wf_b: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The right wavefunction in the expectation value: |b>
                
                z_coord: ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
                        The complex classical coordinates. 
        
        :Returns:
                ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
        
        :See Also:
                dh_c_dzc
                h_c_params
                z_coord

* .. function:: dh_c_dzc(h_c_params: tuple, wf_a: ndarray, wf_b: ndarray, z_coord: ndarray) -> ndarray

        Calculate the expectation value of the derivative of the classical Hamiltonian with-respect-to the
        conjugate z coordinate over the first dimension of z_coord: <a| dH_{c}/dz| b> 

        :Parameters:
                h_c_params : tuple
                        Parameters needed to compute the classical Hamiltonian

                wf_a: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The left wavefunction in the expectation value, not conjugated or transposed: |a>
                
                wf_b: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The right wavefunction in the expectation value: |b>
                
                z_coord: ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
                        The complex classical coordinates. 
        
        :Returns:
                ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
        
        :See Also:
                dh_c_dz
                h_c_params
                z_coord

* .. function:: init_classical(sim: Simulation Class, seed: int) -> ndarray

        Initializes the classical coordinates given a random seed.

        :Parameters:
                sim: Simulation Class

                seed: int, an integer acting as a seed for randomness. 
        
        :Returns:
                ndarray, ``shape=(num_classical_coordinates), dtype=complex``
        
        :See Also:
                z_coord

* .. function:: wf_db (ndarray, ``shape=(num_states), dtype=complex``)

        Initial diabatic wavefunction. 


Surface Hopping Specific Attributes
-----------------------------------

The following functions and constants are required only for surface hopping (FSSH and CFSSH) methods. 

* .. function:: sh_deterministic (bool) -> True (FSSH, CFSSH)

        An boolean value that tells the FSSH or CFSSH algorithm to deterministically (if True) or stochastically (if False) sample
        branches.

* .. function:: num_branches (int) -> num_states (FSSH, CFSSH)

        An integer value telling the FSSH or CFSSH algorithm how many branches to sample if ``sh_deterministic=False``. 
        For FSSH ``num_branches >= 1`` and for CFSSH ``num_branches > 1``. There is in principle no upper bound but a good practice is 
        not to approach ``num_branches = num_states`` too closely after which ``sh_deterministic=True`` is the more efficient option. 

* .. function:: pab_cohere (bool) -> True (FSSH), False (CFSSH)

        A boolean value dictating if hopping probabilities are initially calculated with the adiabatic wavefunction coefficients (True) or with
        coefficients that were initialized as delta functions in the active surface (False). 
        The default values reflect the original formulations of FSSH and CFSSH. 

* .. function:: dmat_const (int) -> 1 (CFSSH)

        An integer (0 or 1) that dictates how the density matrix is to be constructed. 1 yields the original construction
        for CFSSH and 0 yields a less expensive but often similarly accurate construction that neglects the use of branch-pair 
        eigenvectors (note that this method has not yet been explored in a publication).

* .. function:: cfssh_branch_pair_update (int) -> 0 (CFSSH)

        An integer (0, 1, or 2) that determines how frequently branch-pair eigenvectors are updated. If 0, branch-pari eigenvectors
        are updated only when needed, opening the possibility that a gauge (sign) change might take place leading to inaccuracies. 
        If 1 then branch-pairs are updated every ``dt_output`` timestep. If 2 then branch-pairs are updated every ``dt`` timestep. 

* .. function:: hop(sim: Simulation Class, z: ndarray, delta_z: ndarray, ev_diff: float) -> ndarray, bool

        The "hopping" function in Surface Hopping algorithms. Determines if a hop is allowed and if so rescales the 
        complex classical coordinate in the direction of delta_z.

        :Parameters:
                sim : Simulation Class

                z: ndarray, ``shape=(num_classical_coordinates), dtype=complex``
                        The complex classical coordinates.

                delta_z: ndarray, ``shape=(num_classical_coordinates), dtype=complex``
                        The change in complex classical coordinates induced if a hop is successful.

                ev_diff: float
                        The change in energy of the quantum subsystem should a hop take place: e_{final} - e_{initial} 

        :Returns:
                ndarray, ``shape=(num_classical_coordinates), dtype=complex``
                        the classical coordinates that have either been rescaled (if a hop took place) or remained unchanged
                        if a hop did not take place. 
                bool
                        A boolean indicating if a hop took place (True) or not (False)

Example Simulation Class
------------------------

The algorithms implemented impart a number of default options on the Simulation Class if those attributes are left unspecified. 
These default options assume the user is using classical coordinates goverend by a harmonic potential sampled from a Boltzmann distribution. 
Therefore, in addition to the above options an additional attribute (``temp``) is required in the Simulation object. Assuming these default options
are to be used the following example instantiates a Spin-Boson model

.. math::
        \hat{H}_{q} = \left(\begin{array}{cc}
                            E & V \\
                            V & -E 
                            \end{array} \right)

.. math::
        \hat{H}_{qc} = \sum_{i}^{A}g_{i} (2 m_{i}h_{i})^{1/2}(z_{i}+z^{*}_{i})\left(\begin{array}{cc}
                                                                        1 & 0 \\
                                                                        0 & -1 
                                                                        \end{array} \right)

.. math:: 
        H_{c} = \sum_{i}^{A}h_{i}z_{i}^{*}z_{i}

In the above expressions, :math:`h_{i}=w_{i}` is the oscillation frequency of the i-th oscillator sampled from a Debye spectal density, :math:`A` is the total number of classical oscillators,
:math:`g_{i}` is the coupling, and :math:`E` and :math:`V` are diagonal energies and off-diagonal couplings, respectively. The term :math:`(2 m_{i}h_{i})^{1/2}(z_{i}+z^{*}_{i})` is equivalent to the position 
coordinate for real-valued classical coordinates. By reading parameters from an input dictionary (``input_params``) this model can be implemented as follows::
    class SpinBosonModel:
        def __init__(self, input_params):
            # Here we can define some input parameters that the model accepts and use them to construct the relevant aspects of the physical system 
            self.temp = input_params['temp']  # temperature
            self.V = input_params['V']  # offdiagonal coupling
            self.E = input_params['E']  # diagonal energy
            self.A = input_params['A']  # total number of classical oscillators
            self.W = input_params['W']  # characteristic frequency
            self.l = input_params['l']  # reorganization energy
            self.w = self.W * np.tan(((np.arange(self.A) + 1) - (1 / 2)) * np.pi / (2 * self.A))  # classical oscillator frequency
            self.g = self.w * np.sqrt(2 * self.l / self.A)  # electron-phonon coupling
            self.h = self.w
            self.m = np.ones_like(self.w)
            self.num_states = 2  # number of states
            self.h_q_params = (self.E, self.V)
            self.h_qc_params = None
            self.num_classical_coordinates = self.A

            # initialize derivatives of h wrt z and zc
            # tensors have dimension # classical osc \times # quantum states \times # quantum states
            dz_mat = np.zeros((self.A, self.num_states, self.num_states), dtype=complex)
            dzc_mat = np.zeros((self.A, self.num_states, self.num_states), dtype=complex)
            dz_mat[:, 0, 0] = self.g * np.sqrt(1 / (2 * self.m * self.h))
            dz_mat[:, 1, 1] = -self.g * np.sqrt(1 / (2 * self.m * self.h))
            dzc_mat[:, 0, 0] = self.g * np.sqrt(1 / (2 * self.m * self.h))
            dzc_mat[:, 1, 1] = -self.g * np.sqrt(1 / (2 * self.m * self.h))
            dz_shape = np.shape(dz_mat)
            dzc_shape = np.shape(dzc_mat)
            # position of nonzero matrix elements
            dz_ind = np.where(np.abs(dz_mat) > 1e-12)
            dzc_ind = np.where(np.abs(dzc_mat) > 1e-12)
            # nonzero matrix elements
            dz_mels = dz_mat[dz_ind] + 0.0j
            dzc_mels = dzc_mat[dzc_ind] + 0.0j

        @njit
        def dh_qc_dz(h_qc_params, psi_a, psi_b, z_coord):
            """
            Computes <\psi_a| dH_qc/dz  |\psi_b> in each branch
            :param psi_a: left vector in each branch/trajectory with shape (sim.num_trajs*sim.num_branches, sim.num_states)
            :param psi_b: right vector in each branch/trajectory with shape (sim.num_trajs*sim.num_branches, sim.num_states)
            :param z_coord: z coordinate in each branch with shape (sim.num_trajs*sim.num_branches, sim.num_classical_coordinates)
            :return:
            """
            out = np.ascontiguousarray(np.zeros((len(psi_a), dz_shape[0]))) + 0.0j
            for n in range(len(psi_a)):
                out[n] = auxiliary.matprod_sparse(dz_shape, dz_ind, dz_mels, psi_a[n], psi_b[n])
            return out

        @njit
        def dh_qc_dzc(h_qc_params, psi_a, psi_b, z_coord):
            """
            Computes <\psi_a| dH_qc/dzc  |\psi_b> in each branch
            :param psi_a: left vector in each branch/trajectory with shape (sim.num_trajs*sim.num_branches, sim.num_states)
            :param psi_b: right vector in each branch/trajectory with shape (sim.num_trajs*sim.num_branches, sim.num_states)
            :param z_coord: z coordinate in each branch with shape (sim.num_trajs*sim.num_branches, sim.num_classical_coordinates)
            :return:
            """
            out = np.ascontiguousarray(np.zeros((len(psi_a), dzc_shape[0]))) + 0.0j
            for n in range(len(psi_a)):
                out[n] = auxiliary.matprod_sparse(dzc_shape, dzc_ind, dzc_mels, psi_a[n],
                                                  psi_b[n])  # conjugation is done by matprod_sparse
            return out

        def h_q(h_q_params):
            """
            Nearest-neighbor tight-binding Hamiltonian with periodic boundary conditions and dimension num_states.
            :param h_q_params: tuple of parameters for h_q
            :return: h_q Hamiltonian
            """
            e, v = h_q_params
            out = np.zeros((self.num_states, self.num_states), dtype=complex)
            out[0, 0] = e
            out[1, 1] = -e
            out[0, 1] = v
            out[1, 0] = v
            return out

        def h_qc(h_qc_params, z_coord):
            """
            Holstein Hamiltonian on a lattice in real-space, z and zc are frequency weighted
            :param h_qc_params: tuple of parameters for h_qc
            :param z_coord: z coordinate
            :return: h_qc(z,z^{*}) Hamiltonian
            """
            h_qc_out = np.zeros((len(z_coord), self.num_states, self.num_states), dtype=complex)
            h_qc_out[:, 0, 0] = np.sum(self.g[np.newaxis, :] * np.sqrt(1 / (2 * self.m * self.h))[np.newaxis, :] * (
                        z_coord + np.conj(z_coord)), axis=1)
            h_qc_out[:, 1, 1] = np.sum(-self.g[np.newaxis, :] * np.sqrt(1 / (2 * self.m * self.h))[np.newaxis, :] * (
                        z_coord + np.conj(z_coord)), axis=1)
            return h_qc_out

        self.dh_qc_dz = dh_qc_dz
        self.dh_qc_dzc = dh_qc_dzc
        self.h_qc = h_qc
        self.h_q = h_q

Obviously the above instantiation of the Spin-Boson model does not include the simulation specific attributes. Because those depend strongly on the simulation being run we leave those
to be specified when the simulation is being run rather than at the level of the Simulation Class. 