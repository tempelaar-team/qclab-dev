Simulation Class Structure
===================================

A simulation Class contains the following required attributes

Simulation Attributes
---------------------
* .. function:: num_trajs (int) -> 1

        An integer value giving the numer of trajectories that the dynamics core should execute at a single time

* .. function:: num_branches (int) -> num_states

        An integer value giving the total number of branches (realizations of a single set of initial classical coordinates) to be executed. 
        if ``sh_deterministic=True`` then ``num_branches = num_states`` otherwise num_branches takes on an algorithm-dependent quantity. 

* .. function:: dt (float) -> 0.01

        A float giving the timestep for the propagation of the mixed quantum-classical dynamics

* .. function:: dt_output (float) -> 0.1

        A float giving the timestep for the calculation of observables, this will be the output resolution. 

* .. function:: tmax (float) -> 10

        A float giving the total simulation time

* .. function:: observables(sim: Simulation Class, dyn: Algorithm Class) -> dict

        A function that calculates observables at the simulation level (for the branches and trajectories of that particular instantiation of the simulation)
        sim and dyn provide the user access to virtually all intermediate quantities in the algorithms. See the algorithms section for details on relevant quantity names. 

        :Parameters:
                sim : Simulation Class

                dyn: Algorithm Class

        :Returns:
                dict: A dictionary containing the calculated observables and their names (determined by the user)
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

* .. function:: h_qc_branch(h_qc_params: tuple, z_coord: ndarray) -> ndarray

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

* .. function:: h_c_branch(h_c_params: tuple, z_coord: ndarray) -> ndarray

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

* .. function:: dh_qc_dz_branch(h_qc_params: tuple, wf_a_branch: ndarray, wf_b_branch: ndarray, z_coord: ndarray) -> ndarray

        Calculate the expectation value of the derivative of the quantum-classical interaction term with-respect-to the
        z coordinate over the first dimension of z_coord: <a| dH_{qc}/dz| b> 

        :Parameters:
                h_qc_params : tuple
                        Parameters needed to compute the quantum-classical interaction

                wf_a_branch: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The left wavefunction in the expectation value, not conjugated or transposed: |a>
                
                wf_b_branch: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The right wavefunction in the expectation value: |b>
                
                z_coord: ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
                        The complex classical coordinates. 
        
        :Returns:
                ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
        
        :See Also:
                dh_qc_dzc_branch
                h_qc_params
                z_coord

* .. function:: dh_qc_dzc_branch(h_qc_params: tuple, wf_a_branch: ndarray, wf_b_branch: ndarray, z_coord: ndarray) -> ndarray

        Calculate the expectation value of the derivative of the quantum-classical interaction term with-respect-to the
        conjugate z coordinate over the first dimension of z_coord: <a| dH_{qc}/dz*| b> 

        :Parameters:
                h_qc_params : tuple
                        Parameters needed to compute the quantum-classical interaction

                wf_a_branch: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The left wavefunction in the expectation value, not conjugated or transposed: |a>
                
                wf_b_branch: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The right wavefunction in the expectation value: |b>
                
                z_coord: ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
                        The complex classical coordinates. 
        
        :Returns:
                ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
        
        :See Also:
                dh_qc_dz_branch
                h_qc_params
                z_coord

* .. function:: dh_c_dz_branch(h_c_params: tuple, wf_a_branch: ndarray, wf_b_branch: ndarray, z_coord: ndarray) -> ndarray

        Calculate the expectation value of the derivative of the classical Hamiltonian with-respect-to the
        z coordinate over the first dimension of z_coord: <a| dH_{c}/dz| b> 

        :Parameters:
                h_c_params : tuple
                        Parameters needed to compute the classical Hamiltonian

                wf_a_branch: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The left wavefunction in the expectation value, not conjugated or transposed: |a>
                
                wf_b_branch: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The right wavefunction in the expectation value: |b>
                
                z_coord: ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
                        The complex classical coordinates. 
        
        :Returns:
                ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
        
        :See Also:
                dh_c_dzc_branch
                h_c_params
                z_coord

* .. function:: dh_c_dzc_branch(h_c_params: tuple, wf_a_branch: ndarray, wf_b_branch: ndarray, z_coord: ndarray) -> ndarray

        Calculate the expectation value of the derivative of the classical Hamiltonian with-respect-to the
        conjugate z coordinate over the first dimension of z_coord: <a| dH_{c}/dz| b> 

        :Parameters:
                h_c_params : tuple
                        Parameters needed to compute the classical Hamiltonian

                wf_a_branch: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The left wavefunction in the expectation value, not conjugated or transposed: |a>
                
                wf_b_branch: ndarray, ``shape=(num_trajs*num_branches, num_states), dtype=complex``
                        The right wavefunction in the expectation value: |b>
                
                z_coord: ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
                        The complex classical coordinates. 
        
        :Returns:
                ndarray, ``shape=(num_trajs*num_branches, num_classical_coordinates), dtype=complex``
        
        :See Also:
                dh_c_dz_branch
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

        Calculate the expectation value of the derivative of the classical Hamiltonian with-respect-to the
        conjugate z coordinate over the first dimension of z_coord: <a| dH_{c}/dz| b> 

        :Parameters:
                sim : tuple
                        Parameters needed to compute the classical Hamiltonian

                z: ndarray, ``shape=(num_classical_coordinates), dtype=complex``
                        The complex classical coordinates.

                delta_z: ndarray, ``shape=(num_classical_coordinates), dtype=complex``
                        The change in complex classical coordinates induced if a hop is successful.

                ev_diff: float
                        The change in energy following a hop: e_{final} - e_{initial} 
                

        :Returns:
                ndarray, ``shape=(num_classical_coordinates), dtype=complex``
                        the classical coordinates following a hop
                bool
                        A boolean indicating if a hop took place (True) or not (False)

Example Simulation Class
-------------------

The algorithms implemented impart a number of default options on the Simulation Class ::

    class ModelClass:
        def __init__(self):
            # here we instantiate the requried attributes of the ModelClass that define the physical system
            self.num_states
            self.num_classical_coordinates
            self.h
            self.h_q_params
            self.h_qc_params
            self.h_c_params
            self.h_q
            self.h_qc_branch
            self.h_c

        


            