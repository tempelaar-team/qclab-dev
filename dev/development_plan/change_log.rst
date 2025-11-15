.. _change_log:

Change Log
=================


Version 1.0.1
-----------------

- Fixed typo in the ``_init_h_qc`` ingredient of ``HolsteinLattice`` model affecting the calculation of the ``diagonal_linear_coupling`` matrix.
  Previously, changing the ``classical_coordinate_weight`` parameter did not correctly influence the coupling terms. Now, the coupling terms properly scale with the ``classical_coordinate_weight`` as intended. This typo had no impact on the correctness of the ``HolsteinLattice`` model where ``classical_coordinate_weight`` was set to ``harmonic_frequency``. If this was changed, however, the results would have been incorrect.
- Fixed typo in the ``h_qc`` ingredient of ``HolsteinLatticeReciprocalSpace`` model affecting the calculation of the quantum-classical coupling Hamiltonian.
  Similar to the previous fix, the coupling terms now correctly scale with the ``classical_coordinate_weight`` parameter. This typo had no impact on the correctness of the ``HolsteinLatticeReciprocalSpace`` model where ``classical_coordinate_weight`` was set to ``harmonic_frequency``. If this was changed, however, the results would have been incorrect.
- Added AdiabaticMeanField algorithm. 
- Added QChemASE Model class that uses the Q-Chem interface of the Atomic Simulation Environment (ASE) to perform ab initio quantum chemistry calculations.
- Added custom ASE Q-Chem calculator. 
- Added numerical constants. 


New Capabilities
--------------------

- Adiabatic Model Objects
  - An Adiabatic Model object is one where the system is defined in an unspecified adiabatic basis in which the Hamiltonian of the quantum subsystem is 
    diagonal.
  - In an Adiabatic Model Object ``h_q`` is the identity operator and ``h_qc`` contains the adiabatic energies on its diagonal and ``h_c`` is the classical Hamiltonian.
  


Version 2.0.0
-----------------

- Change the diagonal_linear_coupling term so that the constant is not changing wrt h