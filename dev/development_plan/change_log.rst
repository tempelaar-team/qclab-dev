.. _change_log:

Change Log
=================


Version 1.0.1
-----------------

- Fixed typo in the ``_init_h_qc`` ingredient of ``HolsteinLattice`` model affecting the calculation of the ``diagonal_linear_coupling`` matrix.
  Previously, changing the ``classical_coordinate_weight`` parameter did not correctly influence the coupling terms. Now, the coupling terms properly scale with the ``classical_coordinate_weight`` as intended. This typo had no impact on the correctness of the ``HolsteinLattice`` model where ``classical_coordinate_weight`` was set to ``harmonic_frequency``. If this was changed, however, the results would have been incorrect.
- Fixed typo in the ``h_qc`` ingredient of ``HolsteinLatticeReciprocalSpace`` model affecting the calculation of the quantum-classical coupling Hamiltonian.
  Similar to the previous fix, the coupling terms now correctly scale with the ``classical_coordinate_weight`` parameter. This typo had no impact on the correctness of the ``HolsteinLatticeReciprocalSpace`` model where ``classical_coordinate_weight`` was set to ``harmonic_frequency``. If this was changed, however, the results would have been incorrect.
