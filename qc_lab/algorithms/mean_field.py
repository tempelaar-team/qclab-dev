"""
This module contains the MF algorithm class.
"""

from qc_lab.algorithm import Algorithm
from qc_lab import tasks


class MeanField(Algorithm):
    """
    Mean-field dynamics algorithm class.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {}
        super().__init__(self.default_settings, settings)

    def _assign_seeds_to_parameters(self, sim, parameters, state):
        return tasks.assign_to_parameters(
            self, sim, parameters, state, name="seed", val=state.seed
        )

    def _initialize_z(self, sim, parameters, state):
        return tasks.initialize_z(self, sim, parameters, state, seed=state.seed)

    def _update_h_quantum(self, sim, parameters, state):
        return tasks.update_h_quantum(self, sim, parameters, state, z=state.z)

    initialization_recipe = [
        tasks.initialize_norm_factor,
        _assign_seeds_to_parameters,
        _initialize_z,
        _update_h_quantum,
    ]

    def _update_z_rk4(self, sim, parameters, state):
        return tasks.update_z_rk4(
            self,
            sim,
            parameters,
            state,
            z=state.z,
            output_name="z",
            wf=state.wf_db,
            use_gauge_field_force=False,
        )

    update_recipe = [
        _update_h_quantum,
        _update_z_rk4,
        tasks.update_wf_db_rk4,
    ]

    def _update_quantum_energy(self, sim, parameters, state):
        return tasks.update_quantum_energy(self, sim, parameters, state, wf=state.wf_db)

    def _update_classical_energy(self, sim, parameters, state):
        return tasks.update_classical_energy(self, sim, parameters, state, z=state.z)

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_mf,
        _update_quantum_energy,
        _update_classical_energy,
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_classical_energy,
        tasks.collect_quantum_energy,
    ]
