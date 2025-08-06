"""
This module contains the MF algorithm class.
"""

from functools import partial
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

    initialization_recipe = [
        tasks.update_variables,
        tasks.initialize_norm_factor,
        partial(tasks.state_to_parameters, state_name="seed", parameters_name="seed"),
        tasks.update_variables,
        partial(tasks.initialize_z, seed="state.seed"),
        partial(tasks.update_h_quantum, z="state.z"),
    ]

    update_recipe = [
        partial(tasks.update_h_quantum, z="state.z"),
        ## Begin RK4 integration steps.
        partial(tasks.update_classical_forces, z="state.z"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="state.wf_db",
            z="state.z",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k1, z="state.z", output_name="z_1"),
        partial(tasks.update_classical_forces, z="state.z_1"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="state.wf_db",
            z="state.z_1",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k2, z="state.z", output_name="z_2"),
        partial(tasks.update_classical_forces, z="state.z_2"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="state.wf_db",
            z="state.z_2",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k3, z="state.z", output_name="z_3"),
        partial(tasks.update_classical_forces, z="state.z_3"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="state.wf_db",
            z="state.z_3",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k4, z="state.z", output_name="z"),
        ## End RK4 integration steps.
        tasks.update_wf_db_rk4,
    ]

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_mf,
        partial(tasks.update_quantum_energy, wf="state.wf_db"),
        partial(tasks.update_classical_energy, z="state.z"),
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_classical_energy,
        tasks.collect_quantum_energy,
    ]
