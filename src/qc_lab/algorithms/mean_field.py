"""
This module contains the MeanField algorithm class.
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
        tasks.initialize_variable_objects,
        tasks.initialize_norm_factor,
        partial(tasks.state_to_parameters, state_name="seed", parameters_name="seed"),
        partial(tasks.initialize_z, seed="seed", name="z"),
        partial(tasks.update_h_quantum, z="z"),
    ]

    update_recipe = [
        partial(tasks.update_h_quantum, z="z"),
        # Begin RK4 integration steps.
        partial(tasks.update_classical_forces, z="z"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="wf_db",
            z="z",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k1, z="z", output_name="z_1"),
        partial(tasks.update_classical_forces, z="z_1"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="wf_db",
            z="z_1",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k2, z="z", output_name="z_2"),
        partial(tasks.update_classical_forces, z="z_2"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="wf_db",
            z="z_2",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k3, z="z", output_name="z_3"),
        partial(tasks.update_classical_forces, z="z_3"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="wf_db",
            z="z_3",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k4, z="z", output_name="z"),
        # End RK4 integration steps.
        tasks.update_wf_db_rk4,
    ]

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_mf,
        partial(tasks.update_quantum_energy, wf="wf_db"),
        partial(tasks.update_classical_energy, z="z"),
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_classical_energy,
        tasks.collect_quantum_energy,
    ]
