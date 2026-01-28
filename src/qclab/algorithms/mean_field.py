"""
This module contains the MeanField algorithm class.
"""

from functools import partial
from qclab.algorithm import Algorithm
from qclab import tasks


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
        tasks.initialize_z,
        tasks.update_h_q_tot,
    ]

    update_recipe = [
        # Begin RK4 integration steps.
        partial(tasks.update_classical_force, z_name="z"),
        tasks.update_quantum_classical_force,
        tasks.update_z_rk4_k123,
        partial(tasks.update_classical_force, z_name="z_1"),
        partial(
            tasks.update_quantum_classical_force,
            z_name="z_1",
            wf_changed=False,
        ),
        partial(tasks.update_z_rk4_k123, z_name="z", z_k_name="z_2", k_name="z_rk4_k2"),
        partial(tasks.update_classical_force, z_name="z_2"),
        partial(
            tasks.update_quantum_classical_force,
            z_name="z_2",
            wf_changed=False,
        ),
        partial(
            tasks.update_z_rk4_k123,
            z_name="z",
            z_k_name="z_3",
            k_name="z_rk4_k3",
            dt_factor=1.0,
        ),
        partial(tasks.update_classical_force, z_name="z_3"),
        partial(
            tasks.update_quantum_classical_force,
            z_name="z_3",
            wf_changed=False,
        ),
        tasks.update_z_rk4_k4,
        # End RK4 integration steps.
        tasks.update_wf_db_rk4,
        tasks.update_h_q_tot,
    ]

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_wf,
        tasks.update_quantum_energy_wf,
        tasks.update_classical_energy,
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_classical_energy,
        tasks.collect_quantum_energy,
    ]


class AbInitioMeanField(Algorithm):
    """
    Adiabatic Mean-field dynamics algorithm class.

    Uses velocity verlet integration for the classical degrees of freedom.

    Uses a substep propagation for the quantum wavefunction within each ``dt_update`` timestep.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {
            "update_wf_adb_eig_num_substeps": 10,
        }
        super().__init__(self.default_settings, settings)

    initialization_recipe = [
        tasks.initialize_variable_objects,
        partial(tasks.copy_to_parameters, state_name="seed", parameters_name="seed"),
        tasks.initialize_norm_factor,
        tasks.initialize_z,
        tasks.update_h_q_tot,
        tasks.update_classical_force,
        tasks.update_derivative_coupling_dzc,
        partial(tasks.update_quantum_classical_force, wf_db_name="wf_adb"),
        tasks.update_adb_connection,
    ]

    update_recipe = [
        partial(
            tasks.copy_in_state,
            copy_name="derivative_coupling_dzc_previous",
            orig_name="derivative_coupling_dzc",
        ),
        partial(
            tasks.copy_in_state,
            copy_name="adb_connection_previous",
            orig_name="adb_connection",
        ),
        partial(tasks.copy_in_state, copy_name="h_q_tot_previous", orig_name="h_q_tot"),
        partial(
            tasks.copy_in_state,
            copy_name="quantum_classical_force_previous",
            orig_name="quantum_classical_force",
        ),
        partial(
            tasks.copy_in_state,
            copy_name="classical_force_previous",
            orig_name="classical_force",
        ),
        partial(
            tasks.copy_in_state,
            copy_name="dh_qc_dzc_previous",
            orig_name="dh_qc_dzc",
        ),
        partial(
            tasks.copy_in_state,
            copy_name="z_previous",
            orig_name="z",
        ),
        tasks.update_q_velocity_verlet,
        tasks.update_ab_initio_properties,
        tasks.update_derivative_coupling_dzc,
        tasks.update_adb_connection,
        partial(
            tasks.update_wf_adb_hop_prob,
            calculate_hopping_probabilities=False,
        ),
        tasks.update_dh_qc_dzc,
        tasks.update_h_q_tot,
        partial(
            tasks.update_quantum_classical_force,
            wf_db_name="wf_adb",
            dh_qc_dzc_name="dh_qc_dzc",
            update_dh_qc_dzc_flag=False,
        ),
        # Should recalculate classical forces here
        tasks.update_p_velocity_verlet,
        tasks.update_classical_force,
    ]

    collect_recipe = [
        tasks.update_t,
        partial(tasks.update_dm_db_wf, wf_db_name="wf_adb"),
        partial(tasks.update_quantum_energy_wf, wf_db_name="wf_adb"),
        tasks.update_classical_energy,
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_classical_energy,
        tasks.collect_quantum_energy,
    ]
