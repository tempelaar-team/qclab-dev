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


class AdiabaticMeanField(Algorithm):
    """
    Adiabatic Mean-field dynamics algorithm class.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {}
        super().__init__(self.default_settings, settings)

    initialization_recipe = [
        tasks.initialize_variable_objects,
        partial(tasks.copy_to_parameters, state_name="seed", parameters_name="seed"),
        tasks.initialize_norm_factor,
        tasks.initialize_z,
        tasks.update_h_q_tot,
    ]

    update_recipe = [
        # Begin RK4 integration steps.
        partial(tasks.update_classical_force, z_name="z"),
        partial(tasks.update_quantum_classical_force, wf_db_name="wf_adb"),
        tasks.update_adb_connection,
        tasks.update_z_rk4_k123,
        partial(tasks.update_classical_force, z_name="z_1"),
        partial(
            tasks.update_quantum_classical_force,
            z_name="z_1",
            wf_db_name="wf_adb",
            wf_changed=False,
        ),
        partial(tasks.update_z_rk4_k123, z_name="z", z_k_name="z_2", k_name="z_rk4_k2"),
        partial(tasks.update_classical_force, z_name="z_2"),
        partial(
            tasks.update_quantum_classical_force,
            z_name="z_2",
            wf_db_name="wf_adb",
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
            wf_db_name="wf_adb",
            wf_changed=False,
        ),
        tasks.update_z_rk4_k4,
        # End RK4 integration steps.
        tasks.update_wf_adb_rk4,
        tasks.update_h_q_tot,
    ]

    collect_recipe = [
        tasks.update_t,
        partial(tasks.update_dm_db_wf, wf_db_name="wf_adb", dm_db_name="dm_adb"),
        partial(tasks.update_quantum_energy_wf, wf_db_name="wf_adb"),
        tasks.update_classical_energy,
        tasks.collect_t,
        partial(tasks.collect_dm_db, dm_db_name="dm_adb", dm_db_output_name="dm_adb"),
        tasks.collect_classical_energy,
        tasks.collect_quantum_energy,
    ]


class AbInitioMeanField(Algorithm):
    """
    Adiabatic Mean-field dynamics algorithm class.

    Uses velocity verlet integration for the classical degrees of freedom,
    suitable for ab inito calculations or any other problem where the quantum-classical
    Hamiltonian only depends on the position (real part of z) coordinate.

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
        tasks.update_h_q_tot, # t=n
        tasks.update_classical_force, # t=n
        partial(tasks.update_quantum_classical_force, wf_db_name="wf_adb"), # t=n
        tasks.update_adb_connection, # t=n
    ]

    update_recipe = [
        # adb_connection_prev(n), adb_connection(n)
        partial(
            tasks.copy_in_state,
            copy_name="adb_connection_prev",
            orig_name="adb_connection",
        ),
        # h_prev(n), h(n)
        partial(tasks.copy_in_state, copy_name="h_q_tot_prev", orig_name="h_q_tot"),
        # qf_prev(n), qf(n)
        partial(
            tasks.copy_in_state,
            copy_name="quantum_classical_force_prev",
            orig_name="quantum_classical_force",
        ),
        partial(
            tasks.copy_in_state,
            copy_name="classical_force_prev",
            orig_name="classical_force",
        ),
        # q(n+1) = q(n) + dt*qf_p(n) + 0.5*qf_q(n)*dt**2
        tasks.update_q_velocity_verlet,
        # c(q(n+1), p(n))
        #tasks.update_wf_adb_coeffs,
        # qf_q(n+1), qf_p(n+1)
        partial(tasks.update_quantum_classical_force, wf_db_name="wf_adb"),
        # p(n+1) = p(n) + 0.5*(qf_q(n) + qf_q(n+1))*dt
        tasks.update_p_velocity_verlet,
        partial(tasks.update_classical_force, z_name="z"),
        partial(tasks.update_adb_connection, update_derivative_coupling=False),
        tasks.update_h_q_tot,
        tasks.update_wf_adb_eig,
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
