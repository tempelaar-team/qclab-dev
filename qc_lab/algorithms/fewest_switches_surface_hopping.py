"""
This module contains the FSSH algorithm class.
"""

from functools import partial
from qc_lab.algorithm import Algorithm
from qc_lab import tasks


class FewestSwitchesSurfaceHopping(Algorithm):
    """
    Fewest switches surface hopping algorithm class.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {
            "fssh_deterministic": False,
            "gauge_fixing": "sign_overlap",
        }
        super().__init__(self.default_settings, settings)

    initialization_recipe = [
        tasks.initialize_norm_factor,
        tasks.initialize_branch_seeds,
        partial(tasks.initialize_z, seed="state.seed"),
        partial(tasks.update_h_quantum, z="state.z"),
        partial(
            tasks.diagonalize_matrix,
            matrix="state.h_quantum",
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        partial(
            tasks.gauge_fix_eigs,
            eigvals="state.eigvals",
            eigvecs="state.eigvecs",
            eigvecs_previous="state.eigvecs",
            output_eigvecs_name="eigvecs",
            z="state.z",
            gauge_fixing="phase_der_couple",
        ),
        partial(tasks.assign_to_state, name="eigvecs_previous", val="state.eigvecs"),
        partial(
            tasks.basis_transform_vec,
            input_vec="state.wf_db",
            basis="state.eigvecs",
            output_name="wf_adb",
        ),
        tasks.initialize_random_values_fssh,
        tasks.initialize_active_surface,
        tasks.initialize_dm_adb_0_fssh,
        tasks.update_act_surf_wf,
    ]

    update_recipe = [
        partial(tasks.assign_to_state, name="eigvecs_previous", val="state.eigvecs"),
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
        partial(
            tasks.update_wf_db_eigs,
            wf_db="state.wf_db",
            eigvals="state.eigvals",
            eigvecs="state.eigvecs",
            adb_name="wf_adb",
            output_name="wf_db",
        ),
        partial(tasks.update_h_quantum, z="state.z"),
        partial(
            tasks.diagonalize_matrix,
            matrix="state.h_quantum",
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        partial(
            tasks.gauge_fix_eigs,
            eigvals="state.eigvals",
            eigvecs="state.eigvecs",
            eigvecs_previous="state.eigvecs",
            output_eigvecs_name="eigvecs",
            z="state.z",
        ),
        tasks.update_hop_probs_fssh,
        tasks.update_hop_inds_fssh,
        tasks.update_hop_vals_fssh,
        tasks.update_z_hop_fssh,
        tasks.update_act_surf_hop_fssh,
        tasks.update_act_surf_wf,
    ]

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_fssh,
        partial(tasks.update_quantum_energy_fssh, wf="state.act_surf_wf"),
        partial(tasks.update_classical_energy_fssh, z="state.z"),
        # _update_quantum_energy_fssh,
        # _update_classical_energy_fssh,
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_quantum_energy,
        tasks.collect_classical_energy,
    ]
