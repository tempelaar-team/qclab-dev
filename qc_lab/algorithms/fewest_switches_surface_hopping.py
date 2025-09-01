"""
This module contains the FewestSwitchesSurfaceHopping algorithm class.
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
        partial(tasks.initialize_z, seed="seed", name="z"),
        partial(tasks.update_h_quantum, z="z"),
        partial(
            tasks.diagonalize_matrix,
            matrix="h_quantum",
            eigvals="eigvals",
            eigvecs="eigvecs",
        ),
        partial(
            tasks.gauge_fix_eigs,
            eigvals="eigvals",
            eigvecs="eigvecs",
            eigvecs_previous="eigvecs",
            output_eigvecs_name="eigvecs",
            z="z",
            gauge_fixing="phase_der_couple",
        ),
        partial(tasks.copy_in_state, dest_name="eigvecs_previous", orig_name="eigvecs"),
        partial(
            tasks.basis_transform_vec,
            input_vec="wf_db",
            basis="eigvecs",
            output_name="wf_adb",
            db_to_adb=True,
        ),
        tasks.initialize_random_values_fssh,
        tasks.initialize_active_surface,
        tasks.initialize_dm_adb_0_fssh,
        tasks.update_act_surf_wf,
    ]

    update_recipe = [
        partial(tasks.copy_in_state, dest_name="eigvecs_previous", orig_name="eigvecs"),
        # Begin RK4 integration steps.
        partial(tasks.update_classical_forces, z="z"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="act_surf_wf",
            z="z",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k1, z="z", output_name="z_1"),
        partial(tasks.update_classical_forces, z="z_1"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="act_surf_wf",
            z="z_1",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k2, z="z", output_name="z_2"),
        partial(tasks.update_classical_forces, z="z_2"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="act_surf_wf",
            z="z_2",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k3, z="z", output_name="z_3"),
        partial(tasks.update_classical_forces, z="z_3"),
        partial(
            tasks.update_quantum_classical_forces,
            wf="act_surf_wf",
            z="z_3",
            use_gauge_field_force=False,
        ),
        partial(tasks.update_z_rk4_k4, z="z", output_name="z"),
        # End RK4 integration steps.
        partial(
            tasks.update_wf_db_eigs,
            wf_db="wf_db",
            eigvals="eigvals",
            eigvecs="eigvecs",
            adb_name="wf_adb",
            output_name="wf_db",
        ),
        partial(tasks.update_h_quantum, z="z"),
        partial(
            tasks.diagonalize_matrix,
            matrix="h_quantum",
            eigvals="eigvals",
            eigvecs="eigvecs",
        ),
        partial(
            tasks.gauge_fix_eigs,
            eigvals="eigvals",
            eigvecs="eigvecs",
            eigvecs_previous="eigvecs_previous",
            output_eigvecs_name="eigvecs",
            z="z",
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
        partial(tasks.update_quantum_energy_fssh, wf="act_surf_wf"),
        partial(tasks.update_classical_energy_fssh, z="z"),
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_quantum_energy,
        tasks.collect_classical_energy,
    ]
