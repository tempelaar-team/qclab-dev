"""
This module contains the FSSH algorithm class.
"""

import numpy as np
from qclab.algorithm import Algorithm
from qclab import tasks



class FewestSwitchesSurfaceHopping(Algorithm):
    """
    Fewest switches surface hopping algorithm class.
    """
    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}
        self.default_parameters = {"fssh_deterministic": False,
                                    "num_branches": 1, 
                                    "gauge_fixing": 0}
        # add default_params to params if not already in params
        super().__init__(self.default_parameters, parameters)

        self.initialization_recipe = [
            lambda sim, state: tasks.initialize_z_coord(
                sim=sim, state=state, seed=state.seed),
            lambda sim, state: tasks.broadcast_var_to_branch_vectorized(
                sim=sim, state=state, val=state.z_coord,name='z_coord_branch'),
            lambda sim, state: tasks.broadcast_var_to_branch_vectorized(
                sim=sim, state=state, val=state.wf_db,name='wf_db_branch'),
            lambda sim, state: tasks.update_h_quantum_vectorized(
                sim=sim, state=state, z_coord=state.z_coord_branch),
            lambda sim, state: tasks.diagonalize_matrix_vectorized(
                sim=sim, state=state, matrix=state.h_quantum,
                eigvals_name='eigvals', eigvecs_name='eigvecs'),
            lambda sim, state: tasks.gauge_fix_eigs_vectorized(
                sim=sim, state=state, eigvals=state.eigvals,
                eigvecs=state.eigvecs, eigvecs_previous=state.eigvecs,
                output_eigvecs_name='eigvecs', z_coord=state.z_coord_branch,
                gauge_fixing=2),
            lambda sim, state: tasks.copy_value_vectorized(
                sim=sim, state=state, val=state.eigvecs,name='eigvecs_previous'),
            lambda sim, state: tasks.copy_value_vectorized(
                sim=sim, state=state, val=state.eigvals,name='eigvals_previous'),
            lambda sim, state: tasks.basis_transform_vec_vectorized(
                sim=sim, state=state, input_vec=state.wf_db_branch,
                basis=np.einsum('...ij->...ji', state.eigvecs).conj(),output_name='wf_adb_branch'),
            lambda sim, state: tasks.initialize_random_values_fssh(
                sim=sim, state=state),
            lambda sim, state: tasks.initialize_active_surface(
                sim=sim, state=state),
            lambda sim, state: tasks.initialize_dm_adb_0_fssh_vectorized(
                sim=sim, state=state),
            lambda sim, state: tasks.update_act_surf_wf_vectorized(
                sim=sim, state=state),
            lambda sim, state: tasks.update_quantum_energy_fssh_vectorized(
                sim=sim, state=state),
            lambda sim, state: tasks.initialize_timestep_index(
                sim=sim, state=state),
        ]
        self.update_recipe = [
            lambda sim, state: tasks.copy_value_vectorized(
                sim=sim, state=state, val=state.eigvecs,name='eigvecs_previous'),
            lambda sim, state: tasks.copy_value_vectorized(
                sim=sim, state=state, val=state.eigvals,name='eigvals_previous'),
            lambda sim, state: tasks.update_z_coord_rk4_vectorized(
                sim=sim, state=state, wf=state.act_surf_wf,
                z_coord=state.z_coord_branch,
                output_name='z_coord_branch',
                update_quantum_classical_forces_bool=False),
            lambda sim, state: tasks.update_wf_db_eigs_vectorized(
                sim=sim, state=state, wf_db=state.wf_db_branch,
                eigvals=state.eigvals, eigvecs=state.eigvecs,
                adb_name='wf_adb_branch', output_name='wf_db_branch'),
            lambda sim, state: tasks.update_h_quantum_vectorized(
                sim=sim, state=state, z_coord=state.z_coord_branch),
            lambda sim, state: tasks.diagonalize_matrix_vectorized(
                sim=sim, state=state, matrix=state.h_quantum,
                eigvals_name='eigvals', eigvecs_name='eigvecs'),
            lambda sim, state: tasks.gauge_fix_eigs_vectorized(
                sim=sim, state=state, eigvals=state.eigvals,
                eigvecs=state.eigvecs,
                eigvecs_previous=state.eigvecs_previous,
                output_eigvecs_name='eigvecs', z_coord=state.z_coord_branch,
                gauge_fixing=sim.algorithm.parameters.gauge_fixing),
            lambda sim, state: tasks.update_active_surface_fssh(
                sim=sim, state=state),
            lambda sim, state: tasks.update_act_surf_wf_vectorized(
                sim=sim, state=state),
            lambda sim, state: tasks.update_timestep_index(sim=sim, state=state),
        ]
        self.output_recipe = [
            lambda sim, state: tasks.update_dm_db_fssh_vectorized(
                sim=sim, state=state),
            lambda sim, state: tasks.update_quantum_energy_fssh_vectorized(
                sim=sim, state=state),
            lambda sim, state: tasks.update_classical_energy_fssh_vectorized(
                sim=sim, state=state, z_coord=state.z_coord_branch),
        ]
        self.output_variables = [
            'quantum_energy',
            'classical_energy',
            'dm_db',
        ]
