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
    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {"fssh_deterministic": False,
                                    "num_branches": 1, 
                                    "gauge_fixing": 0}
        # add default_params to params if not already in params
        super().__init__(self.default_settings, settings)

        self.initialization_recipe = [
            lambda sim, parameters, state: tasks.initialize_z_coord(sim=sim, parameters=parameters, state=state, seed=state.seed),
            lambda sim, parameters, state: tasks.broadcast_var_to_branch_vectorized(sim=sim, parameters=parameters, state=state, val=state.z_coord,name='z_coord_branch'),
            lambda sim, parameters, state: tasks.broadcast_var_to_branch_vectorized(sim=sim, parameters=parameters, state=state, val=state.wf_db,name='wf_db_branch'),
            lambda sim, parameters, state: tasks.update_h_quantum(sim=sim, parameters=parameters, state=state, z_coord=state.z_coord_branch),
            lambda sim, parameters, state: tasks.diagonalize_matrix_vectorized(sim=sim, parameters=parameters, state=state, matrix=state.h_quantum,eigvals_name='eigvals', eigvecs_name='eigvecs'),
            lambda sim, parameters, state: tasks.gauge_fix_eigs_vectorized(sim=sim, parameters=parameters, state=state, eigvals=state.eigvals,eigvecs=state.eigvecs, eigvecs_previous=state.eigvecs,output_eigvecs_name='eigvecs', z_coord=state.z_coord_branch,gauge_fixing=2),
            lambda sim, parameters, state: tasks.copy_value_vectorized(sim=sim, parameters=parameters, state=state, val=state.eigvecs,name='eigvecs_previous'),
            lambda sim, parameters, state: tasks.copy_value_vectorized(sim=sim, parameters=parameters, state=state, val=state.eigvals,name='eigvals_previous'),
            lambda sim, parameters, state: tasks.basis_transform_vec_vectorized(sim=sim, parameters=parameters, state=state, input_vec=state.wf_db_branch,basis=np.einsum('...ij->...ji', state.eigvecs).conj(),output_name='wf_adb_branch'),
            tasks.initialize_random_values_fssh,
            tasks.initialize_active_surface,
            tasks.initialize_dm_adb_0_fssh_vectorized,
            tasks.update_act_surf_wf_vectorized,
            tasks.update_quantum_energy_fssh_vectorized,
            tasks.initialize_timestep_index,
        ]
        self.update_recipe = [
            lambda sim, parameters, state: tasks.copy_value_vectorized(sim=sim, parameters=parameters, state=state, val=state.eigvecs,name='eigvecs_previous'),
            lambda sim, parameters, state: tasks.copy_value_vectorized(sim=sim, parameters=parameters, state=state, val=state.eigvals,name='eigvals_previous'),
            lambda sim, parameters, state: tasks.update_z_coord_rk4(sim=sim, parameters=parameters, state=state, wf=state.act_surf_wf,z_coord=state.z_coord_branch,output_name='z_coord_branch',update_quantum_classical_forces_bool=False),
            lambda sim, parameters, state: tasks.update_wf_db_eigs_vectorized(
                sim=sim, parameters=parameters, state=state,
                wf_db=state.wf_db_branch,
                eigvals=state.eigvals,
                eigvecs=state.eigvecs,
                adb_name='wf_adb_branch',
                output_name='wf_db_branch'),
            lambda sim, parameters, state: tasks.update_h_quantum(sim=sim, parameters=parameters, state=state, z_coord=state.z_coord_branch),
            lambda sim, parameters, state: tasks.diagonalize_matrix_vectorized(sim=sim, parameters=parameters, state=state, matrix=state.h_quantum,eigvals_name='eigvals', eigvecs_name='eigvecs'),
            lambda sim, parameters, state: tasks.gauge_fix_eigs_vectorized(sim=sim, parameters=parameters, state=state, eigvals=state.eigvals,eigvecs=state.eigvecs,eigvecs_previous=state.eigvecs_previous,output_eigvecs_name='eigvecs', z_coord=state.z_coord_branch,gauge_fixing=sim.algorithm.settings.gauge_fixing),
            tasks.update_active_surface_fssh,
            tasks.update_act_surf_wf_vectorized,
            tasks.update_timestep_index,
        ]
        self.output_recipe = [
            tasks.update_dm_db_fssh_vectorized,
            tasks.update_quantum_energy_fssh_vectorized,
            lambda sim, parameters, state: tasks.update_classical_energy_fssh_vectorized(sim=sim, parameters=parameters, state=state, z_coord=state.z_coord_branch),
        ]
        self.output_variables = [
            'quantum_energy',
            'classical_energy',
            'dm_db',
        ]
