"""
This module contains the MF algorithm class.
"""

from qclab.algorithm import Algorithm
import qclab.tasks as tasks


class MeanField(Algorithm):
    """
    Mean-field dynamics algorithm class. 

    The algorithm class has a set of parameters that define 
    the algorithm Some of these parameters depends on the
    model i.e. num_branches is always the same as the number 
    of quantum states in the model for deterministic surface
    hopping methods.
    
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {}
        # add default_params to params if not already in params

        super().__init__(self.default_settings, settings)

        self.initialization_recipe = [
            lambda sim, parameters, state: tasks.initialize_z_coord(
                sim=sim, parameters=parameters, state=state, seed=state.seed),
            lambda sim, parameters, state: tasks.update_h_quantum_vectorized(
                sim=sim, parameters=parameters, state=state, z_coord=state.z_coord),
        ]
        self.update_recipe = [
            lambda sim, parameters, state: tasks.update_h_quantum_vectorized(
                sim=sim, parameters=parameters, state=state, z_coord=state.z_coord),
            lambda sim, parameters, state: tasks.update_z_coord_rk4_vectorized(
                sim=sim, parameters=parameters, state=state, z_coord=state.z_coord,
                output_name='z_coord', wf=state.wf_db,
                update_quantum_classical_forces_bool=False),
            tasks.update_wf_db_rk4_vectorized,
        ]
        self.output_recipe = [
            #lambda sim, parameters, state: tasks.update_dm_db_mf_vectorized(sim=sim,parameters=parameters, state=state),
            tasks.update_dm_db_mf_vectorized,
            lambda sim, parameters, state: tasks.update_quantum_energy_mf_vectorized(
                sim=sim, parameters=parameters, state=state, wf=state.wf_db),
            lambda sim, parameters, state: tasks.update_classical_energy_vectorized(
                sim=sim, parameters=parameters, state=state, z_coord=state.z_coord),
        ]
        self.output_variables = [
            'dm_db',
            'classical_energy',
            'quantum_energy',
        ]
