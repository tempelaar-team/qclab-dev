from qclab.algorithm import AlgorithmClass
import qclab.tasks as tasks
from qclab.parameter import ParameterClass


class MeanField(AlgorithmClass):
    """
    Mean-field dynamics algorithm class. 

    The algorithm class has a set of parameters that define the algorithm Some of these parameters depends on the
    model i.e. num_branches is always the same as the number of quantum states in the model for deterministic surface
    hopping methods.
    
    """

    def __init__(self, parameters=dict()):
        self.default_parameters = dict()
        # add default_params to params if not already in params
        parameters = {**self.default_parameters, **parameters}
        self.parameters = ParameterClass()
        for key, val in parameters.items():
            setattr(self.parameters, key, val)

    initialization_recipe = [
        lambda sim, state: tasks.initialize_z_coord(sim=sim, state=state, seed=state.seed),
        lambda sim, state: tasks.update_h_quantum_vectorized(sim=sim, state=state, z_coord=state.z_coord),
    ]
    update_recipe = [
        lambda sim, state: tasks.update_h_quantum_vectorized(sim=sim, state=state, z_coord=state.z_coord),
        lambda sim, state: tasks.update_z_coord_rk4_vectorized(sim=sim, state=state, z_coord=state.z_coord,
                                                               output_name='z_coord', wf=state.wf_db,
                                                               update_quantum_classical_forces_bool=False),
        lambda sim, state: tasks.update_wf_db_rk4_vectorized(sim=sim, state=state),
    ]
    output_recipe = [
        lambda sim, state: tasks.update_dm_db_mf_vectorized(sim=sim, state=state),
        lambda sim, state: tasks.update_quantum_energy_mf_vectorized(sim=sim, state=state, wf=state.wf_db),
        lambda sim, state: tasks.update_classical_energy_vectorized(sim=sim, state=state, z_coord=state.z_coord),
    ]
    output_variables = [
        'dm_db',
        'classical_energy',
        'quantum_energy',
    ]
