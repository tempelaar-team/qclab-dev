from qclab.algorithm import AlgorithmClass
import qclab.tasks as tasks
from qclab.parameter import ParameterClass

class FewestSwitchesSurfaceHopping(AlgorithmClass):
    def __init__(self, parameters=dict()):
        default_parameters = dict(fssh_deterministic=False)
        # add default_params to params if not already in params
        parameters = {**default_parameters, **parameters}
        self.parameters = ParameterClass()
        for key, val in parameters.items():
            setattr(self.parameters, key, val)
    initialization_recipe = [
        lambda sim, state: tasks.initialize_z_coord(sim = sim, state = state, seed = state.seed),
    ]
    update_recipe = [
        lambda sim, state: tasks.update_h_quantum_vectorized(sim = sim, state = state, z_coord = state.z_coord),
    ]
    output_recipe = [
        lambda sim, state: tasks.update_dm_db_mf(sim = sim, state = state),
    ]
    output_variables = [
        'dm_db',
        'classical_energy',
        'quantum_energy',
    ]