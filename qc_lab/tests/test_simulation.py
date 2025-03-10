import pytest
import numpy as np
from qc_lab import Simulation, Vector, Data

def test_simulation_initialization():
    """
    Test the initialization of the Simulation class.

    This test verifies that the Simulation class is correctly initialized with the given parameters.
    It checks that the parameters are correctly assigned to the simulation instance.
    """
    parameters = {'tmax': 20, 'dt': 0.02, 'dt_output': 0.2, 'num_trajs': 5}
    sim = Simulation(parameters)
    assert sim.settings.tmax == 20
    assert sim.settings.dt == 0.02
    assert sim.settings.dt_output == 0.2
    assert sim.settings.num_trajs == 5
    assert sim.algorithm is None
    assert sim.model is None
    assert isinstance(sim.state, Vector)

def test_initialize_timesteps():
    """
    Test the initialize_timesteps method of the Simulation class.

    This test verifies that the timesteps are correctly initialized based on the simulation parameters.
    """
    parameters = {'tmax': 10, 'dt': 0.01, 'dt_output': 0.1}
    sim = Simulation(parameters)
    sim.initialize_timesteps()
    assert sim.settings.tmax_n == 1000
    assert sim.settings.dt_output_n == 10
    assert np.allclose(sim.settings.tdat, np.arange(0, 10.01, 0.01))
    assert np.allclose(sim.settings.tdat_n, np.arange(0, 1001, 1))
    assert np.allclose(sim.settings.tdat_output, np.arange(0, 10.1, 0.1))
    assert np.allclose(sim.settings.tdat_output_n, np.arange(0, 1010, 10))

def test_generate_seeds():
    """
    Test the generate_seeds method of the Simulation class.

    This test verifies that new seeds are correctly generated based on the existing seeds in the data object.
    """
    parameters = {'num_trajs': 5}
    sim = Simulation(parameters)
    data = Data()
    data.data_dic['seed'] = sim.generate_seeds(data)
    assert np.array_equal(data.data_dic['seed'], np.arange(5))


    data.data_dic['seed'] = np.array([1, 2, 3])
    data.data_dic['seed'] = sim.generate_seeds(data)
    assert np.array_equal(data.data_dic['seed'], np.array([4, 5, 6, 7, 8]))

if __name__ == "__main__":
    pytest.main()