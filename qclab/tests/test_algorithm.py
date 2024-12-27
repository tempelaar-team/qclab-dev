import pytest
import numpy as np
from qclab.algorithm import AlgorithmClass
import qclab.simulation as simulation


def test_algorithm_initialization():
    """
    Test the initialization of the AlgorithmClass.

    This test verifies that the AlgorithmClass is correctly initialized with the given parameters.
    It checks that the parameters are correctly assigned to the algorithm instance.

    Assertions:
        - The 'param1' attribute of the algorithm's parameters should be 1.
        - The 'param2' attribute of the algorithm's parameters should be 2.
    """
    parameters = {'param1': 1, 'param2': 2}
    algorithm = AlgorithmClass(parameters)
    assert algorithm.parameters.param1 == 1
    assert algorithm.parameters.param2 == 2


def test_is_vectorized():
    """
    Test the _is_vectorized method of the AlgorithmClass.

    This test verifies the following scenarios:
    1. A lambda function wrapping a non-vectorized dummy function should return False.
    2. A lambda function wrapping a non-vectorized dummy function assigned to the algorithm should return False.
    3. A lambda function wrapping a vectorized dummy function should return True.
    4. A lambda function wrapping a vectorized dummy function assigned to the algorithm should return True.
    """
    algorithm = AlgorithmClass()

    def dummy_func(x):
        pass

    lambda_dummy_func = lambda x: dummy_func(x)
    assert algorithm._is_vectorized(lambda_dummy_func) is False

    algorithm.dummy_func = dummy_func
    lambda_dummy_func = lambda x: algorithm.dummy_func(x)
    assert algorithm._is_vectorized(lambda_dummy_func) is False

    def dummy_func_vectorized(x):
        pass

    lambda_dummy_func_vectorized = lambda x: dummy_func_vectorized(x)
    assert algorithm._is_vectorized(lambda_dummy_func_vectorized) is True

    algorithm.dummy_func_vectorized = dummy_func_vectorized
    lambda_dummy_func_vectorized = lambda x: algorithm.dummy_func_vectorized(x)
    assert algorithm._is_vectorized(lambda_dummy_func_vectorized) is True


def test_determine_vectorized():
    """
    Test the determine_vectorized method of the AlgorithmClass.

    This test initializes an instance of AlgorithmClass and sets up dummy
    functions for the initialization, update, and output recipes. It then
    calls the determine_vectorized method and asserts that the
    initialization_recipe_vectorized_bool, update_recipe_vectorized_bool,
    and output_recipe_vectorized_bool attributes are correctly set to
    [False, True].

    The test ensures that the determine_vectorized method correctly
    identifies which functions in the recipes are vectorized.
    """
    algorithm = AlgorithmClass()

    def dummy_func():
        pass

    def dummy_func_vectorized():
        pass

    algorithm.initialization_recipe = [dummy_func, dummy_func_vectorized]
    algorithm.update_recipe = [dummy_func, dummy_func_vectorized]
    algorithm.output_recipe = [dummy_func, dummy_func_vectorized]
    algorithm.determine_vectorized()
    assert algorithm.initialization_recipe_vectorized_bool == [False, True]
    assert algorithm.update_recipe_vectorized_bool == [False, True]
    assert algorithm.output_recipe_vectorized_bool == [False, True]


def test_execute_initialization_recipe():
    """
    Test the execution of the initialization recipe in the AlgorithmClass.

    This test verifies that the initialization recipe correctly modifies the state
    and full_state objects. It uses dummy functions to increment the 'non_vectorized_output'
    and 'vectorized_output' fields in the state.

    The test performs the following steps:
    1. Create an instance of AlgorithmClass and define dummy initialization functions.
    2. Set the initialization recipe with the dummy functions.
    3. Create a batch of seeds and initialize state objects.
    4. Add 'non_vectorized_output' and 'vectorized_output' fields to the full_state.
    5. Execute the initialization recipe and verify the outputs.

    Assertions:
    - Verify that the 'non_vectorized_output' and 'vectorized_output' fields in each state
      are correctly incremented.
    - Verify that the 'non_vectorized_output' and 'vectorized_output' fields in the full_state
      are correctly incremented.
    - Verify that the 'non_vectorized_output' and 'vectorized_output' fields retrieved using
      the get method are correctly incremented.
    """
    algorithm = AlgorithmClass()

    def dummy_func(sim, state):
        state.modify('non_vectorized_output', state.get('non_vectorized_output') + 1)
        return state

    def dummy_func_vectorized(sim, state):
        state.modify('vectorized_output', state.get('vectorized_output') + 1)
        return state

    algorithm.initialization_recipe = [dummy_func, dummy_func_vectorized]
    batch_seeds_single = np.arange(3)
    state = simulation.State()
    state_list = [state.copy() for _ in batch_seeds_single]

    # Initialize seed in each state
    for n in range(len(batch_seeds_single)):
        state_list[n].add('seed', batch_seeds_single[n][np.newaxis])

    # Create a full_state
    full_state = simulation.new_full_state(state_list)
    full_state.add('non_vectorized_output', np.arange(3).reshape(3, 1))
    full_state.add('vectorized_output', np.arange(3).reshape(3, 1))
    state_list = simulation.new_state_list(full_state)
    sim = simulation.SimulationClass()
    sim.algorithm = algorithm
    sim.algorithm.determine_vectorized()
    state_list, full_state = algorithm.execute_initialization_recipe(sim, state_list, full_state)
    assert state_list[0].non_vectorized_output == np.array([1])
    assert state_list[1].non_vectorized_output == np.array([2])
    assert state_list[2].non_vectorized_output == np.array([3])
    assert np.all(full_state.non_vectorized_output == np.array([1, 2, 3]).reshape(3, 1))
    assert state_list[0].vectorized_output == np.array([1])
    assert state_list[1].vectorized_output == np.array([2])
    assert state_list[2].vectorized_output == np.array([3])
    assert np.all(full_state.vectorized_output == np.array([1, 2, 3]).reshape(3, 1))
    assert state_list[0].get('non_vectorized_output') == np.array([1])
    assert state_list[1].get('non_vectorized_output') == np.array([2])
    assert state_list[2].get('non_vectorized_output') == np.array([3])
    assert np.all(full_state.get('non_vectorized_output') == np.array([1, 2, 3]).reshape(3, 1))
    assert state_list[0].get('vectorized_output') == np.array([1])
    assert state_list[1].get('vectorized_output') == np.array([2])
    assert state_list[2].get('vectorized_output') == np.array([3])
    assert np.all(full_state.get('vectorized_output') == np.array([1, 2, 3]).reshape(3, 1))


def test_execute_update_recipe():
    """
    Test the execution of the update recipe in the AlgorithmClass.

    This test verifies that the update_recipe method correctly modifies the state
    and full_state objects using both non-vectorized and vectorized functions.

    The test performs the following steps:
    1. Initializes an instance of AlgorithmClass and defines two dummy functions
       for non-vectorized and vectorized updates.
    2. Sets the update_recipe attribute of the algorithm instance to the dummy functions.
    3. Creates a batch of seeds and initializes a list of state objects.
    4. Adds the seeds and initial output values to the state and full_state objects.
    5. Executes the update_recipe method and verifies that the state and full_state
       objects are correctly updated.

    Assertions:
    - The non-vectorized output in each state object is incremented by 1.
    - The vectorized output in each state object is incremented by 1.
    - The non-vectorized output in the full_state object matches the expected values.
    - The vectorized output in the full_state object matches the expected values.
    - The 'non_vectorized_output' and 'vectorized_output' keys in each state object
      and the full_state object match the expected values.
    """
    algorithm = AlgorithmClass()

    def dummy_func(sim, state):
        state.modify('non_vectorized_output', state.get('non_vectorized_output') + 1)
        return state

    def dummy_func_vectorized(sim, state):
        state.modify('vectorized_output', state.get('vectorized_output') + 1)
        return state

    algorithm.update_recipe = [dummy_func, dummy_func_vectorized]
    batch_seeds_single = np.arange(3)
    state = simulation.State()
    state_list = [state.copy() for _ in batch_seeds_single]

    # Initialize seed in each state
    for n in range(len(batch_seeds_single)):
        state_list[n].add('seed', batch_seeds_single[n][np.newaxis])

    # Create a full_state
    full_state = simulation.new_full_state(state_list)
    full_state.add('non_vectorized_output', np.arange(3).reshape(3, 1))
    full_state.add('vectorized_output', np.arange(3).reshape(3, 1))
    state_list = simulation.new_state_list(full_state)
    sim = simulation.SimulationClass()
    sim.algorithm = algorithm
    sim.algorithm.determine_vectorized()
    state_list, full_state = algorithm.execute_update_recipe(sim, state_list, full_state)
    assert state_list[0].non_vectorized_output == np.array([1])
    assert state_list[1].non_vectorized_output == np.array([2])
    assert state_list[2].non_vectorized_output == np.array([3])
    assert np.all(full_state.non_vectorized_output == np.array([1, 2, 3]).reshape(3, 1))
    assert state_list[0].vectorized_output == np.array([1])
    assert state_list[1].vectorized_output == np.array([2])
    assert state_list[2].vectorized_output == np.array([3])
    assert np.all(full_state.vectorized_output == np.array([1, 2, 3]).reshape(3, 1))
    assert state_list[0].get('non_vectorized_output') == np.array([1])
    assert state_list[1].get('non_vectorized_output') == np.array([2])
    assert state_list[2].get('non_vectorized_output') == np.array([3])
    assert np.all(full_state.get('non_vectorized_output') == np.array([1, 2, 3]).reshape(3, 1))
    assert state_list[0].get('vectorized_output') == np.array([1])
    assert state_list[1].get('vectorized_output') == np.array([2])
    assert state_list[2].get('vectorized_output') == np.array([3])
    assert np.all(full_state.get('vectorized_output') == np.array([1, 2, 3]).reshape(3, 1))


def test_execute_output_recipe():
    """
    Test the execution of the output recipe in the AlgorithmClass.

    This test verifies that the output recipe functions correctly modify the state
    and full_state objects as expected. It checks both non-vectorized and vectorized
    outputs.

    The test performs the following steps:
    1. Initializes an AlgorithmClass instance and sets its output_recipe with two dummy functions.
    2. Creates a batch of seeds and initializes a list of state objects.
    3. Adds 'seed', 'non_vectorized_output', and 'vectorized_output' attributes to the states.
    4. Creates a full_state object from the state list and adds the same attributes.
    5. Initializes a Simulation instance and assigns the algorithm to it.
    6. Executes the output recipe using the algorithm and verifies the modifications
       to the state_list and full_state objects.

    Assertions:
    - Checks that the 'non_vectorized_output' and 'vectorized_output' attributes in
      state_list and full_state are correctly modified by the dummy functions.
    - Ensures that the values in the state_list and full_state match the expected
      results after executing the output recipe.
    """
    algorithm = AlgorithmClass()

    def dummy_func(sim, state):
        state.modify('non_vectorized_output', state.get('non_vectorized_output') + 1)
        return state

    def dummy_func_vectorized(sim, state):
        state.modify('vectorized_output', state.get('vectorized_output') + 1)
        return state

    algorithm.output_recipe = [dummy_func, dummy_func_vectorized]
    batch_seeds_single = np.arange(3)
    state = simulation.State()
    state_list = [state.copy() for _ in batch_seeds_single]

    # Initialize seed in each state
    for n in range(len(batch_seeds_single)):
        state_list[n].add('seed', batch_seeds_single[n][np.newaxis])

    # Create a full_state
    full_state = simulation.new_full_state(state_list)
    full_state.add('non_vectorized_output', np.arange(3).reshape(3, 1))
    full_state.add('vectorized_output', np.arange(3).reshape(3, 1))
    state_list = simulation.new_state_list(full_state)
    sim = simulation.SimulationClass()
    sim.algorithm = algorithm
    sim.algorithm.determine_vectorized()
    state_list, full_state = algorithm.execute_output_recipe(sim, state_list, full_state)
    assert state_list[0].non_vectorized_output == np.array([1])
    assert state_list[1].non_vectorized_output == np.array([2])
    assert state_list[2].non_vectorized_output == np.array([3])
    assert np.all(full_state.non_vectorized_output == np.array([1, 2, 3]).reshape(3, 1))
    assert state_list[0].vectorized_output == np.array([1])
    assert state_list[1].vectorized_output == np.array([2])
    assert state_list[2].vectorized_output == np.array([3])
    assert np.all(full_state.vectorized_output == np.array([1, 2, 3]).reshape(3, 1))
    assert state_list[0].get('non_vectorized_output') == np.array([1])
    assert state_list[1].get('non_vectorized_output') == np.array([2])
    assert state_list[2].get('non_vectorized_output') == np.array([3])
    assert np.all(full_state.get('non_vectorized_output') == np.array([1, 2, 3]).reshape(3, 1))
    assert state_list[0].get('vectorized_output') == np.array([1])
    assert state_list[1].get('vectorized_output') == np.array([2])
    assert state_list[2].get('vectorized_output') == np.array([3])
    assert np.all(full_state.get('vectorized_output') == np.array([1, 2, 3]).reshape(3, 1))

if __name__ == "__main__":
    pytest.main()