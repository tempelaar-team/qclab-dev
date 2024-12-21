import numpy as np
from qclab.parameter import ParameterClass
import inspect

class AlgorithmClass:
    """
    Base class for algorithms in the simulation framework.

    Attributes:
        parameters (ParameterClass): The parameters of the algorithm.
        initialization_recipe (list): List of functions for initialization.
        update_recipe (list): List of functions for updating the state.
        output_recipe (list): List of functions for generating output.
        output_variables (list): List of variables to be output.
    """

    def __init__(self, parameters=dict()):
        """
        Initializes the AlgorithmClass with given parameters.

        Args:
            parameters (dict): A dictionary of parameters to initialize the algorithm.
        """
        default_parameters = dict()
        # Add default parameters to the provided parameters if not already present
        parameters = {**default_parameters, **parameters}
        self.parameters = ParameterClass()
        for key, val in parameters.items():
            setattr(self.parameters, key, val)  # Set attributes

    initialization_recipe = []
    update_recipe = []
    output_recipe = []
    output_variables = []

    def _check_vectorized_old(self, func):
        """
        Check if a function is vectorized.

        Args:
            func (function): The function to check.

        Returns:
            bool: True if the function is vectorized, False otherwise.
        """
        
        func_name = func.__code__.co_names[1]
        return func_name.endswith('_vectorized')
    
    def _is_vectorized(self, func):
        if '_vectorized' in inspect.getsource(func):
            return True
        else:
            return False

        

    def determine_vectorized(self):
        """
        Determine which functions in the recipes are vectorized.
        """
        self.initialization_recipe_vectorized_bool = list(map(self._is_vectorized, self.initialization_recipe))
        self.update_recipe_vectorized_bool = list(map(self._is_vectorized, self.update_recipe))
        self.output_recipe_vectorized_bool = list(map(self._is_vectorized, self.output_recipe))

    def execute_initialization_recipe(self, sim, state_list, full_state):
        """
        Execute the initialization recipe.

        Args:
            sim (Simulation): The simulation object.
            state_list (list): List of state objects for each trajectory.
            full_state (State): The full state object containing all trajectories.

        Returns:
            tuple: Updated state_list and full_state.
        """
        for ind, func in enumerate(sim.algorithm.initialization_recipe):
            if sim.algorithm.initialization_recipe_vectorized_bool[ind]:
                full_state = func(sim, full_state)
            else:
                state_list = [func(sim, state) for state in state_list]
        return state_list, full_state

    def execute_update_recipe(self, sim, state_list, full_state):
        """
        Execute the update recipe.

        Args:
            sim (Simulation): The simulation object.
            state_list (list): List of state objects for each trajectory.
            full_state (State): The full state object containing all trajectories.

        Returns:
            tuple: Updated state_list and full_state.
        """
        for ind, func in enumerate(sim.algorithm.update_recipe):
            if sim.algorithm.update_recipe_vectorized_bool[ind]:
                full_state = func(sim, full_state)
            else:
                state_list = [func(sim, state) for state in state_list]
        return state_list, full_state

    def execute_output_recipe(self, sim, state_list, full_state):
        """
        Execute the output recipe.

        Args:
            sim (Simulation): The simulation object.
            state_list (list): List of state objects for each trajectory.
            full_state (State): The full state object containing all trajectories.

        Returns:
            tuple: Updated state_list and full_state.
        """
        for ind, func in enumerate(sim.algorithm.output_recipe):
            if sim.algorithm.output_recipe_vectorized_bool[ind]:
                full_state = func(sim, full_state)
            else:
                state_list = [func(sim, state) for state in state_list]
        return state_list, full_state