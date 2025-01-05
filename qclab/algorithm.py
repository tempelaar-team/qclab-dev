"""
This module contains the Algorithm class, which is the base class for Algorithm objects in QC Lab.
"""

import inspect
from qclab.parameter import Parameter


class Algorithm:
    """
    Base class for algorithms in the simulation framework.

    Attributes:
        parameters (ParameterClass): The parameters of the algorithm.
        initialization_recipe (list): List of functions for initialization.
        update_recipe (list): List of functions for updating the state.
        output_recipe (list): List of functions for generating output.
        output_variables (list): List of variables to be output.
    """

    def __init__(self, default_parameters=None, parameters=None):
        """
        Initializes the AlgorithmClass with given parameters.

        Args:
            parameters (dict): A dictionary of parameters to initialize the algorithm.
        """
        if parameters is None:
            parameters = {}
        if default_parameters is None:
            default_parameters = {}
        # Add default parameters to the provided parameters if not already present
        parameters = {**default_parameters, **parameters}
        self.parameters = Parameter(self.update_algorithm_parameters)
        for key, val in parameters.items():
            setattr(self.parameters, key, val)
        self.parameters._init_complete = True
        self.output_recipe_vectorized_bool = None
        self.update_recipe_vectorized_bool = None
        self.initialization_recipe_vectorized_bool = None
        self.initialization_recipe = []
        self.update_recipe = []
        self.output_recipe = []
        self.output_variables = []
        self.parameters._init_complete = True
        self.update_algorithm_parameters()

    def update_algorithm_parameters(self):
        """
        Update algorithm parameters. This method should be overridden by subclasses.
        """

    def _is_vectorized(self, func):
        if '_vectorized' in inspect.getsource(func):
            return True
        else:
            return False

    def determine_vectorized(self):
        """
        Determine which functions in the recipes are vectorized.
        """
        self.initialization_recipe_vectorized_bool = list(
            map(self._is_vectorized, self.initialization_recipe))
        self.update_recipe_vectorized_bool = list(
            map(self._is_vectorized, self.update_recipe))
        self.output_recipe_vectorized_bool = list(
            map(self._is_vectorized, self.output_recipe))

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
