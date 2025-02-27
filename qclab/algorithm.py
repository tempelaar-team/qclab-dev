"""
This module contains the Algorithm class, which is the base class for Algorithm objects in QC Lab.
"""

import inspect
from qclab.constants import Constants


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

    def __init__(self, default_settings=None, settings=None):
        """
        Initializes the AlgorithmClass with given parameters.

        Args:
            parameters (dict): A dictionary of parameters to initialize the algorithm.
        """
        if settings is None:
            settings = {}
        if default_settings is None:
            default_settings = {}
        # Add default parameters to the provided parameters if not already present
        settings = {**default_settings, **settings}
        self.settings = Constants(self.update_algorithm_parameters)
        for key, val in settings.items():
            setattr(self.settings, key, val)
        self.settings._init_complete = True
        self.output_recipe_vectorized_bool = None
        self.update_recipe_vectorized_bool = None
        self.initialization_recipe_vectorized_bool = None
        self.initialization_recipe = []
        self.update_recipe = []
        self.output_recipe = []
        self.output_variables = []
        self.settings._init_complete = True
        self.update_algorithm_parameters()

    def update_algorithm_parameters(self):
        """
        Update algorithm parameters. This method should be overridden by subclasses.
        """

    def _is_vectorized(self, func):
        if "_vectorized" in inspect.getsource(func):
            return True
        else:
            return False

    def determine_vectorized(self):
        """
        Determine which functions in the recipes are vectorized.
        """
        self.initialization_recipe_vectorized_bool = list(
            map(self._is_vectorized, self.initialization_recipe)
        )
        self.update_recipe_vectorized_bool = list(
            map(self._is_vectorized, self.update_recipe)
        )
        self.output_recipe_vectorized_bool = list(
            map(self._is_vectorized, self.output_recipe)
        )

    def execute_initialization_recipe(self, sim, parameter_vector, state_vector):
        for ind, func in enumerate(sim.algorithm.initialization_recipe):
            parameter_vector, state_vector = func(
                    sim, parameter_vector, state_vector
                )
        return parameter_vector, state_vector

    def execute_update_recipe(self, sim, parameter_vector, state_vector):
        for ind, func in enumerate(sim.algorithm.update_recipe):
            parameter_vector, state_vector = func(
                sim, parameter_vector, state_vector
            )
        return parameter_vector, state_vector

    def execute_output_recipe(self, sim, parameter_vector, state_vector):
        for ind, func in enumerate(sim.algorithm.output_recipe):
            parameter_vector, state_vector = func(
                sim, parameter_vector, state_vector
            )
        return parameter_vector, state_vector
