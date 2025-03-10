"""
This module contains the Algorithm class, which is the base class for Algorithm objects in QC Lab.
"""

from qc_lab.constants import Constants


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

    def execute_initialization_recipe(self, sim, parameter_vector, state_vector):
        """
        Executes the initialization recipe for the given simulation.

        Args:
            sim (Simulation): The simulation object.
            parameter_vector (Vector Object): The vector object containing trajectory parameters.
            state_vector (Vector Object): The vector object containing dynamics variables.

        Returns:
            tuple: A tuple containing the updated parameter vector and state vector
                after applying all initialization functions.
        """
        for _, func in enumerate(sim.algorithm.initialization_recipe):
            parameter_vector, state_vector = func(sim, parameter_vector, state_vector)
        return parameter_vector, state_vector

    def execute_update_recipe(self, sim, parameter_vector, state_vector):
        """
        Executes the update recipe for the given simulation.

        Args:
            sim (Simulation): The simulation object.
            parameter_vector (Vector Object): The vector object containing trajectory parameters.
            state_vector (Vector Object): The vector object containing dynamics variables.

        Returns:
            tuple: A tuple containing the updated parameter vector and state vector
                after applying all initialization functions.
        """
        for _, func in enumerate(sim.algorithm.update_recipe):
            parameter_vector, state_vector = func(sim, parameter_vector, state_vector)
        return parameter_vector, state_vector

    def execute_output_recipe(self, sim, parameter_vector, state_vector):
        """
        Executes the output recipe for the given simulation.

        Args:
            sim (Simulation): The simulation object.
            parameter_vector (Vector Object): The vector object containing trajectory parameters.
            state_vector (Vector Object): The vector object containing dynamics variables.

        Returns:
            tuple: A tuple containing the updated parameter vector and state vector
                after applying all initialization functions.
        """
        for _, func in enumerate(sim.algorithm.output_recipe):
            parameter_vector, state_vector = func(sim, parameter_vector, state_vector)
        return parameter_vector, state_vector
