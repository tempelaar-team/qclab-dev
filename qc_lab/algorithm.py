"""
This module contains the Algorithm class, which is the base class for Algorithm objects in QC Lab.
"""

from qc_lab.constants import Constants


class Algorithm:
    """
    Base class for algorithms in the simulation framework.
    """

    def __init__(self, default_settings=None, settings=None):
        """
        Initializes the Algorithm class with default settings and provided settings.
        if no settings are provided, the default ones are used instead.
        """
        if settings is None:
            settings = {}
        if default_settings is None:
            default_settings = {}
        # Add default settings to the provided settings if not already present
        settings = {**default_settings, **settings}
        self.settings = Constants(self.update_algorithm_settings)
        for key, val in settings.items():
            setattr(self.settings, key, val)
        self.settings._init_complete = True
        self.update_algorithm_settings()

    def update_algorithm_settings(self):
        """
        Update algorithm settings. This method should be overridden by subclasses.
        """
        pass

    initialization_recipe = []
    update_recipe = []
    output_recipe = []
    output_variables = []

    def execute_initialization_recipe(self, sim, parameter, state):
        """
        Executes the initialization recipe for the given simulation.

        Args:
            sim (Simulation): The simulation object.
            parameter (Vector Object): The vector object containing trajectory parameters.
            state (Vector Object): The vector object containing dynamics variables.

        Returns:
            tuple: A tuple containing the updated parameter vector and state vector
                after applying all initialization functions.
        """
        for _, func in enumerate(sim.algorithm.initialization_recipe):
            parameter, state = func(sim, parameter, state)
        return parameter, state

    def execute_update_recipe(self, sim, parameter, state):
        """
        Executes the update recipe for the given simulation.

        Args:
            sim (Simulation): The simulation object.
            parameter (Vector Object): The vector object containing trajectory parameters.
            state (Vector Object): The vector object containing dynamics variables.

        Returns:
            tuple: A tuple containing the updated parameter vector and state vector
                after applying all initialization functions.
        """
        for _, func in enumerate(sim.algorithm.update_recipe):
            parameter, state = func(sim, parameter, state)
        return parameter, state

    def execute_output_recipe(self, sim, parameter, state):
        """
        Executes the output recipe for the given simulation.

        Args:
            sim (Simulation): The simulation object.
            parameter (Vector Object): The vector object containing trajectory parameters.
            state (Vector Object): The vector object containing dynamics variables.

        Returns:
            tuple: A tuple containing the updated parameter vector and state vector
                after applying all initialization functions.
        """
        for _, func in enumerate(sim.algorithm.output_recipe):
            parameter, state = func(sim, parameter, state)
        return parameter, state
