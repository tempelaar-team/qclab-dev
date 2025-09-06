"""
This module contains the Algorithm class.
"""

import copy
from qc_lab.constants import Constants


class Algorithm:
    """
    Base class for algorithms in QC Lab.
    """

    def __init__(self, default_settings=None, settings=None):
        if settings is None:
            settings = {}
        if default_settings is None:
            default_settings = {}
        # Merge default settings with user-provided settings.
        settings = {**default_settings, **settings}
        # Construct a Constants object to hold settings.
        # Pass a function to update settings when attributes change.
        self.settings = Constants(self.update_algorithm_settings)
        # Put settings from the dictionary into the Constants object.
        for key, val in settings.items():
            setattr(self.settings, key, val)
        # Indicate that initialization is complete.
        self.settings._init_complete = True
        # Call the method to update algorithm settings.
        self.update_algorithm_settings()
        # Copy the recipes and output variables to ensure they are not shared
        # across instances.
        self.initialization_recipe = copy.deepcopy(self.initialization_recipe)
        self.update_recipe = copy.deepcopy(self.update_recipe)
        self.collect_recipe = copy.deepcopy(self.collect_recipe)

    initialization_recipe = []
    update_recipe = []
    collect_recipe = []

    def update_algorithm_settings(self):
        """
        Update algorithm settings.

        This method should be overridden by subclasses.
        """

    def execute_recipe(self, sim, parameter, state, recipe):
        """
        Carry out the given recipe for the simulation by running 
        each task in the recipe.
        """
        for func in recipe:
            parameter, state = func(sim.algorithm, sim, parameter, state)
        return parameter, state
