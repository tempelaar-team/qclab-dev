"""
This module defines the Model class in QC Lab.
"""

import copy
from qc_lab.constants import Constants


class Model:
    """
    The Model object provides the framework for defining the default constants of a model and
    retreiving the ingredients of the model.
    """

    def __init__(self, default_constants=None, constants=None):
        if constants is None:
            constants = {}
        internal_defaults = {}
        constants = {**internal_defaults, **default_constants, **constants}
        self.constants = Constants(self.initialize_constants)
        for key, val in constants.items():
            setattr(self.constants, key, val)
        self.constants._init_complete = True
        self.initialization_functions = copy.deepcopy(self.initialization_functions)
        self.ingredients = copy.deepcopy(self.ingredients)
        self.update_h_q = True
        self.update_dh_qc_dzc = True

    def get(self, ingredient_name):
        """
        Get the ingredient by its name. Returns the first instance of the ingredient in reverse order.
        """
        for ingredient in self.ingredients[::-1]:
            if ingredient[0] == ingredient_name and ingredient[1] is not None:
                return ingredient[1], True
        return None, False

    def initialize_constants(self):
        """
        Initialize the constants for the model and ingredients.
        """
        for func in self.initialization_functions:
            func(self)

    initialization_functions = []
    ingredients = []
