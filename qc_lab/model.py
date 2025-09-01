"""
This module contains the Model class in QC Lab.
"""

import copy
from qc_lab.constants import Constants


class Model:
    """
    The Model object provides the framework for defining the default constants of a
    model and retrieving the ingredients of the model.
    """

    def __init__(self, default_constants=None, constants=None):
        if constants is None:
            constants = {}
        constants = {**default_constants, **constants}
        self.constants = Constants(self.initialize_constants)
        for key, val in constants.items():
            setattr(self.constants, key, val)
        self.constants._init_complete = True
        self.ingredients = copy.deepcopy(self.ingredients)
        self.update_h_q = True
        self.update_dh_qc_dzc = True

    def get(self, ingredient_name):
        """
        Retrieve an ingredient by name.

        Args:
            ingredient_name (str): Name of the ingredient to search for.

        Returns:
            tuple[callable | None, bool]: The ingredient function (or None if
            not found) and a flag indicating whether it exists.
        """
        for ingredient in self.ingredients[::-1]:
            if ingredient[0] == ingredient_name and ingredient[1] is not None:
                return ingredient[1], True
        return None, False

    def initialize_constants(self):
        """
        Initialize the constants for the model and ingredients.
        """
        for ingredient in self.ingredients[::-1]:
            if ingredient[0].startswith("_init_") and ingredient[1] is not None:
                ingredient[1](self, None)
        return

    ingredients = []
