"""
This file contains the Model class, which is the base class for Model objects in QC Lab.
"""

from qc_lab.constants import Constants
from qc_lab.vector import Vector
import qc_lab.ingredients as ingredients
import numpy as np


class Model:
    """
    Base class for models in the simulation framework.

    Attributes:
        parameters (ParameterClass): The parameters of the model.
    """

    def __init__(self, default_constants=None, constants=None):
        if constants is None:
            constants = {}
        internal_defaults = {
        }
        # Add default constants to the provided constants if not already present
        constants = {**internal_defaults, **default_constants, **constants}
        self.constants = Constants(self.initialize_constants)
        for key, val in constants.items():
            setattr(self.constants, key, val)
        self.constants._init_complete = True
        self.initialize_constants()
        self.parameters = Vector()

    def initialize_constants(self):
        for func in self.initialization_functions:
            func(self)

    def h_q(self):
        """
        Quantum Hamiltonian function. This method should be overridden by subclasses.
        """
        return

    def h_qc(self):
        """
        Quantum-classical Hamiltonian function. This method should be overridden by subclasses.
        """
        return
    
    def h_c(self):
        """
        Classical Hamiltonian function. This method should be overridden by subclasses.
        """
        return

    def initialize_constants_h_c(self):
        pass

    def initialize_constants_h_qc(self):
        pass

    def initialize_constants_h_q(self):
        pass

    def initialize_constants_model(self):
        pass

    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_c,
        initialize_constants_h_qc,
        initialize_constants_h_q,
    ]
