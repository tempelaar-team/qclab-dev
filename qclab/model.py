"""
This file contains the Model class, which is the base class for Model objects in QC Lab.
"""

from qclab.parameter import Constants
from qclab.simulation import VectorObject
import qclab.ingredients as ingredients
import numpy as np

class Model:
    """
    Base class for models in the simulation framework.

    Attributes:
        parameters (ParameterClass): The parameters of the model.
    """

    def __init__(self, default_constants = None, constants=None):
        if constants is None:
            constants = {}

        internal_defaults = {'temp': 1,'classical_mass':1,'classical_frequency':1,
                             'num_classical_coordinates':1,
                                 'numerical_boltzmann_init_classical_num_points':100,
                                 'numerical_boltzmann_init_classical_max_amplitude':5,
                                 'numerical_fssh_hop_gamma_range':5,
                                 'numerical_fssh_hop_num_iter':10,
                                 'numerical_fssh_hop_num_points':100}
        # Add default constants to the provided constants if not already present
        constants = {**internal_defaults, **default_constants, **constants}
        self.constants = Constants(self.initialize_constants)
        for key, val in constants.items():
            setattr(self.constants, key, val)
        self.constants._init_complete = True
        self.initialize_constants()
        self.parameters = VectorObject()

    def initialize_constants(self):
        for func in self.initialization_functions:
            func(self)


    def h_q(self):
        """
        Quantum Hamiltonian function. This method should be overridden by subclasses.

        Returns:
            None
        """
        return

    def h_qc(self):
        """
        Quantum-classical Hamiltonian function. This method should be overridden by subclasses.

        Returns:
            None
        """
        return

    def h_c(self):
        """
        Classical Hamiltonian function. This method should be overridden by subclasses.

        Returns:
            None
        """
        return
    h_c = ingredients.harmonic_oscillator_h_c
    dh_c_dzc = ingredients.dh_c_dzc_finite_differences
    dh_qc_dzc = ingredients.dh_qc_dzc_finite_differences
    init_classical = ingredients.default_numerical_boltzmann_init_classical
    #init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical
    #hop_function = ingredients.default_numerical_fssh_hop
    initialization_functions = []
