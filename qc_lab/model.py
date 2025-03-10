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
            "temp": 1,
            "classical_mass": 1,
            "classical_frequency": 1,
            "num_classical_coordinates": 1,
            "num_quantum_states": 2,
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

    def initialize_constants_h_c(self):
        self.constants.harmonic_oscillator_frequency = self.constants.w

    def initialize_constants_h_qc(self):
        pass

    def initialize_constants_h_q(self):
        pass

    def initialize_constants_hop(self):
        self.constants.numerical_fssh_hop_gamma_range = 5
        self.constants.numerical_fssh_hop_num_iter = 10
        self.constants.numerical_fssh_hop_num_points = 100

    def initialize_constants_init_classical(self):
        self.constants.numerical_boltzmann_init_classical_num_points = 100
        self.constants.numerical_boltzmann_init_classical_max_amplitude = 5

    def initialize_constants_model(self):
        self.constants.num_classical_coordinates = (
            self.constants.num_classical_coordinates
        )
        self.constants.num_quantum_states = self.constants.num_quantum_states
        self.constants.classical_coordinate_weight = (
            np.ones(self.constants.num_classical_coordinates)
            * self.constants.classical_frequency
        )
        self.constants.classical_coordinate_mass = (
            np.ones(self.constants.num_classical_coordinates)
            * self.constants.classical_mass
        )

    h_c = ingredients.harmonic_oscillator_h_c
    dh_c_dzc = ingredients.dh_c_dzc_finite_differences
    dh_qc_dzc = ingredients.dh_qc_dzc_finite_differences
    init_classical = ingredients.default_numerical_boltzmann_init_classical
    hop_function = ingredients.default_numerical_fssh_hop
    linear_h_qc = False
    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_c,
        initialize_constants_h_qc,
        initialize_constants_h_q,
        initialize_constants_hop,
        initialize_constants_init_classical,
    ]
