"""
This module contains the spin-boson Model class.
"""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients


class SpinBoson(Model):
    """
    Spin-Boson model class for the simulation framework.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}

        self.default_constants = {
            "kBT": 1,
            "V": 0.5,
            "E": 0.5,
            "A": 100,
            "W": 0.1,
            "l_reorg": 0.005,
            "boson_mass": 1,
        }
        super().__init__(self.default_constants, constants)
        self.dh_qc_dzc_inds = None
        self.dh_qc_dzc_mels = None
        self.dh_qc_dzc_shape = None
        self.update_dh_qc_dzc = False
        self.update_h_q = False

    def initialize_constants_model(self):
        """
        Initialize the model-specific constants.
        """
        A = self.constants.get("A", self.default_constants.get("A"))
        W = self.constants.get("W", self.default_constants.get("W"))
        boson_mass = self.constants.get(
            "boson_mass", self.default_constants.get("boson_mass")
        )
        self.constants.w = W * np.tan(((np.arange(A) + 1) - 0.5) * np.pi / (2 * A))
        self.constants.num_classical_coordinates = A
        self.constants.num_quantum_states = 2
        self.constants.classical_coordinate_weight = self.constants.w
        self.constants.classical_coordinate_mass = boson_mass * np.ones(A)

    def initialize_constants_h_c(self):
        """
        Initialize the constants for the classical Hamiltonian.
        """
        self.constants.harmonic_frequency = self.constants.get("w")

    def initialize_constants_h_qc(self):
        """
        Initialize the constants for the quantum-classical coupling Hamiltonian.
        """
        A = self.constants.get("A", self.default_constants.get("A"))
        l_reorg = self.constants.get("l_reorg", self.default_constants.get("l_reorg"))
        boson_mass = self.constants.get(
            "boson_mass", self.default_constants.get("boson_mass")
        )
        h = self.constants.classical_coordinate_weight
        w = self.constants.w
        self.constants.diagonal_linear_coupling = np.zeros((2, A))
        self.constants.diagonal_linear_coupling[0] = (
            w * np.sqrt(2 * l_reorg / A) * (1 / np.sqrt(2 * boson_mass * h))
        )
        self.constants.diagonal_linear_coupling[1] = (
            -w * np.sqrt(2 * l_reorg / A) * (1 / np.sqrt(2 * boson_mass * h))
        )

    def initialize_constants_h_q(self):
        """
        Initialize the constants for the quantum Hamiltonian.
        """
        self.constants.two_level_00 = self.constants.get(
            "E", self.default_constants.get("E")
        )
        self.constants.two_level_11 = -self.constants.get(
            "E", self.default_constants.get("E")
        )
        self.constants.two_level_01_re = self.constants.get(
            "V", self.default_constants.get("V")
        )
        self.constants.two_level_01_im = 0

    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_c,
        initialize_constants_h_qc,
        initialize_constants_h_q,
    ]
    ingredients = [
        ("h_q", ingredients.h_q_two_level),
        ("h_qc", ingredients.h_qc_diagonal_linear),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", ingredients.dh_qc_dzc_diagonal_linear),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop_function", ingredients.hop_harmonic),
    ]
