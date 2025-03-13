""" 
This file contains the spin-boson model class.
"""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients


class SpinBosonModel(Model):
    """
    Spin-Boson model class for the simulation framework.
    """

    def __init__(self, constants=None):
        """
        Initializes the SpinBosonModel with given constants.

        Args:
            constants (dict, optional): A dictionary of constants for the model.
        """
        if constants is None:
            constants = {}
        self.default_constants = {
            "temp": 1,
            "V": 0.5,
            "E": 0.5,
            "A": 100,
            "W": 0.1,
            "l_reorg": 0.02 / 4,
            "boson_mass": 1,
        }
        super().__init__(self.default_constants, constants)
        self.dh_qc_dzc_inds = None
        self.dh_qc_dzc_mels = None

    def initialize_constants_model(self):
        """
        Initialize the model-specific constants.
        """
        num_bosons = self.constants.get("A", self.default_constants.get("A"))
        char_freq = self.constants.get("W", self.default_constants.get("W"))
        w = self.constants.get("w", self.default_constants.get("w"))
        boson_mass = self.constants.get(
            "boson_mass", self.default_constants.get("boson_mass")
        )
        self.constants.num_classical_coordinates = num_bosons
        self.constants.num_quantum_states = 2
        self.constants.w = char_freq * np.tan(
            ((np.arange(num_bosons) + 1) - 0.5) * np.pi / (2 * num_bosons)
        )
        self.constants.classical_coordinate_weight = w
        self.constants.classical_coordinate_mass = boson_mass * np.ones(num_bosons)

    def initialize_constants_h_c(self):
        """
        Initialize the constants for the classical Hamiltonian.
        """
        w = self.constants.get("w", self.default_constants.get("w"))
        self.constants.harmonic_oscillator_frequency = w

    def initialize_constants_h_qc(self):
        """
        Initialize the constants for the quantum-classical coupling Hamiltonian.
        """
        num_bosons = self.constants.get("A", self.default_constants.get("A"))
        w = self.constants.get("w", self.default_constants.get("w"))
        l_reorg = self.constants.get("l_reorg", self.default_constants.get("l_reorg"))
        self.constants.g = w * np.sqrt(
            2 * l_reorg / num_bosons
        )  # Electron-phonon coupling

    def initialize_constants_h_q(self):
        """
        Initialize the constants for the quantum Hamiltonian.
        """
        self.constants.two_level_system_a = self.constants.get(
            "E", self.default_constants.get("E")
        )
        self.constants.two_level_system_b = -self.constants.get(
            "E", self.default_constants.get("E")
        )
        self.constants.two_level_system_c = self.constants.get(
            "V", self.default_constants.get("V")
        )
        self.constants.two_level_system_d = 0

# @ingredients.vectorize_ingredient
    # def h_qc(self, constants, parameters, **kwargs):
    #     z = kwargs.get("z_coord")
    #     g = constants.g
    #     m = constants.classical_coordinate_mass
    #     h = constants.classical_coordinate_weight
    #     h_qc = np.zeros((2, 2), dtype=complex)
    #     h_qc[0, 0] = np.sum(g * np.sqrt(1 / (2 * m * h)) * (z + np.conj(z)))
    #     h_qc[1, 1] = -h_qc[0, 0]
    #     return h_qc

    # @ingredients.make_ingredient_sparse
    # @ingredients.vectorize_ingredient
    # def dh_qc_dzc(self, constants, parameters, **kwargs):
    #     m = constants.classical_coordinate_mass
    #     g = constants.g
    #     h = constants.classical_coordinate_weight
    #     dh_qc_dzc = np.zeros((constants.A, 2, 2), dtype=complex)
    #     dh_qc_dzc[:, 0, 0] = g * np.sqrt(1 / (2 * m * h))
    #     dh_qc_dzc[:, 1, 1] = -dh_qc_dzc[:, 0, 0]
    #     return dh_qc_dzc

    # @ingredients.vectorize_ingredient
    # def h_qc(self, constants, parameters, **kwargs):
        #     z = kwargs.get("z_coord")
    #     g = constants.g
    #     m = constants.classical_coordinate_mass
    #     h = constants.classical_coordinate_weight
    #     h_qc = np.zeros((2, 2), dtype=complex)
    #     h_qc[0, 0] = np.sum(g * np.sqrt(1 / (2 * m * h)) * (z + np.conj(z)))
    #     h_qc[1, 1] = -h_qc[0, 0]
    #     return h_qc

    # @ingredients.make_ingredient_sparse
    # @ingredients.vectorize_ingredient
    # def dh_qc_dzc(self, constants, parameters, **kwargs):
    #     m = constants.classical_coordinate_mass
    #     g = constants.g
    #     h = constants.classical_coordinate_weight
    #     dh_qc_dzc = np.zeros((constants.A, 2, 2), dtype=complex)
    #     dh_qc_dzc[:, 0, 0] = g * np.sqrt(1 / (2 * m * h))
    #     dh_qc_dzc[:, 1, 1] = -dh_qc_dzc[:, 0, 0]
    #     return dh_qc_dzc


    def h_qc(self, constants, parameters, **kwargs):
        z = kwargs.get("z_coord")
        g = constants.g
        m = constants.classical_coordinate_mass
        h = constants.classical_coordinate_weight
        h_qc = np.zeros((len(z), 2, 2), dtype=complex)
        h_qc[:, 0, 0] = np.sum(
            g * np.sqrt(1 / (2 * m * h))[np.newaxis, :] * (z + np.conj(z)), axis=-1
        )
        h_qc[:, 1, 1] = -h_qc[:, 0, 0]
        return h_qc

    def dh_qc_dzc(self, constants, parameters, **kwargs):
        """
        Calculate the derivative of the quantum-classical coupling Hamiltonian
        with respect to the z-coordinates.

        Args:
            constants: The constants object.
            parameters: The parameters object.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: Indices and values of the non-zero elements of the derivative.
        """
        del kwargs
        if self.dh_qc_dzc_inds is None or self.dh_qc_dzc_mels is None:
            m = constants.classical_coordinate_mass
            g = constants.g
            h = constants.classical_coordinate_weight
            batch_size = len(parameters.seed)
            dh_qc_dzc = np.zeros((batch_size, constants.A, 2, 2), dtype=complex)
            dh_qc_dzc[:, :, 0, 0] = (g * np.sqrt(1 / (2 * m * h)))[..., :]
            dh_qc_dzc[:, :, 1, 1] = -dh_qc_dzc[..., :, 0, 0]
            inds = np.where(dh_qc_dzc != 0)
            mels = dh_qc_dzc[inds]
            self.dh_qc_dzc_inds = inds
            self.dh_qc_dzc_mels = dh_qc_dzc[inds]
        else:
            inds = self.dh_qc_dzc_inds
            mels = self.dh_qc_dzc_mels
        return inds, mels

    # Assigning functions from ingredients module
    init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical
    hop_function = ingredients.harmonic_oscillator_hop
    h_c = ingredients.harmonic_oscillator_h_c
    h_q = ingredients.two_level_system_h_q
    dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc
    linear_h_qc = True
    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_c,
        initialize_constants_h_qc,
        initialize_constants_h_q,
    ]
