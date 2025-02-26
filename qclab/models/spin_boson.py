""" 
This file contains the spin-boson model class.
"""

import numpy as np
from qclab.model import Model
import qclab.ingredients as ingredients


class SpinBosonModel(Model):
    """
    Spin-Boson model class for the simulation framework.

    """

    def __init__(self, constants=None):
        """
        Initializes the SpinBosonModel with given constants.
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

    def initialize_constants_model(self):
        self.constants.num_classical_coordinates = self.constants.A
        self.constants.num_quantum_states = 2
        self.constants.w = self.constants.W * np.tan(
            ((np.arange(self.constants.A) + 1) - 0.5) * np.pi / (2 * self.constants.A)
        )
        self.constants.classical_coordinate_weight = self.constants.w
        self.constants.classical_coordinate_mass = self.constants.boson_mass * np.ones(
            self.constants.A
        )

    def initialize_constants_h_c(self):
        self.constants.harmonic_oscillator_frequency = self.constants.w

    def initialize_constants_h_qc(self):
        self.constants.g = self.constants.w * np.sqrt(
            2 * self.constants.l_reorg / self.constants.A
        )  # Electron-phonon coupling

    def initialize_constants_h_q(self):
        self.constants.two_level_system_a = self.constants.E
        self.constants.two_level_system_b = -self.constants.E
        self.constants.two_level_system_c = self.constants.V
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
        m = constants.classical_coordinate_mass
        g = constants.g
        h = constants.classical_coordinate_weight
        batch_size = len(parameters.seed)
        dh_qc_dzc = np.zeros((batch_size, constants.A, 2, 2), dtype=complex)
        dh_qc_dzc[:, :, 0, 0] = (g * np.sqrt(1 / (2 * m * h)))[..., :]
        dh_qc_dzc[:, :, 1, 1] = -dh_qc_dzc[..., :, 0, 0]
        return dh_qc_dzc

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
