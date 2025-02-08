""" 
This file contains the spin-boson model class.
"""


import numpy as np
from qclab.model import Model
import qclab.ingredients as ingredients


class SpinBosonModel(Model):
    """
    Spin-Boson model class for the simulation framework.

    Attributes:
        constants (ParameterClass): The constants of the model.
    """

    def __init__(self, constants=None):
        """
        Initializes the SpinBosonModel with given constants.

        Args:
            constants (dict): A dictionary of constants to initialize the model.
        """
        if constants is None:
            constants = {}
        self.default_constants = {
            'temp': 1, 'V': 0.5, 'E': 0.5, 'A': 100, 'W': 0.1,
            'l_reorg': 0.02 / 4, 'boson_mass': 1
        }
        super().__init__(self.default_constants, constants)

    def update_model_constants(self):
        """
        Update model constants based on the current constant values.
        """
        self.constants.w = self.constants.W * np.tan(
            ((np.arange(self.constants.A) + 1) - 0.5) *
            np.pi / (2 * self.constants.A)
        )  # Classical oscillator frequency
        self.constants.g = self.constants.w * np.sqrt(
            2 * self.constants.l_reorg / self.constants.A
        )  # Electron-phonon coupling
        # Diagonal energy of state 0
        self.constants.two_level_system_a = self.constants.E
        # Diagonal energy of state 1
        self.constants.two_level_system_b = -self.constants.E
        # Real part of the off-diagonal coupling
        self.constants.two_level_system_c = self.constants.V
        # Imaginary part of the off-diagonal coupling
        self.constants.two_level_system_d = 0
        self.constants.pq_weight = self.constants.w
        self.constants.num_classical_coordinates = self.constants.A
        self.constants.mass = np.ones(
            self.constants.A) * self.constants.boson_mass
        self.constants.harmonic_oscillator_frequency = self.constants.w
        self.constants.harmonic_oscillator_mass = self.constants.mass

    def h_qc(self, constants, parameters, **kwargs):
        z = kwargs.get('z_coord', parameters.z_coord)
        g = constants.g
        m = constants.mass
        h = constants.pq_weight
        h_qc = np.zeros((2, 2), dtype=complex)
        h_qc[0, 0] = np.sum(g * np.sqrt(1 / (2 * m * h)) * (z + np.conj(z)))
        h_qc[1, 1] = -h_qc[0, 0]
        return h_qc
    

    def h_qc_vectorized(self, constants, parameters, **kwargs):
        z = kwargs.get('z_coord', parameters.z_coord)
        g = constants.g
        m = constants.mass
        h = constants.pq_weight
        h_qc = np.zeros((len(z), 2, 2), dtype=complex)
        h_qc[:, 0, 0] = np.sum(g * np.sqrt(1 / (2 * m * h))[np.newaxis, :] * (z + np.conj(z)), axis=-1)
        h_qc[:, 1, 1] = -h_qc[:, 0, 0]
        return h_qc

    def dh_qc_dzc(self, constants, parameters, **kwargs):
        m = constants.mass
        g = constants.g
        h = constants.pq_weight
        dh_qc_dzc = np.zeros((constants.A, 2, 2), dtype=complex)
        dh_qc_dzc[:, 0, 0] = g * np.sqrt(1 / (2 * m * h))
        dh_qc_dzc[:, 1, 1] = -dh_qc_dzc[:, 0, 0]
        return dh_qc_dzc

    def dh_qc_dzc_vectorized(self, constants, parameters, **kwargs):
        m = constants.mass
        g = constants.g
        h = constants.pq_weight
        batch_size = parameters._size
        dh_qc_dzc = np.zeros((batch_size, constants.A, 2, 2), dtype=complex)
        dh_qc_dzc[:, :, 0, 0] = (g * np.sqrt(1 / (2 * m * h)))[..., :]
        dh_qc_dzc[:, :, 1, 1] = -dh_qc_dzc[..., :, 0, 0]
        return dh_qc_dzc

    # Assigning functions from ingredients module
    h_q = ingredients.two_level_system_h_q
    h_c = ingredients.harmonic_oscillator_h_c
    dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc
    init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical
    hop_function = ingredients.harmonic_oscillator_hop

    h_c_vectorized = ingredients.harmonic_oscillator_h_c_vectorized
    h_q_vectorized = ingredients.two_level_system_h_q_vectorized
    dh_c_dzc_vectorized = ingredients.harmonic_oscillator_dh_c_dzc_vectorized
