"""
The module defines the Model class for Tully's first problem, a simple avoided crossing.
"""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients
from qc_lab import functions


class TullyProblemOne(Model):
    """
    Tully's first problem: a simple avoided crossing.

    Reference publication:
    Tully. J. Chem. Phys. 93, 2, 1061-1071. (1990); https://doi.org/10.1063/1.459170
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}

        self.default_constants = {
            "init_momentum": 10.0,
            "init_position": -25.0,
            "mass": 2000.0,
            "A": 0.01,
            "B": 1.6,
            "C": 0.005,
            "D": 1.0,
        }

        super().__init__(self.default_constants, constants)

        self.update_dh_qc_dzc = True
        self.update_h_q = False

    def _init_model(self, parameters, **kwargs):
        self.constants.num_quantum_states = 2
        self.constants.num_classical_coordinates = 1
        self.constants.classical_coordinate_mass = np.array(
            [self.constants.get("mass")]
        )
        self.constants.classical_coordinate_weight = np.array([1.0])
        return

    def _init_h_qc(self, parameters, **kwargs):
        self.constants.gradient_weight = 1.0 / np.sqrt(
            2.0
            * self.constants.classical_coordinate_mass
            * self.constants.classical_coordinate_weight
        )
        return

    def h_qc(self, parameters, **kwargs):
        num_quantum_states = self.constants.num_quantum_states
        A = self.constants.get("A")
        B = self.constants.get("B")
        C = self.constants.get("C")
        D = self.constants.get("D")

        z = kwargs["z"]
        batch_size = kwargs.get("batch_size", len(z))

        m = self.constants.classical_coordinate_mass[np.newaxis, :]
        h = self.constants.classical_coordinate_weight[np.newaxis, :]
        q = functions.z_to_q(z, m, h)

        h_qc = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )

        v_11 = np.zeros(np.shape(z), dtype=complex)
        indices_pos = np.real(z) >= 0.0
        v_11[indices_pos] = A * (1.0 - np.exp(-1.0 * B * q[indices_pos]))
        indices_neg = np.real(z) < 0.0
        v_11[indices_neg] = -1.0 * A * (1.0 - np.exp(B * q[indices_neg]))
        v_12 = C * (np.exp(-1.0 * D * (q**2)))

        h_qc[:, 0, 0] = v_11.flatten()
        h_qc[:, 0, 1] = v_12.flatten()
        h_qc[:, 1, 0] = v_12.flatten()
        h_qc[:, 1, 1] = -1.0 * v_11.flatten()

        return h_qc

    def dh_qc_dzc(self, parameters, **kwargs):
        num_quantum_states = self.constants.num_quantum_states
        num_classical_coordinates = self.constants.num_classical_coordinates
        gradient_weight = self.constants.gradient_weight
        A = self.constants.get("A")
        B = self.constants.get("B")
        C = self.constants.get("C")
        D = self.constants.get("D")

        z = kwargs["z"]
        batch_size = kwargs.get("batch_size", len(z))

        dh_qc_dzc = np.zeros(
            (
                batch_size,
                num_classical_coordinates,
                num_quantum_states,
                num_quantum_states,
            ),
            dtype=complex,
        )

        dv_11_dzc = np.zeros(np.shape(z), dtype=complex)
        indices_pos = np.real(z) >= 0.0
        dv_11_dzc[indices_pos] = (A * B * gradient_weight) * (
            np.exp(
                (-1.0 * B * gradient_weight)
                * (np.conj(z[indices_pos]) + z[indices_pos])
            )
        )
        indices_neg = np.real(z) < 0.0
        dv_11_dzc[indices_neg] = (A * B * gradient_weight) * (
            np.exp((B * gradient_weight) * (np.conj(z[indices_neg]) + z[indices_neg]))
        )
        dv_12_dzc = (
            (-2.0 * C * D * (gradient_weight**2))
            * (z + np.conj(z))
            * (np.exp(-1.0 * D * (((z + np.conj(z)) * gradient_weight) ** 2)))
        )

        dh_qc_dzc[:, 0, 0, 0] = dv_11_dzc.flatten()
        dh_qc_dzc[:, 0, 0, 1] = dv_12_dzc.flatten()
        dh_qc_dzc[:, 0, 1, 0] = dv_12_dzc.flatten()
        dh_qc_dzc[:, 0, 1, 1] = -1.0 * dv_11_dzc.flatten()

        inds = np.where(dh_qc_dzc != 0)
        mels = dh_qc_dzc[inds]
        shape = np.shape(dh_qc_dzc)

        return inds, mels, shape

    ingredients = [
        ("h_q", ingredients.h_q_two_level),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_free),
        ("dh_qc_dzc", dh_qc_dzc),
        ("dh_c_dzc", ingredients.dh_c_dzc_free),
        ("init_classical", ingredients.init_classical_definite_position_momentum),
        ("hop", ingredients.hop_free),
        ("_init_h_qc", _init_h_qc),
        ("_init_model", _init_model),
    ]
