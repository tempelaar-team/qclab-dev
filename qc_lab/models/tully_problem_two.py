"""
This module contains the Model class for Tully's second problem, a dual avoided crossing.
"""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients
from qc_lab import functions


class TullyProblemTwo(Model):
    """
    Tully's second problem: a dual avoided crossing.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}

        self.default_constants = {
            "init_momentum": 10.0,
            "init_position": -25.0,
            "mass": 2000.0,
            "A": 0.1,
            "B": 0.28,
            "C": 0.015,
            "D": 0.06,
            "E_0": 0.05,
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
        E_0 = self.constants.get("E_0")
        z = kwargs["z"]
        batch_size = kwargs.get("batch_size", len(z))

        m = self.constants.classical_coordinate_mass[np.newaxis, :]
        h = self.constants.classical_coordinate_weight[np.newaxis, :]
        q = functions.z_to_q(z, m, h)

        h_qc = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )

        v_12 = C * (np.exp(-1.0 * D * (q**2)))
        v_22 = -1.0 * A * (np.exp(-1.0 * B * (q**2))) + E_0

        h_qc[:, 0, 1] = v_12.flatten()
        h_qc[:, 1, 0] = v_12.flatten()
        h_qc[:, 1, 1] = v_22.flatten()

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

        dv_12_dzc = (
            (-2 * C * D * (gradient_weight**2))
            * (z + np.conj(z))
            * (np.exp(-1.0 * D * (((z + np.conj(z)) * gradient_weight) ** 2)))
        )
        dv_22_dzc = (
            (2.0 * A * B * (gradient_weight**2))
            * (z + np.conj(z))
            * (np.exp(-1.0 * B * (((z + np.conj(z)) * gradient_weight) ** 2)))
        )

        dh_qc_dzc[:, 0, 0, 1] = dv_12_dzc.flatten()
        dh_qc_dzc[:, 0, 1, 0] = dv_12_dzc.flatten()
        dh_qc_dzc[:, 0, 1, 1] = dv_22_dzc.flatten()

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
