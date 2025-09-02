"""
This module contains the Model class for Tully's third problem,
an extended coupling with reflection.
"""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients
from qc_lab import functions


class TullyProblemThree(Model):
    """
    Tully's third problem: an extended coupling with reflection.

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
            "A": 0.0006,
            "B": 0.1,
            "C": 0.9,
        }

        super().__init__(self.default_constants, constants)

        self.update_dh_qc_dzc = True
        self.update_h_q = False

    def _init_model(self, parameters, **kwargs):
        self.constants.num_quantum_states = 2
        self.constants.num_classical_coordinates = 1
        self.constants.classical_coordinate_mass = np.array(
            [self.constants.get("mass", self.default_constants.get("mass"))]
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
        """
        Quantum-Classical Hamiltonian for Tully's third problem.
        """
        z = kwargs["z"]
        batch_size = len(z)
        num_quantum_states = self.constants.num_quantum_states
        A = self.constants.get("A")
        B = self.constants.get("B")
        C = self.constants.get("C")

        z = kwargs["z"]
        batch_size = kwargs.get("batch_size", len(z))

        m = self.constants.classical_coordinate_mass[np.newaxis, :]
        h = self.constants.classical_coordinate_weight[np.newaxis, :]
        q = functions.z_to_q(z, m, h)[:,0]

        v_12 = np.zeros(batch_size, dtype=complex)

        v_12[q >= 0] = B * (2.0 - np.exp(-C * q))[q >= 0]
        v_12[q < 0] = B * np.exp(C * q)[q < 0]
        v_11 = np.ones(batch_size) * A

        h_qc = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )

        # v_11 = np.ones(np.shape(z)) * A
        # v_12 = np.zeros(np.shape(z), dtype=complex)
        # indices_pos = np.real(z) >= 0
        # v_12[indices_pos] = B * (2.0 - np.exp(-1.0 * C * q[indices_pos]))
        # indices_neg = np.real(z) < 0
        # v_12[indices_neg] = B * np.exp(C * q[indices_neg])

        h_qc[:, 0, 0] = v_11
        h_qc[:, 0, 1] = v_12
        h_qc[:, 1, 0] = v_12
        h_qc[:, 1, 1] = -v_11

        return h_qc

    def dh_qc_dzc(self, parameters, **kwargs):
        """
        Gradient w.r.t. to the conjugate z coordinate of the quantum-classical Hamiltonian
        for Tully's third problem.
        """
        z = kwargs["z"]
        batch_size = len(z)
        num_quantum_states = self.constants.num_quantum_states
        num_classical_coordinates = self.constants.num_classical_coordinates
        gradient_weight = self.constants.gradient_weight
        B = self.constants.get("B")
        C = self.constants.get("C")
        z = kwargs["z"]
        batch_size = kwargs.get("batch_size", len(z))

        m = self.constants.classical_coordinate_mass[np.newaxis, :]
        h = self.constants.classical_coordinate_weight[np.newaxis, :]
        q = functions.z_to_q(z, m, h)[:,0]

        dh_qc_dzc = np.zeros(
            (
                batch_size,
                num_classical_coordinates,
                num_quantum_states,
                num_quantum_states,
            ),
            dtype=complex,
        )

        dv_12_dq = np.zeros(batch_size, dtype=complex)
        dv_12_dq[q >= 0.0] = B * C * np.exp(-C * q)[q >= 0.0]
        dv_12_dq[q < 0.0] = B * C * np.exp(C * q)[q < 0.0]

        dv_12_dzc = functions.dqdp_to_dzc(dv_12_dq, None, m[0], h[0])

        dh_qc_dzc[:, 0, 0, 1] = dv_12_dzc
        dh_qc_dzc[:, 0, 1, 0] = dv_12_dzc

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
