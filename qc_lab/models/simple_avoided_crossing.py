"""Model representing Tully's first problem, a simple avoided crossing."""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients


class SimpleAvoidedCrossing(Model):
    """Tully's first problem, a simple avoided crossing."""

    def __init__(self, constants=None):
        if constants is None:
            constants = {}

        self.default_constants = dict(
            init_momentum=10,
            init_position=-25,
            mass=2000,
            A=0.01,
            B=1.6,
            C=0.005,
            D=1.0,
        )

        super().__init__(self.default_constants, constants)

        self.update_dh_qc_dzc = True
        self.update_h_q = False

    def initialize_constants_model(self):
        self.constants.num_quantum_states = 2
        self.constants.num_classical_coordinates = 1
        self.constants.classical_coordinate_mass = np.array(
            [self.constants.get("mass", self.default_constants.get("mass"))]
        )
        self.constants.classical_coordinate_weight = np.array([1])

    def initialize_constants_h_qc(self):
        self.constants.gradient_weight = 1 / np.sqrt(
            2
            * self.constants.classical_coordinate_mass
            * self.constants.classical_coordinate_weight
        )

    def h_qc(self, parameters, **kwargs):
        num_quantum_states = self.constants.num_quantum_states
        mass = self.constants.classical_coordinate_mass.flatten()
        h = self.constants.classical_coordinate_weight.flatten()
        A = self.constants.A
        B = self.constants.B
        C = self.constants.C
        D = self.constants.D

        z = kwargs["z"]
        batch_size = kwargs.get("batch_size", len(z))

        q = ((z + np.conj(z)) / 2) / ((mass * h / 2) ** (1 / 2))

        h_qc = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )

        V_11 = np.zeros(np.shape(z), dtype=complex)
        indices_pos = np.real(z) >= 0
        V_11[indices_pos] = A * (1 - np.exp(-1 * B * q[indices_pos]))
        indices_neg = np.real(z) < 0
        V_11[indices_neg] = -1 * A * (1 - np.exp(B * q[indices_neg]))
        V_12 = C * (np.exp(-1 * D * (q**2)))

        h_qc[:, 0, 0] = V_11.flatten()
        h_qc[:, 0, 1] = V_12.flatten()
        h_qc[:, 1, 0] = V_12.flatten()
        h_qc[:, 1, 1] = -1 * V_11.flatten()

        return h_qc

    def dh_qc_dzc(self, parameters, **kwargs):
        num_quantum_states = self.constants.num_quantum_states
        num_classical_coordinates = self.constants.num_classical_coordinates
        gradient_weight = self.constants.gradient_weight
        A = self.constants.A
        B = self.constants.B
        C = self.constants.C
        D = self.constants.D

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

        dV_11_dzc = np.zeros(np.shape(z), dtype=complex)
        indices_pos = np.real(z) >= 0
        dV_11_dzc[indices_pos] = (A * B * gradient_weight) * (
            np.exp(
                (-1 * B * gradient_weight) * (np.conj(z[indices_pos]) + z[indices_pos])
            )
        )
        indices_neg = np.real(z) < 0
        dV_11_dzc[indices_neg] = (A * B * gradient_weight) * (
            np.exp((B * gradient_weight) * (np.conj(z[indices_neg]) + z[indices_neg]))
        )
        dV_12_dzc = (
            (-2 * C * D * (gradient_weight**2))
            * (z + np.conj(z))
            * (np.exp(-1 * D * (((z + np.conj(z)) * gradient_weight) ** 2)))
        )

        dh_qc_dzc[:, 0, 0, 0] = dV_11_dzc.flatten()
        dh_qc_dzc[:, 0, 0, 1] = dV_12_dzc.flatten()
        dh_qc_dzc[:, 0, 1, 0] = dV_12_dzc.flatten()
        dh_qc_dzc[:, 0, 1, 1] = -1 * dV_11_dzc.flatten()

        inds = np.where(dh_qc_dzc != 0)
        mels = dh_qc_dzc[inds]
        shape = np.shape(dh_qc_dzc)
        self.dh_qc_dzc_inds = inds
        self.dh_qc_dzc_mels = dh_qc_dzc[inds]
        self.dh_qc_dzc_shape = shape

        return inds, mels, shape

    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_qc,
    ]

    ingredients = [
        ("h_q", ingredients.h_q_two_level),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_free),
        ("dh_qc_dzc", dh_qc_dzc),
        ("dh_c_dzc", ingredients.dh_c_dzc_free),
        ("init_classical", ingredients.init_classical_definite_position_momentum),
        ("hop", ingredients.hop_free),
    ]
