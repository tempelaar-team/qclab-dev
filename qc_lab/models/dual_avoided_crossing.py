"""Model for Tully's second problem, a dual avoided crossing."""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients


class DualAvoidedCrossing(Model):
    """Tully's second problem, a dual avoided crossing."""

    def __init__(self, constants=None):
        if constants is None:
            constants = {}

        self.default_constants = dict(
            init_momentum=10,
            init_position=-25,
            mass=2000,
            A=0.1,
            B=0.28,
            C=0.015,
            D=0.06,
            E_0=0.05,
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
        E_0 = self.constants.E_0
        z = kwargs["z"]
        batch_size = kwargs.get("batch_size", len(z))

        q = ((z + np.conj(z)) / 2) / ((mass * h / 2) ** (1 / 2))

        h_qc = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )

        V_12 = C * (np.exp(-1 * D * (q**2)))
        V_22 = -1 * A * (np.exp(-1 * B * (q**2))) + E_0

        h_qc[:, 0, 1] = V_12.flatten()
        h_qc[:, 1, 0] = V_12.flatten()
        h_qc[:, 1, 1] = V_22.flatten()

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

        dV_12_dzc = (
            (-2 * C * D * (gradient_weight**2))
            * (z + np.conj(z))
            * (np.exp(-1 * D * (((z + np.conj(z)) * gradient_weight) ** 2)))
        )
        dV_22_dzc = (
            (2 * A * B * (gradient_weight**2))
            * (z + np.conj(z))
            * (np.exp(-1 * B * (((z + np.conj(z)) * gradient_weight) ** 2)))
        )

        dh_qc_dzc[:, 0, 0, 1] = dV_12_dzc.flatten()
        dh_qc_dzc[:, 0, 1, 0] = dV_12_dzc.flatten()
        dh_qc_dzc[:, 0, 1, 1] = dV_22_dzc.flatten()

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
        ("hop_function", ingredients.hop_free),
    ]
