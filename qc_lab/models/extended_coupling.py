"""Model for Tully's third problem with extended coupling and reflection."""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients


class ExtendedCoupling(Model):
    """Tully's third problem, an extended coupling with reflection."""

    def __init__(self, constants=None):
        if constants is None:
            constants = {}

        self.default_constants = dict(
            init_momentum=10, init_position=-25, mass=2000, A=0.0006, B=0.1, C=0.9
        )

        super().__init__(self.default_constants, constants)

        self.dh_qc_dzc_inds = None
        self.dh_qc_dzc_mels = None
        self.dh_qc_dzc_shape = None
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

        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = len(parameters.seed)
        z = kwargs["z"]

        q = ((z + np.conj(z)) / 2) / ((mass * h / 2) ** (1 / 2))

        h_qc = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )

        V_11 = np.ones(np.shape(z)) * A
        V_12 = np.zeros(np.shape(z), dtype=complex)
        indices_pos = np.real(z) >= 0
        V_12[indices_pos] = B * (2 - np.exp(-1 * C * q[indices_pos]))
        indices_neg = np.real(z) < 0
        V_12[indices_neg] = B * np.exp(C * q[indices_neg])

        h_qc[:, 0, 0] = V_11.flatten()
        h_qc[:, 0, 1] = V_12.flatten()
        h_qc[:, 1, 0] = V_12.flatten()
        h_qc[:, 1, 1] = -1 * V_11.flatten()

        return h_qc

    def dh_qc_dzc(self, parameters, **kwargs):
        num_quantum_states = self.constants.num_quantum_states
        num_classical_coordinates = self.constants.num_classical_coordinates
        gradient_weight = self.constants.gradient_weight
        B = self.constants.B
        C = self.constants.C

        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = len(parameters.seed)
        z = kwargs["z"]

        dh_qc_dzc = np.zeros(
            (
                batch_size,
                num_classical_coordinates,
                num_quantum_states,
                num_quantum_states,
            ),
            dtype=complex,
        )

        dV_12_dzc = np.zeros(np.shape(z), dtype=complex)
        indices_pos = np.real(z) >= 0
        dV_12_dzc[indices_pos] = (B * C * gradient_weight) * (
            np.exp(
                -1 * C * gradient_weight * (z[indices_pos] + np.conj(z[indices_pos]))
            )
        )
        indices_neg = np.real(z) < 0
        dV_12_dzc[indices_neg] = (B * C * gradient_weight) * (
            np.exp(C * gradient_weight * (z[indices_neg] + np.conj(z[indices_neg])))
        )

        dh_qc_dzc[:, 0, 0, 1] = dV_12_dzc.flatten()
        dh_qc_dzc[:, 0, 1, 0] = dV_12_dzc.flatten()

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
        ("h_q", ingredients.two_level_system_h_q),
        ("h_qc", h_qc),
        ("h_c", ingredients.free_particle_h_c),
        ("dh_qc_dzc", dh_qc_dzc),
        ("dh_c_dzc", ingredients.free_particle_dh_c_dzc),
        ("init_classical", ingredients.definite_position_momentum_init_classical),
        ("hop_function", ingredients.free_particle_hop_function),
    ]
