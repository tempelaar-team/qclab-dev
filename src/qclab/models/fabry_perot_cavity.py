"""
This module contains the cavity Model class.
"""

import numpy as np
from qclab.model import Model
from qclab import ingredients
from qclab import numerical_constants
import qclab.functions as functions

class FPCavity(Model):
    """
    Fabry-Perot Cavity Model class.

    Reference publication:
    Hsieh, M.-H., Krotz, A., Tempelaar, Roel. J. Phys. Chem. Lett. 2023, 14, 1253–1258. https://doi.org/10.1021/acs.jpclett.2c03724
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = { # Constants are in atomic units.
            "kBT": 0.0,
            "epsilon_g": -0.6738,
            "epsilon_e": -0.2798,
            "N": 400,
            "L": 2.362E5,
            "mu": 1.034,
            "r_atom": 0.5,
            "c0": 137.036,
            "epsilon0": 1/4/np.pi
        }
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = False
        self.update_h_q = False
        self.diagonal_h_q = True

    def _init_h_q(self, parameters, **kwargs):
        energy_level = self.constants.get("epsilon_e") - self.constants.get("epsilon_g")
        self.constants.two_level_00 = energy_level
        self.constants.two_level_11 = 0.0
        return
    
    def h_qc(model, parameters, **kwargs):
        z = kwargs['z'] # shape (B, A) where B is the batch size and A is the number of classical coordinates
        batch_size = z.shape[0]
        m = model.constants.classical_coordinate_mass
        h = model.constants.classical_coordinate_weight
        w = model.constants.harmonic_frequency
        mu = model.constants.get("mu")
        lambda_alpha = model.constants.lambda_alpha #shape (N,)
        h_qc = np.zeros((batch_size, 2, 2), dtype=complex)
        h_qc[:, 0, 1] = np.sum(mu * w[np.newaxis, :] * lambda_alpha[np.newaxis, :] * functions.z_to_q(z, m, h), axis=1)
        h_qc[:, 1, 0] = np.conj(h_qc[:, 0, 1])
        return h_qc 

    def _init_h_c(self, parameters, **kwargs):
        N = self.constants.get("N")
        c0 = self.constants.get("c0")
        alpha = np.arange(1, N+1, 1)
        L = self.constants.get("L")
        self.constants.harmonic_frequency = (np.pi * c0 * alpha)/L
        return

    def _init_model(self, parameters, **kwargs):
        N = self.constants.get("N")
        self.constants.num_classical_coordinates = N
        self.constants.num_quantum_states = 2
        self.constants.classical_coordinate_weight = self.constants.harmonic_frequency
        self.constants.classical_coordinate_mass = np.ones(N)
        alpha = np.arange(1, N+1, 1)
        r_atom = self.constants.get("r_atom")
        L = self.constants.get("L")
        epsilon0 = self.constants.get("epsilon0")
        self.constants.lambda_alpha = np.sqrt(2/(epsilon0 * L)) * np.sin(np.pi * alpha * r_atom) #shape (N,)
        return

    def dh_qc_dzc(model, parameters, **kwargs):
        """
        Gradient of the diagonal linear quantum-classical coupling Hamiltonian
        in sparse format.

        :math:`[\\partial_{z} H_{qc}]_{ijkl} = \\delta_{kl}\\gamma_{kj}`

        .. rubric:: Keyword Args
        z : ndarray
            Complex classical coordinate.

        .. rubric:: Model Constants
        diagonal_linear_coupling : ndarray
            Coupling constants :math:`\\gamma`.

        .. rubric:: Returns
        inds : tuple of ndarray
            Indices of the non-zero elements of the gradient.
            ``(batch_index, coordinate_index, row_index, column_index)``.
        mels : ndarray
            Values of the non-zero elements of the gradient.
        shape : tuple
            Shape of the full gradient array.
            ``(batch_size, num_classical_coordinates, num_states, num_states)``.
        """
        z = kwargs["z"]
        batch_size = len(z)
        num_states = model.constants.num_quantum_states
        num_classical_coordinates = model.constants.num_classical_coordinates

        mu = model.constants.get("mu")
        w = model.constants.harmonic_frequency
        lambda_alpha = model.constants.get("lambda_alpha") #shape (N,)

        dh_qc_dzc = np.zeros(
            (num_classical_coordinates, num_states, num_states), dtype=complex
        )

        dh_qc_dzc[:, 0, 1] = mu * lambda_alpha * np.sqrt(w/2)
        dh_qc_dzc[:, 1, 0] = np.conj(dh_qc_dzc[:, 0, 1])

        dh_qc_dzc = dh_qc_dzc[np.newaxis, :, :, :] + np.zeros(
            (batch_size, num_classical_coordinates, num_states, num_states),
            dtype=complex,
        )
        inds = np.where(dh_qc_dzc != 0)
        mels = dh_qc_dzc[inds]
        shape = np.shape(dh_qc_dzc)
        return inds, mels, shape


    ingredients = [
        ("h_q", ingredients.h_q_two_level),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", dh_qc_dzc),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop", ingredients.hop_harmonic),
        ("_init_h_q", _init_h_q),
        ("_init_model", _init_model),
        ("_init_h_c", _init_h_c),
    ]
