"""
This module contains ingredients for use in Model classes.

It also contains any functions that the ingredients depend on for low-level operations.
"""

import functools
import numpy as np
from qc_lab.utils import njit
from qc_lab.functions import (
    z_to_q,
    z_to_p,
    qp_to_z,
    dh_c_dzc_harmonic_jit,
    h_qc_diagonal_linear_jit,
)


def h_c_harmonic(model, parameters, **kwargs):
    """
    Harmonic oscillator classical Hamiltonian function.

    Required constants:
        - `harmonic_frequency`: Array of harmonic frequencies.
    """
    del parameters
    z = kwargs["z"]
    w = model.constants.harmonic_frequency[np.newaxis, :]
    m = model.constants.classical_coordinate_mass[np.newaxis, :]
    h = model.constants.classical_coordinate_weight[np.newaxis, :]
    q = z_to_q(z, m, h)
    p = z_to_p(z, m, h)
    h_c = np.sum(0.5 * (((p**2) / m) + m * (w**2) * (q**2)), axis=-1)
    return h_c


def h_c_free(model, parameters, **kwargs):
    """
    Free particle classical Hamiltonian function.

    Required constants:
        - `classical_coordinate_mass`: Mass of the classical coordinates.
    """
    del parameters
    z = kwargs["z"]
    m = model.constants.classical_coordinate_mass[np.newaxis, :]
    h = model.constants.classical_coordinate_weight[np.newaxis, :]
    p = z_to_p(z, m, h)
    h_c = np.sum((0.5 / m) * (p**2), axis=-1)
    return h_c


def dh_c_dzc_harmonic(model, parameters, **kwargs):
    """
    Derivative of the harmonic oscillator classical Hamiltonian function with respect to
    the conjugate `z` coordinate. This is an ingredient that calls the low-level
    function `dh_c_dzc_harmonic_jit`.

    Required constants:
        - `harmonic_frequency`: Array of harmonic frequencies.
    """
    z = kwargs["z"]
    h = model.constants.classical_coordinate_weight
    w = model.constants.harmonic_frequency
    return dh_c_dzc_harmonic_jit(z, h, w)


def dh_c_dzc_free(model, parameters, **kwargs):
    """
    Derivative of the free particle classical Hamiltonian function with respect to the
    conjugate `z` coordinate.

    Required constants:
        - `classical_coordinate_mass`: Mass of the classical coordinates.
    """
    del parameters
    z = kwargs["z"]
    m = model.constants.classical_coordinate_mass[np.newaxis, :]
    h = model.constants.classical_coordinate_weight[np.newaxis, :]
    p = z_to_p(z, m, h)
    # return -(0.5 * h[..., :]) * (np.conj(z) - z)
    return p / m


def h_q_two_level(model, parameters, **kwargs):
    """
    Quantum Hamiltonian for a two-level system.

    H = [[two_level_00, two_level_01_re + i * two_level_01_im],
        [two_level_01_re - i * two_level_01_im, two_level_11]]

    Required constants:
        - `two_level_00`: Energy of the first level.
        - `two_level_11`: Energy of the second level.
        - `two_level_01_re`: Real part of the coupling between levels.
        - `two_level_01_im`: Imaginary part of the coupling between levels.
    """
    batch_size = kwargs.get("batch_size", len(parameters.seed))
    h_q = np.zeros((batch_size, 2, 2), dtype=complex)
    h_q[:, 0, 0] = model.constants.get("two_level_00", 0.0)
    h_q[:, 1, 1] = model.constants.get("two_level_11", 0.0)
    h_q[:, 0, 1] = model.constants.get(
        "two_level_01_re", 0.0
    ) + 1j * model.constants.get("two_level_01_im", 0.0)
    h_q[:, 1, 0] = model.constants.get(
        "two_level_01_re", 0.0
    ) - 1j * model.constants.get("two_level_01_im", 0.0)
    return h_q


def h_q_nearest_neighbor(model, parameters, **kwargs):
    """
    Quantum Hamiltonian for a nearest-neighbor lattice.

    Required constants:
        - `nearest_neighbor_hopping_energy`: Hopping energy between sites.
        - `nearest_neighbor_periodic`: Boolean indicating periodic boundary conditions.
    """
    batch_size = kwargs.get("batch_size", len(parameters.seed))
    num_sites = model.constants.num_quantum_states
    hopping_energy = model.constants.nearest_neighbor_hopping_energy
    periodic = model.constants.nearest_neighbor_periodic
    h_q = np.zeros((num_sites, num_sites), dtype=complex)
    # Fill the Hamiltonian matrix with hopping energies.
    for n in range(num_sites - 1):
        h_q[n, n + 1] += -hopping_energy
        h_q[n + 1, n] += np.conj(h_q[n, n + 1])
    # Apply periodic boundary conditions if specified.
    if periodic:
        h_q[0, num_sites - 1] += -hopping_energy
        h_q[num_sites - 1, 0] += np.conj(h_q[0, num_sites - 1])
    out = h_q[np.newaxis, :, :] + np.zeros(
        (batch_size, num_sites, num_sites), dtype=complex
    )
    return out


def h_qc_diagonal_linear(model, parameters, **kwargs):
    """
    Diagonal linear quantum-classical Hamiltonian.

    Diagonal elements are given by

    :math:`H_{ii} = \sum_{j} \gamma_{ij} (z_{j} + z_{j}^*)`

    Required constants:
        - `diagonal_linear_coupling`: Array of coupling constants
          (num_quantum_states, num_classical_coordinates).
    """
    del parameters
    z = kwargs["z"]
    batch_size = kwargs.get("batch_size", len(z))
    num_sites = model.constants.num_quantum_states
    num_classical_coordinates = model.constants.num_classical_coordinates
    gamma = model.constants.diagonal_linear_coupling
    return h_qc_diagonal_linear_jit(
        batch_size, num_sites, num_classical_coordinates, z, gamma
    )


def dh_qc_dzc_diagonal_linear(model, parameters, **kwargs):
    """
    Gradient of the diagonal linear quantum-classical coupling Hamiltonian.

    Required constants:
        - `diagonal_linear_coupling`: Array of coupling constants
          (num_quantum_states, num_classical_coordinates).
    """
    batch_size = kwargs.get("batch_size", len(parameters.seed))
    num_states = model.constants.num_quantum_states
    num_classical_coordinates = model.constants.num_classical_coordinates
    gamma = model.constants.diagonal_linear_coupling
    dh_qc_dzc = np.zeros(
        (num_classical_coordinates, num_states, num_states), dtype=complex
    )
    for i in range(num_states):
        for j in range(num_classical_coordinates):
            dh_qc_dzc[j, i, i] = gamma[i, j]
    dh_qc_dzc = dh_qc_dzc[np.newaxis, :, :, :] + np.zeros(
        (batch_size, num_classical_coordinates, num_states, num_states),
        dtype=complex,
    )
    inds = np.where(dh_qc_dzc != 0)
    mels = dh_qc_dzc[inds]
    shape = np.shape(dh_qc_dzc)
    return inds, mels, shape


def hop_harmonic(model, parameters, **kwargs):
    """
    FSSH hop function for taking the classical coordinates to represent harmonic
    oscillators.

    Determines the shift in the classical coordinates required to conserve energy
    following a hop between quantum states. The quantity ev_diff = e_final - e_initial
    is the energy difference between the final and initial quantum states and
    delta_z is the rescaling direction of the z coordinate.

    If enough energy is available, the function returns the shift in the classical
    coordinates such that the new classical coordinate is z + shift and a Boolean
    equaling True if the hop has occurred. If not enough energy is available,
    the shift becomes zero and the Boolean is False.

    Required constants:
        - `harmonic_frequency`: Array of harmonic frequencies.
    """
    z = kwargs["z"]
    delta_z = kwargs["delta_z"]
    ev_diff = kwargs["ev_diff"]
    delta_zc = np.conj(delta_z)
    zc = np.conj(z)
    a_const = 0.25 * (
        (
            (model.constants.harmonic_frequency**2)
            / model.constants.classical_coordinate_weight
        )
        - model.constants.classical_coordinate_weight
    )
    b_const = 0.25 * (
        (
            (model.constants.harmonic_frequency**2)
            / model.constants.classical_coordinate_weight
        )
        + model.constants.classical_coordinate_weight
    )
    # Here, akj_z, bkj_z, ckj_z are the coefficients of the quadratic equation
    # akj_z * gamma^2 - bkj_z * gamma + ckj_z = 0
    akj_z = np.sum(
        2.0 * delta_zc * delta_z * b_const - a_const * (delta_z**2 + delta_zc**2)
    )
    bkj_z = 2.0j * np.sum(
        (z * delta_z - delta_zc * zc) * a_const
        + (delta_z * zc - delta_zc * z) * b_const
    )
    ckj_z = ev_diff
    disc = bkj_z**2 - 4.0 * akj_z * ckj_z
    if disc >= 0:
        if bkj_z < 0:
            gamma = bkj_z + np.sqrt(disc)
        else:
            gamma = bkj_z - np.sqrt(disc)
        if akj_z == 0:
            gamma = 0
        else:
            gamma = 0.5 * gamma / akj_z
        shift = -1j * gamma * delta_z
        return shift, True
    shift = np.zeros_like(z)
    return shift, False


def hop_free(model, parameters, **kwargs):
    """
    FSSH hop function taking the classical coordinates to represent free particles.

    Determines the shift in the classical coordinates required to conserve energy
    following a hop between quantum states. The quantity ev_diff = e_final - e_initial
    is the energy difference between the final and initial quantum states and
    delta_z is the rescaling direction of the z coordinate.

    If enough energy is available, the function returns the shift in the classical
    coordinates such that the new classical coordinate is z + shift and a Boolean
    equaling True if the hop has occurred. If not enough energy is available,
    the shift becomes zero and the Boolean is False.

    Required constants:
        - `classical_coordinate_weight`: Mass of the classical coordinates.
    """
    z = kwargs["z"]
    delta_z = kwargs["delta_z"]
    ev_diff = kwargs["ev_diff"]
    delta_zc = np.conj(delta_z)
    zc = np.conj(z)

    f = 1j * (delta_zc + delta_z)
    g = zc - z

    h = model.constants.classical_coordinate_weight

    # Here, akj_z, bkj_z, ckj_z are the coefficients of the quadratic equation
    # akj_z * gamma^2 - bkj_z * gamma + ckj_z = 0

    akj_z = np.sum(0.25 * h * f * f)
    bkj_z = -np.sum(0.5 * h * f * g)
    ckj_z = -ev_diff

    disc = bkj_z**2 - 4.0 * akj_z * ckj_z
    if disc >= 0:
        if bkj_z < 0:
            gamma = bkj_z + np.sqrt(disc)
        else:
            gamma = bkj_z - np.sqrt(disc)
        if akj_z == 0:
            gamma = 0
        else:
            gamma = 0.5 * gamma / akj_z
        shift = -1j * gamma * delta_z
        return shift, True
    shift = np.zeros_like(z)
    return shift, False


def init_classical_boltzmann_harmonic(model, parameters, **kwargs):
    """
    Initialize classical coordinates according to Boltzmann statistics taking the
    classical coordinates to represent harmonic oscillators.

    Required constants:
        - `kBT`: Thermal quantum.
        - `harmonic_frequency`: Array of harmonic frequencies.
    """
    seed = kwargs["seed"]
    kBT = model.constants.kBT
    w = model.constants.harmonic_frequency
    m = model.constants.classical_coordinate_mass
    h = model.constants.classical_coordinate_weight
    out = np.zeros(
        (len(seed), model.constants.num_classical_coordinates), dtype=complex
    )
    # Calculate the standard deviations for q and p.
    std_q = np.sqrt(kBT / (m * (w**2)))
    std_p = np.sqrt(m * kBT)
    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        # Generate random q and p values.
        q = np.random.normal(
            loc=0, scale=std_q, size=model.constants.num_classical_coordinates
        )
        p = np.random.normal(
            loc=0, scale=std_p, size=model.constants.num_classical_coordinates
        )
        # Calculate the complex-valued classical coordinate.
        out[s] = qp_to_z(q, p, m, h)
    return out


def init_classical_wigner_harmonic(model, parameters, **kwargs):
    """
    Initialize classical coordinates according to the Wigner distribution of the ground
    state of a harmonic oscillator.

    Required constants:
        - `kBT`: Thermal quantum.
        - `harmonic_frequency`: Array of harmonic frequencies.
    """
    del parameters
    seed = kwargs["seed"]
    w = model.constants.harmonic_frequency
    m = model.constants.classical_coordinate_mass
    h = model.constants.classical_coordinate_weight
    kBT = model.constants.kBT
    out = np.zeros(
        (len(seed), model.constants.num_classical_coordinates), dtype=complex
    )
    # Calculate the standard deviations for q and p.
    if kBT > 0:
        std_q = np.sqrt(0.5 / (w * m * np.tanh(0.5 * w / kBT)))
        std_p = np.sqrt(0.5 * m * w / np.tanh(0.5 * w / kBT))
    else:
        std_q = np.sqrt(0.5 / (w * m))
        std_p = np.sqrt(0.5 * m * w)
    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        # Generate random q and p values.
        q = np.random.normal(
            loc=0, scale=std_q, size=model.constants.num_classical_coordinates
        )
        p = np.random.normal(
            loc=0, scale=std_p, size=model.constants.num_classical_coordinates
        )
        # Calculate the complex-valued classical coordinate.
        out[s] = qp_to_z(q, p, m, h)
    return out


def init_classical_definite_position_momentum(model, parameters, **kwargs):
    """
    Initialize classical coordinates with definite position and momentum. The quantities
    init_position and init_momentum are the initial position and momentum, and so should
    be numpy arrays of shape (num_classical_coordinates).

    Required constants:
        - `classical_coordinate_mass`: Mass of the classical coordinates.
        - `start_position`: Initial position of the classical coordinates.
        - `start_momentum`: Initial momentum of the classical coordinates.
    """
    seed = kwargs["seed"]
    q = model.constants.init_position
    p = model.constants.init_momentum
    m = model.constants.classical_coordinate_mass
    h = model.constants.classical_coordinate_weight
    z = np.zeros((len(seed), model.constants.num_classical_coordinates), dtype=complex)
    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        z[s] = qp_to_z(q, p, m, h)
    return z


def init_classical_wigner_coherent_state(model, parameters, **kwargs):
    """
    Initialize classical coordinates according to the Wigner distribution of a coherent
    state of a harmonic oscillator.

    :math:`exp(a\hat{b}^{\dagger} - a^{*}\hat{b})\vert 0\rangle`

    where `a` is the complex displacement parameter of the coherent state.

    Required constants:
        - `coherent_state_displacement`: Array of complex displacement
          parameter for the coherent state.
        - `harmonic_frequency`: Array of harmonic frequencies.
    """
    seed = kwargs["seed"]
    a = model.constants.coherent_state_displacement
    w = model.constants.harmonic_frequency
    m = model.constants.classical_coordinate_mass
    h = model.constants.classical_coordinate_weight
    out = np.zeros(
        (len(seed), model.constants.num_classical_coordinates), dtype=complex
    )
    # Calculate the standard deviations for q and p.
    std_q = np.sqrt(0.5 / (w * m))
    std_p = np.sqrt(0.5 * m * w)
    mu_q = np.sqrt(2.0 / (m * w)) * np.real(a)
    mu_p = np.sqrt(2.0 / (m * w)) * np.imag(a)
    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        # Generate random q and p values.
        q = np.random.normal(
            loc=mu_q, scale=std_q, size=model.constants.num_classical_coordinates
        )
        p = np.random.normal(
            loc=mu_p, scale=std_p, size=model.constants.num_classical_coordinates
        )
        # Calculate the complex-valued classical coordinate.
        z = qp_to_z(q, p, m, h)
        out[s] = z
    return out
