"""
This file contains ingredient functions for use in Model classes.
"""

import numpy as np
from tqdm import tqdm
import functools


def make_ingredient_sparse(ingredient):
    """
    Converts a vectorized ingredient output to a sparse format
    """

    @functools.wraps(ingredient)
    def sparse_ingredient(*args, **kwargs):
        (model, constants, parameters) = args
        out = ingredient(model, constants, parameters, **kwargs)
        inds = np.where(out != 0)
        mels = out[inds]
        return inds, mels

    return sparse_ingredient


def vectorize_ingredient(ingredient):
    """
    Vectorize an ingredient function.
    assumes any kwarg that is a np.ndarray is vectorized over its firts index.
    non np.ndarray kwargs are assumed to not be vectorized.

    Args:
        ingredient (function): The ingredient function to vectorize.

    Returns:
        function: The vectorized ingredient function.
    """

    @functools.wraps(ingredient)
    def vectorized_ingredient(*args, **kwargs):
        (model, constants, parameters) = args
        batch_size = len(parameters.seed)
        keys = kwargs.keys()
        kwargs_list = []
        for n in range(batch_size):
            kwargs_n = {}
            for key in keys:
                if isinstance(kwargs[key], np.ndarray):
                    kwargs_n[key] = kwargs[key][n]
                else:
                    kwargs_n[key] = kwargs[key]
            kwargs_list.append(kwargs_n)
        out = np.array(
            [
                ingredient(model, constants, parameters, **kwargs_list[n])
                for n in range(batch_size)
            ]
        )
        return out

    return vectorized_ingredient


def harmonic_oscillator_h_c(model, constants, parameters, **kwargs):
    """
    Harmonic oscillator classical Hamiltonian function.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The classical Hamiltonian values.

    Required attributes of constants:
        - classical_coordinate_weight
        - harmonic_oscillator_frequency
        - classical_coordinate_mass

    Required attributes of parameters:
        - seed
    """
    z = kwargs.get("z_coord")
    h = constants.classical_coordinate_weight[np.newaxis, :]
    w = constants.harmonic_oscillator_frequency[np.newaxis, :]
    m = constants.classical_coordinate_mass[np.newaxis, :]
    q = np.sqrt(2 / (m * h)) * np.real(z)
    p = np.sqrt(2 * m * h) * np.imag(z)
    h_c = np.sum((1 / 2) * (((p**2) / m) + m * (w**2) * (q**2)), axis=-1)
    return h_c


def harmonic_oscillator_dh_c_dzc(model, constants, parameters, **kwargs):
    """
    Calculate the derivative of the classical harmonic oscillator Hamiltonian
    with respect to the z-coordinates.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The derivative of the classical Hamiltonian.

    Required attributes of constants:
        - classical_coordinate_weight

    Required attributes of parameters:
        - None
    """
    z_coord = kwargs["z_coord"]
    dh_c_dzc = constants.classical_coordinate_weight[..., :] * z_coord + 0.0j
    return dh_c_dzc


def two_level_system_h_q(model, constants, parameters, **kwargs):
    """
    Calculate the quantum Hamiltonian for a two-level system.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The quantum Hamiltonian.

    Required attributes of constants:
        - two_level_system_a
        - two_level_system_b
        - two_level_system_c
        - two_level_system_d

    Required attributes of parameters:
        - seed
    """
    del kwargs
    batch_size = len(parameters.seed)
    h_q = np.zeros((batch_size, 2, 2), dtype=complex)
    h_q[:, 0, 0] = constants.two_level_system_a
    h_q[:, 1, 1] = constants.two_level_system_b
    h_q[:, 0, 1] = constants.two_level_system_c + 1j * constants.two_level_system_d
    h_q[:, 1, 0] = constants.two_level_system_c - 1j * constants.two_level_system_d
    return h_q


def nearest_neighbor_lattice_h_q(model, constants, parameters, **kwargs):
    """
    Calculate the quantum Hamiltonian for a nearest-neighbor lattice.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The quantum Hamiltonian.

    Required attributes of constants:
        - num_quantum_states
        - nearest_neighbor_lattice_hopping_energy
        - nearest_neighbor_lattice_periodic_boundary

    Required attributes of parameters:
        - seed
    """
    del kwargs
    num_sites = constants.num_quantum_states
    hopping_energy = constants.nearest_neighbor_lattice_hopping_energy
    periodic_boundary = constants.nearest_neighbor_lattice_periodic_boundary
    h_q = np.zeros((num_sites, num_sites), dtype=complex)

    # Fill the Hamiltonian matrix with hopping energies
    for n in range(num_sites - 1):
        h_q[n, n + 1] += -hopping_energy
        h_q[n + 1, n] += np.conj(h_q[n, n + 1])

    # Apply periodic boundary conditions if specified
    if periodic_boundary:
        h_q[0, num_sites - 1] += -hopping_energy
        h_q[num_sites - 1, 0] += np.conj(h_q[0, num_sites - 1])

    return h_q[np.newaxis, :, :] + np.zeros(
        (len(parameters.seed), num_sites, num_sites), dtype=complex
    )


def holstein_coupling_h_qc(model, constants, parameters, **kwargs):
    """
    Calculate the Holstein coupling Hamiltonian.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The Holstein coupling Hamiltonian.

    Required attributes of constants:
        - holstein_coupling_oscillator_frequency
        - holstein_coupling_dimensionless_coupling
        - num_quantum_states

    Required attributes of parameters:
        - None
    """
    z_coord = kwargs["z_coord"]
    num_sites = constants.num_quantum_states
    oscillator_frequency = constants.holstein_coupling_oscillator_frequency
    dimensionless_coupling = constants.holstein_coupling_dimensionless_coupling
    h_qc = np.zeros((*np.shape(z_coord)[:-1], num_sites, num_sites), dtype=complex)
    np.einsum("...ii->...i", h_qc)[...] = (
        dimensionless_coupling * oscillator_frequency
    )[..., :] * (z_coord + np.conj(z_coord)) + 0.0j
    return h_qc


def holstein_coupling_dh_qc_dzc(model, constants, parameters, **kwargs):
    """
    Calculate the derivative of the Holstein coupling Hamiltonian with
    respect to the z-coordinates.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The derivative of the Holstein coupling Hamiltonian.

    Required attributes of constants:
        - holstein_coupling_oscillator_frequency
        - holstein_coupling_dimensionless_coupling
        - num_quantum_states

    Required attributes of parameters:
        - None
    """

    if hasattr(model, "dh_qc_dzc_mels") and hasattr(model, "dh_qc_dzc_inds"):
        return model.dh_qc_dzc_inds, model.dh_qc_dzc_mels
    else:
        z_coord = kwargs["z_coord"]
        num_sites = constants.num_quantum_states
        oscillator_frequency = constants.holstein_coupling_oscillator_frequency
        dimensionless_coupling = constants.holstein_coupling_dimensionless_coupling

        dh_qc_dzc = np.zeros(
            (len(z_coord), num_sites, num_sites, num_sites), dtype=complex
        )

        np.einsum("tiii->ti", dh_qc_dzc, optimize="greedy")[...] = (
            dimensionless_coupling * oscillator_frequency
        )[..., :] * (np.ones_like(z_coord)) + 0.0j
        inds = np.where(dh_qc_dzc != 0)
        mels = dh_qc_dzc[inds]
        model.dh_qc_dzc_inds = inds
        model.dh_qc_dzc_mels = dh_qc_dzc[inds]
        return inds, mels


def harmonic_oscillator_hop(model, constants, parameters, **kwargs):
    """
    Perform a hopping operation for the harmonic oscillator.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: The updated z-coordinates and a boolean indicating if the hop was accepted.

    Required attributes of constants:
        - harmonic_oscillator_frequency
        - classical_coordinate_weight

    Required attributes of parameters:
        - None
    """
    z_coord = kwargs["z_coord"]
    delta_z_coord = kwargs["delta_z_coord"]
    ev_diff = kwargs["ev_diff"]
    hopped = False
    delta_zc_coord = np.conj(delta_z_coord)
    zc_coord = np.conj(z_coord)

    a_const = (1 / 4) * (
        (
            (constants.harmonic_oscillator_frequency**2)
            / constants.classical_coordinate_weight
        )
        - constants.classical_coordinate_weight
    )

    b_const = (1 / 4) * (
        (
            (constants.harmonic_oscillator_frequency**2)
            / constants.classical_coordinate_weight
        )
        + constants.classical_coordinate_weight
    )

    akj_z = np.sum(
        2 * delta_zc_coord * delta_z_coord * b_const
        - a_const * (delta_z_coord**2 + delta_zc_coord**2)
    )
    bkj_z = 2j * np.sum(
        (z_coord * delta_z_coord - delta_zc_coord * zc_coord) * a_const
        + (delta_z_coord * zc_coord - delta_zc_coord * z_coord) * b_const
    )
    ckj_z = ev_diff

    disc = bkj_z**2 - 4 * akj_z * ckj_z
    if disc >= 0:
        if bkj_z < 0:
            gamma = bkj_z + np.sqrt(disc)
        else:
            gamma = bkj_z - np.sqrt(disc)
        if akj_z == 0:
            gamma = 0
        else:
            gamma = gamma / (2 * akj_z)
        # adjust classical coordinate
        z_coord = z_coord - 1.0j * np.real(gamma) * delta_z_coord
        hopped = True
    return z_coord, hopped


def harmonic_oscillator_boltzmann_init_classical(
    model, constants, parameters, **kwargs
):
    """
    Initialize classical coordinates according to Boltzmann statistics for the Harmonic oscillator.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The initialized classical coordinates.

    Required attributes of constants:
        - temp
        - classical_coordinate_weight
        - harmonic_oscillator_frequency
        - classical_coordinate_mass
        - num_classical_coordinates

    Required attributes of parameters:
        - seed
    """
    seed = kwargs.get("seed", None)
    kBT = constants.temp

    h = constants.classical_coordinate_weight
    w = constants.harmonic_oscillator_frequency
    m = constants.classical_coordinate_mass
    out = np.zeros((len(seed), constants.num_classical_coordinates), dtype=complex)
    for s in range(len(seed)):
        np.random.seed(seed[s])
        q = np.random.normal(
            loc=0,
            scale=np.sqrt(kBT / (m * (w**2))),
            size=(constants.num_classical_coordinates),
        )
        p = np.random.normal(
            loc=0, scale=np.sqrt(kBT), size=(constants.num_classical_coordinates)
        )
        z = np.sqrt(h * m / 2) * (q + 1.0j * (p / (h * m)))
        out[s] = z
    return out


def default_numerical_boltzmann_init_classical(model, constants, parameters, **kwargs):
    """
    Initialize classical coordinates according to Boltzmann statistics using a numerical method.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The initialized classical coordinates.

    Required attributes of constants:
        - numerical_boltzmann_init_classical_num_points
        - numerical_boltzmann_init_classical_max_amplitude
        - temp
        - num_classical_coordinates

    Required attributes of parameters:
        - seed
    """
    seed = kwargs.get("seed", None)
    out = np.zeros((len(seed), constants.num_classical_coordinates), dtype=complex)
    for s in tqdm(range(len(seed))):
        np.random.seed(seed[s])
        rand_val = np.random.rand()
        num_points = constants.numerical_boltzmann_init_classical_num_points
        max_amplitude = constants.numerical_boltzmann_init_classical_max_amplitude

        z_out = np.zeros((constants.num_classical_coordinates), dtype=complex)

        for n in range(constants.num_classical_coordinates):
            grid = (
                2 * max_amplitude * (np.random.rand(num_points) - 0.5)
            )  # np.linspace(-max_amplitude, max_amplitude, num_points)
            kinetic_grid = 1.0j * grid
            potential_grid = grid
            # construct grid for kinetic points
            kinetic_points = np.zeros(
                (num_points, constants.num_classical_coordinates), dtype=complex
            )
            # construct grid for potential points
            potential_points = np.zeros(
                (num_points, constants.num_classical_coordinates), dtype=complex
            )
            for p in range(num_points):
                kinetic_points[p, n] = kinetic_grid[p]
                potential_points[p, n] = potential_grid[p]
            # calculate kinetic energies on the grid
            kinetic_energies = np.array(
                [
                    model.h_c(constants, parameters, z_coord=kinetic_points[p])
                    for p in range(num_points)
                ]
            )
            boltz_facs = np.exp(-kinetic_energies / constants.temp)
            boltz_facs = boltz_facs / np.sum(boltz_facs)
            # calculate cumulative distribution
            tot = 0
            for k in range(num_points):
                tot += boltz_facs[k]
                if rand_val <= tot:
                    z_out[n] += kinetic_grid[k]
                    break
            # calculate potential energies on the grid
            potential_energies = np.array(
                [
                    model.h_c(constants, parameters, z_coord=potential_points[p])
                    for p in range(num_points)
                ]
            )
            boltz_facs = np.exp(-potential_energies / constants.temp)
            boltz_facs = boltz_facs / np.sum(boltz_facs)
            # calculate cumulative distribution
            tot = 0
            for p in range(num_points):
                tot += boltz_facs[p]
                if rand_val <= tot:
                    z_out[n] += potential_grid[p]
                    break
        out[s] = z_out
    return out


def harmonic_oscillator_wigner_init_classical(model, constants, parameters, **kwargs):
    """
    Initialize classical coordinates according to the Wigner distribution
    of the ground state of a harmonic oscillator.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The initialized classical coordinates.

    Required attributes of constants:
        - classical_coordinate_weight
        - mass
        - temp
        - num_classical_coordinates

    Required attributes of parameters:
        - seed
    """
    seed = kwargs.get("seed", None)
    np.random.seed(seed)

    # Calculate the standard deviations for q and p
    std_q = np.sqrt(
        1
        / (
            2
            * constants.classical_coordinate_weight
            * constants.mass
            * np.tanh(constants.classical_coordinate_weight / (2 * constants.temp))
        )
    )
    std_p = np.sqrt(
        (constants.mass * constants.classical_coordinate_weight)
        / (2 * np.tanh(constants.classical_coordinate_weight / (2 * constants.temp)))
    )

    # Generate random q and p values
    q = np.random.normal(loc=0, scale=std_q, size=constants.num_classical_coordinates)
    p = np.random.normal(loc=0, scale=std_p, size=constants.num_classical_coordinates)

    # Calculate the classical coordinates z
    z = np.sqrt(constants.classical_coordinate_weight * constants.mass / 2) * (
        q + 1.0j * (p / (constants.classical_coordinate_weight * constants.mass))
    )

    return z
