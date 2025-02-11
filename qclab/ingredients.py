"""
This file contains ingredient functions for use in Model classes.
"""

import warnings
import numpy as np
from tqdm import tqdm

def harmonic_oscillator_h_c(model, constants, parameters, **kwargs):
    """
    Harmonic oscillator classical Hamiltonian function.
    """

    z = kwargs.get("z_coord", parameters.z_coord)
    h = constants.pq_weight[np.newaxis, :]
    w = constants.harmonic_oscillator_frequency[np.newaxis, :]
    m = constants.harmonic_oscillator_mass[np.newaxis, :]
    q = np.sqrt(2 / (m * h)) * np.real(z)
    p = np.sqrt(2 * m * h) * np.imag(z)
    h_c = np.sum((1 / 2) * (((p**2) / m) + m * (w**2) * (q**2)), axis=-1)
    return h_c


def harmonic_oscillator_dh_c_dzc(model, constants, parameters, **kwargs):
    """
    Calculate the derivative of the classical harmonic oscillator Hamiltonian
    with respect to the z-coordinates.
    """
    z_coord = kwargs["z_coord"]
    dh_c_dzc = constants.pq_weight[..., :] * z_coord + 0.0j
    return dh_c_dzc


def two_level_system_h_q(model, constants, parameters, **kwargs):
    """
    Calculate the quantum Hamiltonian for a two-level system.
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
    """
    del kwargs
    num_sites = constants.nearest_neighbor_lattice_num_sites
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

    return h_q[np.newaxis, :, :] * np.ones((len(parameters.seed), num_sites, num_sites)).astype(complex)


def holstein_coupling_h_qc(model, constants, parameters, **kwargs):
    """
    Calculate the Holstein coupling Hamiltonian.
    """
    z_coord = kwargs["z_coord"]
    num_sites = constants.holstein_coupling_num_sites
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
    """
    z_coord = kwargs["z_coord"]
    num_sites = constants.holstein_coupling_num_sites
    oscillator_frequency = constants.holstein_coupling_oscillator_frequency
    dimensionless_coupling = constants.holstein_coupling_dimensionless_coupling
    dh_qc_dzc = np.zeros(
        (*np.shape(z_coord)[:-1], num_sites, num_sites, num_sites), dtype=complex
    )
    np.einsum("...iii->...i", dh_qc_dzc)[...] = (
        dimensionless_coupling * oscillator_frequency
    )[..., :] * (np.ones_like(z_coord)) + 0.0j
    return dh_qc_dzc


def default_numerical_fssh_hop(model, constants, parameters, **kwargs):
    z_coord = kwargs["z_coord"]
    delta_z_coord = kwargs["delta_z_coord"]
    ev_diff = kwargs["ev_diff"]
    hopped = False
    delta_zc_coord = np.conj(delta_z_coord)
    zc = np.conj(z_coord)
    warnings.warn("Hop function excludes mass, check it", UserWarning)

    gamma_range = constants.numerical_fssh_hop_gamma_range
    num_iter = constants.numerical_fssh_hop_num_iter
    num_points = constants.numerical_fssh_hop_num_points

    init_energy = model.h_c(constants, parameters, z_coord=z_coord)

    min_gamma = 0
    for iter in range(num_iter):
        gamma_list = np.linspace(
            min_gamma - gamma_range, min_gamma + gamma_range, num_points
        )
        new_energies = np.abs(
            ev_diff
            - np.array(
                [
                    init_energy
                    - model.h_c(
                        constants,
                        parameters,
                        z_coord=z_coord - 1.0j * gamma * delta_z_coord,
                    )
                    for gamma in gamma_list
                ]
            )
        )
        min_gamma = gamma_list[np.argmin(new_energies)]
        min_energy = np.min(new_energies)
        gamma_range = gamma_range / 2

    if min_energy > 1 / num_points:
        # print('rejected hop', min_energy)

        return z_coord, False
    else:
        # print('accepted hop', min_energy)
        return z_coord - 1.0j * min_gamma * delta_z_coord, True

def harmonic_oscillator_hop(model, constants, parameters, **kwargs):
    """
    Perform a hopping operation for the harmonic oscillator.

    Model Ingredient:
        - model.hop

    Required keyword arguments:
        - z_coord (np.ndarray): The z-coordinates.
        - delta_z_coord (np.ndarray): The change in z-coordinates.
        - ev_diff (float): The energy difference.

    Model parameters:
        - constants.pq_weight (np.ndarray): The weight parameters.

    Related functions:
        - :func:`harmonic_oscillator_boltzmann_init_classical`
        - :func:`harmonic_oscillator_wigner_init_classical`
    """
    z_coord = kwargs["z_coord"]
    delta_z_coord = kwargs["delta_z_coord"]
    ev_diff = kwargs["ev_diff"]
    hopped = False
    delta_zc_coord = np.conj(delta_z_coord)
    zc = np.conj(z_coord)
    warnings.warn("Hop function excludes mass, check it", UserWarning)
    akj_z = np.real(np.sum(constants.pq_weight * delta_zc_coord * delta_z_coord))
    bkj_z = np.real(
        np.sum(
            1j * constants.pq_weight * (zc * delta_z_coord - z_coord * delta_zc_coord)
        )
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
    Now vectorized
    Initialize classical coordinates according to Boltzmann statistics.

    Model Ingredient:
        - model.init_classical

    Required keyword arguments:
        - seed (int): The random seed.

    Model parameters:
        - constants.temp (float): Temperature.
        - constants.mass (float): Mass.
        - constants.pq_weight (np.ndarray): The weight parameters.
        - constants.num_classical_coordinates (int): Number of classical coordinates.

    Related functions:
        - :func:`harmonic_oscillator_wigner_init_classical`
    """
    seed = kwargs.get("seed", None)
    kBT = constants.temp

    h = constants.pq_weight
    w = constants.harmonic_oscillator_frequency
    m = constants.harmonic_oscillator_mass
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

def default_numerical_boltzmann_init_classical_(model, constants, parameters, **kwargs):
    seed = kwargs.get("seed", None)
    out = np.zeros((len(seed), constants.num_classical_coordinates), dtype=complex)
    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        rand_val = np.random.rand()
        num_points = constants.numerical_boltzmann_init_classical_num_points
        max_amplitude = constants.numerical_boltzmann_init_classical_max_amplitude
        grid = 2 * max_amplitude * (np.random.rand(num_points) - 0.5)
        kinetic_grid = 1.0j * grid
        potential_grid = grid
        z_out = np.zeros((constants.num_classical_coordinates), dtype=complex)

        parameters.z_coord = np.zeros(constants.num_classical_coordinates) + 0.0j
        for n in range(constants.num_classical_coordinates):
            kinetic_points = np.zeros((num_points, constants.num_classical_coordinates), dtype=complex)
            potential_points = np.zeros((num_points, constants.num_classical_coordinates), dtype=complex)
            for p in range(num_points):
                kinetic_points[p, n] = kinetic_grid[p]
                potential_points[p, n] = potential_grid[p]

            kinetic_energies = np.array([
                model.h_c(constants, parameters, z_coord=kinetic_points[p])
                for p in range(num_points)
            ])
            boltz_facs = np.exp(-kinetic_energies / constants.temp)
            boltz_facs /= np.sum(boltz_facs)

            tot = 0
            for k, boltz_fac in enumerate(boltz_facs):
                tot += boltz_fac
                if rand_val <= tot:
                    z_out[n] += kinetic_grid[k]
                    break

            potential_energies = np.array([
                model.h_c(constants, parameters, z_coord=potential_points[p])
                for p in range(num_points)
            ])
            boltz_facs = np.exp(-potential_energies / constants.temp)
            boltz_facs /= np.sum(boltz_facs)

            tot = 0
            for p, boltz_fac in enumerate(boltz_facs):
                tot += boltz_fac
                if rand_val <= tot:
                    z_out[n] += potential_grid[p]
                    break

        out[s] = z_out
    return out

def default_numerical_boltzmann_init_classical(model, constants, parameters, **kwargs):
    seed = kwargs.get("seed", None)
    out = np.zeros((len(seed), constants.num_classical_coordinates), dtype=complex)
    for s in tqdm(range(len(seed))):
        np.random.seed(seed[s])
        rand_val = np.random.rand()
        num_points = constants.numerical_boltzmann_init_classical_num_points
        max_amplitude = constants.numerical_boltzmann_init_classical_max_amplitude
        
        z_out = np.zeros((constants.num_classical_coordinates), dtype=complex)

        parameters.z_coord = np.zeros(constants.num_classical_coordinates) + 0.0j
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

    Model Ingredient:
        - model.init_classical

    Required constants attributes:
        - pq_weight (float): The pq weight parameter.
        - mass (float): The mass of the harmonic oscillator.
        - temp (float): The temperature.
        - num_classical_coordinates (int): The number of classical coordinates.

    Related functions:
        - :func:`harmonic_oscillator_boltzmann_init_classical`
    """
    seed = kwargs.get("seed", None)
    np.random.seed(seed)

    # Calculate the standard deviations for q and p
    std_q = np.sqrt(
        1
        / (
            2
            * constants.pq_weight
            * constants.mass
            * np.tanh(constants.pq_weight / (2 * constants.temp))
        )
    )
    std_p = np.sqrt(
        (constants.mass * constants.pq_weight)
        / (2 * np.tanh(constants.pq_weight / (2 * constants.temp)))
    )

    # Generate random q and p values
    q = np.random.normal(loc=0, scale=std_q, size=constants.num_classical_coordinates)
    p = np.random.normal(loc=0, scale=std_p, size=constants.num_classical_coordinates)

    # Calculate the classical coordinates z
    z = np.sqrt(constants.pq_weight * constants.mass / 2) * (
        q + 1.0j * (p / (constants.pq_weight * constants.mass))
    )

    return z


def dh_c_dzc_finite_differences(model, constants, parameters, **kwargs):
    z_coord = kwargs["z_coord"]
    # Approximate the gradient using finite differences
    delta_z = 1e-6
    offset_z_coord_re = z_coord[np.newaxis, :] + np.identity(len(z_coord)) * delta_z
    offset_z_coord_im = (
        z_coord[np.newaxis, :] + 1j * np.identity(len(z_coord)) * delta_z
    )

    h_c_0 = model.h_c(constants, parameters, z_coord=z_coord)
    dh_c_dzc = np.zeros((len(z_coord), *np.shape(h_c_0)), dtype=complex)

    for n in range(len(z_coord)):
        h_c_offset_re = model.h_c(constants, parameters, z_coord=offset_z_coord_re[n])
        diff_re = (h_c_offset_re - h_c_0) / delta_z
        h_c_offset_im = model.h_c(constants, parameters, z_coord=offset_z_coord_im[n])
        diff_im = (h_c_offset_im - h_c_0) / delta_z
        dh_c_dzc[n] = 0.5 * (diff_re + 1j * diff_im)
    return dh_c_dzc


def finite_differences_dh_c_dzc_vectorized(model, constants, parameters, **kwargs):
    z_coord = kwargs["z_coord"]
    delta_z = np.ones((len(z_coord),))


def dh_qc_dzc_finite_differences(model, constants, parameters, **kwargs):
    z_coord = kwargs["z_coord"]
    # Approximate the gradient using finite differences
    delta_z = 1e-6
    offset_z_coord_re = z_coord[np.newaxis, :] + np.identity(len(z_coord)) * delta_z
    offset_z_coord_im = (
        z_coord[np.newaxis, :] + 1j * np.identity(len(z_coord)) * delta_z
    )

    h_qc_0 = model.h_qc(constants, parameters, z_coord=z_coord)
    dh_qc_dzc = np.zeros((len(z_coord), *np.shape(h_qc_0)), dtype=complex)

    for n in range(len(z_coord)):
        h_qc_offset_re = model.h_qc(constants, parameters, z_coord=offset_z_coord_re[n])
        diff_re = (h_qc_offset_re - h_qc_0) / delta_z
        h_qc_offset_im = model.h_qc(constants, parameters, z_coord=offset_z_coord_im[n])
        diff_im = (h_qc_offset_im - h_qc_0) / delta_z
        dh_qc_dzc[n] = 0.5 * (diff_re + 1j * diff_im)
    return dh_qc_dzc


