"""
This module contains functions used by tasks.
"""

import logging
import numpy as np
from numba import njit
from qc_lab.constants import SMALL

logger = logging.getLogger(__name__)


def gen_sample_gaussian(constants, z0=None, seed=None, separable=True):
    """
    Generate a sample from a Gaussian distribution.

    Required constants:
        - num_classical_coordinates (int): Number of classical coordinates. Default: None.
        - mcmc_std (float): Standard deviation for sampling. Default: 1.
    """
    if seed is not None:
        np.random.seed(seed)
    num_classical_coordinates = constants.num_classical_coordinates
    if separable:
        rand = np.random.rand(num_classical_coordinates)
    else:
        rand = np.random.rand()
    std_re = constants.get("mcmc_std", 1)
    std_im = constants.get("mcmc_std", 1)
    # Generate random real and imaginary parts of z.
    z_re = np.random.normal(loc=0, scale=std_re, size=num_classical_coordinates)
    z_im = np.random.normal(loc=0, scale=std_im, size=num_classical_coordinates)
    z = z_re + 1j * z_im
    if z0 is None:
        return (
            np.random.rand(num_classical_coordinates)
            + 1j * np.random.rand(num_classical_coordinates),
            rand,
        )
    return z0 + z, rand


@njit
def calc_sparse_inner_product(inds, mels, shape, vec_l, vec_r):
    """
    Given the indices, matrix elements and shape of a sparse matrix,
    calculate the expectation value with a vector.

    Required constants:
        - None.
    """
    out = np.zeros((shape[:2]), dtype=np.complex128)
    for i in range(len(inds[0])):
        out[inds[0][i], inds[1][i]] = (
            out[inds[0][i], inds[1][i]]
            + np.conj(vec_l[inds[0][i], inds[2][i]])
            * mels[i]
            * vec_r[inds[0][i], inds[3][i]]
        )
    return out


def analytic_der_couple_phase(algorithm, sim, parameters, state, eigvals, eigvecs):
    """
    Calculates the phase change needed to fix the gauge using analytical derivative couplings.

    Required constants:
        - None.
    """
    der_couple_q_phase = np.ones(
        (
            sim.settings.batch_size,
            sim.model.constants.num_quantum_states,
        ),
        dtype=complex,
    )
    der_couple_p_phase = np.ones(
        (
            sim.settings.batch_size,
            sim.model.constants.num_quantum_states,
        ),
        dtype=complex,
    )
    for i in range(sim.model.constants.num_quantum_states - 1):
        j = i + 1
        evec_i = eigvecs[..., i]
        evec_j = eigvecs[..., j]
        eval_i = eigvals[..., i]
        eval_j = eigvals[..., j]
        ev_diff = eval_j - eval_i
        plus = np.zeros_like(ev_diff)
        if np.any(np.abs(ev_diff) < SMALL):
            plus[np.where(np.abs(ev_diff) < SMALL)] = 1
            logger.error("Degenerate eigenvalues detected.")
        der_couple_zc = np.zeros(
            (
                sim.settings.batch_size,
                sim.model.constants.num_classical_coordinates,
            ),
            dtype=complex,
        )
        der_couple_z = np.zeros(
            (
                sim.settings.batch_size,
                sim.model.constants.num_classical_coordinates,
            ),
            dtype=complex,
        )
        inds, mels, _ = state.dh_qc_dzc
        np.add.at(
            der_couple_zc,
            (inds[0], inds[1]),
            np.conj(evec_i)[inds[0], inds[2]]
            * mels
            * evec_j[inds[0], inds[3]]
            / ((ev_diff + plus)[inds[0]]),
        )
        np.add.at(
            der_couple_z,
            (inds[0], inds[1]),
            np.conj(evec_i)[inds[0], inds[3]]
            * np.conj(mels)
            * evec_j[inds[0], inds[2]]
            / ((ev_diff + plus)[inds[0]]),
        )
        der_couple_p = (
            1j
            * np.sqrt(
                0.5
                / (
                    sim.model.constants.classical_coordinate_weight
                    * sim.model.constants.classical_coordinate_mass
                )
            )[..., :]
            * (der_couple_z - der_couple_zc)
        )
        der_couple_q = np.sqrt(
            sim.model.constants.classical_coordinate_weight
            * sim.model.constants.classical_coordinate_mass
            * 0.5
        )[..., :] * (der_couple_z + der_couple_zc)
        der_couple_q_angle = np.angle(
            der_couple_q[
                np.arange(len(der_couple_q)),
                np.argmax(np.abs(der_couple_q), axis=-1),
            ]
        )
        der_couple_p_angle = np.angle(
            der_couple_p[
                np.arange(len(der_couple_p)),
                np.argmax(np.abs(der_couple_p), axis=-1),
            ]
        )
        der_couple_q_angle[np.where(np.abs(der_couple_q_angle) < SMALL)] = 0
        der_couple_p_angle[np.where(np.abs(der_couple_p_angle) < SMALL)] = 0
        der_couple_q_phase[..., i + 1 :] = (
            np.exp(1j * der_couple_q_angle[..., np.newaxis])
            * der_couple_q_phase[..., i + 1 :]
        )
        der_couple_p_phase[..., i + 1 :] = (
            np.exp(1j * der_couple_p_angle[..., np.newaxis])
            * der_couple_p_phase[..., i + 1 :]
        )
    return der_couple_q_phase, der_couple_p_phase


@njit
def matprod(mat, vec):
    """
    Perform matrix-vector multiplication.

    Required constants:
        - None.
    """
    out = np.zeros(np.shape(vec), dtype=np.complex128)
    for t in range(len(mat)):
        for i in range(len(mat[0])):
            accum = 0j
            for j in range(len(mat[0,])):
                accum = accum + mat[t, i, j] * vec[t, j]
            out[t, i] = accum
    return out


@njit
def wf_db_rk4(h_quantum, wf_db, dt_update):
    """
    Low-level function for quantum RK4 propagation.

    Required constants:
        - None.
    """
    k1 = -1j * matprod(h_quantum, wf_db)
    k2 = -1j * matprod(h_quantum, (wf_db + 0.5 * dt_update * k1))
    k3 = -1j * matprod(h_quantum, (wf_db + 0.5 * dt_update * k2))
    k4 = -1j * matprod(h_quantum, (wf_db + dt_update * k3))
    return wf_db + dt_update * (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def calc_delta_z_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the rescaling direction state.delta_z in FSSH.
    """
    traj_ind, final_state_ind, init_state_ind = (
        kwargs["traj_ind"],
        kwargs["final_state_ind"],
        kwargs["init_state_ind"],
    )
    rescaling_direction_fssh, has_rescaling_direction_fssh = sim.model.get(
        "rescaling_direction_fssh"
    )
    if has_rescaling_direction_fssh:
        delta_z = rescaling_direction_fssh(
            parameters,
            z=state.z[traj_ind],
            init_state_ind=init_state_ind,
            final_state_ind=final_state_ind,
        )
        return delta_z

    inds, mels, _ = state.dh_qc_dzc
    eigvecs_flat = state.eigvecs
    eigvals_flat = state.eigvals
    evec_init_state = eigvecs_flat[traj_ind][:, init_state_ind]
    evec_final_state = eigvecs_flat[traj_ind][:, final_state_ind]
    eval_init_state = eigvals_flat[traj_ind][init_state_ind]
    eval_final_state = eigvals_flat[traj_ind][final_state_ind]
    ev_diff = eval_final_state - eval_init_state
    inds_traj_ind = (
        inds[0][inds[0] == traj_ind],
        inds[1][inds[0] == traj_ind],
        inds[2][inds[0] == traj_ind],
        inds[3][inds[0] == traj_ind],
    )
    mels_traj_ind = mels[inds[0] == traj_ind]
    dkj_z = np.zeros((sim.model.constants.num_classical_coordinates), dtype=complex)
    dkj_zc = np.zeros((sim.model.constants.num_classical_coordinates), dtype=complex)
    np.add.at(
        dkj_z,
        (inds_traj_ind[1]),
        np.conj(evec_init_state)[inds_traj_ind[2]]
        * mels_traj_ind
        * evec_final_state[inds_traj_ind[3]]
        / ev_diff,
    )
    np.add.at(
        dkj_zc,
        (inds_traj_ind[1]),
        np.conj(evec_init_state)[inds_traj_ind[3]]
        * np.conj(mels_traj_ind)
        * evec_final_state[inds_traj_ind[2]]
        / ev_diff,
    )
    dkj_p = (
        1j
        * np.sqrt(
            0.5
            / (
                sim.model.constants.classical_coordinate_weight
                * sim.model.constants.classical_coordinate_mass
            )
        )
        * (dkj_z - dkj_zc)
    )
    dkj_q = np.sqrt(
        0.5
        * sim.model.constants.classical_coordinate_weight
        * sim.model.constants.classical_coordinate_mass
    ) * (dkj_z + dkj_zc)

    max_pos_q = np.argmax(np.abs(dkj_q))
    max_pos_p = np.argmax(np.abs(dkj_p))
    # Check for complex nonadiabatic couplings.
    if (
        np.abs(dkj_q[max_pos_q]) > SMALL
        and np.abs(np.sin(np.angle(dkj_q[max_pos_q]))) > SMALL
    ):
        logger.error("dkj_q Nonadiabatic coupling is complex, needs gauge fixing!")
    if (
        np.abs(dkj_p[max_pos_p]) > SMALL
        and np.abs(np.sin(np.angle(dkj_p[max_pos_p]))) > SMALL
    ):
        logger.error("dkj_p Nonadiabatic coupling is complex, needs gauge fixing!")
    delta_z = dkj_zc
    return delta_z


def numerical_fssh_hop(model, parameters, **kwargs):
    """
    Determines the coordinate rescaling in FSSH numerically.

    Required constants:
        - numerical_fssh_hop_gamma_range (float): Range for gamma. Default: 5.0.
        - numerical_fssh_hop_num_iter (int): Number of iterations. Default: 20.
        - numerical_fssh_hop_num_points (int): Number of points. Default: 10.
    """
    z = kwargs["z"]
    delta_z = kwargs["delta_z"]
    ev_diff = kwargs["ev_diff"]
    gamma_range = model.constants.get("numerical_fssh_hop_gamma_range", 5.0)
    max_iter = model.constants.get("numerical_fssh_hop_max_iter", 20)
    num_points = model.constants.get("numerical_fssh_hop_num_points", 10)
    thresh = model.constants.get("numerical_fssh_hop_threshold", 1e-6)
    h_c, _ = model.get("h_c")
    init_energy = h_c(model, parameters, z=np.array([z]), batch_size=1)[0]
    min_gamma = 0.0
    num_iter = 0
    min_energy = 1.0
    while min_energy > thresh and num_iter < max_iter:
        gamma_list = np.linspace(
            min_gamma - gamma_range, min_gamma + gamma_range, num_points
        )
        new_energies = np.abs(
            ev_diff
            - np.array(
                [
                    init_energy
                    - h_c(
                        model,
                        parameters,
                        z=np.array([z - 1j * gamma * delta_z]),
                        batch_size=1,
                    )[0]
                    for gamma in gamma_list
                ]
            )
        )
        min_gamma = gamma_list[np.argmin(new_energies)]
        min_energy = np.min(new_energies)
        gamma_range = gamma_range * 0.5
        num_iter += 1
    if min_energy > thresh:
        return np.zeros_like(z), False
    return -1j * min_gamma * delta_z, True
