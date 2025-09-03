"""
This module contains functions used in QC Lab. This includes
functions that are used by tasks, ingredients, and drivers.
"""

import logging
import functools
import numpy as np
from qc_lab.utils import njit
from qc_lab.constants import SMALL
from qc_lab.variable import Variable

logger = logging.getLogger(__name__)


@njit
def update_z_rk4_k123_sum(z_0, classical_forces, quantum_classical_forces, dt_update):
    """
    Low-level function to calculate the intermediate z coordinate and k values
    for RK4 update. Applies to steps 1-3.
    """
    batch_size, num_classical_coordinates = z_0.shape
    k = np.empty((batch_size, num_classical_coordinates), dtype=np.complex128)
    out = np.empty((batch_size, num_classical_coordinates), dtype=np.complex128)
    for i in range(z_0.shape[0]):
        for j in range(z_0.shape[1]):
            k[i, j] = -1j * (classical_forces[i, j] + quantum_classical_forces[i, j])
            out[i, j] = z_0[i, j] + dt_update * k[i, j]
    return out, k


@njit
def update_z_rk4_k4_sum(
    z_0, k1, k2, k3, classical_forces, quantum_classical_forces, dt_update
):
    """
    Low-level function to calculate the fourth and final step for the RK4 update.
    """
    for i in range(z_0.shape[0]):
        for j in range(z_0.shape[1]):
            z_0[i, j] = z_0[i, j] + (dt_update / 6.0) * (
                k1[i, j]
                + 2.0 * k2[i, j]
                + 2.0 * k3[i, j]
                - 1j * (classical_forces[i, j] + quantum_classical_forces[i, j])
            )
    return z_0


@njit()
def dqdp_to_dzc(dq, dp, m, h):
    """
    Convert derivatives w.r.t. q and p (dq and dp, respectively) to
    the derivative w.r.t. zc.
    """
    dp_present = dp is not None
    dq_present = dq is not None
    if dp_present:
        return 1j * np.sqrt(m * h / 2) * dp
    if dq_present:
        return np.sqrt(0.5 / (m * h)) * dq.astype(np.complex128)
    if dq_present and dp_present:
        return np.sqrt(0.5 / (m * h)) * dq + 1j * np.sqrt(m * h / 2) * dp
    raise ValueError("At least one of dq or dp must be provided.")


@njit()
def z_to_q(z, m, h):
    """
    Convert complex coordinates to position coordinate.
    """
    return np.real((1.0 / np.sqrt(2.0 * m * h)) * (z + np.conj(z)))


@njit()
def z_to_p(z, m, h):
    """
    Convert complex coordinates to momentum coordinate.
    """
    return np.real(1j * np.sqrt(0.5 * m * h) * (np.conj(z) - z))


@njit()
def qp_to_z(q, p, m, h):
    """
    Convert real coordinates to complex coordinates.
    """
    return np.sqrt(0.5 * m * h) * q + 1j * np.sqrt(0.5 / (m * h)) * p


def make_ingredient_sparse(ingredient):
    """
    Wrapper that converts a vectorized ingredient output to a sparse format consisting
    of the indices (inds), nonzero elements (mels), and shape.
    """

    @functools.wraps(ingredient)
    def sparse_ingredient(*args, **kwargs):
        (model, constants, parameters) = args
        out = ingredient(model, constants, parameters, **kwargs)
        inds = np.where(out != 0)
        mels = out[inds]
        shape = np.shape(out)
        return inds, mels, shape

    return sparse_ingredient


def vectorize_ingredient(ingredient):
    """
    Wrapper that vectorize an ingredient function.

    It assumes that any kwarg is an numpy.ndarray that is vectorized over its first
    index. Other kwargs are assumed to not be vectorized.
    """

    @functools.wraps(ingredient)
    def vectorized_ingredient(*args, **kwargs):
        (model, parameters) = args
        batch_size = kwargs.get("batch_size", len(parameters.seed))
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
            [ingredient(model, parameters, **kwargs_list[n]) for n in range(batch_size)]
        )
        return out

    return vectorized_ingredient


@njit()
def dh_c_dzc_harmonic_jit(z, h, w):
    """
    Derivative of the harmonic oscillator classical Hamiltonian function with respect to
    the conjugate `z` coordinate.

    This is a low-level function accelerated using Numba.
    """
    # a = 0.5 * (((w**2) / h) - h)
    # b = 0.5 * (((w**2) / h) + h)
    # out = b[..., :] * z + a[..., :] * np.conj(z)

    batch_size, num_classical_coordinates = z.shape
    out = np.empty((batch_size, num_classical_coordinates), dtype=np.complex128)
    w2_over_h = (w**2) / h
    for i in range(batch_size):
        for j in range(num_classical_coordinates):
            zij = z[i, j]
            out[i, j] = complex(w2_over_h[j] * zij.real, h[j] * zij.imag)

    return out


@njit()
def h_qc_diagonal_linear_jit(z, gamma):
    """
    Low level function to generate the diagonal linear quantum-classical Hamiltonian.
    """
    batch_size = z.shape[0]
    num_classical_coordinates = z.shape[1]
    num_sites = gamma.shape[0]
    h_qc = np.zeros((batch_size, num_sites, num_sites), dtype=np.complex128)
    for b in range(batch_size):
        for i in range(num_sites):
            acc = 0.0
            for j in range(num_classical_coordinates):
                acc += gamma[i, j] * (2.0 * z[b, j].real)
            h_qc[b, i, i] = acc
    return h_qc


def gen_sample_gaussian(constants, z0=None, seed=None, separable=True):
    """
    Generate a sample from a Gaussian distribution.

    Required constants:
        - num_classical_coordinates (int): Number of classical coordinates.
          Default: None.
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


@njit(cache=True)
def calc_sparse_inner_product(inds, mels, shape, vec_l_conj, vec_r):
    """
    Take a sparse gradient of a matrix (batch_size, num_classical_coordinates,
    num_quantum_state, num_quantum_states) and calculate the matrix element of
    the vectors vec_l_conj and vec_r with  shape (batch_size, num_quantum_states).
    """
    batch_size, num_classical_coordinates, num_quantum_states = (
        shape[0],
        shape[1],
        shape[2],
    )
    r = inds[0]
    c = inds[1]
    a = inds[2]
    b = inds[3]

    out = np.zeros(batch_size * num_classical_coordinates, dtype=np.complex128)

    l_flat = vec_l_conj.reshape(batch_size * num_quantum_states)
    r_flat = vec_r.reshape(batch_size * num_quantum_states)

    for i in range(mels.shape[0]):
        ri = r[i]
        ci = c[i]
        idx_out = ri * num_classical_coordinates + ci
        idx_l = ri * num_quantum_states + a[i]
        idx_r = ri * num_quantum_states + b[i]
        out[idx_out] += l_flat[idx_l] * mels[i] * r_flat[idx_r]

    return out.reshape((batch_size, num_classical_coordinates))


def analytic_der_couple_phase(algorithm, sim, parameters, state, eigvals, eigvecs):
    """
    Calculates the phase change needed to fix the gauge using analytical derivative
    couplings.

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
    # Check positions where the nonadiabatic coupling is greater than SMALL.
    big_pos = np.arange(sim.model.constants.num_classical_coordinates, dtype=int)[
        np.abs(dkj_zc) > SMALL
    ]
    # Calculate a weighting factor to rescale real and imaginary parts appropriately.
    imag_weight = np.sqrt(
        0.5
        / (
            sim.model.constants.classical_coordinate_weight
            * sim.model.constants.classical_coordinate_mass
        )
    )
    real_weight = np.sqrt(
        0.5
        * (
            sim.model.constants.classical_coordinate_weight
            * sim.model.constants.classical_coordinate_mass
        )
    )
    # Determine if the real and imaginary parts are properly aligned.
    if not (
        np.allclose(
            (imag_weight * np.imag(dkj_z))[big_pos],
            (-imag_weight * np.imag(dkj_zc))[big_pos],
            atol=SMALL,
        )
    ) or not (
        np.allclose(
            (real_weight * np.real(dkj_z))[big_pos],
            (real_weight * np.real(dkj_zc))[big_pos],
            atol=SMALL,
        )
    ):
        logger.error("Nonadiabatic coupling is complex, needs gauge fixing!")
    return dkj_zc


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


def initialize_variable_objects(sim, seed):
    """
    Generate the `parameter` and `state` variables for a simulation.

    Args:
        sim (Simulation): The simulation instance.
        seed (Iterable[int]): Array of trajectory seeds.

    Returns:
        tuple[Variable, Variable]: The `parameter` and `state` objects.
    """
    state_variable = Variable()
    state_variable.seed = seed
    logger.info("Initializing state variable with seed %s.", state_variable.seed)
    for name in sim.state.__dict__.keys():
        obj = getattr(sim.state, name)
        if isinstance(obj, np.ndarray) and name[0] != "_":
            init_shape = np.shape(obj)
            new_obj = (
                np.zeros((len(seed), *init_shape), dtype=obj.dtype) + obj[np.newaxis]
            )
            logger.info(
                "Initializing state variable %s with shape %s.", name, new_obj.shape
            )
            setattr(state_variable, name, new_obj)
        elif name[0] != "_":
            logger.warning(
                "Variable %s in sim.state is not an array, "
                "skipping initialization in state Variable object.",
                name,
            )
    parameter_variable = Variable()
    return parameter_variable, state_variable
