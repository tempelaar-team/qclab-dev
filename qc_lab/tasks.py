import warnings
import numpy as np
from numba import njit
from qc_lab import ingredients


def initialize_branch_seeds(sim, parameters, state, **kwargs):
    """
    Initialize the seeds in each branch
    """
    num_branches = sim.algorithm.settings.num_branches
    batch_size = sim.settings.batch_size
    orig_seeds = state.seed
    min_seed = orig_seeds.min()
    if min_seed != orig_seeds[0]:
        warnings.warn(
            "Minimum seed is not the first, this could lead to redundancies.",
            UserWarning,
        )
    num_prev_trajs = min_seed * num_branches
    state.branch_ind = (
        np.zeros((batch_size // num_branches, num_branches), dtype=int)
        + np.arange(num_branches)[np.newaxis, :]
    ).flatten()
    new_seeds = (num_prev_trajs + np.arange(len(orig_seeds))) // num_branches
    parameters.seed = new_seeds
    state.seed = new_seeds
    return parameters, state


def initialize_z_coord(sim, parameters, state, **kwargs):
    """
    Initialize the classical coordinate by using the init_classical function from the model object.
    """
    seed = kwargs["seed"]
    state.z_coord = sim.model.init_classical(sim.model.constants, parameters, seed=seed)
    return parameters, state


def assign_to_parameters(sim, parameters, state, **kwargs):
    """
    Assign the value of the variable "val" to the parameters object with the name "name".
    """
    name = kwargs["name"]
    val = kwargs["val"]
    setattr(parameters, name, val)
    return parameters, state


def dh_c_dzc_finite_differences(model, constants, parameters, **kwargs):
    """
    Calculate the gradient of the classical Hamiltonian using finite differences.

    Args:
        model: The model object.
        constants: The constants object.
        parameters: The parameters object.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The gradient of the classical Hamiltonian.

    Required attributes of constants:
        - None

    Required attributes of parameters:
        - seed

    Required model ingredients:
        - h_c
    """
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

def dh_c_dzc_finite_differences(model, constants, parameters, **kwargs):
    z_coord = kwargs["z_coord"]
    delta_z = 1e-6
    batch_size = len(parameters.seed)
    num_classical_coordinates = model.constants.num_classical_coordinates
    num_quantum_states = model.constants.num_quantum_states
    offset_z_coord_re = (
        z_coord[:, np.newaxis, :]
        + np.identity(num_classical_coordinates)[np.newaxis, :, :]
        * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    offset_z_coord_im = (
        z_coord[:, np.newaxis, :]
        + 1.0j
        * np.identity(num_classical_coordinates)[np.newaxis, :, :]
        * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    h_c_0 = model.h_c(constants, parameters, z_coord=z_coord)
    h_c_offset_re = model.h_c(
        constants, parameters, z_coord=offset_z_coord_re
    ).reshape(
        batch_size,
        num_classical_coordinates
    )
    h_c_offset_im = model.h_c(
        constants, parameters, z_coord=offset_z_coord_im
    ).reshape(
        batch_size,
        num_classical_coordinates
    )
    diff_re = (h_c_offset_re - h_c_0[:, np.newaxis]) / delta_z
    diff_im = (h_c_offset_im - h_c_0[:, np.newaxis]) / delta_z
    dh_c_dzc = 0.5 * (diff_re + 1.0j * diff_im)
    return dh_c_dzc


def dh_qc_dzc_finite_differences(model, constants, parameters, **kwargs):
    z_coord = kwargs["z_coord"]
    delta_z = 1e-6
    batch_size = len(parameters.seed)
    num_classical_coordinates = model.constants.num_classical_coordinates
    num_quantum_states = model.constants.num_quantum_states
    offset_z_coord_re = (
        z_coord[:, np.newaxis, :]
        + np.identity(num_classical_coordinates)[np.newaxis, :, :]
        * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    offset_z_coord_im = (
        z_coord[:, np.newaxis, :]
        + 1.0j
        * np.identity(num_classical_coordinates)[np.newaxis, :, :]
        * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    h_qc_0 = model.h_qc(constants, parameters, z_coord=z_coord)
    h_qc_offset_re = model.h_qc(
        constants, parameters, z_coord=offset_z_coord_re
    ).reshape(
        batch_size,
        num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    h_qc_offset_im = model.h_qc(
        constants, parameters, z_coord=offset_z_coord_im
    ).reshape(
        batch_size,
        num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    diff_re = (h_qc_offset_re - h_qc_0[:, np.newaxis, :, :]) / delta_z
    diff_im = (h_qc_offset_im - h_qc_0[:, np.newaxis, :, :]) / delta_z
    dh_qc_dzc = 0.5 * (diff_re + 1.0j * diff_im)
    inds = np.where(dh_qc_dzc != 0)
    mels = dh_qc_dzc[inds]
    return inds, mels


def update_dh_c_dzc(sim, parameters, state, **kwargs):
    """
    Update the gradient of the classical Hamiltonian w.r.t the conjugate classical coordinate.
    """
    z_coord = kwargs["z_coord"]
    if hasattr(sim.model, "dh_c_dzc"):
        state.dh_c_dzc = sim.model.dh_c_dzc(
            sim.model.constants, parameters, z_coord=z_coord
        )
    else:
        state.dh_c_dzc = dh_c_dzc_finite_differences(
            sim.model, sim.model.constants, parameters, z_coord=z_coord
        )
    return parameters, state


def update_dh_qc_dzc(sim, parameters, state, **kwargs):
    """
    Update the gradient of the quantum-classical Hamiltonian w.r.t the conjugate classical coordinate.
    """
    z_coord = kwargs["z_coord"]
    if hasattr(sim.model, "dh_qc_dzc"):
        state.dh_qc_dzc = sim.model.dh_qc_dzc(
            sim.model.constants, parameters, z_coord=z_coord
        )
    else:
        state.dh_qc_dzc = dh_qc_dzc_finite_differences(
            sim.model, sim.model.constants, parameters, z_coord=z_coord
        )

    return parameters, state


def update_classical_forces(sim, parameters, state, **kwargs):
    """
    Update the classical forces.
    """
    z_coord = kwargs["z_coord"]
    parameters, state = update_dh_c_dzc(sim, parameters, state, z_coord=z_coord)
    state.classical_forces = state.dh_c_dzc
    return parameters, state


def update_quantum_classical_forces(sim, parameters, state, **kwargs):
    """
    Update the quantum-classical forces w.r.t the state defined by wf.
    """
    z_coord = kwargs["z_coord"]
    wf = kwargs["wf"]
    parameters, state = update_dh_qc_dzc(sim, parameters, state, z_coord=z_coord)
    inds, mels = state.dh_qc_dzc

    state.quantum_classical_forces = np.zeros(  # TODO this should not need len(z_coord)
        (sim.settings.batch_size, sim.model.constants.num_classical_coordinates),
        dtype=complex,
    )
    np.add.at(
        state.quantum_classical_forces,
        (inds[0], inds[1]),
        np.conj(wf)[inds[0], inds[2]] * mels * wf[inds[0], inds[3]],
    )

    # state.quantum_classical_forces = np.einsum(
    #    "tnj,tj->tn",
    #    np.einsum("tnji,ti->tnj", state.dh_qc_dzc, wf, optimize="greedy"),
    #    np.conj(wf),
    #    optimize="greedy",
    # )
    return parameters, state


def update_z_coord_rk4(sim, parameters, state, **kwargs):
    """
    Update the z-coordinates using the 4th-order Runge-Kutta method.
    If the gradient of the quantum-classical Hamiltonian depends on z then
    update_quantum_classical_forces_bool should be set to True.
    """
    dt = sim.settings.dt
    wf = kwargs["wf"]
    if hasattr(sim.model, "linear_h_qc"):
        update_quantum_classical_forces_bool = not (sim.model.linear_h_qc)
    else:
        update_quantum_classical_forces_bool = True
    z_coord_0 = kwargs["z_coord"]
    output_name = kwargs["output_name"]
    parameters, state = update_classical_forces(
        sim, parameters, state, z_coord=z_coord_0
    )
    parameters, state = update_quantum_classical_forces(
        sim, parameters, state, wf=wf, z_coord=z_coord_0
    )
    k1 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    parameters, state = update_classical_forces(
        sim, parameters, state, z_coord=z_coord_0 + 0.5 * dt * k1
    )
    if update_quantum_classical_forces_bool:
        parameters, state = update_quantum_classical_forces(
            sim, parameters, state, wf=wf, z_coord=z_coord_0 + 0.5 * dt * k1
        )
    k2 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    parameters, state = update_classical_forces(
        sim, parameters, state, z_coord=z_coord_0 + 0.5 * dt * k2
    )
    if update_quantum_classical_forces_bool:
        parameters, state = update_quantum_classical_forces(
            sim, parameters, state, wf=wf, z_coord=z_coord_0 + 0.5 * dt * k2
        )
    k3 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    parameters, state = update_classical_forces(
        sim, parameters, state, z_coord=z_coord_0 + dt * k3
    )
    if update_quantum_classical_forces_bool:
        parameters, state = update_quantum_classical_forces(
            sim, parameters, state, wf=wf, z_coord=z_coord_0 + dt * k3
        )
    k4 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    setattr(state, output_name, z_coord_0 + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4))
    return parameters, state


def update_z_coord_parameter(sim, parameters, state, **kwargs):
    """
    Put the current z-coordinate into the parameters object.
    """
    z_coord = kwargs.get("z_coord", state.z_coord)
    parameters.z_coord = z_coord
    return parameters, state


def update_h_quantum(sim, parameters, state, **kwargs):
    """
    Update the quantum + quantum-classical Hamiltonian.
    """
    z_coord = kwargs.get("z_coord", state.z_coord)
    h_q = sim.model.h_q(sim.model.constants, parameters)
    h_qc = sim.model.h_qc(sim.model.constants, parameters, z_coord=z_coord)
    state.h_quantum = h_q + h_qc
    return parameters, state


@njit
def mat_vec_branch(mat, vec):
    """
    Perform matrix-vector multiplication for each branch.
    """
    return np.sum(mat * vec[:, np.newaxis, :], axis=-1)


def update_wf_db_rk4(sim, parameters, state, **kwargs):
    """
    Update the wavefunction using the 4th-order Runge-Kutta method.
    """
    del kwargs
    dt = sim.settings.dt
    wf_db = state.wf_db
    h_quantum = state.h_quantum
    k1 = -1j * mat_vec_branch(h_quantum, wf_db)
    k2 = -1j * mat_vec_branch(h_quantum, (wf_db + 0.5 * dt * k1))
    k3 = -1j * mat_vec_branch(h_quantum, (wf_db + 0.5 * dt * k2))
    k4 = -1j * mat_vec_branch(h_quantum, (wf_db + dt * k3))
    state.wf_db = wf_db + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return parameters, state


def update_dm_db_mf(sim, parameters, state, **kwargs):
    """
    Update the density matrix in the mean-field approximation.
    """
    del sim, kwargs
    wf_db = state.wf_db
    state.dm_db = np.einsum("ti,tj->tij", wf_db, np.conj(wf_db), optimize="greedy")
    return parameters, state


def update_classical_energy(sim, parameters, state, **kwargs):
    """
    Update the classical energy.
    """
    z_coord = kwargs["z_coord"]
    state.classical_energy = sim.model.h_c(
        sim.model.constants, parameters, z_coord=z_coord
    )
    return parameters, state


def update_classical_energy_fssh(sim, parameters, state, **kwargs):
    """
    Update the classical energy as a sum of equally-weighted contributions from each branch.
    """
    z_coord = kwargs["z_coord"]
    num_branches = sim.algorithm.settings.num_branches
    batch_size = sim.settings.batch_size // num_branches
    num_states = sim.model.constants.num_quantum_states
    if sim.algorithm.settings.fssh_deterministic:
        state.classical_energy = 0
        branch_weights = np.sqrt(
            np.einsum(
                "tbbb->tb",
                state.dm_adb_0.reshape(
                    (batch_size, num_branches, num_states, num_states)
                ),
            )
        )
        for branch_ind in range(num_branches):
            z_coord_branch = (
                z_coord[state.branch_ind == branch_ind]
                * branch_weights[:, branch_ind][:, np.newaxis]
            )
            state.classical_energy = state.classical_energy + sim.model.h_c(
                sim.model.constants, parameters, z_coord=z_coord_branch
            )
    else:
        state.classical_energy = 0
        for branch_ind in range(num_branches):
            z_coord_branch = z_coord[state.branch_ind == branch_ind]
            state.classical_energy = state.classical_energy + sim.model.h_c(
                sim.model.constants, parameters, z_coord=z_coord_branch
            )
        state.classical_energy = state.classical_energy / num_branches
    return parameters, state


def update_quantum_energy(sim, parameters, state, **kwargs):
    """
    Update the quantum energy w.r.t the wavefunction specified by wf.
    """
    wf = kwargs["wf"]
    state.quantum_energy = np.einsum(
        "ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy"
    )
    return parameters, state


def update_quantum_energy_fssh(sim, parameters, state, **kwargs):
    """
    Update the quantum energy w.r.t the wavefunction specified by wf.
    """
    wf = kwargs["wf"]

    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.algorithm.settings.num_branches
        batch_size = sim.settings.batch_size // num_branches
        num_states = sim.model.constants.num_quantum_states
        wf = wf * np.sqrt(
            np.einsum(
                "tbbb->tb",
                state.dm_adb_0.reshape(
                    (batch_size, num_branches, num_states, num_states)
                ),
            ).flatten()[:, np.newaxis]
        )
        state.quantum_energy = np.einsum(
            "ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy"
        )
    else:
        state.quantum_energy = np.einsum(
            "ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy"
        )
        state.quantum_energy = (
            state.quantum_energy / sim.algorithm.settings.num_branches
        )
    return parameters, state


def broadcast_var_to_branch(sim, parameters, state, **kwargs):
    """
    Broadcasts a variable to an equivalent with a new internal "branch" index. Each branch will be identical.
    Also generates a new set of indices named "var_branch_ind" and "var_traj_ind" which can be used to index the new variable.
    The shape will be (batch_size * num_branches, *var.shape[1:])
    """
    name = kwargs["name"]
    val = kwargs["val"]
    out = (
        np.zeros(
            (
                sim.settings.batch_size,
                sim.algorithm.settings.num_branches,
                *np.shape(val)[1:],
            ),
            dtype=val.dtype,
        )
        + val[:, np.newaxis, ...]
    ).reshape(
        (
            sim.settings.batch_size * sim.algorithm.settings.num_branches,
            *np.shape(val)[1:],
        )
    )
    setattr(state, name, out)
    branch_ind = (
        (
            np.arange(sim.algorithm.settings.num_branches)[np.newaxis, :]
            + np.zeros((sim.settings.batch_size, sim.algorithm.settings.num_branches))
        )
        .astype(int)
        .reshape(sim.settings.batch_size * sim.algorithm.settings.num_branches)
    )
    setattr(state, name + "_branch_ind", branch_ind)
    traj_ind = (
        (
            np.arange(sim.settings.batch_size)[:, np.newaxis]
            + np.zeros((sim.settings.batch_size, sim.algorithm.settings.num_branches))
        )
        .astype(int)
        .reshape(sim.settings.batch_size * sim.algorithm.settings.num_branches)
    )
    setattr(state, name + "_traj_ind", traj_ind)
    return parameters, state


def diagonalize_matrix(sim, parameters, state, **kwargs):
    """
    Diagonalizes a given matrix and stores the eigenvalues and eigenvectors in the state object.
    """
    del sim
    matrix = kwargs["matrix"]
    eigvals_name = kwargs["eigvals_name"]
    eigvecs_name = kwargs["eigvecs_name"]
    eigvals, eigvecs = np.linalg.eigh(matrix + 0.0j)
    setattr(state, eigvals_name, eigvals)
    setattr(state, eigvecs_name, eigvecs)
    return parameters, state


def analytic_der_couple_phase(sim, parameters, state, eigvals, eigvecs):
    """
    Calculates the phase change needed to fix the gauge using analytic derivative couplings.
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
        if np.any(np.abs(ev_diff) < 1e-10):
            plus[np.where(np.abs(ev_diff) < 1e-10)] = 1
            warnings.warn("Degenerate eigenvalues detected.")
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
        inds, mels = state.dh_qc_dzc
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
            1.0j
            * np.sqrt(
                1
                / (
                    2
                    * sim.model.constants.classical_coordinate_weight
                    * sim.model.constants.classical_coordinate_mass
                )
            )[..., :]
            * (der_couple_z - der_couple_zc)
        )
        der_couple_q = np.sqrt(
            sim.model.constants.classical_coordinate_weight
            * sim.model.constants.classical_coordinate_mass
            / 2
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
        der_couple_q_angle[np.where(np.abs(der_couple_q_angle) < 1e-12)] = 0
        der_couple_p_angle[np.where(np.abs(der_couple_p_angle) < 1e-12)] = 0
        der_couple_q_phase[..., i + 1 :] = (
            np.exp(1.0j * der_couple_q_angle[..., np.newaxis])
            * der_couple_q_phase[..., i + 1 :]
        )
        der_couple_p_phase[..., i + 1 :] = (
            np.exp(1.0j * der_couple_p_angle[..., np.newaxis])
            * der_couple_p_phase[..., i + 1 :]
        )
    return der_couple_q_phase, der_couple_p_phase


def gauge_fix_eigs(sim, parameters, state, **kwargs):
    """
    Fixes the gauge of the eigenvectors as specified by the gauge_fixing parameter.

    if gauge_fixing >= 0:
        Only the sign of the eigenvector is changed

    if gauge_fixing >= 1:
        The phase of the eigenvector is determined from its overlap with the previous eigenvector and the phase is fixed.

    if gauge_fixing >= 2:
        The phase of the eigenvector is determined by calculating the derivative couplings.

    """
    eigvals = kwargs["eigvals"]
    eigvecs = kwargs["eigvecs"]
    eigvecs_previous = kwargs["eigvecs_previous"]
    output_eigvecs_name = kwargs["output_eigvecs_name"]
    if kwargs["gauge_fixing"] >= 1:
        phase = np.exp(
            -1.0j * np.angle(np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2))
        )
        eigvecs = np.einsum("tai,ti->tai", eigvecs, phase, optimize="greedy")
    if kwargs["gauge_fixing"] >= 2:
        z_coord = kwargs["z_coord"]
        parameters, state = update_dh_qc_dzc(sim, parameters, state, z_coord=z_coord)
        der_couple_q_phase, _ = analytic_der_couple_phase(
            sim, parameters, state, eigvals, eigvecs
        )
        eigvecs = np.einsum(
            "tai,ti->tai", eigvecs, np.conj(der_couple_q_phase), optimize="greedy"
        )
    if kwargs["gauge_fixing"] >= 0:
        signs = np.sign(np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2))
        eigvecs = np.einsum("tai,ti->tai", eigvecs, signs, optimize="greedy")
    if kwargs["gauge_fixing"] == 2:
        der_couple_q_phase_new, der_couple_p_phase_new = analytic_der_couple_phase(
            sim, parameters, state, eigvals, eigvecs
        )
        if (
            np.sum(
                np.abs(np.imag(der_couple_q_phase_new)) ** 2
                + np.abs(np.imag(der_couple_p_phase_new)) ** 2
            )
            > 1e-10
        ):
            # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
            # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
            warnings.warn(
                "Phase error encountered when fixing gauge analytically.", UserWarning
            )
    setattr(state, output_eigvecs_name, eigvecs)
    return parameters, state


def assign_to_state(sim, parameters, state, **kwargs):
    """
    Creates a new state variable with the name "name" and the value "val".
    """
    del sim
    name = kwargs["name"]
    val = kwargs["val"]
    setattr(state, name, np.copy(val))
    return parameters, state


def basis_transform_vec(sim, parameters, state, **kwargs):
    """
    Transforms a vector "input_vec" to a new basis defined by "basis".
    """
    del sim
    # default is adb to db
    input_vec = kwargs["input_vec"]
    basis = kwargs["basis"]
    output_name = kwargs["output_name"]
    setattr(
        state,
        output_name,
        np.einsum("tij,tj->ti", basis, input_vec, optimize="greedy"),
    )
    return parameters, state


def basis_transform_mat(sim, parameters, state, **kwargs):
    """
    Transforms a matrix "input_mat" to a new basis defined by "basis" and stores it in the state object
    with name "output_name".
    """
    del sim
    # default is adb to db
    input_mat = kwargs["input_mat"]
    basis = kwargs["basis"]
    output_name = kwargs["output_name"]
    setattr(
        state,
        output_name,
        np.einsum(
            "tij,tjl->til",
            basis,
            np.einsum("tjk,tlk->tjl", input_mat, np.conj(basis), optimize="greedy"),
            optimize="greedy",
        ),
    )
    return parameters, state


def initialize_active_surface(sim, parameters, state, **kwargs):
    """
    Initializes the active surface (act_surf), active surface index (act_surf_ind) and initial active surface index (act_surf_ind_0)
    for FSSH.

    If fssh_deterministic is true it will set act_surf_ind_0 to be the same as the branch index and assert that the number of branches (num_branches)
    is equal to the number of quantum states (num_states).

    If fssh_deterministic is fasle it will stochastically sample the active surface from the density specified by the initial quantum wavefunction in the
    adiabatic basis. This implementation is capable of stochastically sampling an arbitrary number of branches.
    """
    del kwargs
    num_states = sim.model.constants.num_quantum_states
    num_branches = sim.algorithm.settings.num_branches
    num_trajs = sim.settings.batch_size // num_branches
    if sim.algorithm.settings.fssh_deterministic:
        if num_branches != num_states:
            raise ValueError(
                "num_branches must be equal to the quantum dimension for deterministic FSSH."
            )
        act_surf_ind_0 = np.arange(sim.algorithm.settings.num_branches, dtype=int)[
            np.newaxis, :
        ] + np.zeros((num_trajs, num_branches)).astype(int)
    else:
        intervals = np.cumsum(
            np.real(
                np.abs(state.wf_adb.reshape((num_trajs, num_branches, num_states))) ** 2
            ),
            axis=-1,
        )
        bool_mat = intervals > state.stochastic_sh_rand_vals[:, :, np.newaxis]
        act_surf_ind_0 = np.argmax(bool_mat, axis=-1).astype(int)
    state.act_surf_ind_0 = np.copy(act_surf_ind_0)
    state.act_surf_ind = np.copy(act_surf_ind_0)
    act_surf = np.zeros((num_trajs, num_branches, num_states), dtype=int)
    traj_inds = (
        (np.arange(num_trajs)[:, np.newaxis] * np.ones((num_trajs, num_branches)))
        .flatten()
        .astype(int)
    )
    branch_inds = (
        (np.arange(num_branches)[np.newaxis, :] * np.ones((num_trajs, num_branches)))
        .flatten()
        .astype(int)
    )
    act_surf[traj_inds, branch_inds, act_surf_ind_0.flatten()] = 1
    state.act_surf = act_surf.astype(int).reshape(
        (num_trajs * num_branches, num_states)
    )
    return parameters, state


def initialize_random_values_fssh(sim, parameters, state, **kwargs):
    """
    Initialize a set of random variables using the trajectory seeds for FSSH.
    """
    del kwargs
    num_branches = sim.algorithm.settings.num_branches
    batch_size = sim.settings.batch_size // num_branches
    state.hopping_probs_rand_vals = np.zeros((batch_size, len(sim.settings.tdat)))
    state.stochastic_sh_rand_vals = np.zeros((batch_size, num_branches))
    # this for loop is important so each seed is used
    for nt in range(batch_size):
        np.random.seed(state.seed[nt])
        state.hopping_probs_rand_vals[nt] = np.random.rand(len(sim.settings.tdat))
        state.stochastic_sh_rand_vals[nt] = np.random.rand(num_branches)
    return parameters, state


def initialize_dm_adb_0_fssh(sim, parameters, state, **kwargs):
    """
    Initialize the initial adiabatic density matrix for FSSH.
    """
    del sim, kwargs
    state.dm_adb_0 = (
        np.einsum(
            "...i,...j->...ij",
            state.wf_adb,
            np.conj(state.wf_adb),
            optimize="greedy",
        )
        + 0.0j
    )
    return parameters, state


def update_act_surf_wf(sim, parameters, state, **kwargs):
    """
    Update the wavefunction corresponding to the active surface.
    """
    del kwargs
    num_trajs = sim.settings.batch_size
    act_surf_wf = state.eigvecs[
        np.arange(num_trajs, dtype=int),
        :,
        state.act_surf_ind.flatten().astype(int),
    ]
    state.act_surf_wf = act_surf_wf
    return parameters, state


def update_dm_db_fssh(sim, parameters, state, **kwargs):
    """
    Update the diabatic density matrix for FSSH.
    """
    del kwargs
    dm_adb_branch = np.einsum(
        "...i,...j->...ij",
        state.wf_adb,
        np.conj(state.wf_adb),
        optimize="greedy",
    )
    num_branches = sim.algorithm.settings.num_branches
    batch_size = sim.settings.batch_size // num_branches
    num_quantum_states = sim.model.constants.num_quantum_states
    for nt in range(len(dm_adb_branch)):
        np.einsum("...jj->...j", dm_adb_branch[nt])[...] = state.act_surf[nt]
    if sim.algorithm.settings.fssh_deterministic:
        dm_adb_branch = (
            np.einsum(
                "tbbb->tb",
                state.dm_adb_0.reshape(
                    (batch_size, num_branches, num_quantum_states, num_quantum_states)
                ),
            ).flatten()[:, np.newaxis, np.newaxis]
            * dm_adb_branch
        )
    else:
        dm_adb_branch = dm_adb_branch / num_branches
    state.dm_adb = np.sum(dm_adb_branch, axis=-3) + 0.0j  # this is wrong....
    parameters, state = basis_transform_mat(
        sim,
        parameters,
        state,
        input_mat=dm_adb_branch.reshape(
            (
                batch_size * num_branches,
                num_quantum_states,
                num_quantum_states,
            )
        ),
        basis=state.eigvecs,
        output_name="dm_db_branch",
    )
    state.dm_db = (
        np.sum(
            state.dm_db_branch.reshape(
                (
                    batch_size,
                    num_branches,
                    num_quantum_states,
                    num_quantum_states,
                )
            ),
            axis=-3,
        )
        + 0.0j
    )
    return parameters, state


def update_wf_db_eigs(sim, parameters, state, **kwargs):
    """
    Evolve the diabatic wavefunction using the electronic eigenbasis.
    """
    wf_db = kwargs["wf_db"]
    adb_name = kwargs["adb_name"]
    output_name = kwargs["output_name"]
    eigvals = kwargs["eigvals"]
    eigvecs = kwargs["eigvecs"]
    evals_exp = np.exp(-1.0j * eigvals * sim.settings.dt)
    parameters, state = basis_transform_vec(
        sim=sim,
        parameters=parameters,
        state=state,
        input_vec=wf_db,
        basis=np.einsum("...ij->...ji", eigvecs).conj(),
        output_name=adb_name,
    )
    setattr(state, adb_name, (state.wf_adb * evals_exp))
    parameters, state = basis_transform_vec(
        sim=sim,
        parameters=parameters,
        state=state,
        input_vec=state.wf_adb,
        basis=eigvecs,
        output_name=output_name,
    )
    return parameters, state


@njit
def nan_num(num):
    """
    converts nan to a large or small number using numba acceleration
    """
    if np.isnan(num):
        return 0.0
    if num == np.inf:
        return 100e100
    if num == -np.inf:
        return -100e100
    else:
        return num


def update_active_surface_fssh(sim, parameters, state, **kwargs):
    """
    Update the active surface in FSSH. If a hopping function is not specified in the model
    class a numerical hopping procedure is used instead.
    """
    del kwargs
    rand = state.hopping_probs_rand_vals[:, sim.t_ind]
    act_surf_ind = state.act_surf_ind
    act_surf_ind_flat = act_surf_ind.flatten().astype(int)
    num_branches = sim.algorithm.settings.num_branches
    num_trajs = sim.settings.batch_size // num_branches
    traj_ind = (
        (np.arange(num_trajs)[:, np.newaxis] * np.ones((num_trajs, num_branches)))
        .flatten()
        .astype(int)
    )

    prod = np.einsum(
        "bn,bni->bi",
        np.conj(
            state.eigvecs[
                np.arange(num_trajs * num_branches, dtype=int), :, act_surf_ind_flat
            ]
        ),
        state.eigvecs_previous,
        optimize="greedy",
    )
    hop_prob = -2 * np.real(
        prod
        * state.wf_adb
        / state.wf_adb[np.arange(num_trajs * num_branches), act_surf_ind_flat][
            :, np.newaxis
        ]
    )
    hop_prob[np.arange(num_branches * num_trajs), act_surf_ind_flat] *= 0
    cumulative_probs = np.cumsum(
        np.nan_to_num(hop_prob, nan=0, posinf=100e100, neginf=-100e100), axis=1
    )
    rand_branch = (rand[:, np.newaxis] * np.ones((num_trajs, num_branches))).flatten()
    traj_hop_ind = np.where(
        np.sum((cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1) > 0
    )[0]
    if len(traj_hop_ind) > 0:
        inds, mels = state.dh_qc_dzc
        eigvecs_flat = state.eigvecs
        eigvals_flat = state.eigvals
        z_coord = state.z_coord
        act_surf_flat = state.act_surf
        for traj_ind in traj_hop_ind:
            k = np.argmax(
                (cumulative_probs[traj_ind] > rand_branch[traj_ind]).astype(int)
            )
            j = act_surf_ind_flat[traj_ind]
            evec_k = eigvecs_flat[traj_ind][:, j]
            evec_j = eigvecs_flat[traj_ind][:, k]
            eval_k = eigvals_flat[traj_ind][j]
            eval_j = eigvals_flat[traj_ind][k]
            ev_diff = eval_j - eval_k
            inds_traj_ind = (
                inds[0][inds[0] == traj_ind],
                inds[1][inds[0] == traj_ind],
                inds[2][inds[0] == traj_ind],
                inds[3][inds[0] == traj_ind],
            )
            mels_traj_ind = mels[inds[0] == traj_ind]
            dkj_z = np.zeros(
                (sim.model.constants.num_classical_coordinates), dtype=complex
            )
            dkj_zc = np.zeros(
                (sim.model.constants.num_classical_coordinates), dtype=complex
            )
            np.add.at(
                dkj_z,
                (inds_traj_ind[1]),
                np.conj(evec_k)[inds_traj_ind[2]]
                * mels_traj_ind
                * evec_j[inds_traj_ind[3]]
                / ev_diff,
            )
            np.add.at(
                dkj_zc,
                (inds_traj_ind[1]),
                np.conj(evec_k)[inds_traj_ind[3]]
                * np.conj(mels_traj_ind)
                * evec_j[inds_traj_ind[2]]
                / ev_diff,
            )
            dkj_p = (
                1.0j
                * np.sqrt(
                    1
                    / (
                        2
                        * sim.model.constants.classical_coordinate_weight
                        * sim.model.constants.classical_coordinate_mass
                    )
                )
                * (dkj_z - dkj_zc)
            )
            dkj_q = np.sqrt(
                sim.model.constants.classical_coordinate_weight
                * sim.model.constants.classical_coordinate_mass
                / 2
            ) * (dkj_z + dkj_zc)

            max_pos_q = np.argmax(np.abs(dkj_q))
            max_pos_p = np.argmax(np.abs(dkj_p))
            # Check for complex nonadiabatic couplings
            if (
                np.abs(dkj_q[max_pos_q]) > 1e-8
                and np.abs(np.sin(np.angle(dkj_q[max_pos_q]))) > 1e-2
            ):
                warnings.warn(
                    "dkj_q Nonadiabatic coupling is complex, needs gauge fixing!",
                    UserWarning,
                )
            if (
                np.abs(dkj_p[max_pos_p]) > 1e-8
                and np.abs(np.sin(np.angle(dkj_p[max_pos_p]))) > 1e-2
            ):
                warnings.warn(
                    "dkj_p Nonadiabatic coupling is complex, needs gauge fixing!",
                    UserWarning,
                )
            delta_z = dkj_zc

            # Perform hopping using the model's hop function or the default harmonic oscillator hop function
            if hasattr(sim.model, "hop_function"):
                z_coord_branch_out, hopped = sim.model.hop_function(
                    sim.model.constants,
                    parameters,
                    z_coord=z_coord[traj_ind],
                    delta_z_coord=delta_z,
                    ev_diff=ev_diff,
                )
            else:
                z_coord_branch_out, hopped = ingredients.numerical_fssh_hop(
                    sim.model,
                    sim.model.constants,
                    parameters,
                    z_coord=z_coord[traj_ind],
                    delta_z_coord=delta_z,
                    ev_diff=ev_diff,
                )
            if hopped:
                act_surf_ind_flat[traj_ind] = k
                act_surf_flat[traj_ind] = np.zeros_like(act_surf_flat[traj_ind])
                act_surf_flat[traj_ind][k] = 1
                z_coord[traj_ind] = z_coord_branch_out
                state.act_surf_ind = np.copy(
                    act_surf_ind_flat.reshape((num_trajs, num_branches))
                )
                state.act_surf = np.copy(act_surf_flat)
                state.z_coord_branch = np.copy(z_coord)

    return parameters, state
