"""Tasks that update the simulation state during propagation."""

import numpy as np
import warnings
from qc_lab.jit import njit
from qc_lab.tasks.default_ingredients import *


def update_t(algorithm, sim, parameters, state):
    """
    Update the time in the state object with the time index in each trajectory
    multiplied by the update timestep.

    Required constants:
        - None.
    """
    batch_size = len(parameters.seed)
    # the variable should store the time in each trajectory and should therefore be 
    # and array with length batch_size.
    state.t = np.ones(batch_size) * sim.t_ind * sim.settings.dt_update
    return parameters, state


def update_dh_c_dzc(algorithm, sim, parameters, state, **kwargs):
    """
    Update the gradient of the classical Hamiltonian
    w.r.t the conjugate classical coordinate.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    dh_c_dzc, has_dh_c_dzc = sim.model.get("dh_c_dzc")
    if has_dh_c_dzc:
        state.dh_c_dzc = dh_c_dzc(sim.model, parameters, z=z)
        return parameters, state
    state.dh_c_dzc = dh_c_dzc_finite_differences(sim.model, parameters, z=z)
    return parameters, state


def update_dh_qc_dzc(algorithm, sim, parameters, state, **kwargs):
    """
    Update the gradient of the quantum-classical Hamiltonian
    w.r.t the conjugate classical coordinate.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    dh_qc_dzc, has_dh_qc_dzc = sim.model.get("dh_qc_dzc")
    if has_dh_qc_dzc:
        state.dh_qc_dzc = dh_qc_dzc(sim.model, parameters, z=z)
        return parameters, state
    state.dh_qc_dzc = dh_qc_dzc_finite_differences(sim.model, parameters, z=z)
    return parameters, state


def update_classical_forces(algorithm, sim, parameters, state, **kwargs):
    """
    Update the classical forces.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    parameters, state = update_dh_c_dzc(algorithm, sim, parameters, state, z=z)
    state.classical_forces = state.dh_c_dzc
    return parameters, state


@njit
def calc_sparse_inner_product(inds, mels, shape, vec_l, vec_r):
    """
    Given the indices, matrix elements and shape of a sparse matrix, calculate the expectation value with a vector.

    Required constants:
        - None.
    """
    out = np.zeros((shape[:2])) + 0.0j
    for i in range(len(inds[0])):
        out[inds[0][i], inds[1][i]] = (
            out[inds[0][i], inds[1][i]]
            + np.conj(vec_l[inds[0][i], inds[2][i]])
            * mels[i]
            * vec_r[inds[0][i], inds[3][i]]
        )
    return out


def update_quantum_classical_forces(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum-classical forces w.r.t the state defined by wf.

    If the model has a gauge_field_force ingredient, this term will be added
    to the quantum-classical forces.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    wf = kwargs["wf"]
    use_gauge_field_force = kwargs.get("use_gauge_field_force", False)
    parameters, state = update_dh_qc_dzc(algorithm, sim, parameters, state, z=z)
    inds, mels, shape = state.dh_qc_dzc
    state.quantum_classical_forces = calc_sparse_inner_product(
        inds, mels, shape, wf, wf
    )
    gauge_field_force, has_gauge_field_force = sim.model.get("gauge_field_force")
    if has_gauge_field_force and use_gauge_field_force:
        state.quantum_classical_forces += gauge_field_force(parameters, z=z, wf=wf)
    return parameters, state


def diagonalize_matrix(algorithm, sim, parameters, state, **kwargs):
    """
    Diagonalizes a given matrix and stores the eigenvalues and eigenvectors in the state object.

    Required constants:
        - None.
    """
    matrix = kwargs["matrix"]
    eigvals_name = kwargs["eigvals_name"]
    eigvecs_name = kwargs["eigvecs_name"]
    eigvals, eigvecs = np.linalg.eigh(matrix)
    setattr(state, eigvals_name, eigvals)
    setattr(state, eigvecs_name, eigvecs)
    return parameters, state


def analytic_der_couple_phase(algorithm, sim, parameters, state, eigvals, eigvecs):
    """
    Calculates the phase change needed to fix the gauge using analytic derivative couplings.

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


def gauge_fix_eigs(algorithm, sim, parameters, state, **kwargs):
    """
    Fixes the gauge of the eigenvectors as specified by the gauge_fixing parameter.

    if gauge_fixing == "sign_overlap":
        Only the sign of the eigenvector is changed so its overlap
        with the previous eigenvector is positive.

    if gauge_fixing == "phase_overlap":
        The phase of the eigenvector is determined from its overlap
        with the previous eigenvector and used to maximize the overlap.

    if gauge_fixing == "phase_der_couple":
        The phase of the eigenvector is determined by calculating the derivative couplings
        and changed so that all the derivative couplings are real-valued.

    Required constants:
        - None.
    """
    eigvals = kwargs["eigvals"]
    eigvecs = kwargs["eigvecs"]
    eigvecs_previous = kwargs["eigvecs_previous"]
    output_eigvecs_name = kwargs["output_eigvecs_name"]
    gauge_fixing = kwargs["gauge_fixing"]
    gauge_fixing_numerical_values = {
        "sign_overlap": 0,
        "phase_overlap": 1,
        "phase_der_couple": 2,
    }
    gauge_fixing_value = gauge_fixing_numerical_values[gauge_fixing]
    if gauge_fixing_value >= 1:
        phase = np.exp(
            -1.0j * np.angle(np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2))
        )
        eigvecs = np.einsum("tai,ti->tai", eigvecs, phase, optimize="greedy")
    if gauge_fixing_value >= 2:
        z = kwargs["z"]
        parameters, state = update_dh_qc_dzc(algorithm, sim, parameters, state, z=z)
        der_couple_q_phase, _ = analytic_der_couple_phase(
            algorithm, sim, parameters, state, eigvals, eigvecs
        )
        eigvecs = np.einsum(
            "tai,ti->tai", eigvecs, np.conj(der_couple_q_phase), optimize="greedy"
        )
    if gauge_fixing_value >= 0:
        signs = np.sign(np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2))
        eigvecs = np.einsum("tai,ti->tai", eigvecs, signs, optimize="greedy")
    if gauge_fixing_value == 2:
        der_couple_q_phase_new, der_couple_p_phase_new = analytic_der_couple_phase(
            algorithm, sim, parameters, state, eigvals, eigvecs
        )
        if (
            np.sum(
                np.abs(np.imag(der_couple_q_phase_new)) ** 2
                + np.abs(np.imag(der_couple_p_phase_new)) ** 2
            )
            > 1e-10
        ):
            warnings.warn(
                "Phase error encountered when fixing gauge analytically.", UserWarning
            )
    setattr(state, output_eigvecs_name, eigvecs)
    return parameters, state


def basis_transform_vec(algorithm, sim, parameters, state, **kwargs):
    """
    Transforms a vector "input_vec" to a new basis defined by "basis".

    Required constants:
        - None.
    """
    # Default transformation is adiabatic to diabatic.
    input_vec = kwargs["input_vec"]
    basis = kwargs["basis"]
    output_name = kwargs["output_name"]
    setattr(
        state,
        output_name,
        np.einsum("tij,tj->ti", basis, input_vec, optimize="greedy"),
    )
    return parameters, state


def basis_transform_mat(algorithm, sim, parameters, state, **kwargs):
    """
    Transforms a matrix "input_mat" to a new basis
    defined by "basis" and stores it in the state object
    with name "output_name".

    Required constants:
        - None.
    """
    # Default transformation is adiabatic to diabatic.
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


def update_act_surf_wf(algorithm, sim, parameters, state, **kwargs):
    """
    Update the wavefunction corresponding to the active surface.

    Required constants:
        - None.
    """
    num_trajs = sim.settings.batch_size
    act_surf_wf = state.eigvecs[
        np.arange(num_trajs, dtype=int),
        :,
        state.act_surf_ind.flatten().astype(int),
    ]
    state.act_surf_wf = act_surf_wf
    return parameters, state


def update_wf_db_eigs(algorithm, sim, parameters, state, **kwargs):
    """
    Evolve the diabatic wavefunction using the electronic eigenbasis.

    Required constants:
        - None.
    """
    wf_db = kwargs["wf_db"]
    adb_name = kwargs["adb_name"]
    output_name = kwargs["output_name"]
    eigvals = kwargs["eigvals"]
    eigvecs = kwargs["eigvecs"]
    evals_exp = np.exp(-1.0j * eigvals * sim.settings.dt_update)
    parameters, state = basis_transform_vec(
        algorithm,
        sim=sim,
        parameters=parameters,
        state=state,
        input_vec=wf_db,
        basis=np.einsum("...ij->...ji", eigvecs).conj(),
        output_name=adb_name,
    )
    setattr(state, adb_name, (state.wf_adb * evals_exp))
    parameters, state = basis_transform_vec(
        algorithm,
        sim=sim,
        parameters=parameters,
        state=state,
        input_vec=state.wf_adb,
        basis=eigvecs,
        output_name=output_name,
    )
    return parameters, state


@njit
def matprod(mat, vec):
    """
    Perform matrix-vector multiplication.

    Required constants:
        - None.
    """
    out = np.zeros(np.shape(vec)) + 0.0j
    for t in range(len(mat)):
        for i in range(len(mat[0])):
            accum = 0 + 0.0j
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
    return wf_db + dt_update * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)


def update_wf_db_rk4(algorithm, sim, parameters, state, **kwargs):
    """
    Update the wavefunction using the 4th-order Runge-Kutta method.

    Required constants:
        - None.
    """
    dt_update = sim.settings.dt_update
    wf_db = state.wf_db
    h_quantum = state.h_quantum
    state.wf_db = wf_db_rk4(h_quantum, wf_db, dt_update)
    return parameters, state


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
    # Check for complex nonadiabatic couplings.
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
    return delta_z


def update_hop_probs_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    This task calculates the hopping probabilities for FSSH.

    P_{a->b} = -2 * Re((C_{b}/C_{a}) * < a(t)| b(t-dt)>)

    Stores the probabilities in state.hop_prob.
    """
    act_surf_ind = state.act_surf_ind
    act_surf_ind_flat = act_surf_ind.flatten().astype(int)
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
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
    state.hop_prob = hop_prob

    return parameters, state


def update_hop_inds_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Determines which trajectories hop based on the hopping probabilities and which state they hop to.
    Note that these will only hop if they are not frustrated by the hopping function.

    Stores the indices of the hopping trajectories in state.hop_ind.
    Stores the destination indices of the hops in state.hop_dest.
    """
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    hop_prob = state.hop_prob
    rand = state.hopping_probs_rand_vals[:, sim.t_ind]
    cumulative_probs = np.cumsum(
        np.nan_to_num(hop_prob, nan=0, posinf=100e100, neginf=-100e100), axis=1
    )
    rand_branch = (rand[:, np.newaxis] * np.ones((num_trajs, num_branches))).flatten()
    hop_ind = np.where(
        np.sum((cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1) > 0
    )[
        0
    ]  # trajectory indices that hop
    # destination indices of the hops in each hoping trajectory
    hop_dest = np.argmax(
        (cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1
    )[hop_ind]
    state.hop_ind = hop_ind
    state.hop_dest = hop_dest
    return parameters, state


def update_hop_vals_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Executes the hopping function for the hopping trajectories.
    It stores the rescaled coordinates in state.z_rescaled
    and the a boolean registering if the hop was successful in state.hop_successful.
    """

    hop_ind = state.hop_ind
    hop_dest = state.hop_dest
    state.z_shift = np.zeros(
        (len(hop_ind), sim.model.constants.num_classical_coordinates), dtype=complex
    )
    state.hop_successful = np.zeros(len(hop_ind), dtype=bool)
    eigvals_flat = state.eigvals
    z = np.copy(state.z)
    act_surf_ind = state.act_surf_ind
    act_surf_ind_flat = act_surf_ind.flatten().astype(int)
    hop_traj_ind = 0
    for traj_ind in hop_ind:
        final_state_ind = hop_dest[hop_traj_ind]
        init_state_ind = act_surf_ind_flat[traj_ind]
        eval_init_state = eigvals_flat[traj_ind][init_state_ind]
        eval_final_state = eigvals_flat[traj_ind][final_state_ind]
        ev_diff = eval_final_state - eval_init_state
        delta_z = calc_delta_z_fssh(
            algorithm,
            sim,
            parameters,
            state,
            traj_ind=traj_ind,
            final_state_ind=final_state_ind,
            init_state_ind=init_state_ind,
        )
        hop_function, has_hop_function = sim.model.get("hop_function")
        if has_hop_function:
            z_shift, hopped = hop_function(
                sim.model,
                parameters,
                z=z[traj_ind],
                delta_z=delta_z,
                ev_diff=ev_diff,
            )
        else:
            z_shift, hopped = numerical_fssh_hop(
                sim.model,
                parameters,
                z=z[traj_ind],
                delta_z=delta_z,
                ev_diff=ev_diff,
            )
        state.hop_successful[hop_traj_ind] = hopped
        state.z_shift[hop_traj_ind] = z_shift
        hop_traj_ind += 1
    return parameters, state


def update_z_hop_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Executes the post-hop updates for FSSH, rescaling the z coordinates and updating
    the active surface indices, and wavefunctions.
    """
    # idx = state.hop_ind[state.hop_successful]
    # dz = state.z_shift[state.hop_successful]
    state.z[state.hop_ind] += state.z_shift  # [idx] += dz
    return parameters, state


def update_act_surf_hop_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the active surface, active surface index, and active surface wavefunction
    following a hop in FSSH.

    """
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    act_surf_flat = state.act_surf
    act_surf_ind = state.act_surf_ind
    act_surf_ind_flat = act_surf_ind.flatten().astype(int)

    idx = state.hop_ind[state.hop_successful]
    act_surf_ind_flat[idx] = state.hop_dest[state.hop_successful]
    act_surf_flat[idx] = np.zeros_like(act_surf_flat[idx])
    act_surf_flat[idx, state.hop_dest[state.hop_successful]] = 1
    state.act_surf_ind = np.copy(act_surf_ind_flat.reshape((num_trajs, num_branches)))
    state.act_surf = np.copy(act_surf_flat)
    parameters.act_surf_ind = state.act_surf_ind
    return parameters, state


def update_h_quantum(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum + quantum-classical Hamiltonian.

    Required constants:
        - None.
    """
    z = kwargs.get("z", state.z)
    h_q, _ = sim.model.get("h_q")
    h_qc, _ = sim.model.get("h_qc")
    if sim.model.update_h_q or state.h_q is None:
        # Update the quantum Hamiltonian if required or if it is not set.
        state.h_q = h_q(sim.model, parameters)
    # Update the quantum-classical Hamiltonian.
    state.h_qc = h_qc(sim.model, parameters, z=z)
    state.h_quantum = state.h_q + state.h_qc
    return parameters, state


def update_z_rk4(algorithm, sim, parameters, state, **kwargs):
    """
    Update the z-coordinates using the 4th-order Runge-Kutta method.
    If the gradient of the quantum-classical Hamiltonian depends on z then
    update_quantum_classical_forces_bool should be set to True.

    Required constants:
        - None.
    """
    dt_update = sim.settings.dt_update
    wf = kwargs["wf"]
    use_gauge_field_force = kwargs.get("use_gauge_field_force", False)
    if hasattr(sim.model, "update_dh_qc_dzc"):
        update_quantum_classical_forces_bool = sim.model.update_dh_qc_dzc
    else:
        update_quantum_classical_forces_bool = True
    z_0 = kwargs["z"]
    output_name = kwargs["output_name"]
    parameters, state = update_classical_forces(
        algorithm, sim, parameters, state, z=z_0
    )
    parameters, state = update_quantum_classical_forces(
        algorithm,
        sim,
        parameters,
        state,
        wf=wf,
        z=z_0,
        use_gauge_field_force=use_gauge_field_force,
    )
    k1 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    parameters, state = update_classical_forces(
        algorithm, sim, parameters, state, z=z_0 + 0.5 * dt_update * k1
    )
    if update_quantum_classical_forces_bool:
        parameters, state = update_quantum_classical_forces(
            algorithm,
            sim,
            parameters,
            state,
            wf=wf,
            z=z_0 + 0.5 * dt_update * k1,
            use_gauge_field_force=use_gauge_field_force,
        )
    k2 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    parameters, state = update_classical_forces(
        algorithm, sim, parameters, state, z=z_0 + 0.5 * dt_update * k2
    )
    if update_quantum_classical_forces_bool:
        parameters, state = update_quantum_classical_forces(
            algorithm,
            sim,
            parameters,
            state,
            wf=wf,
            z=z_0 + 0.5 * dt_update * k2,
            use_gauge_field_force=use_gauge_field_force,
        )
    k3 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    parameters, state = update_classical_forces(
        algorithm, sim, parameters, state, z=z_0 + dt_update * k3
    )
    if update_quantum_classical_forces_bool:
        parameters, state = update_quantum_classical_forces(
            algorithm,
            sim,
            parameters,
            state,
            wf=wf,
            z=z_0 + dt_update * k3,
            use_gauge_field_force=use_gauge_field_force,
        )
    k4 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    setattr(
        state, output_name, z_0 + dt_update * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    )
    return parameters, state


def update_dm_db_mf(algorithm, sim, parameters, state, **kwargs):
    """
    Update the density matrix in the mean-field approximation.

    Required constants:
        - None.
    """
    wf_db = state.wf_db
    state.dm_db = np.einsum("ti,tj->tij", wf_db, np.conj(wf_db), optimize="greedy")
    return parameters, state


def update_classical_energy(algorithm, sim, parameters, state, **kwargs):
    """
    Update the classical energy.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    h_c, _ = sim.model.get("h_c")
    state.classical_energy = np.real(h_c(sim.model, parameters, z=z, batch_size=len(z)))
    return parameters, state


def update_classical_energy_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the classical energy as a sum of equally-weighted contributions from each branch.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_states = sim.model.constants.num_quantum_states
    h_c, _ = sim.model.get("h_c")
    if sim.algorithm.settings.fssh_deterministic:
        state.classical_energy = 0
        branch_weights = np.sqrt(
            num_branches * np.einsum(
                "tbbb->tb",
                state.dm_adb_0.reshape(
                    (batch_size, num_branches, num_states, num_states)
                ),
            )
        )
        for branch_ind in range(num_branches):
            z_branch = (
                z[state.branch_ind == branch_ind]
                * branch_weights[:, branch_ind][:, np.newaxis]
            )
            state.classical_energy = state.classical_energy + h_c(
                sim.model,
                parameters,
                z=z_branch,
                batch_size=len(z_branch),
            )
    else:
        state.classical_energy = 0
        for branch_ind in range(num_branches):
            z_branch = z[state.branch_ind == branch_ind]
            state.classical_energy = state.classical_energy + h_c(
                sim.model,
                parameters,
                z=z_branch,
                batch_size=len(z_branch),
            )
        state.classical_energy = state.classical_energy / num_branches
    state.classical_energy = np.real(state.classical_energy)
    return parameters, state


def update_quantum_energy(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum energy w.r.t the wavefunction specified by wf.

    Required constants:
        - None.
    """
    wf = kwargs["wf"]
    state.quantum_energy = np.real(
        np.einsum("ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy")
    )
    return parameters, state


def update_quantum_energy_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum energy w.r.t the wavefunction specified by wf.

    Required constants:
        - None.
    """
    wf = kwargs["wf"]

    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
        batch_size = sim.settings.batch_size // num_branches
        num_states = sim.model.constants.num_quantum_states
        wf = wf * np.sqrt(
            num_branches * np.einsum(
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
        state.quantum_energy = state.quantum_energy
    state.quantum_energy = np.real(state.quantum_energy)
    return parameters, state


def update_dm_db_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the diabatic density matrix for FSSH.

    Required constants:
        - None.
    """
    dm_adb_branch = np.einsum(
        "...i,...j->...ij",
        state.wf_adb,
        np.conj(state.wf_adb),
        optimize="greedy",
    )
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_quantum_states = sim.model.constants.num_quantum_states
    for nt, _ in enumerate(dm_adb_branch):
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
    parameters, state = basis_transform_mat(
        algorithm,
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
    state.dm_db = num_branches * np.sum(
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
    return parameters, state
