"""
Tasks that update the simulation state during propagation.
"""

import logging
import numpy as np
from qc_lab.functions import (
    calc_sparse_inner_product,
    analytic_der_couple_phase,
    numerical_fssh_hop,
    wf_db_rk4,
    calc_delta_z_fssh,
)
from qc_lab.constants import SMALL

logger = logging.getLogger(__name__)


def update_t(algorithm, sim, parameters, state):
    """
    Update the time in the state object with the time index in each trajectory
    multiplied by the update timestep.

    Required constants:
        - None.
    """
    batch_size = len(parameters.seed)
    # the variable should store the time in each trajectory and should
    # therefore be an array with length batch_size.
    state.t = np.ones(batch_size) * sim.t_ind * sim.settings.dt_update
    return parameters, state


def update_dh_c_dzc_finite_differences(algorithm, sim, parameters, state, **kwargs):
    """
    Calculate the gradient of the classical Hamiltonian using finite differences.

    Required constants:
        - num_classical_coordinates (int): Number of classical
          coordinates. Default: None.
    """
    z = kwargs["z"]
    name = kwargs.get("name", "dh_c_dzc")
    delta_z = sim.model.constants.get("dh_c_dzc_finite_difference_delta", 1e-6)
    batch_size = len(parameters.seed)
    num_classical_coordinates = sim.model.constants.num_classical_coordinates
    offset_z_re = (
        z[:, np.newaxis, :]
        + np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    offset_z_im = (
        z[:, np.newaxis, :]
        + 1j * np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    h_c, _ = sim.model.get("h_c")
    h_c_0 = h_c(sim.model, parameters, z=z, batch_size=len(z))
    h_c_offset_re = h_c(
        sim.model,
        parameters,
        z=offset_z_re,
        batch_size=batch_size * num_classical_coordinates,
    ).reshape(batch_size, num_classical_coordinates)
    h_c_offset_im = h_c(
        sim.model,
        parameters,
        z=offset_z_im,
        batch_size=batch_size * num_classical_coordinates,
    ).reshape(batch_size, num_classical_coordinates)
    diff_re = (h_c_offset_re - h_c_0[:, np.newaxis]) / delta_z
    diff_im = (h_c_offset_im - h_c_0[:, np.newaxis]) / delta_z
    dh_c_dzc = 0.5 * (diff_re + 1j * diff_im)
    setattr(state, name, dh_c_dzc)
    return parameters, state


def update_classical_forces(algorithm, sim, parameters, state, **kwargs):
    """
    Update the gradient of the classical Hamiltonian w.r.t. the conjugate classical
    coordinate.

    Required constants:
        - None.
    """
    z = getattr(state, kwargs["z"])
    dh_c_dzc, has_dh_c_dzc = sim.model.get("dh_c_dzc")
    if has_dh_c_dzc:
        state.classical_forces = dh_c_dzc(sim.model, parameters, z=z)
        return parameters, state
    return update_dh_c_dzc_finite_differences(
        algorithm, sim, parameters, state, name="classical_forces", z=z
    )


def update_dh_qc_dzc_finite_differences(algorithm, sim, parameters, state, **kwargs):
    """
    Calculate the gradient of the quantum-classical Hamiltonian using finite
    differences.

    Required constants:
        - num_classical_coordinates (int): Number of classical
          coordinates. Default: None.
        - num_quantum_states (int): Number of quantum states. Default: None.
        - finite_difference_dz (float): Step size for finite differences.
          Default: 1e-6.
    """
    z = kwargs["z"]
    batch_size = kwargs.get("batch_size", len(z))
    delta_z = sim.model.constants.get("dh_qc_dzc_finite_difference_delta", 1e-6)
    num_classical_coordinates = sim.model.constants.num_classical_coordinates
    num_quantum_states = sim.model.constants.num_quantum_states
    offset_z_re = (
        z[:, np.newaxis, :]
        + np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    offset_z_im = (
        z[:, np.newaxis, :]
        + 1j * np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    h_qc, _ = sim.model.get("h_qc")
    h_qc_0 = h_qc(sim.model, parameters, z=z)
    h_qc_offset_re = h_qc(
        sim.model,
        parameters,
        z=offset_z_re,
        batch_size=batch_size * num_classical_coordinates,
    ).reshape(
        batch_size,
        num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    h_qc_offset_im = h_qc(
        sim.model,
        parameters,
        z=offset_z_im,
        batch_size=batch_size * num_classical_coordinates,
    ).reshape(
        batch_size,
        num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    diff_re = (h_qc_offset_re - h_qc_0[:, np.newaxis, :, :]) / delta_z
    diff_im = (h_qc_offset_im - h_qc_0[:, np.newaxis, :, :]) / delta_z
    dh_qc_dzc = 0.5 * (diff_re + 1j * diff_im)
    inds = np.where(dh_qc_dzc != 0)
    mels = dh_qc_dzc[inds]
    shape = np.shape(dh_qc_dzc)
    state.dh_qc_dzc = (inds, mels, shape)
    return parameters, state


def update_dh_qc_dzc(algorithm, sim, parameters, state, **kwargs):
    """
    Update the gradient of the quantum-classical Hamiltonian w.r.t. the conjugate
    classical coordinate.

    Required constants:
        - None.
    """
    z = getattr(state, kwargs["z"])
    if state.dh_qc_dzc is None or sim.model.update_dh_qc_dzc:
        # If dh_qc_dzc has not been claculated yet, or if the
        # model requires it to be updated, calculate it.
        dh_qc_dzc, has_dh_qc_dzc = sim.model.get("dh_qc_dzc")
        if has_dh_qc_dzc:
            state.dh_qc_dzc = dh_qc_dzc(sim.model, parameters, z=z)
            return parameters, state
        return update_dh_qc_dzc_finite_differences(
            algorithm, sim, parameters, state, z=z
        )
    # If dh_qc_dzc has already been calculated and does not need to be updated,
    # return the existing parameters and state objects.
    return parameters, state


def update_quantum_classical_forces(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum-classical forces w.r.t. the state defined by wf.

    If the model has a gauge_field_force ingredient, this term will be added
    to the quantum-classical forces.

    Required constants:
        - None.
    """
    z = getattr(state, kwargs["z"])
    wf = getattr(state, kwargs["wf"])
    use_gauge_field_force = kwargs.get("use_gauge_field_force", False)
    parameters, state = update_dh_qc_dzc(
        algorithm, sim, parameters, state, z=kwargs["z"]
    )
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
    Diagonalizes a given matrix from the state object and stores the eigenvalues and
    eigenvectors in the state object.

    Required constants:
        - None.
    """
    matrix = getattr(state, kwargs["matrix"])
    eigvals, eigvecs = np.linalg.eigh(matrix)
    setattr(state, kwargs["eigvals"], eigvals)
    setattr(state, kwargs["eigvecs"], eigvecs)
    return parameters, state


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
        The phase of the eigenvector is determined by calculating the
        derivative couplings and changed so that all the derivative
        couplings are real-valued.

    Required constants:
        - None.
    """
    eigvals = getattr(state, kwargs["eigvals"])
    eigvecs = getattr(state, kwargs["eigvecs"])
    eigvecs_previous = getattr(state, kwargs["eigvecs_previous"])
    gauge_fixing = kwargs.get("gauge_fixing", sim.algorithm.settings.gauge_fixing)
    gauge_fixing_numerical_values = {
        "sign_overlap": 0,
        "phase_overlap": 1,
        "phase_der_couple": 2,
    }
    gauge_fixing_value = gauge_fixing_numerical_values[gauge_fixing]
    if gauge_fixing_value >= 1:
        overlap = np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)
        phase = np.exp(-1j * np.angle(overlap))
        eigvecs = np.einsum("tai,ti->tai", eigvecs, phase, optimize="greedy")
    if gauge_fixing_value >= 2:
        parameters, state = update_dh_qc_dzc(
            algorithm, sim, parameters, state, z=kwargs["z"]
        )
        der_couple_q_phase, _ = analytic_der_couple_phase(
            algorithm, sim, parameters, state, eigvals, eigvecs
        )
        eigvecs = np.einsum(
            "tai,ti->tai", eigvecs, np.conj(der_couple_q_phase), optimize="greedy"
        )
    if gauge_fixing_value >= 0:
        overlap = np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)
        signs = np.sign(overlap)
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
            > SMALL
        ):
            logger.error("Phase error encountered when fixing gauge analytically.")
    setattr(state, kwargs["output_eigvecs_name"], eigvecs)
    return parameters, state


def basis_transform_vec(algorithm, sim, parameters, state, **kwargs):
    """
    Transforms a vector "input_vec" to a new basis defined by "basis".

    Required constants:
        - None.
    """
    # Default transformation is adiabatic to diabatic.
    input_vec = getattr(state, kwargs["input_vec"])
    basis = getattr(state, kwargs["basis"])
    if kwargs.get("db_to_adb", False):
        basis = np.einsum("...ij->...ji", basis).conj()

    setattr(
        state,
        kwargs["output_name"],
        np.einsum("tij,tj->ti", basis, input_vec, optimize="greedy"),
    )
    return parameters, state


def basis_transform_mat(algorithm, sim, parameters, state, **kwargs):
    """
    Transforms a matrix "input_mat" to a new basis defined by "basis" and stores it in
    the state object with name "output_name".

    Required constants:
        - None.
    """
    # Default transformation is adiabatic to diabatic.
    input_mat = getattr(state, kwargs["input_mat"])
    basis = getattr(state, kwargs["basis"])
    setattr(
        state,
        kwargs["output_name"],
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
    adb_name = kwargs["adb_name"]
    update_wf_db_eigvals = getattr(state, kwargs["eigvals"])
    evals_exp = np.exp(-1j * update_wf_db_eigvals * sim.settings.dt_update)
    parameters, state = basis_transform_vec(
        algorithm,
        sim=sim,
        parameters=parameters,
        state=state,
        input_vec=kwargs["wf_db"],
        basis=kwargs["eigvecs"],
        output_name=adb_name,
        db_to_adb=True,
    )
    setattr(state, adb_name, (state.wf_adb * evals_exp))
    parameters, state = basis_transform_vec(
        algorithm,
        sim=sim,
        parameters=parameters,
        state=state,
        input_vec=adb_name,
        basis=kwargs["eigvecs"],
        output_name=kwargs["output_name"],
        db_to_adb=False,
    )
    return parameters, state


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
    if np.any(
        state.wf_adb[np.arange(num_trajs * num_branches), act_surf_ind_flat] == 0
    ):
        logger.critical(
            "Wavefunction in active surface is zero, cannot calculate "
            "hopping probabilities."
        )
    hop_prob = -2.0 * np.real(
        prod
        * state.wf_adb
        / state.wf_adb[np.arange(num_trajs * num_branches), act_surf_ind_flat][
            :, np.newaxis
        ]
    )
    hop_prob[np.arange(num_branches * num_trajs), act_surf_ind_flat] *= 0.0
    state.hop_prob = hop_prob

    return parameters, state


def update_hop_inds_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Determines which trajectories hop based on the hopping probabilities and which state
    they hop to. Note that these will only hop if they are not frustrated by the hopping
    function.

    Stores the indices of the hopping trajectories in state.hop_ind. Stores the
    destination indices of the hops in state.hop_dest.
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


def update_z_shift_fssh(algorithm, sim, parameters, state, **kwargs):
    z = kwargs["z"]
    delta_z = kwargs["delta_z"]
    ev_diff = kwargs["ev_diff"]
    hop, has_hop = sim.model.get("hop")
    if has_hop:
        z_shift, hopped = hop(
            sim.model,
            parameters,
            z=z,
            delta_z=delta_z,
            ev_diff=ev_diff,
        )
    else:
        z_shift, hopped = numerical_fssh_hop(
            sim.model,
            parameters,
            z=z,
            delta_z=delta_z,
            ev_diff=ev_diff,
        )
    state.hop_successful_traj = hopped
    state.z_shift_traj = z_shift
    return parameters, state


def update_hop_vals_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Executes the hopping function for the hopping trajectories.

    It stores the rescaled coordinates in state.z_rescaled and the a Boolean registering
    if the hop was successful in state.hop_successful.
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
        parameters, state = update_z_shift_fssh(
            algorithm,
            sim,
            parameters,
            state,
            z=z[traj_ind],
            delta_z=delta_z,
            ev_diff=ev_diff,
        )
        state.hop_successful[hop_traj_ind] = state.hop_successful_traj
        state.z_shift[hop_traj_ind] = state.z_shift_traj
        hop_traj_ind += 1
    return parameters, state


def update_z_hop_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Executes the post-hop updates for FSSH, rescaling the z coordinates and updating the
    active surface indices, and wavefunctions.
    """
    state.z[state.hop_ind] += state.z_shift
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
    z = getattr(state, kwargs["z"])
    h_q, _ = sim.model.get("h_q")
    h_qc, _ = sim.model.get("h_qc")
    if sim.model.update_h_q or state.h_q is None:
        # Update the quantum Hamiltonian if required or if it is not set.
        state.h_q = h_q(sim.model, parameters)
    # Update the quantum-classical Hamiltonian.
    state.h_qc = h_qc(sim.model, parameters, z=z)
    state.h_quantum = state.h_q + state.h_qc
    return parameters, state


def update_z_rk4_k1(algorithm, sim, parameters, state, **kwargs):
    dt_update = sim.settings.dt_update
    z_0 = getattr(state, kwargs["z"])
    output_name = kwargs["output_name"]
    k1 = -1j * (state.classical_forces + state.quantum_classical_forces)
    setattr(state, output_name, z_0 + 0.5 * dt_update * k1)
    state.z_rk4_k1 = k1
    return parameters, state


def update_z_rk4_k2(algorithm, sim, parameters, state, **kwargs):
    dt_update = sim.settings.dt_update
    z_0 = getattr(state, kwargs["z"])
    output_name = kwargs["output_name"]
    k2 = -1j * (state.classical_forces + state.quantum_classical_forces)
    setattr(state, output_name, z_0 + 0.5 * dt_update * k2)
    state.z_rk4_k2 = k2
    return parameters, state


def update_z_rk4_k3(algorithm, sim, parameters, state, **kwargs):
    dt_update = sim.settings.dt_update
    z_0 = getattr(state, kwargs["z"])
    output_name = kwargs["output_name"]
    k3 = -1j * (state.classical_forces + state.quantum_classical_forces)
    setattr(state, output_name, z_0 + dt_update * k3)
    state.z_rk4_k3 = k3
    return parameters, state


def update_z_rk4_k4(algorithm, sim, parameters, state, **kwargs):
    dt_update = sim.settings.dt_update
    z_0 = getattr(state, kwargs["z"])
    output_name = kwargs["output_name"]
    k4 = -1j * (state.classical_forces + state.quantum_classical_forces)
    setattr(
        state,
        output_name,
        z_0
        + dt_update
        * (1.0 / 6.0)
        * (state.z_rk4_k1 + 2.0 * state.z_rk4_k2 + 2.0 * state.z_rk4_k3 + k4),
    )
    return parameters, state


def update_dm_db_mf(algorithm, sim, parameters, state, **kwargs):
    """
    Update the diabatic density matrix based on the wavefunction.

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
    z = getattr(state, kwargs["z"])
    h_c, _ = sim.model.get("h_c")
    state.classical_energy = np.real(h_c(sim.model, parameters, z=z, batch_size=len(z)))
    return parameters, state


def update_classical_energy_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the classical energy for FSSH simulations. If deterministic, the energy in
    each branch is summed together with weights determined by the initial adiabatic
    populations. If not deterministic (and so there is only one branch), the energy is
    computed for the single branch.

    Required constants:
        - None.
    """
    z = getattr(state, kwargs["z"])
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_states = sim.model.constants.num_quantum_states
    h_c, _ = sim.model.get("h_c")
    if sim.algorithm.settings.fssh_deterministic:
        state.classical_energy = 0.0
        branch_weights = num_branches * np.einsum(
            "tbbb->tb",
            state.dm_adb_0.reshape((batch_size, num_branches, num_states, num_states)),
        )
        for branch_ind in range(num_branches):
            z_branch = z[state.branch_ind == branch_ind]
            state.classical_energy = state.classical_energy + branch_weights[
                :, branch_ind
            ] * h_c(
                sim.model,
                parameters,
                z=z_branch,
                batch_size=len(z_branch),
            )
    else:
        state.classical_energy = 0.0
        z_branch = z[state.branch_ind == 0]
        state.classical_energy = state.classical_energy + h_c(
            sim.model,
            parameters,
            z=z_branch,
            batch_size=len(z_branch),
        )
        state.classical_energy = state.classical_energy
    state.classical_energy = np.real(state.classical_energy)
    return parameters, state


def update_quantum_energy(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum energy w.r.t. the wavefunction specified by wf.

    Required constants:
        - None.
    """
    wf = getattr(state, kwargs["wf"])
    state.quantum_energy = np.real(
        np.einsum("ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy")
    )
    return parameters, state


def update_quantum_energy_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum energy w.r.t. the wavefunction specified by wf.

    Required constants:
        - None.
    """
    wf = getattr(state, kwargs["wf"])

    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
        batch_size = sim.settings.batch_size // num_branches
        num_states = sim.model.constants.num_quantum_states
        wf = wf * np.sqrt(
            num_branches
            * np.einsum(
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
    state.dm_adb_branch_flat = dm_adb_branch.reshape(
        (
            batch_size * num_branches,
            num_quantum_states,
            num_quantum_states,
        )
    )
    parameters, state = basis_transform_mat(
        algorithm,
        sim,
        parameters,
        state,
        input_mat="dm_adb_branch_flat",
        basis="eigvecs",
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
