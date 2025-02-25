import warnings
import numpy as np
from numba import njit
from qclab import ingredients


def apply_non_vectorized_ingredient(sim, ingredient, constants, parameters, arg_dict):
    val_vec = np.array(
        [
            ingredient(
                constants,
                parameters._element_list[n],
                **{key: val[n] for key, val in arg_dict.items()},
            )
            for n in range(sim.settings.batch_size)
        ]
    )
    return val_vec


def apply_nonvectorized_ingredient_over_internal_axes(
    sim, ingredient, constants, parameters, arg_dict, internal_shape
):
    num_axes = len(internal_shape)
    if num_axes > 0:
        for name, val in arg_dict.items():
            init_shape = np.shape(val)
            arg_dict[name] = val.reshape(
                (
                    sim.settings.batch_size * np.prod(internal_shape),
                    *init_shape[num_axes + 1 :],
                )
            )
        indexing = (slice(None),) + (np.newaxis,) * num_axes
        parameter_index = (
            np.arange(sim.settings.batch_size)[indexing]
            * np.ones((sim.settings.batch_size, *internal_shape))
        ).astype(int)
        parameter_index = parameter_index.reshape(
            (sim.settings.batch_size * np.prod(internal_shape))
        )
        val_vec = np.array(
            [
                ingredient(
                    constants,
                    parameters._element_list[parameter_index[n]],
                    **{name: val[n, ...] for name, val in arg_dict.items()},
                )
                for n in range(sim.settings.batch_size * np.prod(internal_shape))
            ]
        )
        return val_vec.reshape(
            (sim.settings.batch_size, *internal_shape, *np.shape(val_vec)[1:])
        )
    else:
        val_vec = np.array(
            [
                ingredient(
                    constants,
                    parameters._element_list[n],
                    **{key: val[n] for key, val in arg_dict.items()},
                )
                for n in range(sim.settings.batch_size)
            ]
        )
        return val_vec


def apply_vectorized_ingredient_over_internal_axes(
    sim, ingredient, constants, parameters, arg_dict, internal_shape
):
    # applies a vectorized ingredient over an argument that may have internal indices
    # if num_axes = 1 the ingredient is applied over th
    # assumes all arguments in arg_dict have the same internal dimensions.
    num_axes = len(internal_shape)
    if num_axes > 0:
        for name, val in arg_dict.items():
            init_shape = np.shape(val)
            arg_dict[name] = val.reshape(
                (
                    sim.settings.batch_size,
                    np.prod(internal_shape),
                    *init_shape[num_axes + 1 :],
                )
            )

        val_vec = np.einsum(
            "ij...->ji...",
            np.array(
                [
                    ingredient(
                        constants,
                        parameters,
                        **{name: val[:, n, ...] for name, val in arg_dict.items()},
                    )
                    for n in range(np.prod(internal_shape))
                ]
            ),
            optimize="greedy",
        )
        val_vec = np.array(
            [
                val_vec[n].reshape((*internal_shape, *np.shape(val_vec)[2:]))
                for n in range(sim.settings.batch_size)
            ]
        )
        return val_vec

    else:
        return ingredient(constants, parameters, **arg_dict)


def apply_ingredient_over_internal_axes(
    sim, ingredient, constants, parameters, arg_dict, internal_shape, vectorized
):
    if vectorized:
        return apply_vectorized_ingredient_over_internal_axes(
            sim, ingredient, constants, parameters, arg_dict, internal_shape
        )
    else:
        return apply_nonvectorized_ingredient_over_internal_axes(
            sim, ingredient, constants, parameters, arg_dict, internal_shape
        )


def initialize_z_coord(sim, parameters, state, **kwargs):
    """
    Initialize the z-coordinate.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    seed = kwargs["seed"]
    state.z_coord = sim.model.init_classical(sim.model.constants, parameters, seed=seed)
    return parameters, state


def update_dh_c_dzc(sim, parameters, state, **kwargs):
    z_coord = kwargs["z_coord"]
    state.dh_c_dzc = sim.model.dh_c_dzc(
        sim.model.constants, parameters, z_coord=z_coord
    )
    return parameters, state


def update_dh_qc_dzc(sim, parameters, state, **kwargs):
    z_coord = kwargs["z_coord"]
    state.dh_qc_dzc = sim.model.dh_qc_dzc(
        sim.model.constants, parameters, z_coord=z_coord
    )
    return parameters, state


def update_classical_forces(sim, parameters, state, **kwargs):
    """
    Update the classical forces (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs["z_coord"]
    parameters, state = update_dh_c_dzc(sim, parameters, state, z_coord=z_coord)
    state.classical_forces = state.dh_c_dzc
    return parameters, state


def update_quantum_classical_forces(sim, parameters, state, **kwargs):
    """
    Update the quantum-classical forces (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs["z_coord"]
    wf = kwargs["wf"]
    parameters, state = update_dh_qc_dzc(sim, parameters, state, z_coord=z_coord)
    state.quantum_classical_forces = np.einsum(
        "tnj,tj->tn",
        np.einsum("tnji,ti->tnj", state.dh_qc_dzc, wf, optimize="greedy"),
        np.conj(wf),
        optimize="greedy",
    )
    return parameters, state


def update_z_coord_rk4(sim, parameters, state, **kwargs):
    """
    Update the z-coordinates using the 4th-order Runge-Kutta method (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    dt = sim.settings.dt
    wf = kwargs["wf"]
    update_quantum_classical_forces_bool = kwargs[
        "update_quantum_classical_forces_bool"
    ]
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


def update_h_quantum(sim, parameters, state, **kwargs):
    z_coord = kwargs.get("z_coord", state.z_coord)
    parameters.z_coord = z_coord
    h_q = sim.model.h_q(sim.model.constants, parameters)
    h_qc = sim.model.h_qc(sim.model.constants, parameters, z_coord=z_coord)
    state.h_quantum = h_q + h_qc
    return parameters, state


@njit
def mat_vec_branch(mat, vec):
    """
    Perform matrix-vector multiplication for each branch.

    Args:
        mat (ndarray): The matrix.
        vec (ndarray): The vector.

    Returns:
        ndarray: The result of the matrix-vector multiplication.
    """
    return np.sum(mat * vec[:, np.newaxis, :], axis=-1)


def update_wf_db_rk4(sim, parameters, state, **kwargs):
    """
    Update the wavefunction using the 4th-order Runge-Kutta method (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
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
    Update the density matrix in the mean-field approximation (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    del sim, kwargs
    wf_db = state.wf_db
    state.dm_db = np.einsum("ti,tj->tij", wf_db, np.conj(wf_db), optimize="greedy")
    return parameters, state


def update_classical_energy(sim, parameters, state, **kwargs):
    z_coord = kwargs["z_coord"]
    state.classical_energy = sim.model.h_c(
        sim.model.constants, parameters, z_coord=z_coord
    )
    return parameters, state


def update_classical_energy_fssh(sim, parameters, state, **kwargs):
    """
    Update the classical energy (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs["z_coord"]
    state.classical_energy = 0
    for branch_ind in range(sim.algorithm.settings.num_branches):
        z_coord_branch = z_coord[state.z_coord_branch_branch_ind == branch_ind]
        state.classical_energy = state.classical_energy + sim.model.h_c(
            sim.model.constants, parameters, z_coord=z_coord_branch
        )
    return parameters, state


def update_quantum_energy_mf(sim, parameters, state, **kwargs):
    """
    Update the quantum energy.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    del sim
    wf = kwargs["wf"]
    state.quantum_energy = np.einsum(
        "ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy"
    )
    return parameters, state


def update_quantum_energy_fssh(sim, parameters, state, **kwargs):
    del sim, kwargs
    state.quantum_energy = np.einsum(
        "ti,tij,tj->t",
        np.conj(state.act_surf_wf),
        state.h_quantum,
        state.act_surf_wf,
        optimize="greedy",
    )
    return parameters, state


def broadcast_var_to_branch(sim, parameters, state, **kwargs):
    name = kwargs["name"]
    val = kwargs["val"]
    out = (
        np.zeros(
            (
                sim.settings.batch_size,
                sim.algorithm.settings.num_branches,
                *np.shape(val)[1:],
            )
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
    del sim
    matrix = kwargs["matrix"]
    eigvals_name = kwargs["eigvals_name"]
    eigvecs_name = kwargs["eigvecs_name"]
    eigvals, eigvecs = np.linalg.eigh(matrix + 0.0j)
    setattr(state, eigvals_name, eigvals)
    setattr(state, eigvecs_name, eigvecs)
    return parameters, state


def analytic_der_couple_phase(sim, parameters, state, eigvals, eigvecs):
    # TODO vectorize this??
    eigvals = eigvals.reshape(
        (
            sim.settings.batch_size,
            sim.algorithm.settings.num_branches,
            *np.shape(eigvals)[1:],
        )
    )
    eigvecs = eigvecs.reshape(
        (
            sim.settings.batch_size,
            sim.algorithm.settings.num_branches,
            *np.shape(eigvecs)[1:],
        )
    )
    dh_qc_dzc = state.dh_qc_dzc.reshape(
        (
            sim.settings.batch_size,
            sim.algorithm.settings.num_branches,
            *np.shape(state.dh_qc_dzc)[1:],
        )
    )
    der_couple_q_phase = np.ones(np.shape(eigvals), dtype=complex)
    der_couple_p_phase = np.ones(np.shape(eigvals), dtype=complex)
    for i in range(np.shape(eigvals)[-1] - 1):
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
        der_couple_zc = np.ascontiguousarray(
            np.einsum(
                "tbi,tbnij,tbj->tbn",
                np.conj(evec_i),
                dh_qc_dzc,
                evec_j,
                optimize="greedy",
            )
            / ((ev_diff + plus)[..., np.newaxis])
        )
        der_couple_z = np.ascontiguousarray(
            np.einsum(
                "tbi,tbnij,tbj->tbn",
                np.conj(evec_i),
                np.einsum("tbnij->tbnji", dh_qc_dzc).conj(),
                evec_j,
                optimize="greedy",
            )
            / ((ev_diff + plus)[..., np.newaxis])
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
                np.arange(der_couple_q.shape[0])[:, None],
                np.arange(der_couple_q.shape[1]),
                np.argmax(np.abs(der_couple_q), axis=-1),
            ]
        )
        der_couple_p_angle = np.angle(
            der_couple_p[
                np.arange(der_couple_p.shape[0])[:, None],
                np.arange(der_couple_p.shape[1]),
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
    return der_couple_q_phase.reshape(
        (
            sim.settings.batch_size * sim.algorithm.settings.num_branches,
            *np.shape(der_couple_q_phase)[2:],
        )
    ), der_couple_p_phase.reshape(
        (
            sim.settings.batch_size * sim.algorithm.settings.num_branches,
            *np.shape(der_couple_p_phase)[2:],
        )
    )


def gauge_fix_eigs(sim, parameters, state, **kwargs):
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


def copy_value(sim, parameters, state, **kwargs):
    del sim
    name = kwargs["name"]
    val = kwargs["val"]
    setattr(state, name, np.copy(val))
    return parameters, state


def basis_transform_vec(sim, parameters, state, **kwargs):
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


def basis_transform_mat_(sim, parameters, state, **kwargs):
    del sim
    # default is adb to db
    input_mat = kwargs["input_mat"]
    basis = kwargs["basis"]
    output_name = kwargs["output_name"]
    setattr(
        state,
        output_name,
        np.einsum(
            "...ij,...jk,...lk->...il",
            basis,
            input_mat,
            np.conj(basis),
        ),
    )
    return parameters, state


def basis_transform_mat(sim, parameters, state, **kwargs):
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
    del kwargs
    num_states = sim.model.constants.num_quantum_states
    num_branches = sim.algorithm.settings.num_branches
    num_trajs = sim.settings.batch_size
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
                np.abs(
                    state.wf_adb_branch.reshape((num_trajs, num_branches, num_states))
                )
                ** 2
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
    state.act_surf = act_surf.astype(int)
    return parameters, state


def initialize_active_surface__(sim, parameters, state, **kwargs):
    # TODO vectorize this
    del kwargs
    num_states = sim.model.constants.num_quantum_states
    if sim.algorithm.settings.fssh_deterministic:
        if sim.algorithm.settings.num_branches != num_states:
            raise ValueError(
                "num_branches must be equal to the quantum dimension for deterministic FSSH."
            )
        act_surf_ind_0 = np.arange(sim.algorithm.settings.num_branches, dtype=int)
    else:
        intervals = np.zeros(num_states)
        for state_n in range(num_states):
            intervals[state_n] = np.real(
                np.sum((np.abs(state.wf_adb_branch[0]) ** 2)[0 : state_n + 1])
            )
        act_surf_ind_0 = np.zeros((sim.algorithm.settings.num_branches), dtype=int)
        for branch_n in range(sim.algorithm.settings.num_branches):
            act_surf_ind_0[branch_n] = np.arange(num_states, dtype=int)[
                intervals > state.stochastic_sh_rand_vals[branch_n]
            ][0]
        act_surf_ind_0 = np.sort(act_surf_ind_0)
    # initialize active surface and active surface index in each branch
    state.act_surf_ind_0 = act_surf_ind_0.astype(int)
    state.act_surf_ind = act_surf_ind_0.astype(int)
    act_surf = np.zeros((sim.algorithm.settings.num_branches, num_states), dtype=int)
    act_surf[
        np.arange(sim.algorithm.settings.num_branches, dtype=int), state.act_surf_ind
    ] = 1
    state.act_surf = act_surf.reshape(
        (sim.algorithm.settings.num_branches, num_states)
    ).astype(int)
    return parameters, state


def initialize_random_values_fssh(sim, parameters, state, **kwargs):
    del kwargs
    state.hopping_probs_rand_vals = np.zeros(
        (sim.settings.batch_size, len(sim.settings.tdat))
    )
    state.stochastic_sh_rand_vals = np.zeros(
        (sim.settings.batch_size, sim.algorithm.settings.num_branches)
    )
    # this for loop is important so each seed is used
    for nt in range(sim.settings.batch_size):
        np.random.seed(state.seed[nt])
        state.hopping_probs_rand_vals[nt] = np.random.rand(len(sim.settings.tdat))
        state.stochastic_sh_rand_vals[nt] = np.random.rand(
            sim.algorithm.settings.num_branches
        )
    return parameters, state


def initialize_dm_adb_0_fssh(sim, parameters, state, **kwargs):
    del sim, kwargs
    state.dm_adb_0 = (
        np.einsum(
            "...i,...j->...ij",
            state.wf_adb_branch,
            np.conj(state.wf_adb_branch),
            optimize="greedy",
        )
        + 0.0j
    )
    return parameters, state


def update_act_surf_wf(sim, parameters, state, **kwargs):
    del kwargs
    num_trajs = sim.settings.batch_size
    num_branches = sim.algorithm.settings.num_branches
    act_surf_wf = state.eigvecs[
        np.arange(num_trajs * num_branches, dtype=int),
        :,
        state.act_surf_ind.flatten().astype(int),
    ]
    state.act_surf_wf = act_surf_wf
    return parameters, state


def update_dm_db_fssh(sim, parameters, state, **kwargs):
    del kwargs
    dm_adb_branch = np.einsum(
        "...i,...j->...ij",
        state.wf_adb_branch,
        np.conj(state.wf_adb_branch),
        optimize="greedy",
    )
    dm_adb_branch = dm_adb_branch.reshape(
        (
            sim.settings.batch_size,
            sim.algorithm.settings.num_branches,
            sim.model.constants.num_quantum_states,
            sim.model.constants.num_quantum_states,
        )
    )
    for nt in range(len(dm_adb_branch)):
        np.einsum("...jj->...j", dm_adb_branch[nt])[...] = state.act_surf[nt]
    if sim.algorithm.settings.fssh_deterministic:
        dm_adb_branch = (
            np.einsum("tbb->tb", state.dm_adb_0[..., 0, :, :])[
                ..., np.newaxis, np.newaxis
            ]
            * dm_adb_branch
        )
    else:
        dm_adb_branch = dm_adb_branch / sim.algorithm.settings.num_branches
    state.dm_adb = np.sum(dm_adb_branch, axis=-3) + 0.0j
    parameters, state = basis_transform_mat(
        sim,
        parameters,
        state,
        input_mat=dm_adb_branch.reshape(
            (
                sim.settings.batch_size * sim.algorithm.settings.num_branches,
                sim.model.constants.num_quantum_states,
                sim.model.constants.num_quantum_states,
            )
        ),
        basis=state.eigvecs,
        output_name="dm_db_branch",
    )
    print(np.shape(state.dm_db_branch))
    state.dm_db = (
        np.sum(
            state.dm_db_branch.reshape(
                (
                    sim.settings.batch_size,
                    sim.algorithm.settings.num_branches,
                    sim.model.constants.num_quantum_states,
                    sim.model.constants.num_quantum_states,
                )
            ),
            axis=-3,
        )
        + 0.0j
    )
    return parameters, state


def update_wf_db_eigs(sim, parameters, state, **kwargs):
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
    setattr(state, adb_name, (state.wf_adb_branch * evals_exp))
    parameters, state = basis_transform_vec(
        sim=sim,
        parameters=parameters,
        state=state,
        input_vec=state.wf_adb_branch,
        basis=eigvecs,
        output_name=output_name,
    )
    return parameters, state


def initialize_timestep_index(sim, parameters, state, **kwargs):
    # TODO vectorize this
    """
    Initialize the timestep index for the simulation.

    This function sets the timestep index (`t_ind`) in the state object to an array with a single element [0].

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    del sim, kwargs
    state.t_ind = 0
    return parameters, state


def update_timestep_index(sim, parameters, state, **kwargs):
    """
    Update the timestep index for the simulation.

    This function increments the timestep index (`t_ind`) in the state object.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    del sim, kwargs
    state.t_ind = state.t_ind + 1
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
    del kwargs
    rand = state.hopping_probs_rand_vals[:, state.t_ind]
    act_surf_ind = state.act_surf_ind
    act_surf = state.act_surf
    act_surf_ind_flat = act_surf_ind.flatten().astype(int)
    num_trajs = sim.settings.batch_size
    num_branches = sim.algorithm.settings.num_branches
    num_states = sim.model.constants.num_quantum_states
    traj_ind = (
        (np.arange(num_trajs)[:, np.newaxis] * np.ones((num_trajs, num_branches)))
        .flatten()
        .astype(int)
    )
    branch_ind = (
        (np.arange(num_branches)[np.newaxis, :] * np.ones((num_trajs, num_branches)))
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
        * state.wf_adb_branch
        / state.wf_adb_branch[np.arange(num_trajs * num_branches), act_surf_ind_flat][
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
        dh_qc_dzc = state.dh_qc_dzc
        eigvecs_flat = state.eigvecs
        eigvals_flat = state.eigvals
        z_coord_branch_flat = state.z_coord_branch
        init_shape = np.shape(state.act_surf)
        act_surf_flat = state.act_surf.reshape(
            (num_trajs * num_branches, *init_shape[2:])
        )
        # return parameters, state
        for traj_ind in traj_hop_ind:
            # print(traj_ind)
            k = np.argmax(
                (cumulative_probs[traj_ind] > rand_branch[traj_ind]).astype(int)
            )
            j = act_surf_ind_flat[traj_ind]
            evec_k = eigvecs_flat[traj_ind][:, j]
            evec_j = eigvecs_flat[traj_ind][:, k]
            eval_k = eigvals_flat[traj_ind][j]
            eval_j = eigvals_flat[traj_ind][k]
            ev_diff = eval_j - eval_k
            dkj_z = (
                np.einsum(
                    "i,nij,j->n",
                    np.conj(evec_k),
                    np.einsum("nij->nji", dh_qc_dzc[traj_ind]).conj(),
                    evec_j,
                    optimize="greedy",
                )
                / ev_diff[..., np.newaxis]
            )
            dkj_zc = (
                np.einsum(
                    "i,nij,j->n",
                    np.conj(evec_k),
                    dh_qc_dzc[traj_ind],
                    evec_j,
                    optimize="greedy",
                )
                / ev_diff[..., np.newaxis]
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
                    z_coord=z_coord_branch_flat[traj_ind],
                    delta_z_coord=delta_z,
                    ev_diff=ev_diff,
                )
            else:
                z_coord_branch_out, hopped = ingredients.numerical_fssh_hop(
                    sim.model,
                    sim.model.constants,
                    parameters,
                    z_coord=z_coord_branch_flat[traj_ind],
                    delta_z_coord=delta_z,
                    ev_diff=ev_diff,
                )

            if hopped:
                act_surf_ind_flat[traj_ind] = k
                act_surf_flat[traj_ind] = np.zeros_like(act_surf_flat[traj_ind])
                act_surf_flat[traj_ind][k] = 1
                z_coord_branch_flat[traj_ind] = z_coord_branch_out
                state.act_surf_ind = np.copy(
                    act_surf_ind_flat.reshape((num_trajs, num_branches))
                )
                state.act_surf = np.copy(
                    act_surf_flat.reshape((num_trajs, num_branches, num_states))
                )
                state.z_coord_branch = np.copy(z_coord_branch_flat)

    return parameters, state
