import numpy as np
import qclab.auxiliary as auxiliary


############################################################
#                  MEAN-FIELD INGREDIENTS                  #
############################################################


def initialize_wf_db(sim, state):
    state.wf_db = np.zeros((sim.num_branches * sim.num_trajs, sim.num_states), dtype=complex) + sim.wf_db[np.newaxis, :]
    return state


def initialize_z_coord(sim, state):
    z_coord = np.zeros((sim.num_trajs, sim.num_branches, sim.num_classical_coordinates), dtype=complex)
    # load initial values of the z coordinate 
    for traj_n in range(sim.num_trajs):
        z_coord[traj_n, :, :] = sim.init_classical(sim, sim.seeds[traj_n])
    state.z_coord = z_coord.reshape((sim.num_trajs * sim.num_branches, sim.num_classical_coordinates))
    return state


def update_h_quantum(sim, state):
    state.h_quantum = np.zeros((sim.num_branches * sim.num_trajs, sim.num_states, sim.num_states),
                               dtype=complex) + sim.h_q(sim.h_q_params)[np.newaxis, :, :] + \
                      sim.h_qc(sim.h_qc_params, state.z_coord)
    return state


def update_quantum_force_wf_db(sim, state):
    state.quantum_force = np.zeros((sim.num_branches * sim.num_trajs, sim.num_classical_coordinates), dtype=complex) + \
                          auxiliary.quantum_force_branch(state.wf_db, None, state.z_coord, sim)
    return state


def update_z_coord_rk4(sim, state):
    state.z_coord = auxiliary.rk4_c(state.z_coord, state.quantum_force, sim.dt, sim)
    return state


def update_wf_db_rk4(sim, state):
    # evolve wf_db using an RK4 solver
    state.wf_db = auxiliary.rk4_q_branch(state.h_quantum, state.wf_db, sim.dt)
    return state


def update_dm_db_mf(sim, state):
    state.dm_db_branch = np.einsum('ni,nj->nij', state.wf_db, np.conj(state.wf_db))
    state.dm_db = np.sum(state.dm_db_branch, axis=0)
    return state


def update_e_c(sim, state):
    state.e_c_branch = np.sum(sim.h_c(sim.h_c_params, state.z_coord).reshape((sim.num_trajs, sim.num_branches)), axis=0)
    state.e_c = np.sum(state.e_c_branch)
    return state


def update_e_q_mf(sim, state):
    state.e_q_branch = np.sum(np.einsum('nj,nji,ni->n', np.conj(state.wf_db), state.h_quantum, state.wf_db).reshape(
        (sim.num_trajs, sim.num_branches)), axis=0)
    state.e_q = np.sum(state.e_q_branch, axis=0)
    return state


############################################################
#                     FSSH INGREDIENTS                    #
############################################################


def initialize_random_values(sim, state):
    # initialize random numbers needed in each trajectory
    state.hopping_probs_rand_vals = np.zeros((sim.num_trajs, len(sim.tdat)))
    state.stochastic_sh_rand_vals = np.zeros((sim.num_trajs, sim.num_branches))
    for nt in range(sim.num_trajs):
        np.random.seed(sim.seeds[nt])
        state.hopping_probs_rand_vals[nt, :] = np.random.rand(len(sim.tdat))
        state.stochastic_sh_rand_vals[nt, :] = np.random.rand(sim.num_branches)
    return state


def update_eigs(sim, state):
    state.eigvals, state.eigvecs = np.linalg.eigh(state.h_quantum)
    return state


def analytic_gauge_fix_eigs(sim, state):
    # compute initial eigenvalues and eigenvectors in each branch
    for n in range(sim.num_trajs * sim.num_branches):
        # compute initial gauge shift for real-valued derivative couplings
        dab_q_phase, dab_p_phase = auxiliary.get_dab_phase(state.eigvals[n], state.eigvecs[n], state.z_coord[n], sim)
        # execute phase shift
        state.eigvecs[n] = np.copy(np.matmul(state.eigvecs[n], np.diag(np.conjugate(dab_q_phase))))
        # recalculate phases and check that they are zero
        dab_q_phase, dab_p_phase = auxiliary.get_dab_phase(state.eigvals[n], state.eigvecs[n], state.z_coord[n], sim)
        if np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2) > 1e-10:
            # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
            # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
            print('Warning: phase init',
                  np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2))
    return state


def update_eigs_previous(sim, state):
    state.eigvecs_previous = np.copy(state.eigvecs)
    return state


def initialize_wf_adb(sim, state):
    state.wf_adb = auxiliary.psi_db_to_adb_branch(state.wf_db, state.eigvecs)
    return state


def initialize_active_surface(sim, state):
    # Options for deterministic branch simulation, num_branches==num_states
    if sim.sh_deterministic:
        assert sim.num_branches == sim.num_states
        act_surf_ind_0 = np.zeros((sim.num_trajs, sim.num_branches)) + np.arange(sim.num_branches, dtype=int)[
                                                                       np.newaxis, :]
    else:
        # determine initial active surfaces
        intervals = np.zeros((sim.num_trajs, sim.num_states))
        for traj_n in range(sim.num_trajs):
            for state_n in range(sim.num_states):
                intervals[traj_n, state_n] = np.real(np.sum((np.abs(state.wf_adb[traj_n]) ** 2)[0:state_n + 1]))
        # initialize active surface index
        act_surf_ind_0 = np.zeros((sim.num_trajs, sim.num_branches), dtype=int)
        for traj_n in range(sim.num_trajs):
            for branch_n in range(sim.num_branches):
                act_surf_ind_0[traj_n, branch_n] = np.arange(sim.num_states, dtype=int)[
                    intervals[traj_n] > state.stochastic_sh_rand_vals[traj_n, branch_n]][0]
            act_surf_ind_0[traj_n] = np.sort(act_surf_ind_0[traj_n])
    # initialize active surface and active surface index in each branch
    state.act_surf_ind_0 = act_surf_ind_0.reshape(sim.num_trajs * sim.num_branches).astype(int)
    state.act_surf_ind = np.copy(state.act_surf_ind_0).astype(int)
    act_surf = np.zeros((sim.num_trajs, sim.num_branches, sim.num_states), dtype=int)
    for nt in range(sim.num_trajs):
        act_surf[nt][np.arange(sim.num_branches, dtype=int), state.act_surf_ind[
                                                             nt * sim.num_branches:(nt + 1) * sim.num_branches]] = 1
    state.act_surf = act_surf.reshape((sim.num_trajs * sim.num_branches, sim.num_states)).astype(int)
    return state


def update_quantum_force_act_surf(sim, state):
    state.quantum_force = np.zeros((sim.num_branches * sim.num_trajs, sim.num_classical_coordinates), dtype=complex) + \
                          auxiliary.quantum_force_branch(state.eigvecs, state.act_surf_ind, state.z_coord, sim)
    return state


def initialize_dm_adb_0_fssh(sim, state):
    state.dm_adb_0 = np.zeros((sim.num_trajs, sim.num_states, sim.num_states), dtype=complex)
    for n in range(sim.num_trajs):
        state.dm_adb_0[n] = np.outer(np.conj(state.wf_adb[n * sim.num_branches]), state.wf_adb[n * sim.num_branches])
    return state


def update_wf_db_eigs(sim, state):
    state.wf_db, state.wf_adb = auxiliary.evolve_wf_eigs(state.wf_db, state.eigvals, state.eigvecs, sim.dt)
    return state


def gauge_fix_eigs(sim, state):
    for nt in range(sim.num_trajs):
        # adjust gauge of eigenvectors
        state.eigvecs[nt * sim.num_branches:(nt + 1) * sim.num_branches], _ = auxiliary.sign_adjust_branch(
            state.eigvecs[nt * sim.num_branches:(nt + 1) * sim.num_branches],
            state.eigvecs_previous[nt * sim.num_branches:(nt + 1) * sim.num_branches],
            state.eigvals[nt * sim.num_branches:(nt + 1) * sim.num_branches],
            state.z_coord[nt * sim.num_branches:(nt + 1) * sim.num_branches], sim)
    return state


def update_active_surface_fssh(sim, state):
    ############################################################
    #                         HOPPING PROCEDURE                #
    ############################################################
    for nt in range(sim.num_trajs):
        rand = state.hopping_probs_rand_vals[nt, state.t_ind]
        for i in range(sim.num_branches):
            # compute hopping probabilities
            prod = np.matmul(np.conjugate(state.eigvecs[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][:,
                                          state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]]),
                             state.eigvecs_previous[nt * sim.num_branches:(nt + 1) * sim.num_branches][i])

            hop_prob = -2 * np.real(prod * (state.wf_adb[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]
                                            / state.wf_adb[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][
                                                state.act_surf_ind[
                                                nt * sim.num_branches:(nt + 1) * sim.num_branches][i]]))
            hop_prob[state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]] = 0
            bin_edge = 0
            # hop if possible
            for k in range(len(hop_prob)):
                hop_prob[k] = auxiliary.nan_num(hop_prob[k])
                bin_edge = bin_edge + hop_prob[k]
                if rand < bin_edge:
                    # compute nonadiabatic coupling d_{kj}= <k|\nabla H|j>/(e_{j} - e_{k})
                    evec_k = state.eigvecs[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][:,
                             state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]]
                    evec_j = state.eigvecs[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][:, k]
                    eval_k = state.eigvals[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][
                        state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]]
                    eval_j = state.eigvals[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][k]
                    ev_diff = eval_j - eval_k
                    # dkj_q is wrt q dkj_p is wrt p.
                    dkj_z, dkj_zc = auxiliary.get_dab(evec_k, evec_j, ev_diff,
                                                      state.z_coord[nt * sim.num_branches:(nt + 1) * sim.num_branches][
                                                          i],
                                                      sim)
                    # check that nonadiabatic couplings are real-valued
                    dkj_q = np.sqrt(sim.h * sim.m / 2) * (dkj_z + dkj_zc)
                    dkj_p = np.sqrt(1 / (2 * sim.h * sim.m)) * 1.0j * (dkj_z - dkj_zc)
                    max_pos_q = np.argmax(np.abs(dkj_q))
                    max_pos_p = np.argmax(np.abs(dkj_p))
                    if np.abs(dkj_q[max_pos_q]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2:
                        raise Exception('dkj_q Nonadiabatic coupling is complex, needs gauge fixing!')
                    if np.abs(dkj_p[max_pos_p]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_p[np.argmax(np.abs(dkj_p))]))) > 1e-2:
                        raise Exception('dkj_p Nonadiabatic coupling is complex, needs gauge fixing!')
                    delta_z = dkj_zc
                    state.z_coord[nt * sim.num_branches:(nt + 1) * sim.num_branches][i], hopped = \
                        sim.hop(sim, state.z_coord[nt * sim.num_branches:(nt + 1) * sim.num_branches][i], delta_z,
                                ev_diff)
                    if hopped:  # adjust active surfaces if a hop has occurred
                        state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i] = k
                        state.act_surf[nt * sim.num_branches:(nt + 1) * sim.num_branches][i] = np.zeros_like(
                            state.act_surf[nt * sim.num_branches:(nt + 1) * sim.num_branches][i])
                        state.act_surf[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][
                            state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]] = 1
                    break
    return state


def update_dm_adb_fssh(sim, state):
    state.dm_adb_branch = np.einsum('ni,nj->nij', state.wf_adb, np.conj(state.wf_adb))
    for nt in range(sim.num_trajs):
        np.einsum('...jj->...j', state.dm_adb_branch[nt * sim.num_branches:(nt + 1) * sim.num_branches])[
            ...] = state.act_surf[nt * sim.num_branches:(nt + 1) * sim.num_branches]
    return state


def update_dm_db_fssh(sim, state):
    state.dm_db_branch = auxiliary.rho_adb_to_db_branch(state.dm_adb_branch, state.eigvecs)
    if sim.sh_deterministic:
        state.dm_db_branch = ((np.einsum('njj->nj', state.dm_adb_0).reshape((sim.num_trajs, sim.num_states)))[:, :,
                              np.newaxis, np.newaxis] * state.dm_db_branch.reshape(sim.num_trajs, sim.num_branches,
                                                                                   sim.num_states, sim.num_states))
    else:
        state.dm_db_branch = state.dm_db_branch / sim.num_branches

    state.dm_db_branch = state.dm_db_branch.reshape(sim.num_trajs * sim.num_branches, sim.num_states, sim.num_states)
    state.dm_db = np.sum(state.dm_db_branch, axis=0)
    return state


def update_e_q_fssh(sim, state):
    e_q_branch = np.zeros((sim.num_branches * sim.num_trajs))
    for n in range(len(state.act_surf_ind)):
        e_q_branch[n] += state.eigvals[n][state.act_surf_ind[n]]
    state.e_q_branch = np.sum(e_q_branch.reshape((sim.num_trajs, sim.num_branches)), axis=0)
    state.e_q = np.sum(state.e_q_branch)
    return state


############################################################
#                     CFSSH INGREDIENTS                    #
############################################################


def update_wf_db_delta_eigs(sim, state):
    state.wf_db_delta, state.wf_adb_delta = auxiliary.evolve_wf_eigs(state.wf_db_delta, state.eigvals, state.eigvecs,
                                                                     sim.dt)
    return state


def initialize_branch_phase(sim, state):
    state.branch_phase = np.zeros((sim.num_trajs * sim.num_branches))
    return state


def update_branch_phase(sim, state):
    state.branch_phase = state.branch_phase + sim.dt * state.eigvals[
        np.arange(sim.num_branches * sim.num_trajs, dtype=int), state.act_surf_ind_0]
    return state


def initialize_wf_adb_delta(sim, state):
    # initialize wavefunction as a delta function in each branch
    wf_adb_delta = np.zeros((sim.num_trajs, sim.num_branches, sim.num_states), dtype=complex)
    for nt in range(sim.num_trajs):
        wf_adb_delta[nt][np.arange(sim.num_branches, dtype=int), state.act_surf_ind_0[nt * sim.num_branches:(
                                                                                                                    nt + 1) * sim.num_branches]] = 1.0 + 0.j
    state.wf_adb_delta = wf_adb_delta.reshape((sim.num_trajs * sim.num_branches, sim.num_states))
    # transform to diabatic basis
    state.wf_db_delta = auxiliary.psi_adb_to_db_branch(state.wf_adb_delta, state.eigvecs)
    return state


def update_active_surface_cfssh(sim, state):
    ############################################################
    #                         HOPPING PROCEDURE                #
    ############################################################
    for nt in range(sim.num_trajs):
        rand = state.hopping_probs_rand_vals[nt, state.t_ind]
        for i in range(sim.num_branches):
            # compute hopping probabilities
            prod = np.matmul(np.conjugate(state.eigvecs[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][:,
                                          state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]]),
                             state.eigvecs_previous[nt * sim.num_branches:(nt + 1) * sim.num_branches][i])

            hop_prob = -2 * np.real(prod * (state.wf_adb_delta[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]
                                            / state.wf_adb_delta[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][
                                                state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][
                                                    i]]))
            hop_prob[state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]] = 0
            bin_edge = 0
            # hop if possible
            for k in range(len(hop_prob)):
                hop_prob[k] = auxiliary.nan_num(hop_prob[k])
                bin_edge = bin_edge + hop_prob[k]
                if rand < bin_edge:
                    # compute nonadiabatic coupling d_{kj}= <k|\nabla H|j>/(e_{j} - e_{k})
                    evec_k = state.eigvecs[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][:,
                             state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]]
                    evec_j = state.eigvecs[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][:, k]
                    eval_k = state.eigvals[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][
                        state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]]
                    eval_j = state.eigvals[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][k]
                    ev_diff = eval_j - eval_k
                    # dkj_q is wrt q dkj_p is wrt p.
                    dkj_z, dkj_zc = auxiliary.get_dab(evec_k, evec_j, ev_diff,
                                                      state.z_coord[nt * sim.num_branches:(nt + 1) * sim.num_branches][
                                                          i],
                                                      sim)
                    # check that nonadiabatic couplings are real-valued
                    dkj_q = np.sqrt(sim.h * sim.m / 2) * (dkj_z + dkj_zc)
                    dkj_p = np.sqrt(1 / (2 * sim.h * sim.m)) * 1.0j * (dkj_z - dkj_zc)
                    max_pos_q = np.argmax(np.abs(dkj_q))
                    max_pos_p = np.argmax(np.abs(dkj_p))
                    if np.abs(dkj_q[max_pos_q]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2:
                        raise Exception('dkj_q Nonadiabatic coupling is complex, needs gauge fixing!')
                    if np.abs(dkj_p[max_pos_p]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_p[np.argmax(np.abs(dkj_p))]))) > 1e-2:
                        raise Exception('dkj_p Nonadiabatic coupling is complex, needs gauge fixing!')
                    delta_z = dkj_zc
                    state.z_coord[nt * sim.num_branches:(nt + 1) * sim.num_branches][i], hopped = \
                        sim.hop(sim, state.z_coord[nt * sim.num_branches:(nt + 1) * sim.num_branches][i], delta_z,
                                ev_diff)
                    if hopped:  # adjust active surfaces if a hop has occured
                        state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i] = k
                        state.act_surf[nt * sim.num_branches:(nt + 1) * sim.num_branches][i] = np.zeros_like(
                            state.act_surf[nt * sim.num_branches:(nt + 1) * sim.num_branches][i])
                        state.act_surf[nt * sim.num_branches:(nt + 1) * sim.num_branches][i][
                            state.act_surf_ind[nt * sim.num_branches:(nt + 1) * sim.num_branches][i]] = 1
                    break
    return state


def update_dm_adb_cfssh(sim, state):
    # state.dm_adb_branch = np.einsum('ni,nj->nij', state.wf_adb, np.conj(state.wf_adb))
    # for nt in range(sim.num_trajs):
    #    np.einsum('...jj->...j', state.dm_adb_branch[nt * sim.num_branches:(nt + 1) * sim.num_branches])[
    #        ...] = state.act_surf[nt * sim.num_branches:(nt + 1) * sim.num_branches]
    dm_adb_coh = np.zeros((sim.num_trajs, sim.num_states, sim.num_states), dtype=complex)
    dm_db_coh = np.zeros((sim.num_trajs, sim.num_states, sim.num_states), dtype=complex)
    for nt in range(sim.num_trajs):
        dm_adb_coh_ij = np.zeros((sim.num_states, sim.num_states), dtype=complex)
        phase_branch_nt = state.branch_phase
    return state


def update_dm_db_cfssh(sim, state):
    state.dm_adb_branch = np.einsum('ni,nj->nij', state.wf_adb, np.conj(state.wf_adb))
    for nt in range(sim.num_trajs):
        np.einsum('...jj->...j', state.dm_adb_branch[nt * sim.num_branches:(nt + 1) * sim.num_branches])[
            ...] = state.act_surf[nt * sim.num_branches:(nt + 1) * sim.num_branches]
    return state


def update_branch_pair_eigs(sim, state):
    state.eigvals_branch_pair = np.zeros((sim.num_trajs, sim.num_branches, sim.num_branches, sim.num_states),
                                         dtype=float)
    state.eigvecs_branch_pair = np.zeros(
        (sim.num_trajs, sim.num_branches, sim.num_branches, sim.num_states, sim.num_states), dtype=complex)
    for nt in range(sim.num_trajs):
        state.eigvals_branch_pair[nt], state.eigvecs_branch_pair[nt] = auxiliary.get_branch_pair_eigs(
            state.z_coord[nt * sim.num_branches:(nt + 1) * sim.num_branches], sim)
    return state


def gauge_fix_branch_pair_eigs(sim, state):
    for nt in range(sim.num_trajs):
        state.eigvals_branch_pair[nt], state.eigvecs_branch_pair[nt] = auxiliary.sign_adjust_branch_pair_eigs(
            state.z_coord[nt * sim.num_branches:(nt + 1) * sim.num_branches],
            state.eigvecs_branch_pair[nt], state.eigvals_branch_pair[nt], state.eigvecs_branch_pair_previous[nt], sim)
    return state


def update_branch_pair_eigs_previous(sim, state):
    state.eigvecs_branch_pair_previous = np.copy(state.eigvecs_branch_pair)
    return state


def analytic_gauge_fix_branch_pair_eigs(sim, state):
    # compute initial eigenvalues and eigenvectors in each branch
    for n in range(sim.num_trajs * sim.num_branches):
        # compute initial gauge shift for real-valued derivative couplings
        dab_q_phase, dab_p_phase = auxiliary.get_dab_phase(state.eigvals[n], state.eigvecs[n], state.z_coord[n], sim)
        # execute phase shift
        state.eigvecs[n] = np.copy(np.matmul(state.eigvecs[n], np.diag(np.conjugate(dab_q_phase))))
        # recalculate phases and check that they are zero
        dab_q_phase, dab_p_phase = auxiliary.get_dab_phase(state.eigvals[n], state.eigvecs[n], state.z_coord[n], sim)
        if np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2) > 1e-10:
            # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
            # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
            print('Warning: phase init',
                  np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2))
    return state


############################################################
#              MANYBODY MEAN FIELD INGREDIENTS             #
############################################################


def initialize_wf_db_mb(sim, state):
    state.wf_db_MB = np.zeros((sim.num_branches * sim.num_trajs, sim.num_states, sim.num_particles),
                              dtype=complex) + sim.wf_db_MB[np.newaxis, :, :]
    return state


def initialize_wf_db_mb_coeffs(sim, state):
    state.wf_db_MB_coeffs = np.zeros((sim.num_branches * sim.num_trajs, sim.num_SD),
                                     dtype=complex) + sim.wf_db_MB_coeffs[np.newaxis, :]
    return state


def update_quantum_force_wf_db_mbmf(sim, state):
    state.quantum_force = np.zeros((sim.num_branches * sim.num_trajs, sim.num_classical_coordinates), dtype=complex)
    for n in range(sim.num_particles):
        state.quantum_force += sim.dh_qc_dzc(sim.h_qc_params, state.wf_db_MB[:, :, n], state.wf_db_MB[:, :, n],
                                             state.z_coord)
    return state


def update_wf_db_mb_rk4(sim, state):
    # evolve wf_db using an RK4 solver
    state.wf_db_MB = auxiliary.rk4_q_branch(state.h_quantum, state.wf_db_MB, sim.dt)
    return state


def update_e_q_mbmf(sim, state):
    state.e_q_branch = np.sum(
        np.einsum('tin,tij,tjn->t', np.conj(state.wf_db_MB), state.h_quantum, state.wf_db_MB).reshape(
            (sim.num_trajs, sim.num_branches)), axis=0)
    state.e_q = np.sum(state.e_q_branch, axis=0)
    return state


############################################################
#           MANYBODY MEAN FIELD ARPES INGREDIENTS          #
############################################################


def update_quantum_force_wf_db_mbmf_arpes(sim, state):
    if state.t_ind >= sim.delay_ind:
        qf_mat_1 = np.zeros(
            (sim.num_branches * sim.num_trajs, sim.num_classical_coordinates, sim.num_particles, sim.num_particles),
            dtype=complex)
        for i in range(sim.num_particles):
            for j in range(sim.num_particles):
                qf_mat_1[:, :, i, j] = sim.dh_qc_dzc(sim.h_qc_params, state.wf_db_MB[:, :, i], state.wf_db_MB[:, :, j],
                                                     state.z_coord)
                if i != j:
                    qf_mat_1[:, :, i, j] *= -1
        qf_mat = np.copy(qf_mat_1)
        for n in range(sim.num_SD):
            qf_mat_1_tmp = np.copy(qf_mat_1)
            qf_mat_1_tmp[:, :, n, n] *= 0
            qf_mat[:, :, n, n] = np.einsum('tann->ta', qf_mat_1, optimize='greedy')
        state.quantum_force = np.einsum('tn,tamn,tm->ta', np.conj(state.wf_db_MB_coeffs), qf_mat, state.wf_db_MB_coeffs,
                                        optimize='greedy')
    else:
        state.quantum_force = np.zeros((sim.num_branches * sim.num_trajs, sim.num_classical_coordinates), dtype=complex)
        for n in range(sim.num_particles):
            state.quantum_force += sim.dh_qc_dzc(sim.h_qc_params, state.wf_db_MB[:, :, n], state.wf_db_MB[:, :, n],
                                                 state.z_coord)
    return state


def update_e_q_mbmf_arpes(sim, state):
    if state.t_ind >= sim.delay_ind:
        mat = 1 + 2 * np.identity(sim.num_SD) - np.ones((sim.num_SD, sim.num_SD)) * 2
        h_mat_1 = np.einsum('tin,tij,tjm->tmn', np.conj(state.wf_db_MB), state.h_quantum, state.wf_db_MB,
                            optimize='greedy') * mat
        h_mat = np.copy(h_mat_1)
        for n in range(sim.num_SD):
            h_mat_1_tmp = np.copy(h_mat_1)
            h_mat_1_tmp[:, n, n] *= 0
            h_mat[:, n, n] = np.einsum('tnn->t', h_mat_1, optimize='greedy')
        state.e_q_branch = np.sum(
            np.einsum('tn,tnm,tm->t', np.conj(state.wf_db_MB_coeffs), h_mat, state.wf_db_MB_coeffs,
                      optimize='greedy').reshape((sim.num_trajs, sim.num_branches)), axis=0)
        state.e_q = np.sum(state.e_q_branch, axis=0)
    else:
        state.e_q_branch = np.sum(
            np.einsum('tin,tij,tjn->t', np.conj(state.wf_db_MB), state.h_quantum, state.wf_db_MB).reshape(
                (sim.num_trajs, sim.num_branches)), axis=0)
        state.e_q = np.sum(state.e_q_branch, axis=0)
    return state



############################################################
#             DECOUPLED MEAN FIELD INGREDIENTS            #
############################################################

def initialize_z_coord_zpe(sim, state):
    z_coord = np.zeros((sim.num_trajs, sim.num_branches, sim.num_classical_coordinates), dtype=complex)
    # load initial values of the z coordinate 
    for traj_n in range(sim.num_trajs):
        z_coord[traj_n, :, :] = sim.init_classical(sim, sim.seeds[traj_n])
    state.z_coord = z_coord.reshape((sim.num_trajs * sim.num_branches, sim.num_classical_coordinates))
    return state


def update_h_quantum_dcmf(sim, state):
    state.h_quantum = np.zeros((sim.num_branches * sim.num_trajs, sim.num_states, sim.num_states),
                               dtype=complex) + sim.h_q(sim.h_q_params)[np.newaxis, :, :] + \
                      sim.h_qc(sim.h_qc_params, state.z_coord)
    return state


def update_quantum_force_wf_db_zpe(sim, state):
    state.quantum_force_zpe = np.zeros((sim.num_branches * sim.num_trajs, sim.num_classical_coordinates), dtype=complex) + \
                          auxiliary.quantum_force_branch(state.wf_db, None, state.z_coord_zpe, sim)
    return state


def update_z_coord_zpe_rk4(sim, state):
    state.z_coord_zpe = auxiliary.rk4_c(state.z_coord_zpe, state.quantum_force, sim.dt, sim)
    return state