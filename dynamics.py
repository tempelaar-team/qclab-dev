import ray
import time
import dill as pickle
from os import path
import os
import simulation
import numpy as np
import auxilliary


def run_dynamics(sim):
    # make calculation directory if needed
    if not (os.path.exists(sim.calc_dir)):
        os.mkdir(sim.calc_dir)
    # initialize ray cluster
    ray.shutdown()
    ray.init(**sim.cluster_args)
    if sim.num_procs > sim.num_trajs:
        sim.num_procs = sim.num_trajs
    ray_sim = ray.put(sim)  # put simulation object in shared memory
    data_filename = sim.calc_dir + '/data.out'  # output data object
    if path.exists(data_filename):
        data_file = open(data_filename, 'rb')
        data_obj = pickle.load(data_file)
        data_file.close()
        # determine index of last trajectory
        last_index = data_obj.index_list[-1] + 1
    else:
        # set last_index to 0 if new run
        last_index = 0
        # initialize data object
        data_obj = simulation.Data(data_filename)
    seeds = np.array([n for n in np.arange(last_index, sim.num_trajs + last_index)])
    for run in range(0, int(sim.num_trajs / sim.num_procs)):
        index_list = [run * sim.num_procs + i + last_index for i in range(sim.num_procs)]
        seed_list = [seeds[run * sim.num_procs + i] for i in range(sim.num_procs)]
        #results = [dynamics.remote(simulation.Trajectory(seed_list[i], index_list[i]), ray_sim) for i in range(sim.num_procs)]
        results = [dynamics(simulation.Trajectory(seed_list[i], index_list[i]), sim) for i in range(sim.num_procs)]
        for r in results:
            #traj_obj, msg = ray.get(r)
            traj_obj, msg = r
            print(msg)
            data_obj.add_data(traj_obj)
        del results
    ray.shutdown()
    data_file = open(data_obj.filename, 'wb')
    pickle.dump(data_obj, data_file)
    data_file.close()
    return sim



#@ray.remote
def dynamics(traj, sim):
    start_time = time.time()
    np.random.seed(traj.seed)
    # initialize classical coordinate
    z = sim.init_classical(sim)
    # initial wavefunction in diabatic basis
    psi_db = sim.psi_db_0
    # store the number of states
    num_states = len(psi_db)
    sim.num_states = num_states
    if sim.num_branches is None:
        sim.num_branches = num_states
    num_branches = sim.num_branches
    # compute initial Hamiltonian
    h_q = sim.h_q(sim)
    h_tot = h_q + sim.h_qc(z, sim)
    # initialize outputs
    tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0, sim.tmax + sim.dt_bath, sim.dt_bath)
    ec = np.zeros((len(tdat)))
    eq = np.zeros((len(tdat)))
    # initialize classical coordinates in each branch
    z_branch = np.zeros((num_branches, *np.shape(z)), dtype=complex)
    z_branch[:] = z
    if sim.dynamics_method == 'MF':
        psi_db_branch = np.zeros((num_branches, num_states), dtype=complex)
        psi_db_branch[:] = psi_db
        #assert sim.pab_cohere is None
        #assert sim.dmat_const is None
        #assert sim.branch_update is None
        assert num_branches == 1
    if sim.dynamics_method == 'CFSSH' or sim.dynamics_method == 'FSSH':
        ############################################################
        #              SURFACE HOPPING SPECIFIC INITIALIZATION     #
        ############################################################
        # compute initial eigenvalues and eigenvectors
        evals_0, evecs_0 = np.linalg.eigh(h_tot)
        # compute initial gauge shift for real-valued derivative couplings
        dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, z, sim)
        # execute phase shift
        evecs_0 = np.matmul(evecs_0, np.diag(np.conjugate(dab_q_phase)))
        # recalculate phases and check that they are zero
        dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, z, sim)
        if np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2) > 1e-10:
            # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
            # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
            print('Warning: phase init', np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2))
        # determine initial adiabatic wavefunction in fixed gauge
        psi_adb = auxilliary.psi_db_to_adb(psi_db, evecs_0)
        # determine initial adiabatic density matrix
        rho_adb_0 = np.outer(psi_adb, np.conj(psi_adb))
        # initial wavefunction in branches
        psi_adb_branch = np.zeros((num_branches, num_states), dtype=complex)
        psi_adb_branch[:] = psi_adb
        # initialize eigenvalues and eigenvectors in each branch
        evals_branch = np.zeros((num_branches, num_states))
        evecs_branch = np.zeros((num_branches, num_states, num_states), dtype=complex)
        evals_branch[:] = evals_0
        evecs_branch[:] = evecs_0
        # initialize branch-pair eigenvalues and eigenvectors
        if sim.dmat_const > 0:
            evecs_branch_pair = np.zeros((num_branches, num_branches, num_states, num_states), dtype=complex)
            evals_branch_pair = np.zeros((num_branches, num_branches, num_states))
            evals_branch_pair[:, :] = evecs_0
            evals_branch_pair[:, :] = evals_0

        ############################################################
        #                   ACTIVE SURFACE INITIALIZATION          #
        ############################################################
        # Options for deterministic branch simulation, num_branches==num_states
        if sim.sh_deterministic:
            assert num_branches == num_states
            act_surf_ind_0 = np.arange(num_branches,dtype=int)
        else:
            if sim.dynamics_method == 'CFSSH':
                assert num_branches > 1
            # determine initial active surfaces
            intervals = np.zeros(num_states)
            for n in range(num_states):
                intervals[n] = np.sum(np.real(np.abs(psi_adb) ** 2)[0:n + 1])
            rand_val = np.random.rand(num_branches)
            # initialize active surface index
            act_surf_ind_0 = np.zeros((num_branches), dtype=int)
            for n in range(num_branches):
                act_surf_ind_0[n] = np.arange(num_states)[intervals > rand_val[n]][0]
            act_surf_ind_0 = np.sort(act_surf_ind_0)
        # initialize active surface and active surface index in each branch
        act_surf_ind_branch = np.copy(act_surf_ind_0)
        act_surf_branch = np.zeros((num_branches, num_states), dtype=int)
        act_surf_branch[np.arange(num_branches, dtype=int), act_surf_ind_branch] = 1

        ############################################################
        #                    WAVEFUNCTION INITIALIZATION           #
        ############################################################
        # initialize wavefunction as a delta function in each branch
        psi_adb_delta_branch = np.zeros((num_branches, num_states), dtype=complex)
        psi_adb_delta_branch[np.arange(num_branches, dtype=int), act_surf_ind_0] = 1.0 + 0.j
        # transform to diabatic basis
        psi_db_branch = auxilliary.psi_adb_to_db_branch(psi_adb_branch, evecs_branch)
        psi_db_delta_branch = auxilliary.psi_adb_to_db_branch(psi_adb_delta_branch, evecs_branch)

        ############################################################
        #         COHERENT SURFACE HOPPING SPECIFIC INITIALIZATION#
        ############################################################
        # store the phase of each branch
        phase_branch = np.zeros(num_branches)

    ############################################################
    #                        TIME EVOLUTION                   #
    ############################################################
    t_ind = 0
    for t_bath_ind in np.arange(0, len(tdat_bath)):
        if t_ind == len(tdat):
            break
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * sim.dt_bath:

        ############################################################
        #                            OUTPUT TIMESTEP               #
        ############################################################

            ############################################################
            #                                 CFSSH                    #
            ############################################################
            if sim.calc_cfssh_obs:
                # calculate overlap matrix
                overlap = auxilliary.get_classical_overlap(z_branch, sim)
                if sim.dmat_const == 0:
                    # Inexpensive density matrix construction
                    rho_adb_cfssh_branch = np.zeros((num_branches, num_states, num_states), dtype=complex)
                    rho_adb_cfssh_coh = np.zeros((num_states, num_states), dtype=complex)
                    for i in range(num_branches):
                        for j in range(i, num_branches):
                            a_i = act_surf_ind_branch[i]
                            a_j = act_surf_ind_branch[j]
                            a_i_0 = act_surf_ind_0[i]
                            a_j_0 = act_surf_ind_0[j]
                            if a_i != a_j and a_i != a_j and a_i == a_i_0 and a_j == a_j_0:
                                if sim.sh_deterministic:
                                    prob_fac = 1
                                else:
                                    prob_fac = 1/(rho_adb_0[a_i,a_i]*rho_adb_0[a_j,a_j]*(num_branches-1))
                                rho_adb_cfssh_coh[a_i, a_j] += prob_fac * rho_adb_0[a_i,a_j] * overlap[i, j] * \
                                    np.exp(-1.0j*(phase_branch[i] - phase_branch[j]))
                                rho_adb_cfssh_coh[a_j, a_i] += np.conj(rho_adb_cfssh_coh[a_i, a_j])
                    if sim.sh_deterministic:
                        # construct diagonal of adiaabtic density matrix
                        rho_adb_cfssh_branch_diag = np.diag(rho_adb_0).reshape((-1, 1)) * act_surf_branch
                        np.einsum('...jj->...j', rho_adb_cfssh_branch)[...] = rho_adb_cfssh_branch_diag
                        rho_adb_cfssh_branch = rho_adb_cfssh_branch + rho_adb_cfssh_coh / num_branches
                    else:
                        for n in range(num_branches):
                            rho_adb_cfssh_branch[n, act_surf_ind_branch[n], act_surf_ind_branch[n]] += 1
                        # add coherences averaged over branches
                        rho_adb_cfssh_branch = (rho_adb_cfssh_branch + rho_adb_cfssh_coh / num_branches) / num_branches
                    rho_db_cfssh_branch = auxilliary.rho_adb_to_db_branch(rho_adb_cfssh_branch, evecs_branch)
                # expensive CFSSH density matrix construction
                if sim.dmat_const == 1:

                    pass
                cfssh_observables_t = sim.cfssh_observables(sim, rho_db_cfssh_branch, z_branch)
                if t_ind == 0 and t_bath_ind == 0:
                    for key in cfssh_observables_t.keys():
                        traj.new_observable(key + '_cfssh', (len(tdat), *np.shape(cfssh_observables_t[key])), cfssh_observables_t[key].dtype)
                traj.add_observable_dic(t_ind, cfssh_observables_t)

            ############################################################
            #                                 FSSH                    #
            ############################################################
            if sim.calc_fssh_obs:
                rho_adb_fssh = np.einsum('ni,nj->nij', psi_adb_branch, np.conj(psi_adb_branch))
                np.einsum('...jj->...j', rho_adb_fssh)[...] = act_surf_branch
                rho_db_fssh_branch = auxilliary.rho_adb_to_db_branch(rho_adb_fssh, evecs_branch)
                if sim.sh_deterministic:
                    rho_db_fssh_branch = np.diag(rho_adb_0)[:,np.newaxis,np.newaxis]*rho_db_fssh_branch
                else:
                    rho_db_fssh_branch = rho_db_fssh_branch/num_branches
                fssh_observables_t = sim.fssh_observables(sim, rho_db_fssh_branch, z_branch)
                if t_ind == 0 and t_bath_ind == 0:
                    for key in fssh_observables_t.keys():
                        traj.new_observable(key + '_fssh', (len(tdat), *np.shape(fssh_observables_t[key])), fssh_observables_t[key].dtype)
                traj.add_observable_dic(t_ind, fssh_observables_t)

            ############################################################
            #                                  MF                     #
            ############################################################
            if sim.calc_mf_obs:
                rho_db_mf_branch = np.einsum('ni,nk->nik', psi_db_branch, np.conj(psi_db_branch))
                mf_observables_t = sim.mf_observables(sim, rho_db_mf_branch, z_branch)
                if t_ind == 0 and t_bath_ind == 0:
                    for key in mf_observables_t.keys():
                        traj.new_observable(key + '_mf', (len(tdat), *np.shape(mf_observables_t[key])), mf_observables_t[key].dtype)
                traj.add_observable_dic(t_ind, mf_observables_t)

            ############################################################
            #                         CLASSICAL OBSERVABLES            #
            ############################################################
            # We will not calculate any purely classical observables because
            # the branch separation in principle implies a dependence of the 
            # contribution of the classical trajectory on the statistics of the 
            # branch sampling. Therefore purely classical terms are only present in MF/FSSH
            # where a single branch is used and so using the ordinary FSSH/MF observables
            # generation should be efficient. 
            #classical_obs_t = sim.classical_observables(sim, z_branch)
            #classical_obs_t, _ = sim.classical_observables(sim, z)
            #for obs_n in range(num_classical_obs):
            #    output_classical_obs[obs_n][t_ind] = classical_obs_t[obs_n]
            

            pass

    ############################################################
    #                     CLASSICAL PROPAGATION                #
    ############################################################
    # calculate quantum force
    if sim.dynamics_method == 'MF':
        qfzc_branch = auxilliary.quantum_force_branch(psi_db_branch, None, z_branch, sim)
    if sim.dynamics_method == 'FSSH' or sim.dynamics_method == 'CFSSH':
        qfzc_branch = auxilliary.quantum_force_branch(evecs_branch, act_surf_ind_branch, z_branch, sim)
    # evolve classical coordinates
    z_branch = auxilliary.rk4_c(z_branch, qfzc_branch, sim.dt_bath, sim)
    # update Hamiltonian
    h_tot_branch = h_q[np.newaxis, :, :] + auxilliary.h_qc_branch(z, sim)

    if sim.dynamics_method == 'MF':
        ############################################################
        #               QUANTUM PROPAGATION IN DIABATIC BASIS      #
        ############################################################
        psi_db_branch = auxilliary.rk4_q(h_tot_branch, psi_db_branch, sim.dt_bath, path='greedy')
    if sim.dynamics_method == 'FSSH' or sim.dynamics_method == 'CFSSH':
        ############################################################
        #              QUANTUM PROPAGATION IN ADIABATIC BASIS     #
        ############################################################
        evecs_branch_previous = np.copy(evecs_branch)
        # obtain branch eigenvalues and eigenvectors
        evals_branch, evecs_branch = np.linalg.eigh(h_tot_branch)
        # adjust gauge of eigenvectors
        evecs_branch,_ = auxilliary.sign_adjust_branch(evecs_branch, evecs_branch_previous, evals_branch, z_branch, sim)
        # propagate phases
        phase_branch = phase_branch + sim.dt_bath * evals_branch[np.arange(num_branches,dtype=int),act_surf_ind_0]
        # construct eigenvalue exponential
        evals_exp_branch = np.exp(-1.0j * evals_branch * sim.dt_bath)
        # evolve wavefunction
        diag_matrix_branch = np.zeros((num_branches, num_states, num_states), dtype=complex)
        diag_matrix_branch[:,range(num_states),range(num_states)] = evals_exp_branch
        psi_adb_branch = np.copy(auxilliary.psi_db_to_adb_branch(psi_db_branch, evecs_branch))
        psi_adb_delta = np.copy(auxilliary.psi_db_to_adb_branch(psi_db_delta_branch, evecs_branch))
        # multiply by propagator
        psi_adb_branch = np.copy(np.einsum('nab,nb->na', diag_matrix_branch, psi_adb_branch, optimize='greedy'))
        psi_adb_delta_branch = np.copy(np.einsum('nab,nb->na', diag_matrix_branch, psi_adb_delta_branch, optimize='greedy'))
        # transform back to diabatic basis
        psi_db_branch = auxilliary.psi_adb_to_db_branch(psi_adb_branch, evecs_branch)
        psi_db_delta_branch = auxilliary.psi_adb_to_db_branch(psi_adb_delta_branch, evecs_branch)

        ############################################################
        #                         HOPPING PROCEDURE                #
        ############################################################
        # draw a random number (same for all branches)
        rand = np.random.rand()
        # TODO -- talk to Roel about rand in CFSSH/FSSH, can we actually do CFSSH with stochastic branch sampling?
        for i in range(num_branches):
            # compute hopping probabilities
            prod = np.matmul(np.conjugate(evecs_branch[i][:, act_surf_ind_branch[i]]), evecs_branch_previous[i])
            if sim.pab_cohere:
                hop_prob = -2 * np.real(prod * (psi_adb_branch[i] / psi_adb_branch[i][act_surf_ind_branch[i]]))
            if not sim.pab_cohere:
                hop_prob = -2 * np.real(
                    prod * (psi_adb_delta_branch[i] / psi_adb_delta_branch[i][act_surf_ind_branch[i]]))
            hop_prob[act_surf_ind_branch[i]] = 0
            bin_edge = 0
            # hop if possible
            for k in range(len(hop_prob)):
                hop_prob[k] = auxilliary.nan_num(hop_prob[k])
                bin_edge = bin_edge + hop_prob[k]
                if rand < bin_edge:
                    # compute nonadiabatic coupling d_{kj}= <k|\nabla H|j>/(e_{j} - e_{k})
                    evec_k = evecs_branch[i][:, act_surf_ind_branch[i]]
                    evec_j = evecs_branch[i][:, k]
                    eval_k = evals_branch[i][act_surf_ind_branch[i]]
                    eval_j = evals_branch[i][k]
                    ev_diff = eval_j - eval_k
                    # dkj_q is wrt q dkj_p is wrt p.
                    dkj_z, dkj_zc = auxilliary.get_dab(evec_k, evec_j, ev_diff, z_branch[i], sim)
                    # check that nonadiabatic couplings are real-valued
                    dkj_q = np.sqrt(sim.h * sim.m / 2) * (dkj_z + dkj_zc)
                    dkj_p = np.sqrt(1 / (2 * sim.h * sim.m)) * 1.0j * (dkj_z - dkj_zc)
                    if np.abs(np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2:
                        print('ERROR IMAGINARY DKKQ: \n', 'angle: ',
                              np.abs(np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))),
                              '\n magnitude: ', np.abs(dkj_q[np.argmax(np.abs(dkj_q))]),
                              '\n value: ', dkj_q[np.argmax(np.abs(dkj_q))])
                    if np.abs(np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2:
                        print('ERROR IMAGINARY DKKP: \n', 'angle: ',
                              np.abs(np.sin(np.angle(dkj_p[np.argmax(np.abs(dkj_p))]))),
                              '\n magnitude: ', np.abs(dkj_p[np.argmax(np.abs(dkj_p))]),
                              '\n value: ', dkj_p[np.argmax(np.abs(dkj_p))])
                    # compute rescalings
                    delta_z = dkj_zc
                    z_branch[i], hopped = sim.hop(z_branch[i], delta_z, ev_diff, sim)
                    if hopped:
                        act_surf_ind_branch[i] = k
                        act_surf_branch[i] = np.zeros_like(act_surf_branch[i])
                        act_surf_branch[i][act_surf_ind_branch[i]] = 1
                    break
    traj.add_to_dic('t', tdat)
    end_time = time.time()
    msg = 'trial index: ' + str(traj.index) +  ' time: ' + str(end_time - start_time) + ' seed: ' + str(traj.seed)
    return traj, msg
