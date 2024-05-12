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
        results = [dynamics.remote(simulation.Trajectory(seed_list[i], index_list[i]), ray_sim) for i in range(sim.num_procs)]
        for r in results:
            traj_obj, msg = ray.get(r)
            print(msg)
            data_obj.add_data(traj_obj)
        del results
    ray.shutdown()
    data_file = open(data_obj.filename, 'wb')
    pickle.dump(data_obj, data_file)
    data_file.close()
    return sim



@ray.remote
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

    rho_db_0 = np.outer(psi_db, np.conj(psi_db))
    ######## Initialize outputs ########
    quantum_obs_0, quantum_obs_names = sim.quantum_observables(sim, rho_db_0)
    assert len(quantum_obs_0) == len(quantum_obs_names)
    num_quantum_obs = len(quantum_obs_0)
    classical_obs_0, classical_obs_names = sim.classical_observables(sim, z)
    assert len(classical_obs_0) == len(classical_obs_names)
    num_classical_obs = len(classical_obs_0)
    if sim.calc_mf_obs: # TODO there are rules associated with what calc_#_obs can be depending on what dynamics_method is
        output_quantum_mf_obs = np.array(
            [np.zeros((len(tdat), np.shape(quantum_obs_0[n])), dtype=quantum_obs_0[n].dtype) \
             for n in range(num_quantum_obs)], dtype='object')
    if sim.calc_fssh_obs:
        output_quantum_fssh_obs = np.array(
            [np.zeros((len(tdat), np.shape(quantum_obs_0[n])), dtype=quantum_obs_0[n].dtype) \
             for n in range(num_quantum_obs)], dtype='object')
    if sim.calc_cfssh_obs:
        output_quantum_cfssh_obs = np.array(
            [np.zeros((len(tdat), np.shape(quantum_obs_0[n])), dtype=quantum_obs_0[n].dtype) \
             for n in range(num_quantum_obs)], dtype='object')
    output_classical_obs = np.array([np.zeros((len(tdat), np.shape(classical_obs_0[n])), dtype=classical_obs_0[n].dtype) \
                                     for n in range(num_classical_obs)], dtype='object')
    ####################################
    if sim.dynamics_method == 'MF':
        assert sim.pab_cohere is None
        assert sim.dmat_const is None
        assert sim.branch_update is None
        assert num_branches == 1
    if sim.dynamics_method == 'CFSSH' or sim.dynamics_method == 'FSSH':
        ######## Surface Hopping specific initialization #######
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
            # or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
            print('Warning: phase init', np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2))
        # determine initial adiabatic wavefunction in fixed gauge
        psi_adb = auxilliary.vec_0_db_to_adb(psi_db, evecs_0)
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

        # initialize branch-pair eigenvalues and eigenvectors
        if sim.dmat_const > 0:
            u_ij = np.zeros((num_branches, num_branches, num_states, num_states), dtype=complex)
            e_ij = np.zeros((num_branches, num_branches, num_states))
            u_ij[:, :] = evecs_0
            e_ij[:, :] = evals_0
        # initialize active surface and active surface index in each branch
        act_surf_ind_branch = np.copy(act_surf_ind_0)
        act_surf_branch = np.zeros((num_branches, num_states), dtype=int)
        act_surf_branch[np.arange(num_branches, dtype=int), act_surf_ind_branch] = 1

        # initialize wavefunction as a delta function in each branch
        psi_adb_delta_branch = np.zeros((num_branches, num_states), dtype=complex)
        psi_adb_delta_branch[np.arange(num_branches, dtype=int), act_surf_ind_0] = 1.0 + 0.j
        # transform to diabatic basis
        psi_db_branch = auxilliary.vec_adb_to_db(psi_adb_branch, evecs_branch)
        psi_db_delta_branch = auxilliary.vec_adb_to_db(psi_adb_delta_branch, evecs_branch)



        ######## Coherent Surface Hopping specific initialization ########
        # store the phase of each branch
        phase_branch = np.zeros(num_branches)

    ######## Time Evolution ########
    t_ind = 0
    for t_bath_ind in np.arange(0, len(tdat_bath)):
        if t_ind == len(tdat):
            break
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * sim.dt_bath:
            ######## Output timestep ########

            ######## CFSSH ########
            if sim.calc_cfssh_obs:
                rho_db_cfssh = np.zeros((num_states, num_states), dtype=complex)
                quantum_cfssh_obs_t,_ = sim.quantum_observables(sim, rho_db_cfssh)
                for obs_n in range(num_quantum_obs):
                    output_quantum_cfssh_obs[n][t_ind] = quantum_cfssh_obs_t[n]
            #######################

            ######## FSSH ########
            if sim.calc_fssh_obs:
                rho_adb_fssh = np.einsum('ni,nj->nij', psi_adb_branch, np.conj(psi_adb_branch))
                np.einsum('...jj->...j', rho_adb_fssh)[...] = act_surf_branch
                rho_db_fssh = auxilliary.rho_adb_to_db(rho_adb_fssh, evecs_branch)
                if sim.sh_deterministic:
                    rho_db_fssh = np.sum(np.diag(rho_adb_0)[:,np.newaxis,np.newaxis]*rho_db_fssh,axis=0)
                else:
                    rho_db_fssh = np.sum(rho_db_fssh/num_branches, axis=0)
                quantum_fssh_obs_t,_ = sim.quantum_observables(sim, rho_db_fssh)
                for obs_n in range(num_quantum_obs):
                    output_quantum_fssh_obs[n][t_ind] = quantum_fssh_obs_t[n]
            ######################

            ######## MF ########
            if sim.calc_mf_obs:
                rho_db_mf = np.einsum('ni,nk->ik', psi_db_branch, np.conj(psi_db_branch))
                quantum_mf_obs_t,_ = sim.quantum_observables(sim, rho_db_mf)
                for obs_n in range(num_quantum_obs):
                    output_quantum_mf_obs[n][t_ind] = quantum_mf_obs_t[n]
            ####################

            ######## classical observables ########
            classical_obs_t, _ = sim.classical_observables(sim, z)
            for obs_n in range(num_classical_obs):
                output_classical_obs[n][t_ind] = classical_obs_t[n]
            #######################################

            pass

    if sim.dynamics_method == 'MF':
        qfzc_branch = auxilliary.quantum_force_branch(psi_db_branch, None, z_branch, sim)
    if sim.dynamics_method == 'FSSH' or sim.dynamics_method == 'CFSSH':
        qfzc_branch = auxilliary.quantum_force_branch(evecs_branch, act_surf_ind_branch, z_branch, sim)

    z_branch = auxilliary.rk4_c(z, qfzc_branch, sim.dt_bath, sim)
    h_tot_branch = h_q[np.newaxis, :, :] + auxilliary.h_qc_branch(z, sim)
    if sim.dynamics_method == 'MF':

        ######## quantum propagation in diabatic basis ########
        psi_db_branch = auxilliary.rk4_q(h_tot_branch, psi_db_branch, sim.dt_bath)
        #######################################################

    if sim.dynamics_method == 'FSSH' or sim.dynamics_method == 'CFSSH':

        ######## quantum propagation in adiabatic basis ########
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
        psi_adb_branch = np.copy(auxilliary.vec_db_to_adb(psi_db_branch, evecs_branch))
        psi_adb_delta = np.copy(auxilliary.vec_db_to_adb(psi_db_delta_branch, evecs_branch))
        # multiply by propagator
        psi_adb_branch = np.copy(np.einsum('nab,nb->na', diag_matrix_branch, psi_adb_branch, optimize='greedy'))
        psi_adb_delta_branch = np.copy(np.einsum('nab,nb->na', diag_matrix_branch, psi_adb_delta_branch, optimize='greedy'))
        # transform back to diabatic basis
        psi_db_branch = auxilliary.vec_adb_to_db(psi_adb_branch, evecs_branch)
        psi_db_delta_branch = auxilliary.vec_adb_to_db(psi_adb_delta_branch, evecs_branch)
        #######################################################

        # draw a random number (same for all branches)
        rand = np.random.rand()
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
    traj.add_to_dic('eq', eq)
    traj.add_to_dic('ec', ec)
    for n in range(num_quantum_obs):
        if sim.calc_mf_obs:
            traj.add_to_dic(quantum_obs_names[n]+'_mf', output_quantum_mf_obs[n])
        if sim.calc_fssh_obs:
            traj.add_to_dic(quantum_obs_names[n]+'_fssh', output_quantum_fssh_obs[n])
        if sim.calc_cfssh_obs:
            traj.add_to_dic(quantum_obs_names[n]+'_cfssh', output_quantum_cfssh_obs[n])
    for n in range(num_classical_obs):
        traj.add_to_dic(classical_obs_names[n], output_classical_obs[n])
    end_time = time.time()
    msg = 'trial index: ' + str(traj.index) +  ' time: ' + str(end_time - start_time) + ' seed: ' + str(traj.seed)
    return traj, msg

@ray.remote
def cfssh_dynamics(traj, sim):
    # TODO -- add phase evolution!!!
    start_time = time.time()
    np.random.seed(traj.seed)
    # initialize classical coordinates
    z, zc = sim.init_classical(sim)
    # compute initial Hamiltonian
    h_q = sim.h_q(sim)
    h_tot = h_q + sim.h_qc(z, zc, sim)
    # compute initial eigenvalues and eigenvectors
    evals_0, evecs_0 = np.linalg.eigh(h_tot)
    num_states = len(evals_0)
    num_branches = num_states
    # compute initial gauge shift for real-valued derivative couplings
    dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, z, zc, sim)
    # execute phase shift
    evecs_0 = np.matmul(evecs_0, np.diag(np.conjugate(dab_q_phase)))
    # recalculate phases and check that they are zero
    dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, z, zc, sim)
    if np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2) > 1e-10:
        # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
        print('Warning: phase init', np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2))
    #  initial wavefunction in diabatic basis
    psi_db = sim.psi_db_0
    # determine initial adiabatic wavefunction in fixed gauge
    psi_adb = auxilliary.vec_db_to_adb(psi_db, evecs_0)
    # initialize branches of classical coordinates
    z_branch = np.zeros((num_branches, *np.shape(z)),dtype=complex)
    zc_branch = np.zeros((num_branches, *np.shape(zc)),dtype=complex)
    z_branch[:] = z
    zc_branch[:] = zc
    # initialize outputs
    tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0, sim.tmax + sim.dt_bath, sim.dt_bath)
    ec = np.zeros((len(tdat)))
    eq = np.zeros((len(tdat)))
    pops_db = np.zeros((len(tdat), num_states))
    pops_db_fssh = np.zeros((len(tdat), num_states))
    # initial adiabatic density matrix
    rho_adb_0 = np.outer(psi_adb, np.conjugate(psi_adb))
    # initial wavefunction in branches
    psi_adb_branch = np.zeros((num_branches, num_states), dtype=complex)
    psi_adb_branch[:] = psi_adb
    # initial wavefunction as a delta function in each branch
    psi_adb_delta_branch = np.diag(np.ones(num_branches)).astype(complex)
    # transform to diabatic basis
    psi_db_branch = np.zeros_like(psi_adb_branch).astype(complex)
    psi_db_delta_branch = np.zeros_like(psi_adb_branch).astype(complex)
    for i in range(num_branches):
        psi_db_branch[i] = auxilliary.vec_adb_to_db(psi_adb_branch[i], evecs_0)
        psi_db_delta_branch[i] = auxilliary.vec_adb_to_db(psi_adb_delta_branch[i], evecs_0)
    # initialize phases on each branch
    phase_branch = np.zeros(num_branches)
    # initialize active surfaces
    act_surf_ind_branch = np.arange(0, num_branches, dtype=int)
    act_surf_branch = np.diag(np.ones(num_branches))
    # initialize Hamiltonian
    #h_q_branch = np.zeros((num_branches, num_states, num_states), dtype=complex)
    #h_q_branch[:] = sim.h_q()
    h_tot_branch = auxilliary.h_tot_branch(z_branch, zc_branch, h_q, sim.h_qc, num_branches, num_states, sim)
    #h_q_branch + auxilliary.h_qc_branch(z_branch, zc_branch, sim.h_qc, num_branches, num_states)
    # initialize eigenvalues and eigenvectors
    evals_branch = np.zeros((num_branches, num_states))
    evecs_branch = np.zeros((num_branches, num_states, num_states), dtype=complex)
    evals_branch[:] = evals_0
    evecs_branch[:] = evecs_0

    # initialize branch-pair eigenvalues and eigenvectors
    if sim.dmat_const > 0:
        u_ij = np.zeros((num_branches, num_branches, num_states, num_states), dtype=complex)
        e_ij = np.zeros((num_branches, num_branches, num_states))
        u_ij[:, :] = evecs_0
        e_ij[:, :] = evals_0
    ec_branch = np.zeros((num_branches))
    eq_branch = np.zeros((num_branches))
    for i in range(num_branches):
        ec_branch[i] = sim.h_c(z_branch[i], zc_branch[i], sim)
        eq_branch[i] = evals_branch[i][act_surf_ind_branch[i]]
    hop_count = 0
    t_ind = 0
    for t_bath_ind in np.arange(0, len(tdat_bath)):
        if t_ind == len(tdat):
            break
        if sim.branch_update == 2 and sim.dmat_const == 1: # update every bath timestep
            u_ij_previous = np.copy(u_ij)
            e_ij, u_ij = auxilliary.get_branch_pair_eigs(z_branch, zc_branch, u_ij_previous, h_q, sim)
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * sim.dt_bath:
            overlap = auxilliary.get_classical_overlap(z_branch, zc_branch, sim)
            rho_db = np.zeros((num_states, num_states), dtype=complex)
            rho_db_fssh = np.zeros((num_states, num_states), dtype=complex)
            # only update branches every output timestep and check that the local gauge is converged
            if sim.branch_update == 1 and sim.dmat_const == 1:
                u_ij_previous = np.copy(u_ij)
                e_ij, u_ij = auxilliary.get_branch_pair_eigs(z_branch, zc_branch, u_ij_previous, h_q, sim)
            if sim.branch_update == 0 and sim.dmat_const == 1:
                u_ij_previous = np.copy(u_ij)
            if sim.dmat_const == 1:
                for i in range(num_states):
                    for j in range(num_states):
                        if i != j:
                            a_i = act_surf_ind_branch[i]
                            a_j = act_surf_ind_branch[j]
                            if a_i == i and a_j == j:
                                if sim.branch_update == 0:
                                    branch_mat = h_q + sim.h_qc((z_branch[i] + z_branch[j])/2, (zc_branch[i] + zc_branch[j])/2, sim)
                                    e_ij[i,j], u_ij[i,j] = np.linalg.eigh(branch_mat)
                                    u_ij[i,j], _ = auxilliary.sign_adjust(u_ij[i,j], u_ij_previous[i,j], e_ij[i,j], \
                                                 (z_branch[i] + z_branch[j])/2, (zc_branch[i] + zc_branch[j])/2, sim)
                                for n in range(num_branches):
                                    for m in range(num_branches):
                                        rho_db[n, m] += u_ij[i,j][n, i]*rho_adb_0[i,j]*overlap[i,j]*\
                                            np.exp(-1.0j*(phase_branch[i] - phase_branch[j])) * np.conjugate(u_ij[i,j])[m, j]
                        if i == j:
                            a_i = act_surf_ind_branch[i]
                            e_ij[i,j], u_ij[i,j] = evals_branch[i], evecs_branch[i]
                            rho_adb_fssh = np.outer(psi_adb_branch[i], np.conjugate(psi_adb_branch[i]))
                            rho_adb_fssh[range(num_states), range(num_states)] = act_surf_branch[i]
                            rho_db_fssh += auxilliary.rho_0_adb_to_db(rho_adb_0[i,i]*rho_adb_fssh, u_ij[i,j])
                            for k in range(num_branches):
                                if a_i == k:
                                    for n in range(num_branches):
                                        for m in range(num_branches):
                                            rho_db[n, m] += u_ij[i,j][n, k]*rho_adb_0[i,i]*np.conjugate(u_ij[i,j])[m,k]
            if sim.dmat_const == 0:
                rho_adb = np.zeros((num_branches, num_states, num_states), dtype=complex)
                rho_adb_coh = np.zeros((num_states, num_states), dtype=complex)
                for i in range(num_states):
                    for j in range(num_states):
                        if i != j:
                            a_i = act_surf_ind_branch[i]
                            a_j = act_surf_ind_branch[j]
                            if a_i == i and a_j == j:
                                rho_adb_coh[i, j] = rho_adb_0[i, j] * overlap[i, j] * np.exp(-1.0j*(phase_branch[i] - phase_branch[j]))
                rho_diag = np.diag(rho_adb_0).reshape((-1,1))*act_surf_branch
                np.einsum('...jj->...j',rho_adb)[...] = rho_diag
                rho_adb = rho_adb + rho_adb_coh/num_branches
                for i in range(num_branches):
                    rho_db += auxilliary.rho_0_adb_to_db(rho_adb[i], evecs_branch[i])
                    rho_adb_fssh = np.outer(psi_adb_branch[i], np.conjugate(psi_adb_branch[i]))
                    rho_db_fssh += auxilliary.rho_0_adb_to_db(rho_adb_0[i,i]*rho_adb_fssh, evecs_branch[i])
            pops_db[t_ind] = np.real(np.diag(rho_db))
            pops_db_fssh[t_ind] = np.real(np.diag(rho_db_fssh))
            for i in range(num_branches):
                ec[t_ind] += sim.h_c(z_branch[i], zc_branch[i], sim)
                eq[t_ind] += evals_branch[i][act_surf_ind_branch[i]]
            e_tot_0 = ec[0] + eq[0]  # energy at t=0
            e_tot_t = ec[t_ind] + eq[t_ind]  # energy at t=t
            # check that energy is conserved within 1% of the initial classical energy
            if np.abs(e_tot_t - e_tot_0) > 0.01 * ec[0]:
                print('ERROR: energy not conserved! % error= ', 100 * np.abs(e_tot_t - e_tot_0) / ec[0])
            t_ind += 1
        fz_branch, fzc_branch = auxilliary.quantum_force_branch(evecs_branch, act_surf_ind_branch, z_branch, zc_branch, sim)
        for i in range(num_branches):
            z_branch[i], zc_branch[i] = auxilliary.rk4_c(z_branch[i], zc_branch[i],(fz_branch[i], fzc_branch[i]), sim.dt_bath, sim)
        evecs_branch_previous = np.copy(evecs_branch)
        # obtain branch eigenvalues and eigenvectors (sign adjust in function)
        evals_branch, evecs_branch, evecs_phases = auxilliary.get_branch_eigs(z_branch, zc_branch, evecs_branch_previous, h_q, sim)
        # check for trivial crossings
        if np.any(np.abs(evecs_phases) < 0.99):
            print('Warning: crossing')
        evals_exp_branch = np.exp(-1j * evals_branch * sim.dt_bath)
        rand = np.random.rand() # same random number for each branch
        for i in range(num_branches):
            # evolve wavefunctions in each branch
            diag_matrix = np.diag(evals_exp_branch[i])

            psi_adb_branch[i] = np.copy(auxilliary.vec_db_to_adb(psi_db_branch[i], evecs_branch[i]))
            psi_adb_delta_branch[i] = np.copy(auxilliary.vec_db_to_adb(psi_db_delta_branch[i], evecs_branch[i]))

            psi_adb_branch[i] = np.copy(np.dot(diag_matrix, psi_adb_branch[i]))
            psi_adb_delta_branch[i] = np.copy(np.dot(diag_matrix, psi_adb_delta_branch[i]))

            psi_db_branch[i] = auxilliary.vec_adb_to_db(psi_adb_branch[i], evecs_branch[i])
            psi_db_delta_branch[i] = auxilliary.vec_adb_to_db(psi_adb_delta_branch[i], evecs_branch[i])
            # compute hopping probabilities
            prod = np.matmul(np.conjugate(evecs_branch[i][:, act_surf_ind_branch[i]]), evecs_branch_previous[i])
            if sim.pab_cohere:
                hop_prob = -2 * np.real(prod * (psi_adb_branch[i]/psi_adb_branch[i][act_surf_ind_branch[i]]))
            if not sim.pab_cohere:
                hop_prob = -2 * np.real(prod * (psi_adb_delta_branch[i]/ psi_adb_delta_branch[i][act_surf_ind_branch[i]]))

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
                    dkj_z, dkj_zc = auxilliary.get_dab(evec_k, evec_j, ev_diff, z_branch[i], zc_branch[i], sim)
                    # check that nonadiabatic couplings are real-valued
                    dkj_q = np.sqrt(sim.h*sim.m / 2) * (dkj_z + dkj_zc)
                    dkj_p = np.sqrt(1 / (2 * sim.h*sim.m)) * 1.0j * (dkj_z - dkj_zc)
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
                    delta_zc = dkj_z
                    z_branch[i], zc_branch[i], hopped = sim.hop(z_branch[i], zc_branch[i], delta_z, delta_zc, ev_diff, sim)
                    if hopped:
                        act_surf_ind_branch[i] = k
                        act_surf_branch[i] = np.zeros_like(act_surf_branch[i])
                        act_surf_branch[i][act_surf_ind_branch[i]] = 1
                        hop_count += 1
                    break
    # save data
    traj.add_to_dic('pops_db', pops_db)
    traj.add_to_dic('pops_db_fssh', pops_db_fssh)
    traj.add_to_dic('t', tdat)
    traj.add_to_dic('eq', eq)
    traj.add_to_dic('ec', ec)
    end_time = time.time()
    msg = 'trial index: ' + str(traj.index) + ' hop count: ' + str(hop_count) + ' time: ' + str(
        end_time - start_time) + ' seed: ' + str(traj.seed)
    return traj, msg


@ray.remote
def fssh_dynamics(traj, sim):
    start_time = time.time()
    np.random.seed(traj.seed)
    #  initialize classical coordinates
    z, zc = sim.init_classical(sim)
    #  compute initial Hamiltonian
    h_q = sim.h_q(sim)
    h_tot = h_q + sim.h_qc(z, zc, sim)
    #  compute eigenvectors
    evals, evecs = np.linalg.eigh(h_tot)
    num_states = len(h_q)
    # compute initial gauge shift for real-valued derivative couplings
    dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals, evecs, z, zc, sim)
    # execute phase shift
    evecs = np.matmul(evecs, np.diag(np.conjugate(dab_q_phase)))
    # recalculate phases and check that they are zero
    dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals, evecs, z, zc, sim)
    if np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2) > 1e-10:
        # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
        print('Warning: phase init', np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2))
    #  initial wavefunction in diabatic basis
    psi_db = sim.psi_db_0
    # determine initial adiabatic wavefunction in fixed gauge
    psi_adb = auxilliary.vec_db_to_adb(psi_db, evecs)
    # initial wavefunction where it is only on active surface
    psi_adb_delta = np.zeros(num_states, dtype=complex)
    # determine initial active surface
    intervals = np.zeros(num_states)
    for n in range(num_states):
        intervals[n] = np.sum(np.real(np.abs(psi_adb) ** 2)[0:n + 1])
    rand_val = np.random.rand()
    # initialize active surface index
    act_surf_ind = np.arange(num_states)[intervals > rand_val][0]
    # intialize psi_adb_delta
    psi_adb_delta[act_surf_ind] = 1.0 + 0.0j
    # transform to diabatic basis
    psi_db_delta = auxilliary.vec_adb_to_db(psi_adb_delta, evecs)
    # initialize active surface
    act_surf = np.zeros(num_states, dtype=int)
    act_surf[act_surf_ind] = 1
    # initialize outputs
    tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0, sim.tmax + sim.dt_bath, sim.dt_bath)
    pops_db = np.zeros((len(tdat), num_states))
    ec = np.zeros((len(tdat)))
    eq = np.zeros((len(tdat)))
    # adjust h_q so that the initial quantum energy is always 0
    eq_init = evals[act_surf_ind]
    h_q = h_q - np.identity(num_states) * eq_init
    h_tot = h_q + sim.h_qc(z, zc, sim)
    # update eigenvalues
    evals, _ = np.linalg.eigh(h_tot)
    # begin timesteps
    t_ind = 0
    hop_count = 0
    for t_bath_ind in np.arange(0, len(tdat_bath)):
        if t_ind == len(tdat):
            break
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * sim.dt_bath:
            # compute adiabatic density matrix
            rho_adb = np.outer(psi_adb, np.conjugate(psi_adb))
            rho_adb[range(num_states), range(num_states)] = act_surf
            # transform to diabatic basis
            rho_db = auxilliary.rho_0_adb_to_db(rho_adb, evecs)
            # save populations
            pops_db[t_ind] = np.real(np.diag(rho_db))
            # save energies
            ec[t_ind] = sim.h_c(z, zc, sim)
            eq[t_ind] = evals[act_surf_ind]
            e_tot_0 = ec[0] + eq[0]  # energy at t=0
            e_tot_t = ec[t_ind] + eq[t_ind]  # energy at t=t
            # check that energy is conserved within 1% of the initial classical energy
            if np.abs(e_tot_t - e_tot_0) > 0.01 * ec[0]:
                print('ERROR: energy not conserved! % error= ', 100 * np.abs(e_tot_t - e_tot_0) / ec[0])
            t_ind += 1
        # compute quantum force
        fz, fzc = auxilliary.quantum_force(evecs[:, act_surf_ind], z, zc, sim)
        # evolve classical coordinates
        z, zc = auxilliary.rk4_c(z, zc, (fz, fzc), sim.dt_bath, sim)
        # evolve quantum subsystem saving previous eigenvector values
        evecs_previous = np.copy(evecs)
        h_tot = h_q + sim.h_qc(z, zc, sim)
        evals, evecs = np.linalg.eigh(h_tot)
        evecs, evec_phases = auxilliary.sign_adjust(evecs, evecs_previous, evals, z, zc, sim)
        evals_exp = np.exp(-1j * evals * sim.dt_bath)
        diag_matrix = np.diag(evals_exp)
        psi_adb = np.copy(np.dot(diag_matrix, auxilliary.vec_db_to_adb(psi_db, evecs)))
        psi_adb_delta = np.copy(np.dot(diag_matrix, auxilliary.vec_db_to_adb(psi_db_delta, evecs)))
        psi_db = auxilliary.vec_adb_to_db(psi_adb, evecs)
        psi_db_delta = auxilliary.vec_adb_to_db(psi_adb_delta, evecs)
        # compute random value
        rand = np.random.rand()
        # compute wavefunction overlaps
        prod = np.matmul(np.conj(evecs[:, act_surf_ind]), evecs_previous)
        # compute hopping probability
        if sim.pab_cohere:
            hop_prob = -2 * np.real(prod * (psi_adb / psi_adb[act_surf_ind]))
        else:
            hop_prob = -2 * np.real(prod * (psi_adb_delta / psi_adb_delta[act_surf_ind]))
        hop_prob[act_surf_ind] = 0
        bin_edge = 0
        # loop over states
        for k in range(len(hop_prob)):
            hop_prob[k] = auxilliary.nan_num(hop_prob[k])
            bin_edge = bin_edge + hop_prob[k]
            if rand < bin_edge:
                # compute nonadiabatic coupling d_{kj}= <k|\nabla H|j>/(e_{j} - e_{k})
                evec_k = evecs[:, act_surf_ind]
                evec_j = evecs[:, k]
                eval_k = evals[act_surf_ind]
                eval_j = evals[k]
                ev_diff = eval_j - eval_k
                # dkj_q is wrt q dkj_p is wrt p.
                dkj_z, dkj_zc = auxilliary.get_dab(evec_k, evec_j, ev_diff, z, zc, sim)
                # check that nonadiabatic couplings are real-valued
                dkj_q = np.sqrt(sim.h*sim.m / 2) * (dkj_z + dkj_zc)
                dkj_p = np.sqrt(1 / (2*sim.h*sim.m)) * 1.0j * (dkj_z - dkj_zc)
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
                delta_zc = dkj_z
                z, zc, hopped = sim.hop(z,zc,delta_z,delta_zc, ev_diff, sim)
                if hopped:
                    act_surf_ind = k
                    act_surf = np.zeros_like(act_surf)
                    act_surf[act_surf_ind] = 1
                    hop_count += 1
                break
    # save data
    traj.add_to_dic('pops_db', pops_db)
    traj.add_to_dic('t', tdat)
    traj.add_to_dic('eq', eq)
    traj.add_to_dic('ec', ec)
    end_time = time.time()
    msg = 'trial index: ' + str(traj.index) + ' hop count: ' + str(hop_count) + ' time: ' + str(
        end_time - start_time) + ' seed: ' + str(traj.seed)
    return traj, msg


@ray.remote
def mf_dynamics(traj, sim):
    start_time = time.time()
    np.random.seed(traj.seed)
    #  initialize classical coordinates
    z, zc = sim.init_classical(sim)
    #  compute initial Hamiltonian
    h_q = sim.h_q(sim)
    h_tot = h_q + sim.h_qc(z, zc, sim)
    num_states = len(h_q)
    # initial wavefunction in diabatic basis
    psi_db = sim.psi_db_0
    #  initialize outputs
    tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0, sim.tmax + sim.dt_bath, sim.dt_bath)
    pops_db = np.zeros((len(tdat), num_states))  # diabatic populations
    ec = np.zeros((len(tdat)))  # classical energy
    eq = np.zeros((len(tdat)))  # quantum energy
    # adjust h_q so that the initial quantum energy is always 0
    eq_init = np.real(np.matmul(np.conjugate(psi_db), np.matmul(h_tot, psi_db)))
    h_q = h_q - np.identity(num_states) * eq_init
    h_tot = h_q + sim.h_qc(z, zc, sim)
    t_ind = 0
    for t_bath_ind in np.arange(0, len(tdat_bath)):
        if t_ind == len(tdat):
            break
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * sim.dt_bath:
            # save diabatic populations
            pops_db[t_ind] = np.abs(psi_db) ** 2
            # save energies
            ec[t_ind] = sim.h_c(z, zc, sim)
            eq[t_ind] = np.real(np.matmul(np.conjugate(psi_db), np.matmul(h_tot, psi_db)))
            e_tot_0 = ec[0] + eq[0]  # energy at t=0
            e_tot_t = ec[t_ind] + eq[t_ind]  # energy at t=t
            # check that energy is conserved within 1% of the initial classical energy
            if np.abs(e_tot_t - e_tot_0) > 0.01 * ec[0]:
                print('ERROR: energy not conserved! % error= ', 100 * np.abs(e_tot_t - e_tot_0) / ec[0])
            t_ind += 1
        fz, fzc = auxilliary.quantum_force(psi_db, z, zc, sim)
        z, zc = auxilliary.rk4_c(z, zc, (fz, fzc), sim.dt_bath, sim)
        h_tot = h_q + sim.h_qc(z, zc, sim)
        psi_db = auxilliary.rk4_q(h_tot, psi_db, sim.dt_bath)
    # add data to trajectory object
    traj.add_to_dic('pops_db', pops_db)
    traj.add_to_dic('t', tdat)
    traj.add_to_dic('eq', eq)
    traj.add_to_dic('ec', ec)
    end_time = time.time()
    msg = 'trial index: ' + str(traj.index) + ' time: ' + str(np.round(end_time - start_time, 3)) + ' seed: ' + str(
        traj.seed)
    return traj, msg
