import ray
import time
import dill as pickle
from os import path
import os
import simulation
import numpy as np
import auxilliary

def run_dynamics(sim):
    start_time = time.time()
    # make calculation directory if needed
    if not(os.path.exists(sim.calc_dir)):
        os.mkdir(sim.calc_dir)
    # initialize ray cluster
    ray.shutdown()
    ray.init(**sim.cluster_args)
    if sim.num_procs > sim.num_trajs:
        sim.num_procs = sim.num_trajs
    ray_sim = ray.put(sim) # put simulation object in shared memory
    data_filename = sim.calc_dir + '/data.out' # output data object
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
        if sim.dynamics_method == "MF":
            results = [mf_dynamics.remote(simulation.Trajectory(seed_list[i], index_list[i]), ray_sim) \
                       for i in range(sim.num_procs)]
        elif sim.dynamics_method == "FSSH":
            results = [fssh_dynamics.remote(simulation.Trajectory(seed_list[i], index_list[i]), ray_sim) \
                       for i in range(sim.num_procs)]
        elif sim.dynamics_method == "CFSSH":
            results = [cfssh_dynamics.remote(simulation.Trajectory(seed_list[i], index_list[i]), ray_sim) \
                       for i in range(sim.num_procs)]
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
def cfssh_dynamics(traj, sim):
    start_time = time.time()
    np.random.seed(traj.seed)
    # initialize classical coordinates
    q, p = sim.init_classical()
    # compute initial Hamiltonian
    h_q = sim.h_q()
    h_tot = h_q + sim.h_qc(q, p)
    # compute initial eigenvalues and eigenvectors
    evals_0, evecs_0 = np.linalg.eigh(h_tot)
    num_states = len(evals_0)
    num_branches = num_states
    # compute initial gauge shift for real-valued derivative couplings
    dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, sim.diff_vars)
    # execute phase shift
    evecs_0 = np.matmul(evecs_0, np.diag(np.conjugate(dab_q_phase)))
    # recalculate phases and check that they are zero
    dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, sim.diff_vars)
    if np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2) > 1e-10:
        print('Warning: phase init', np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2))
    #  initial wavefunction in diabatic basis
    psi_db = sim.psi_db_0
    # determine initial adiabatic wavefunction in fixed gauge
    psi_adb = auxilliary.vec_db_to_adb(psi_db, evecs_0)
    # initialize branches of classical coordinates
    q_branch = np.zeros((num_branches, *np.shape(q)))
    p_branch = np.zeros((num_branches, *np.shape(p)))
    q_branch[:] = q
    p_branch[:] = p
    # initialize outputs
    tdat = np.arange(0,sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0,sim.tmax + sim.dt_bath, sim.dt_bath)
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
    psi_adb_delta_branch = np.diag(np.ones(num_branches), dtype=complex)
    # transform to diabatic basis
    psi_db_branch = np.zeros_like(psi_adb_branch).astype(complex)
    psi_db_delta_branch = np.zeros_like(psi_adb_branch).astype(complex)
    for i in range(num_branches):
        psi_db_branch[i] = auxilliary.vec_adb_to_db(psi_adb_branch[i], evecs_0)
        psi_db_delta_branch[i] = auxilliary.vec_adb_to_db(psi_adb_delta_branch[i], evecs_0)


    # initialize active surfaces
    act_surf_ind_branch = np.arange(0, num_branches, dtype=int)
    act_surf_branch = np.diag(np.ones(num_branches))
    # initialize classical oscillator frequency
    w_c_branch = np.zeros((num_branches, *np.shape(sim.w_c)))
    # initialize Hamiltonian
    h_q_branch = np.zeros((num_branches, num_states, num_states), dtype=complex)
    h_q_branch[:] = sim.h_q()
    h_tot_branch = h_q_branch + auxilliary.h_qc_branch(q_branch, p_branch, sim.h_qc, num_branches, num_states)
    # initialize eigenvalues and eigenvectors
    evals_branch = np.zeros((num_branches, num_states))
    evecs_branch = np.zeros((num_branches, num_states, num_states), dtype=complex)
    evals_branch[:] = evals_0
    evecs_branch[:] = evecs_0
    if sim.dmat_const > 0:
        u_ij = np.zeros((num_branches, num_branches, num_states, num_states), dtype=complex)
        e_ij = np.zeros((num_branches, num_branches, num_states))
        u_ij[:, :] = evecs_0
        e_ij[:, :] = evals_0

    t_ind = 0
    msg = ''
    return traj, msg

@ray.remote
def fssh_dynamics(traj, sim):
    start_time = time.time()
    np.random.seed(traj.seed)
    #  initialize classical coordinates
    z, zc = sim.init_classical()
    #  compute initial Hamiltonian
    h_q = sim.h_q()
    h_tot = h_q + sim.h_qc(z, zc)
    #  compute eigenvectors
    evals, evecs = np.linalg.eigh(h_tot)
    num_states = len(h_q)
    # compute initial gauge shift for real-valued derivative couplings
    dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals, evecs, sim)
    # execute phase shift
    evecs = np.matmul(evecs, np.diag(np.conjugate(dab_q_phase)))
    # recalculate phases and check that they are zero
    dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals, evecs, sim)
    if np.sum(np.abs(np.imag(dab_q_phase))**2 + np.abs(np.imag(dab_p_phase))**2) > 1e-10:
        print('Warning: phase init', np.sum(np.abs(np.imag(dab_q_phase))**2 + np.abs(np.imag(dab_p_phase))**2) )
    #  initial wavefunction in diabatic basis
    psi_db = sim.psi_db_0
    # determine initial adiabatic wavefunction in fixed gauge
    psi_adb = auxilliary.vec_db_to_adb(psi_db, evecs)
    # initial wavefunction where it is only on active surface
    psi_adb_delta = np.zeros((num_states), dtype=complex)
    # determine initial active surface
    intervals = np.zeros(num_states)
    for n in range(num_states):
        intervals[n] = np.sum(np.real(np.abs(psi_adb)**2)[0:n+1])
    rand_val = np.random.rand()
    # initialize active surface index
    act_surf_ind = np.arange(num_states)[intervals > rand_val][0]
    # intialize psi_adb_delta
    psi_adb_delta[act_surf_ind] = 1.0+0.0j
    # transform to diabatic basis
    psi_db_delta = auxilliary.vec_adb_to_db(psi_adb_delta, evecs)
    # initialize active surface
    act_surf = np.zeros(num_states, dtype=int)
    act_surf[act_surf_ind] = 1
    # initialize outputs
    tdat = np.arange(0,sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0, sim.tmax + sim.dt_bath, sim.dt_bath)
    pops_db = np.zeros((len(tdat), num_states))
    ec = np.zeros((len(tdat)))
    eq = np.zeros((len(tdat)))
    # adjust h_q so that the initial quantum energy is always 0
    eq_init = evals[act_surf_ind]
    h_q = h_q - np.identity(num_states) * eq_init
    h_tot = h_q + sim.h_qc(z, zc)
    # update eigenvalues
    evals, _ = np.linalg.eigh(h_tot)
    # begin timesteps
    t_ind = 0
    hop_count = 0
    for t_bath_ind  in np.arange(0, len(tdat_bath)):
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
            ec[t_ind] = sim.h_c(z, zc)
            eq[t_ind] = evals[act_surf_ind]
            e_tot_0 = ec[0] + eq[0]  # energy at t=0
            e_tot_t = ec[t_ind] + eq[t_ind]  # energy at t=t
            # check that energy is conserved within 1% of the classical energy
            if np.abs(e_tot_t - e_tot_0) > 0.01 * ec[t_ind]:
                print('ERROR: energy not conserved! % error= ', 100 * np.abs(e_tot_t - e_tot_0) / ec[t_ind])
            t_ind += 1
        # compute quantum force
        fz, fzc = auxilliary.quantum_force(evecs[:, act_surf_ind], sim.diff_vars)
        # evolve classical coordinates
        z, zc = auxilliary.rk4_c(z, zc, (fz, fzc), sim.w_c, sim.dt_bath)
        # evolve quantum subsystem saving prevous eigenvector values
        evecs_previous = np.copy(evecs)
        h_tot = h_q + sim.h_qc(z, zc)
        evals, evecs = np.linalg.eigh(h_tot)
        evecs, evec_phases = auxilliary.sign_adjust(evecs, evecs_previous, evals, sim)
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
            hop_prob = -2*np.real(prod * (psi_adb / psi_adb[act_surf_ind]))
        else:
            hop_prob = -2*np.real(prod * (psi_adb_delta / psi_adb_delta[act_surf_ind]))
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
                dkj_z, dkj_zc = auxilliary.get_dab(evec_k, evec_j, ev_diff, sim.diff_vars)
                # check that nonadiabatic couplings are real-valued
                dkj_q = np.sqrt(sim.w/2)*(dkj_z + dkj_zc)
                dkj_p = np.sqrt(1/(sim.w*2))*1.0j*(dkj_z - dkj_zc)
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
                akj_z = np.real(np.sum(sim.w_c*delta_z*delta_zc))
                bkj_z = np.real(np.sum(1j*(sim.w_c*zc*dkj_zc - sim.w_c*z*dkj_z)))
                ckj_z = ev_diff
                disc = bkj_z**2 - 4*akj_z*ckj_z
                if disc >= 0:
                    if bkj_z < 0:
                        gamma = bkj_z + np.sqrt(disc)
                    else:
                        gamma = bkj_z - np.sqrt(disc)
                    if akj_z == 0:
                        gamma = 0
                    else:
                        gamma = gamma / (2*akj_z)
                    z = z - 1.0j*np.real(gamma)*delta_zc
                    zc = zc + 1.0j*np.real(gamma)*delta_z
                #delta_q = np.real(dkj_p)
                #delta_p = np.real(dkj_q)
                #akj_q = np.sum((1 / 2) * np.abs(
                #    delta_p) ** 2)
                #akj_p = np.sum((1 / 2) * (np.nan_to_num(sim.w_c) ** 2) * np.abs(
                #    delta_q) ** 2)
                #bkj_q = np.sum((p * delta_p))
                #bkj_p = -np.sum((np.nan_to_num(sim.w_c) ** 2) * q * delta_q)
                #disc = (bkj_q + bkj_p) ** 2 - 4 * (akj_q + akj_p) * ev_diff
                #if disc >= 0:
                #    if bkj_q + bkj_p < 0:
                #        gamma = (bkj_q + bkj_p) + np.sqrt(disc)
                #    else:
                #        gamma = (bkj_q + bkj_p) - np.sqrt(disc)
                #    if akj_p + akj_q == 0:
                #        gamma = 0
                #    else:
                #        gamma = gamma / (2 * (akj_q + akj_p))
                #    # rescale classical coordinates
                #    p = p - np.real(gamma) * delta_p
                #    q = q + np.real(gamma) * delta_q
                    # update active surface
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
    return traj, eq+ec#msg

@ray.remote
def mf_dynamics(traj, sim):
    start_time = time.time()
    np.random.seed(traj.seed)
    #  initialize classical coordinates
    z, zc = sim.init_classical()
    #  compute initial Hamiltonian
    h_q = sim.h_q()
    h_tot = h_q + sim.h_qc(z, zc)
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
    eq_init = np.real(np.matmul(np.conjugate(psi_db),np.matmul(h_tot, psi_db)))
    h_q = h_q - np.identity(num_states)*eq_init
    h_tot = h_q + sim.h_qc(z,zc)
    t_ind = 0
    for t_bath_ind in np.arange(0, len(tdat_bath)):
        if t_ind == len(tdat):
            break
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * sim.dt_bath:
            # save diabatic populations
            pops_db[t_ind] = np.abs(psi_db)**2
            # save energies
            ec[t_ind] = sim.h_c(z, zc)
            eq[t_ind] = np.real(np.matmul(np.conjugate(psi_db),np.matmul(h_tot, psi_db)))
            e_tot_0 = ec[0] + eq[0]  # energy at t=0
            e_tot_t = ec[t_ind] + eq[t_ind]  # energy at t=t
            # check that energy is conserved within 1% of the classical energy
            if np.abs(e_tot_t - e_tot_0) > 0.01 * ec[t_ind]:
                print('ERROR: energy not conserved! % error= ', 100 * np.abs(e_tot_t - e_tot_0) / ec[t_ind])
            t_ind += 1
        fz, fzc = auxilliary.quantum_force(psi_db, sim.diff_vars)
        z, zc = auxilliary.rk4_c(z, zc, (fz, fzc), sim.w_c, sim.dt_bath)
        h_tot = h_q + sim.h_qc(z, zc)
        psi_db = auxilliary.rk4_q(h_tot, psi_db, sim.dt_bath)
    # add data to trajectory object
    traj.add_to_dic('pops_db', pops_db)
    traj.add_to_dic('t', tdat)
    traj.add_to_dic('eq', eq)
    traj.add_to_dic('ec', ec)
    end_time = time.time()
    msg = 'trial index: ' + str(traj.index) + ' time: ' + str(np.round(end_time - start_time, 3)) + ' seed: ' + str(traj.seed)
    return traj, msg