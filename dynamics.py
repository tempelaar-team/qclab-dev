import ray
import time
import dill as pickle
import path
import simulation
import numpy as np
import auxilliary

def run_dynamics(sim):
    start_time = time.time()
    ray.shutdown()
    ray.init(sim.cluster_args)
    if sim.num_procs > sim.num_trials:
        sim.num_procs = sim.num_trials
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
    seeds = np.array([n for n in np.arange(last_index, sim.trials + last_index)])
    for run in range(0, int(sim.num_trials / sim.num_procs)):
        index_list = [run * sim.num_procs + i + last_index for i in range(sim.num_procs)]
        seed_list = [seeds[run * sim.nprocs + i + last_index] for i in range(sim.nprocs)]
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
    dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, sim)
    # execute phase shift
    evecs_0 = np.matmul(evecs_0, np.diag(np.conjugate(dab_q_phase)))
    # recalculate phases and check that they are zero
    dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, sim)
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




    return traj, msg

@ray.remote
def fssh_dynamics(traj, sim):
    start_time = time.time()
    np.random.seed(traj.seed)
    #  initialize classical coordinates
    q, p = sim.init_classical()
    #  compute initial Hamiltonian
    h_q = sim.h_q()
    h_tot = h_q + sim.h_qc(q, p)
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
    # determine initial active surface
    intervals = np.zeros(num_states)
    for n in range(num_states):
        intervals[n] = np.sum(np.real(np.abs(psi_adb)**2)[0:n+1])
    rand_val = np.random.rand()
    # initialize active surface index
    act_surf_ind = np.arange(num_states)[intervals > rand_val][0]
    # initialize active surface
    act_surf = np.zeros(num_states, dtype=int)
    act_surf[act_surf_ind] = 1
    # initialize outputs
    tdat = np.arange(0,sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0, sim.tmax + sim.dt_bath, sim.dt_bath)
    pops_db = np.zeros((len(tdat), num_states))
    ec = np.zeros((len(tdat)))
    eq = np.zeros((len(tdat)))
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
            ec[t_ind] = sim.h_c(q, p)
            eq[t_ind] = np.real(np.matmul(np.conjugate(psi_db), np.matmul(h_tot, psi_db)))
            t_ind += 1
        # compute quantum force
        fq, fp = sim.quantumforce(evecs[:, act_surf_ind], sim)
        # evolve classical coordinates
        q, p = auxilliary.rk4_c(q, p, (fq, fp), sim.w_c, sim.dt_bath)
        # evolve quantum subsystem saving prevous eigenvector values
        evecs_previous = np.copy(evecs)
        h_tot = h_q + sim.h_qc(q, p)
        evals, evecs = np.linalg.eigh(h_tot)
        evecs, evec_phases = auxilliary.sign_adjust(evecs, evecs_previous, evals, sim)
        evals_exp = np.exp(-1j * evals * sim.dt_bath)
        diag_matrix = np.diag(evals_exp)
        psi_adb = np.dot(diag_matrix, psi_adb)
        psi_db = auxilliary.vec_adb_to_db(psi_adb, evecs)
        # compute random value
        rand = np.random.rand()
        # compute wavefunction overlaps
        prod = np.matmul(np.conj(evecs[:, act_surf_ind]), evecs_previous)
        # compute hopping probability
        hop_prob = -2*np.real(prod * (psi_adb / psi_adb[act_surf_ind]))
        hop_prob[act_surf_ind] = 0
        bin_edge = 0
        # loop over states
        for k in range(len(hop_prob)):
            hop_prob[k] = auxilliary.nan_num(hop_prob[k])
            bin_edge = bin_edge + hop_prob[k]
            if rand < bin_edge:
                # compute nonadiabatic couplings
                eig_k = evecs[:, act_surf_ind]
                eig_j = evecs[:, k]
                eigval_k = evecs[act_surf_ind]
                eigval_j = evecs[k]
                ev_diff = eigval_j - eigval_k
                dkkq, dkkp = auxilliary.get_dkk(eig_k, eig_j, ev_diff, sim)
                if np.abs(np.sin(np.angle(dkkq[np.argmax(np.abs(dkkq))]))) > 1e-2:
                    print('ERROR IMAGINARY DKKQ: \n', 'angle: ',
                          np.abs(np.sin(np.angle(dkkq[np.argmax(np.abs(dkkq))]))),
                          '\n magnitude: ', np.abs(dkkq[np.argmax(np.abs(dkkq))]),
                          '\n value: ', dkkq[np.argmax(np.abs(dkkq))])
                if np.abs(np.sin(np.angle(dkkq[np.argmax(np.abs(dkkq))]))) > 1e-2:
                    print('ERROR IMAGINARY DKKP: \n', 'angle: ',
                          np.abs(np.sin(np.angle(dkkp[np.argmax(np.abs(dkkp))]))),
                          '\n magnitude: ', np.abs(dkkp[np.argmax(np.abs(dkkp))]),
                          '\n value: ', dkkp[np.argmax(np.abs(dkkp))])
                # compute rescalings
                delta_q = np.real(dkkp)
                delta_p = np.real(dkkq)
                akkq = np.sum((1 / 2) * np.abs(
                    delta_p) ** 2)
                akkp = np.sum((1 / 2) * (np.nan_to_num(sim.w_c) ** 2) * np.abs(
                    delta_q) ** 2)
                bkkq = np.sum((p * delta_p))
                bkkp = -np.sum((np.nan_to_num(sim.w_c) ** 2) * q * delta_q)
                disc = (bkkq + bkkp) ** 2 - 4 * (akkq + akkp) * ev_diff
                if disc >= 0:
                    if bkkq + bkkp < 0:
                        gamma = (bkkq + bkkp) + np.sqrt(disc)
                    else:
                        gamma = (bkkq + bkkp) - np.sqrt(disc)
                    if akkp + akkq == 0:
                        gamma = 0
                    else:
                        gamma = gamma / (2 * (akkq + akkp))
                    # rescale classical coordinates
                    p = p - np.real(gamma) * delta_p
                    q = q + np.real(gamma) * delta_q
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
    return traj, msg

@ray.remte
def mf_dynamics(traj, sim):
    start_time = time.time()
    np.random.seed(traj.seed)
    #  initialize classical coordinates
    q, p = sim.init_classical()
    #  compute initial Hamiltonian
    h_q = sim.h_q()
    h_tot = h_q + sim.h_qc(q, p)
    num_states = len(h_q)
    # initial wavefunction in diabatic basis
    psi_db = sim.psi_db_0
    #  initialize outputs
    tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0, sim.tmax + sim.dt_bath, sim.dt_bath)
    pops_db = np.zeros((len(tdat), num_states))  # diabatic populations
    ec = np.zeros((len(tdat)))  # classical energy
    eq = np.zeros((len(tdat)))  # quantum energy

    t_ind = 0
    for t_bath_ind in np.arange(0, len(tdat_bath)):
        if t_ind == len(tdat):
            break
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * sim.dt_bath:
            ec[t_ind] = sim.h_c(q, p)
            eq[t_ind] = np.real(np.matmul(np.conjugate(psi_db),np.matmul(h_tot, psi_db)))
            pops_db[t_ind] = np.abs(psi_db)**2
            t_ind += 1
        fq, fp = sim.quantum_force(psi_db)
        q, p = auxilliary.rk4_c(q, p, (fq, fp), sim.w_c, sim.dt_bath)
        h_tot = h_q + sim.h_qc(q, p)
        psi_db = auxilliary.rk4_q(h_tot, psi_db, sim.dt_bath)
    # add data to trajectory object
    traj.add_to_dic('pops_db', pops_db)
    traj.add_to_dic('t', tdat)
    traj.add_to_dic('eq', eq)
    traj.add_to_dic('ec', ec)
    end_time = time.time()
    msg = 'trial index: ', + str(traj.index) + ' time: ' + str(np.round(end_time - start_time, 3)) + ' seed: ' + str(traj.seed)
    return traj, msg
