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
    data_file = open(data_obj.filename, 'wb')
    pickle.dump(data_obj, data_file)
    data_file.close()
    return sim

@ray.remote
def cfssh_dynamics(traj, sim):
    return traj, msg

@ray.remote
def fssh_dynamics(traj, sim):
    return traj, msg

@ray.remte
def mf_dynamics(traj, sim):
    start_time = time.time()
    np.random.seed(traj.seed)
    #  initialize classical coordinates
    q, p = sim.init_classical()

    #  compute initial Hamiltonian
    h_q = sim.H_q()
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
            ec[t_ind] = sim.H_c(q, p)
            eq[t_ind] = np.real(np.matmul(np.conjugate(psi_db),np.matmul(h_tot, psi_db)))
            pops_db[t_ind] = np.abs(psi_db)**2
            t_ind += 1
        fq, fp = sim.quantum_force(psi_db)
        q, p = auxilliary.rk4_c(q, p, (fq, fp), sim.w_list, sim.dt_bath)
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
