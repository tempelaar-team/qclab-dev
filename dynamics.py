import ray
import time
import dill as pickle
import path
import simulation
import numpy as np

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
    return traj, msg
