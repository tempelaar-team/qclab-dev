from qclab.drivers.slurm_driver import dynamics_parallel_slurm
from qclab.drivers.ray_driver import dynamics_parallel_ray
from qclab.models.spin_boson import SpinBosonModel
from qclab.algorithms.fssh import FewestSwitchesSurfaceHoppingDynamics
import sys
import dill as pickle
import numpy as np

args = sys.argv[1:]

ntasks = args[0]
ncpus_per_task = args[1]


input_params = dict(temp = 1, V=0.5, E=0.5, A=100, W=0.1, l=0.02/4)
sim = SpinBosonModel(input_params = input_params)

sim.num_trajs = 200
sim.tmax=int(1/0.0260677)+1
sim.dt_output=0.01
sim.dt=1/(10*sim.w[-1])

sim.wf_db = np.zeros((sim.num_states),dtype=complex)
sim.wf_db[0] = 1

num_seeds = 100*sim.num_trajs
seeds = np.arange(0, num_seeds)



data, index = dynamics_parallel_slurm(algorithm=FewestSwitchesSurfaceHoppingDynamics,sim=sim,ntasks=ntasks,ncpus_per_task=ncpus_per_task,sub_driver=dynamics_parallel_ray)

filename = 'data_' + str(index) + '.out'
file = open(filename, 'wb')
pickle.dump(data, file)
file.close()
print('Finished ', index)
