from qclab.drivers.slurm_driver import dynamics_parallel_slurm
from qclab.drivers.ray_driver import dynamics_parallel_ray
from qclab.drivers.serial_driver import dynamics_serial 
from qclab.models.spin_boson import SpinBosonModel
from qclab.recipes import MeanFieldDynamicsRecipe
import qclab.auxiliary as auxilliary
import sys
import numpy as np

args = sys.argv[1:]

ntasks = int(args[0]) # number of instances in slurm array
ncpus_per_task = int(args[1])


input_params = dict(temp = 1, V=0.5, E=0.5, A=100, W=0.1, l=0.02/4)
model = SpinBosonModel(input_params = input_params)

model.wf_db = np.zeros((model.num_states),dtype=complex)
model.wf_db[0] = 1


recipe = MeanFieldDynamicsRecipe()

recipe.params.num_trajs = 100 # for now this has to be a multiple of ntasks
recipe.params.num_branches = 1
recipe.params.batch_size = 10
recipe.params.tmax = 100
recipe.params.dt_output = 1 
recipe.params.dt = 0.01

data, index = dynamics_parallel_slurm(recipe = recipe, model = model, ntasks = ntasks, ncpus_per_task=ncpus_per_task, sub_driver = dynamics_serial)

filename = 'data_' + str(index) + '.out'
data.save_as_h5(filename)
print('Finished ', index)
