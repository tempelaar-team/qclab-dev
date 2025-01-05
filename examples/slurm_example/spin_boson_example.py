import os
import numpy as np
from qclab import Simulation, Data
from qclab.models import SpinBosonModel
from qclab.algorithms import MeanField
from qclab.dynamics import slurm_driver 





simulation_parameters = dict(dt = 0.01,
                            dt_output=.01,
                            tmax = 10,
                            num_trajs = 10000,
                            batch_size = 100)

sim = Simulation(simulation_parameters)

sim.model = SpinBosonModel()

sim.algorithm = MeanField()

wf_db = np.zeros((2), dtype=complex)
wf_db[0] = 1.0+0.0j
sim.state.modify('wf_db', wf_db)

data, idx = slurm_driver(sim, num_tasks = 10)

data_file = 'data_'+str(idx)+'.h5'

if os.path.exists(data_file):
    data.add_data(Data().load_from_h5(data_file))

data.save_as_h5(data_file)
