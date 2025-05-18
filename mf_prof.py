
import numpy as np 
from qc_lab.algorithms import FewestSwitchesSurfaceHopping
from qc_lab.algorithms import MeanField
from qc_lab.models import HolsteinLattice
from qc_lab.models import FMOComplex
from qc_lab import Simulation
from qc_lab.dynamics import serial_driver


# simulation settings 
simulation_settings = dict(dt = 0.01, dt_output=.1, tmax = 30, num_trajs = 250, batch_size = 250)
sim = Simulation(simulation_settings)
#model_parameters = dict(N = 4, j = 1.0,w = 1.0, g = 1, periodic_boundary = True)
#sim.model = HolsteinLatticeModel(model_parameters)
sim.model = FMOComplex({
    "temp": 1,
    "boson_mass": 1,
    "l_reorg": 35 * 0.00509506, # reorganization energy
    "W": 117 * 0.00509506, # characteristic frequency
    "A": 200,
})

if False:
    algorithm_settings = dict(fssh_deterministic=True, num_branches = 2, gauge_fixing=1)
    sim.algorithm = FewestSwitchesSurfaceHopping(algorithm_settings)
else:
    sim.algorithm = MeanField()


wf_db_0 = np.zeros(7) + 0.0j
wf_db_0[0] = 1.0+0.0j
sim.state.wf_db = wf_db_0

data = serial_driver(sim)
