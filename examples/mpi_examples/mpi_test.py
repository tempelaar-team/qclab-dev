import numpy as np
import matplotlib.pyplot as plt
from qc_lab import Simulation 
from qc_lab.models import SpinBosonModel
from qc_lab.algorithms import MeanField 
from qc_lab.dynamics import serial_driver, mpi_driver

# instantiate a simulation
sim = Simulation()

# change settings to customize simulation
sim.settings.num_trajs =4000
sim.settings.batch_size = 100
sim.settings.tmax = 10
sim.settings.dt = 0.001

# instantiate a model 
sim.model = SpinBosonModel()
# instantiate an algorithm 
sim.algorithm = MeanField()
# define an initial diabatic wavefunction 
sim.state.wf_db = np.array([1, 0], dtype=complex)

data = mpi_driver(sim)
print(data.data_dic['seed'])
